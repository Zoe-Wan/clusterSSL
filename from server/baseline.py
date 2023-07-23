import torch
from network.head import LinearHead
from util.torch_dist_sum import *
from util.meter import *
import time
from network.backbone import *
# from util.EMA import EMA
from util.accuracy import accuracy
from dataset.data import *
import torch.nn.functional as F
from util.dist_init import dist_init
from torch.nn.parallel import DistributedDataParallel
import argparse
import math
from util.mixup import Mixup
from torchvision import datasets
import os
from dataset.imagenet import * 
from dataset.augmentation import * 
from util.loss import SoftTargetCrossEntropy
from torch.utils.tensorboard import SummaryWriter 
 
parser = argparse.ArgumentParser()
parser.add_argument('--port', type=int, default=23457)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--bs', type=int, default=256)
parser.add_argument('--backbone', type=str, default='resnet50')
parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--use_fp16', default=False, action='store_true')
parser.add_argument('--checkpoint', type=str, default='')
parser.add_argument('--wd', type=float, default=1e-4)
parser.add_argument('--mixup', default=False, action='store_true')
parser.add_argument('--tensorboard', type=str,default='')
# parser.add_argument('--use_ema', default=False, action='store_true')
args = parser.parse_args()

epochs = args.epochs
warm_up = 5

mixup_func = Mixup(mixup_alpha=0.1, cutmix_alpha=1, switch_prob=1, label_smoothing=0)
# writer = SummaryWriter(args.tensorboard)
# random erase


def train(train_loader, model, local_rank, rank, optimizer, lr, epoch, scaler, ema=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    #top1 = AverageMeter('Acc@1', ':6.2f')
    #top5 = AverageMeter('Acc@5', ':6.2f') 
    progress = ProgressMeter(
        len(train_loader),
        [batch_time,data_time,losses
     #     ,top1, top5
],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (samples, targets) in enumerate(train_loader):
        #  adjust_learning_rate(optimizer, epoch, lr, i, len(train_loader))
        # measure data loading time
        data_time.update(time.time() - end)
        
        samples = samples.cuda(local_rank, non_blocking=True)
        targets = targets.cuda(local_rank, non_blocking=True)
        if args.mixup:
            samples, targets = mixup_func(samples, targets)
        
        with torch.cuda.amp.autocast(enabled=args.use_fp16):
            output = model(samples)
            if args.mixup:
                criterion=nn.BCEWithLogitsLoss()
            #print(targets)
            else:
                criterion= nn.CrossEntropyLoss()           
            loss = criterion(output, targets)
            # writer.add_scalar('train_loss', loss, epoch)
            #acc1,acc5=accuracy(output, targets, topk=(1,5))
            #top1.update(acc1[0],samples.size(0))
            #top5.update(acc5[0],samples.size(0))
            
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # if args.use_ema:
        #     ema.update(model.parameters())
        losses.update(loss.item(), samples.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % 20 == 0 and rank == 0:
            progress.display(i)



@torch.no_grad()
def test(test_loader, model, local_rank, ema=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(test_loader),
        [batch_time, data_time, top1, top5],
        prefix='Test: ')

    # switch to train mode
    model.eval()

    end = time.time()
    for i, (img, target) in enumerate(test_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        img = img.cuda(local_rank, non_blocking=True)
        target = target.cuda(local_rank, non_blocking=True)
        # if args.use_ema:
        #     ema.store(model.parameters())
        #     ema.copy(model.parameters())
        # compute output
        output = model(img)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        
        top1.update(acc1[0], img.size(0))
        top5.update(acc5[0], img.size(0))
        # if args.use_ema:
        #     ema.copy_back(model.parameters())
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % 20 == 0 and local_rank == 0:
            progress.display(i)

    sum1, cnt1, sum5, cnt5 = torch_dist_sum(local_rank, top1.sum, top1.count, top5.sum, top5.count)
    top1_acc = sum(sum1.float()) / sum(cnt1.float())
    top5_acc = sum(sum5.float()) / sum(cnt5.float())

    return top1_acc, top5_acc


def main():
    rank, local_rank, world_size = dist_init(args.port)
    batch_size = args.bs // world_size
    num_workers = 6
    lr = args.lr * batch_size * world_size / 256

    if rank == 0:
        print(args)
    
    if args.dataset == 'cifar10':
        num_classes = 10
    elif args.dataset == 'cifar100':
        num_classes = 100
    elif args.dataset == 'imagenet':
        num_classes = 1000
    model = backbone_dict[args.backbone]()
    model = LinearHead(net=model, dim_in=dim_dict[args.backbone], dim_out=num_classes, fix=False)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.cuda(local_rank)
    model = DistributedDataParallel(model, device_ids=[local_rank],find_unused_parameters=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=args.wd, momentum=0.9)
    # optimizer = Lamb(model.parameters(), lr=lr, weight_decay=args.wd)

    torch.backends.cudnn.benchmark = True
    train_aug, test_aug = get_train_augment(args.dataset), get_test_augment(args.dataset)

    if args.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root='data', download=True, transform=train_aug)
        test_dataset = datasets.CIFAR10(root='data', train=False, download=True, transform=test_aug)
    elif args.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root='data', download=True, transform=train_aug)
        test_dataset = datasets.CIFAR100(root='data', train=False, download=True, transform=test_aug)
    elif args.dataset == 'imagenet': 
        train_dataset = Imagenet(mode='train', max_class=1000)
        test_dataset = Imagenet(mode='val')  
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
        num_workers=num_workers, pin_memory=True, sampler=train_sampler, drop_last=True)
    
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False, drop_last=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=(test_sampler is None),
        num_workers=num_workers, pin_memory=True, sampler=test_sampler)

    scaler = torch.cuda.amp.GradScaler(enabled=args.use_fp16)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1) 
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs*len(train_loader))
    # if args.use_ema:
    #     ema = EMA(model.paramete rs(), decay_rate=0.995, num_updates=0)
    

    if not os.path.exists('checkpoints') and rank == 0:
        os.makedirs('checkpoints')

    checkpoint_path = os.path.join('checkpoints/', args.checkpoint)
    if os.path.exists(checkpoint_path):
        checkpoint =  torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        # if args.use_ema:
        #     ema.load_state_dict(checkpoint['ema'])
        scaler.load_state_dict(checkpoint['scaler'])
        start_epoch = checkpoint['epoch']
    else:
        start_epoch = 0
    
    # ema.to(torch.device('cuda:'+str(local_rank)))
    best_top1 = 0
    best_top5 = 0
    for epoch in range(start_epoch, epochs):
        train_sampler.set_epoch(epoch)
        # train(train_loader, model, local_rank, rank, optimizer, lr, epoch, scaler, ema)
        train(train_loader, model, local_rank, rank, optimizer, lr, epoch, scaler)
        
        # top1, top5 = test(test_loader, model, local_rank,ema)
        top1, top5 = test(test_loader, model, local_rank)
        # writer.add_scalar('top1', top1, epoch)
        scheduler.step()
        best_top1 = max(best_top1, top1)
        best_top5 = max(best_top5, top5)
        if rank == 0:
            print('Epoch:{} * Acc@1 {:.3f} Acc@5 {:.3f} Best_Acc@1 {:.3f} Best_Acc@5 {:.3f}'.format(epoch, top1, top5, best_top1, best_top5))
            
            state_dict =  {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                # 'ema':ema.state_dict(),
                'scheduler':scheduler.state_dict(),
                'scaler': scaler.state_dict(),
                'epoch': epoch + 1
            }
            # if args.use_ema:
            #     state_dict['ema']=ema.state_dict()
            torch.save(state_dict, checkpoint_path)

if __name__ == "__main__":
    main()


