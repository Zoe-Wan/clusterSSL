# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import math
import os
import shutil
import time
from logging import getLogger

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torchvision.transforms as transforms
from src.utils import (
    bool_flag,
    initialize_exp,
    restart_from_checkpoint,
    fix_random_seeds,
    AverageMeter,
    dist_init,
#    accuracy,
    torch_dist_sum,
    GaussianBlur
)
from src.cifar import ContrastiveCifar,Cifar
from src.imagenet import ImagenetContrastive,Imagenet
from src.Phead import Phead_with_pseudo,Phead
from src.resnet import resnet50


logger = getLogger()

parser = argparse.ArgumentParser(description="Implementation of SwAV")

#########################
#### data parameters ####
#########################
parser.add_argument("--use_cifar", action='store_true')


#########################
## swav specific params #
#########################
parser.add_argument("--alpha", default=0.5, type=float)
parser.add_argument("--temperature", default=0.1, type=float,
                    help="temperature parameter in training loss")
parser.add_argument("--epsilon", default=0.05, type=float,
                    help="regularization parameter for Sinkhorn-Knopp algorithm")
parser.add_argument("--sinkhorn_iterations", default=3, type=int,
                    help="number of iterations in Sinkhorn-Knopp algorithm")
parser.add_argument("--feat_dim", default=256, type=int,
                    help="feature dimension")
parser.add_argument("--K", default=3000, type=int,
                    help="number of prototypes")
parser.add_argument("--simloss", action='store_true')



#########################
#### optim parameters ###
#########################
parser.add_argument("--epochs", default=100, type=int,
                    help="number of total epochs to run")
parser.add_argument("--batch_size", default=32, type=int,
                    help="batch size per gpu, i.e. how many unique instances per gpu")
parser.add_argument("--base_lr", default=0.1, type=float, help="base learning rate")
parser.add_argument("--final_lr", type=float, default=0, help="final learning rate")

parser.add_argument("--wd", default=1e-4, type=float, help="weight decay")
parser.add_argument("--warmup_epochs", default=10, type=int, help="number of warmup epochs")
parser.add_argument("--start_warmup", default=0, type=float,
                    help="initial warmup learning rate")

#########################
#### dist parameters ###
#########################
parser.add_argument("--port", default=23456, type=int)

#########################
#### other parameters ###
#########################
# parser.add_argument("--arch", default="resnet50", type=str, help="convnet architecture")
parser.add_argument("--hidden_mlp", default=2048, type=int,
                    help="hidden layer dimension in projection head")
parser.add_argument("--workers", default=6, type=int,
                    help="number of data loading workers")
parser.add_argument("--checkpoint_freq", type=int, default=25,
                    help="Save the model periodically")
parser.add_argument("--syncbn_process_group_size", type=int, default=8, help=""" see
                    https://github.com/NVIDIA/apex/blob/master/apex/parallel/__init__.py#L58-L67""")
parser.add_argument("--dump_path", type=str, default=".",
                    help="experiment dump path for checkpoints and log")
parser.add_argument("--seed", type=int, default=31, help="seed")

parser.add_argument("--ckpt", type=str, default=".")
parser.add_argument('--e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')


def main():
    global args
    args = parser.parse_args()
    # init_distributed_mode(args)
    args.rank,args.local_rank,args.world_size = dist_init(args.port)
    fix_random_seeds(args.seed)
    logger, training_stats = initialize_exp(args, "epoch", "loss")
    if args.use_cifar:
        args.num_class = 10
        train_dataset = ContrastiveCifar(mode = "train",classes=10)
        test_dataset = Cifar(mode="test",classes=10)
    else:
        args.num_class = 1000
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
        augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([
               transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]
        # build data
        train_dataset = ImagenetContrastive(
            mode="train",
            aug = transforms.Compose(augmentation)
        )

        test_dataset = Imagenet(
            mode="val",
        )
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True
    )
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        sampler=test_sampler,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info("Building data done with {} images loaded.".format(len(train_dataset)))

    # build model
    dim_in=2048
    if args.use_cifar:
        model = resnet18_cifar()
        dim_in=512
    else:
        model = resnet50()
        dim_in=2048
    model = Phead_with_pseudo(
            net=model, 
            dim_in=dim_in, 
            hidden_mlp=args.hidden_mlp, 
            pred_dim=args.feat_dim, 
            pseudo = args.K
            )
    # synchronize batch norm layers
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    # copy model to GPU
    model = model.cuda()
    if args.rank == 0:
        logger.info(model)
    logger.info("Building model done.")

    # build optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.base_lr,
        momentum=0.9,
        weight_decay=args.wd,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=True)

    # wrap model
    model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[args.local_rank]
    )

    # optionally resume from a checkpoint
    to_restore = {"epoch": 0}

    restart_from_checkpoint(
        args.ckpt,
        run_variables=to_restore,
        state_dict=model,
        optimizer=optimizer,
        scaler=scaler
    )
 
    start_epoch = to_restore["epoch"]




    best_acc1=0
    best_acc5=0


    cudnn.benchmark = True
    if args.evaluate:
        acc1,acc5 = test(test_loader, model, epoch=-1)
        if args.rank==0:
            logger.info("Validate Acc@1: {0} Acc@5: {1}".format(
                    acc1,acc5
                ))
        return

    for epoch in range(start_epoch, args.epochs):

        # train the network for one epoch
        logger.info("============ Starting epoch %i ... ============" % epoch)

        # set sampler
        train_loader.sampler.set_epoch(epoch)



        # train the network

        scores = train(train_loader, model, optimizer, epoch, scaler)
        acc1,acc5 = test(test_loader,model,epoch)
        best_acc1 = max(acc1, best_acc1)
        best_acc5 = max(acc5, best_acc5)
        training_stats.update(scores) # logger

        # save checkpoints
        if args.rank == 0:
            logger.info(
                "Epoch: [{0}]\t"
                "Acc@1 {1}, Acc@5 {2}\t"
                "Best Acc@1 {3}, Best Acc@5 {4}"
                .format(
                    epoch,
                    acc1,acc5,best_acc1,best_acc5
                )
            )
            save_dict = {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            torch.save(
                save_dict,
                os.path.join(args.dump_path, "checkpoint.pth.tar"),
            )
            if epoch % args.checkpoint_freq == 0 or epoch == args.epochs - 1:
                shutil.copyfile(
                    os.path.join(args.dump_path, "checkpoint.pth.tar"),
                    os.path.join(args.dump_checkpoints, "ckp-" + str(epoch) + ".pth"),
                )

def test(loader,model,epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    # losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
                                                                                               
    loss_fn = nn.CrossEntropyLoss()

    end = time.time()

    model.eval()
    for it, (img1, label) in enumerate(loader): # 注意这里是enumerate遍历，下面的init里是直接遍历
        with torch.no_grad():
            # measure data loading time

            data_time.update(time.time() - end)

            # ============ multi-res forward passes ... ============
            emb, poutput = model(img1)
            
            label = label.cuda(non_blocking=True)
            acc1, acc5 = accuracy(poutput, label, topk=(1, 5))
            # loss = loss_fn(output, label)
            # losses.update(loss)
            top1.update(acc1[0])
            top5.update(acc5[0])


        # ============ misc ... ============
        batch_time.update(time.time() - end)
        end = time.time()
        if args.rank ==0 and it % 50 == 0:
            logger.info(
                "Epoch: [{0}][{1}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                # "Loss {loss.val:.3f} ({loss.avg:.3f})\t"
                "Acc@1 {acc1.val:.3f}({acc1.avg:.3f})\t"
                "Acc@5 {acc5.val:.3f}({acc5.avg:.3f})"
                .format(
                    epoch,
                    it,
                    batch_time=batch_time,
                    data_time=data_time,
                    # loss=losses,
                    acc1=top1,
                    acc5=top5
                )
            )
    sum1, cnt1, sum5, cnt5 = torch_dist_sum(args.local_rank, top1.sum, top1.count, top5.sum, top5.count)
    top1_acc = sum(sum1.float()) / sum(cnt1.float())
    top5_acc = sum(sum5.float()) / sum(cnt5.float())
    return top1_acc, top5_acc

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        target = target.view(1,-1).expand_as(pred)
        correct = torch.logical_and(pred>=target*args.K , pred<(target+1)*args.K)
        # correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def train(loader, model, optimizer, epoch, scaler):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    # conlosses = AverageMeter()
    # suplosses = AverageMeter()
    loss_fn = nn.CosineSimilarity(dim=1).cuda(args.local_rank)

    model.train()
    use_the_queue = False

    end = time.time()
    for it, (img1, img2, label) in enumerate(loader):
        label = label.cuda(args.local_rank, non_blocking=True)
        img1 = img1.cuda(args.local_rank, non_blocking=True)
        img2 = img2.cuda(args.local_rank, non_blocking=True)
        bs = img1.size(0)
        
        # measure data loading time
        data_time.update(time.time() - end)

        # update learning rate
        if epoch<args.warmup_epochs:
            iteration = epoch * len(loader) + it
            for param_group in optimizer.param_groups:
                param_group["lr"] = args.base_lr*iteration/(args.warmup_epochs*len(loader))
        else:
            iteration = (epoch-args.warmup_epochs) * len(loader) + it
            for param_group in optimizer.param_groups:
                param_group["lr"] = 0.5 * args.base_lr * (1 + math.cos(math.pi * iteration / (len(loader) * (args.epochs - args.warmup_epochs))))

        with torch.cuda.amp.autocast(enabled=True):
            emb1, py1 = model(img1)
            emb1 = emb1.detach().float()
            emb2, py2 = model(img2)
            emb2 = emb2.detach().float()

            loss = 0
            
            if args.simloss:
                with torch.no_grad():
                    emb1 = F.normalize(emb1*label_mask,dim=1)
                    emb2 = F.normalize(emb2*label_mask,dim=1)

                py1 = F.normalize(py1,dim=1)
                py2 = F.normalize(py2,dim=1)
                loss = -((py1*emb2).mean()+(py2*emb1).mean())/2

            else:
                with torch.no_grad():
                    label_mask = make_one_hot(label.view(-1, 1),args.num_class,args.K//args.num_class)
                    emb1 = F.softmax(emb1.view(bs,-1,args.K/args.num_class)/args.temperature,dim=2).view(bs,-1)*label_mask
                    emb2 = F.softmax(emb2.view(bs,-1,args.K/args.num_class)/args.temperature,dim=2).view(bs,-1)*label_mask
                
                py1 = F.log_softmax(py1/args.temperature, dim=1)
                py2 = F.log_softmax(py2/args.temperature, dim=1)
                loss = -((py1*emb2).sum(dim=1).mean()+(py2*emb1).sum(dim=1).mean())/2 


        # ============ backward and optim step ... ============
        optimizer.zero_grad()
        scaler.scale(loss).backward()


        scaler.step(optimizer)
        scaler.update()

        # ============ misc ... ============
        losses.update(loss.item(), img1.size(0))


        batch_time.update(time.time() - end)
        end = time.time()
        if args.rank ==0 and it % 50 == 0:
            logger.info(
                "Epoch: [{0}][{1}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Data {data_time.val:.3f} ({data_time.avg:.3f})\t" 
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                # "ConLoss {conloss.val:.4f} ({conloss.avg:.4f})\t"
                # "SupLoss {suploss.val:.4f} ({suploss.avg:.4f})\t"
                "Lr: {lr:.4f}".format(
                    epoch,
                    it,
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    # conloss=conlosses,
                    # suploss=suplosses,
                    lr=optimizer.param_groups[0]["lr"],
                )
            )
    return (epoch, losses.avg)

@torch.no_grad()
def make_one_hot(label,classes,rep=1):
    
    n = label.size()[0]
    one_hot = torch.FloatTensor(n, classes).zero_().to(label.device)
    target = one_hot.scatter_(1, label.data, 1)
    target = target.unsqueeze(2).repeat(1,1,rep).view(target.size(0),-1)
    return target



if __name__ == "__main__":
    main()



