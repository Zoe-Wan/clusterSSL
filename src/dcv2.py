import argparse
import math
import os
import shutil
import time
from logging import getLogger



import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim

# from torch.optim.lr_scheduler import CosineAnnealingLR
# from torch.optim.lr_scheduler import StepLR
from util.maskcrossentropy import *

from util.dist_init import dist_init
from network.head_p import Phead_with_pseudo
from network.head import Phead
from network.backbone import *
from dataset.imagenet import *
from dataset.cifar import *
# from util.meter import *
# from apex.parallel.LARC import LARC
from optim.lars import LARS
from scipy.sparse import csr_matrix
from util.torch_dist_sum import *
from util.accuracy import *
from util.utils import (
    initialize_exp,
    restart_from_checkpoint,
    fix_random_seeds,
    AverageMeter,
    init_distributed_mode,
)
# from src.multicropdataset import MultiCropDataset
# import src.resnet50 as resnet_models

logger = getLogger()

parser = argparse.ArgumentParser(description="Implementation of DeepCluster-v2")

#########################
#### data parameters ####
#########################
parser.add_argument("--dataset", type=str, default="imagenet")
# parser.add_argument("--nmb_crops", type=int, default=[2], nargs="+",
#                     help="list of number of crops (example: [2, 6])")
# parser.add_argument("--size_crops", type=int, default=[224], nargs="+",
#                     help="crops resolutions (example: [224, 96])")
# parser.add_argument("--min_scale_crops", type=float, default=[0.14], nargs="+",
#                     help="argument in RandomResizedCrop (example: [0.14, 0.05])")
# parser.add_argument("--max_scale_crops", type=float, default=[1], nargs="+",
#                     help="argument in RandomResizedCrop (example: [1., 0.14])")
parser.add_argument("--num_classes", default=1000, type=int)
#########################
## dcv2 specific params #
#########################
# parser.add_argument("--crops_for_assign", type=int, nargs="+", default=[0],
#                     help="list of crops id used for computing assignments")
parser.add_argument("--temperature", default=0.1, type=float,
                    help="temperature parameter in training loss")
parser.add_argument("--feat_dim", default=256, type=int,
                    help="feature dimension")
parser.add_argument("--K", default=3000, type=int,
                    help="cluster num of K-means")
parser.add_argument("--alpha", default=1, type=float)

#########################
#### optim parameters ###
#########################
parser.add_argument("--epochs", default=100, type=int,
                    help="number of total epochs to run")
parser.add_argument("--batch_size", default=32, type=int,
                    help="batch size per gpu, i.e. how many unique instances per gpu")
parser.add_argument("--base_lr", default=0.1, type=float, help="base learning rate") # 后面有改

parser.add_argument("--final_lr", type=float, default=0, help="final learning rate")
parser.add_argument("--freeze_prototypes_niters", default=1e10, type=int,
                    help="freeze the prototypes during this many iterations from the start")
parser.add_argument("--wd", default=1e-4, type=float, help="weight decay")
parser.add_argument("--warmup_epochs", default=10, type=int, help="number of warmup_epochs epochs")
parser.add_argument("--start_warmup", default=0, type=float,
                    help="initial warmup_epochs learning rate")
# parser.add_argument("--scheduler", default="step", type=str)
parser.add_argument("--no_scaler", action='store_true')
parser.add_argument("--use_LARS", action='store_true')
parser.add_argument("--no_cluster", action='store_true')
parser.add_argument("--use_mask", action='store_true')
parser.add_argument("--l1_norm", action='store_true')
parser.add_argument("--use_ema", action='store_true')
parser.add_argument("--kmeans_iters", default=10, type=int)



#########################
#### other parameters ###
#########################
parser.add_argument("--arch", default="resnet50", type=str, help="convnet architecture")

parser.add_argument("--port", default=23456, type=int)
parser.add_argument("--hidden_mlp", default=4096, type=int,
                    help="hidden layer dimension in projection head")
parser.add_argument("--workers", default=6, type=int,
                    help="number of data loading workers")
parser.add_argument("--checkpoint_freq", type=int, default=25,
                    help="Save the model periodically")
parser.add_argument("--sync_bn", type=str, default="pytorch", help="synchronize bn")
parser.add_argument("--syncbn_process_group_size", type=int, default=8, help=""" see
                    https://github.com/NVIDIA/apex/blob/master/apex/parallel/__init__.py#L58-L67""")
parser.add_argument("--dump_path", type=str, default=".",
                    help="experiment dump path for checkpoints and log")
parser.add_argument("--seed", type=int, default=31, help="seed")
parser.add_argument("--ckpt", type=str, default=".")
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

def main():
    global args
    args = parser.parse_args()
    args.rank, args.local_rank, args.world_size = dist_init(port=args.port)
    # args.warmup_epochs=10
    # print(args.rank)
    # args.base_lr = 0.1

    fix_random_seeds(args.seed)
    logger, training_stats = initialize_exp(args, "epoch", "loss")

    # build data 这里需要修改一下ImageNet的dataset，主要是改这个return index=True

    if args.dataset == 'cifar10':
        train_dataset = cifar(mode='train',classes=10)
        test_dataset = cifar(mode='test',classes=10)
    elif args.dataset == 'cifar100':
        train_dataset = cifar(mode='train',classes=100)
        test_dataset = cifar(mode='test',classes=100)
    elif args.dataset == 'imagenet': 
        train_dataset = Imagenet(mode='train',return_index=True)
        test_dataset = Imagenet(mode='val',return_index=True)

    sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True
    )

    logger.info("Building data done with {} images loaded.".format(len(train_dataset)))

    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False, drop_last=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        sampler=test_sampler,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True
    )
    scaler=None
    if not args.no_scaler:
        scaler = torch.cuda.amp.GradScaler(enabled=not args.no_scaler)



    model = backbone_dict[args.arch]()
    if not args.no_cluster:
        model = Phead_with_pseudo(net=model, dim_in=dim_dict[args.arch], dim_feat=args.feat_dim,
        hidden_mlp=args.hidden_mlp, dim_out=args.num_classes, pseudo = args.K*args.num_classes, use_ema = args.use_ema)
    else:
        model = Phead(net=model, dim_in=dim_dict[args.arch], dim_out=args.num_classes)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model = model.cuda()

    if args.rank == 0:
        logger.info(model)
    logger.info("Building model done.")



    # build optimizer
    if args.use_LARS:
        param_dict = {}
        for k, v in model.named_parameters():
            param_dict[k] = v
        bn_params = [v for n, v in param_dict.items() if ('bn' in n or 'bias' in n)]
        rest_params = [v for n, v in param_dict.items() if not ('bn' in n or 'bias' in n)]
        optimizer = torch.optim.SGD(
            [{'params': bn_params, 'weight_decay': 0, 'ignore': True },
            {'params': rest_params, 'weight_decay': 1e-6, 'ignore': False}], 
            lr=args.base_lr, momentum=0.9, weight_decay=1e-6
        )
        optimizer = LARS(optimizer,eps=0.0)
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.base_lr, momentum=0.9, weight_decay=args.wd
        )
    #




    logger.info("Building optimizer done.")

    # wrap model
    model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[args.local_rank],
        find_unused_parameters=True,
    )

    # optionally resume from a checkpoint
    to_restore = {"epoch": 0}
    if not args.no_scaler:
        restart_from_checkpoint(
            args.ckpt,
            run_variables=to_restore,
            state_dict=model,
            optimizer=optimizer,
            scaler=scaler
        )
    else:
        restart_from_checkpoint(
            args.ckpt,
            run_variables=to_restore,
            state_dict=model,
            optimizer=optimizer,
        )   

    start_epoch = to_restore["epoch"]
    # build the memory bank
    local_memory_index=None
    local_memory_embeddings=None
    if not args.no_cluster:
        mb_path = os.path.join(args.dump_path, "mb" + str(args.rank) + ".pth")
        if os.path.isfile(mb_path):
            mb_ckp = torch.load(mb_path)
            local_memory_index = mb_ckp["local_memory_index"]
            local_memory_embeddings = mb_ckp["local_memory_embeddings"]
        else:
            local_memory_index, local_memory_embeddings,local_memory_labels = init_memory(train_loader, model)

    cudnn.benchmark = True
    if args.evaluate:
        acc1,acc5 = test(test_loader, model, epoch=-1)
        if args.rank==0:
            logger.info("Validate Acc@1: {0} Acc@5: {1}".format(
                    acc1,acc5
                ))
        return

    # need restoring
    best_acc1=0
    best_acc5=0



    for epoch in range(start_epoch, args.epochs):
        

        # train the network for one epoch
        if args.rank==0:
            logger.info("============ Starting epoch %i ... ============" % epoch)

        # set sampler
        train_loader.sampler.set_epoch(epoch)

        # train the network
        scores, local_memory_index, local_memory_embeddings,local_memory_labels = train(
            train_loader,
            model,
            optimizer,
            epoch,
            local_memory_index,
            local_memory_embeddings,
            local_memory_labels,
            scaler
        )
        training_stats.update(scores)
        acc1,acc5 = test(test_loader,model,epoch)
        best_acc1 = max(acc1, best_acc1)
        best_acc5 = max(acc5, best_acc5)

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
            if not args.no_scaler:
                save_dict['scaler'] = scaler.state_dict()
                
            torch.save(
                save_dict,
                os.path.join(args.dump_path, "checkpoint.pth.tar"),
            )
            if epoch % args.checkpoint_freq == 0 or epoch == args.epochs - 1:
                shutil.copyfile(
                    os.path.join(args.dump_path, "checkpoint.pth.tar"),
                    os.path.join(args.dump_checkpoints, "ckp-" + str(epoch) + ".pth"),
                )
        if not args.no_cluster:       
            torch.save({"local_memory_embeddings": local_memory_embeddings,
                        "local_memory_index": local_memory_index
                        }, mb_path)


def test(loader,model,epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
                                                                                               
    cross_entropy = nn.CrossEntropyLoss(ignore_index=-100)

    end = time.time()
    start_idx = 0
    model.eval()
    for it, (idx, inputs, labels) in enumerate(loader): # 注意这里是enumerate遍历，下面的init里是直接遍历
        with torch.no_grad():
            # measure data loading time
            data_time.update(time.time() - end)

            # ============ multi-res forward passes ... ============
            if not args.no_cluster:
                emb, output, poutput = model(inputs)
            else:
                output = model(inputs)
            labels = labels.cuda(non_blocking=True)
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            loss = cross_entropy(output, labels)
            losses.update(loss)
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
                "Loss {loss.val:.3f} ({loss.avg:.3f})\t"
                "Acc@1 {acc1.val:.3f}({acc1.avg:.3f})\t"
                "Acc@5 {acc5.val:.3f}({acc5.avg:.3f})"
                .format(
                    epoch,
                    it,
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    acc1=top1,
                    acc5=top5
                )
            )
    sum1, cnt1, sum5, cnt5 = torch_dist_sum(args.local_rank, top1.sum, top1.count, top5.sum, top5.count)
    top1_acc = sum(sum1.float()) / sum(cnt1.float())
    top5_acc = sum(sum5.float()) / sum(cnt5.float())
    return top1_acc, top5_acc


def train(loader, model, optimizer, epoch, local_memory_index, local_memory_embeddings, local_memory_labels, scaler):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses1 = AverageMeter()
    losses2 = AverageMeter()
    model.train()
    cross_entropy = nn.CrossEntropyLoss(ignore_index=-100)
    # assignments是顺序保存的
    if not args.no_cluster:
        assignments = cluster_memory(model, local_memory_index,
         local_memory_embeddings,local_memory_labels, loader.dataset ,nmb_kmeans_iters=args.kmeans_iters)
    logger.info('Clustering for epoch {} done.'.format(epoch))

    end = time.time()
    start_idx = 0
    for it, (idx, inputs, labels) in enumerate(loader): # 注意这里是enumerate遍历，下面的init里是直接遍历
        # measure data loading time

        data_time.update(time.time() - end)

        # # update learning rate
        
        if epoch<args.warmup_epochs:
            iteration = epoch * len(loader) + it
            for param_group in optimizer.param_groups:
                param_group["lr"] = args.base_lr*iteration/(args.warmup_epochs*len(loader))
        else:
            iteration = (epoch-args.warmup_epochs) * len(loader) + it
            for param_group in optimizer.param_groups:
                param_group["lr"] = 0.5 * args.base_lr * (1 + math.cos(math.pi * iteration / (len(loader) * (args.epochs - args.warmup_epochs))))

        bs = inputs.size(0)

        if args.use_ema:
            cur_itr = it + epoch * len(loader)
            total_itr = args.epochs * len(loader)
            m =  1 - (1 - 0.996) * (math.cos(math.pi * cur_itr / float(total_itr)) + 1) / 2
            model.module.momentum_update_key_encoder(m)

        with torch.cuda.amp.autocast(enabled=not args.no_scaler):
            # ============ multi-res forward passes ... ============
            if not args.no_cluster:
                emb, output, poutput = model(inputs)
            else:
                output = model(inputs)
        
            # ============ deepcluster-v2 loss ... ============
            # loss = 0
            labels = labels.cuda(non_blocking=True)
            loss1 = cross_entropy(output, labels)
            
            if not args.no_cluster:
                scores = poutput / args.temperature
                targets = assignments[idx].cuda(non_blocking=True)


                if not args.use_mask:
                    loss2 = cross_entropy(scores, targets)
                else:
                    loss2 = cross_entropy_with_mask(scores,targets,labels,args.num_classes,args.K)
                loss = (loss1+args.alpha*loss2)
                emb = emb.detach().float()  
            else:
                loss = loss1

        

        # ============ backward and optim step ... ============
        optimizer.zero_grad()

        if not args.no_scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # cancel some gradients
        for name, p in model.named_parameters():
            if "pfc" in name:
                p.grad = None

        
        if not args.no_scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()


        # ============ update memory banks ... ============
        if not args.no_cluster:
            local_memory_index[start_idx : start_idx + bs] = idx
            local_memory_embeddings[start_idx : start_idx + bs] = emb
            local_memory_labels[start_idx : start_idx + bs] = labels

            start_idx += bs

        # ============ misc ... ============
        losses.update(loss.item(), inputs.size(0))
        losses1.update(loss1.item(), inputs.size(0))
        if not args.no_cluster:
            losses2.update(loss2.item(), inputs.size(0))
        else:
            losses2.update(0, inputs.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        if args.rank ==0 and it % 50 == 0:
            logger.info(
                "Epoch: [{0}][{1}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Loss1 {loss1.val:.4f} ({loss1.avg:.4f})\t"
                "Loss2 {loss2.val:.4f} ({loss2.avg:.4f})\t"
                "Lr: {lr:.4f}".format(
                    epoch,
                    it,
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    loss1=losses1,
                    loss2=losses2,
                    lr=optimizer.param_groups[0]["lr"],
                )
            )
    
    return (epoch, losses.avg), local_memory_index, local_memory_embeddings,local_memory_labels


def init_memory(dataloader, model):
    size_memory_per_process = len(dataloader) * args.batch_size
    local_memory_index = torch.zeros(size_memory_per_process).long().cuda()
    local_memory_embeddings = torch.zeros(size_memory_per_process, args.feat_dim).cuda()
    local_memory_labels = torch.zeros(size_memory_per_process).long().cuda()
    start_idx = 0
    with torch.no_grad():
        logger.info('Start initializing the memory banks')
        for index, inputs, labels in dataloader:
            nmb_unique_idx = inputs.size(0)
            index = index.cuda(non_blocking=True)

            with torch.cuda.amp.autocast(enabled=not args.no_scaler):
                inp = inputs.cuda(non_blocking=True)
                embs = model(inp)[0]


            # fill the memory bank
            local_memory_index[start_idx : start_idx + nmb_unique_idx] = index

            local_memory_embeddings[start_idx : start_idx + nmb_unique_idx] = embs

            local_memory_labels[start_idx : start_idx + nmb_unique_idx] = labels

            start_idx += nmb_unique_idx
    logger.info('Initializion of the memory banks done.')
    return local_memory_index, local_memory_embeddings, local_memory_labels

def cluster_memory(model, local_memory_index, local_memory_embeddings, local_memory_labels, dataset, nmb_kmeans_iters=10):

    assignments = -100 * torch.ones(len(dataset)).long()
    centroids = torch.empty(args.K*args.num_classes, args.feat_dim).cuda(non_blocking=True)
    

    with torch.no_grad():
        local_assignments = torch.empty(len(local_memory_embeddings)).long().cuda(non_blocking=True)
        for label in range(args.num_classes):
            _mask = local_memory_labels==label
            local_memory_index[_mask]

            centroids_label = torch.empty(args.K,args.feat_dim).cuda(non_blocking=True)
            if args.rank == 0:
                random_idx = torch.randperm(len(local_memory_embeddings[_mask]))[:args.K]
                assert len(random_idx) >= args.K, "please reduce the number of centroids"
                centroids_label = local_memory_embeddings[_mask][random_idx]
            dist.broadcast(centroids_label, 0)

            with torch.cuda.amp.autocast(enabled=not args.no_scaler):
                for n_iter in range(nmb_kmeans_iters + 1):
                    # E step
                    dot_products = torch.mm(local_memory_embeddings[_mask], centroids_label.t())
                    _, local_assignments[_mask] = dot_products.max(dim=1)

                    # finish
                    if n_iter == nmb_kmeans_iters:
                        break

                    # M step
                    where_helper = get_indices_sparse(local_assignments[_mask].cpu().numpy())
                    counts = torch.zeros(args.K).cuda(non_blocking=True).int()
                    emb_sums = torch.zeros(args.K, args.feat_dim).cuda(non_blocking=True)
                    for k in range(len(where_helper)):
                        if len(where_helper[k][0]) > 0:
                            emb_sums[k] = torch.sum(
                                local_memory_embeddings[_mask][where_helper[k][0]],
                                dim=0,
                            )
                            counts[k] = len(where_helper[k][0])
                    dist.all_reduce(counts)
                    mask = counts > 0
                    dist.all_reduce(emb_sums)
                    centroids_label[mask] = emb_sums[mask] / counts[mask].unsqueeze(1)

                    # normalize centroids
                    centroids_label = nn.functional.normalize(centroids_label, dim=1, p=2)
                    centroids[label*args.K:(label+1)*args.K] = centroids_label
                local_assignments[_mask]+=args.K*label


        # gather the assignments
        assignments_all = torch.empty(args.world_size, local_assignments.size(0),
                                        dtype=local_assignments.dtype, device=local_assignments.device)
        assignments_all = list(assignments_all.unbind(0))
        dist_process = dist.all_gather(assignments_all, local_assignments, async_op=True)
        dist_process.wait()
        assignments_all = torch.cat(assignments_all).cpu()

        # gather the indexes
        indexes_all = torch.empty(args.world_size, local_memory_index.size(0),
                                    dtype=local_memory_index.dtype, device=local_memory_index.device)
        indexes_all = list(indexes_all.unbind(0))
        dist_process = dist.all_gather(indexes_all, local_memory_index, async_op=True)
        dist_process.wait()
        indexes_all = torch.cat(indexes_all).cpu()

        # log assignments
        assignments[indexes_all] = assignments_all

        model.module.pfc.weight.copy_(centroids)



        # 保存assignments和centroids
        if args.rank == 0:

            torch.save({"centroids": centroids,
                        "assignments": assignments
                        }, os.path.join(args.dump_path, "centroid.pth"))


    return assignments


def get_indices_sparse(data):
    cols = np.arange(data.size)
    M = csr_matrix((cols, (data.ravel(), cols)), shape=(int(data.max()) + 1, data.size))
    return [np.unravel_index(row.data, data.shape) for row in M]


if __name__ == "__main__":
    main()
