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

from src.utils import (
    bool_flag,
    initialize_exp,
    restart_from_checkpoint,
    fix_random_seeds,
    AverageMeter,
    dist_init,
    accuracy,
    torch_dist_sum
    
)
from src.imagenet import ImagenetContrastive,Imagenet
from src.Phead import Phead_with_pseudo,Phead
from src.resnet import resnet50


logger = getLogger()

parser = argparse.ArgumentParser(description="Implementation of SwAV")

#########################
#### data parameters ####
#########################


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
parser.add_argument("--queue_length", type=int, default=15*256,
                    help="length of the queue (0 for no queue)")
parser.add_argument("--epoch_queue_starts", type=int, default=15,
                    help="from this epoch, we start using a queue")
parser.add_argument("--freeze_prototypes_niters", default=313, type=int,
                    help="freeze the prototypes during this many iterations from the start,1281167/4096=312.7")
parser.add_argument("--use_swav", action='store_true')


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
parser.add_argument("--hidden_mlp", default=4096, type=int,
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
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')


def main():
    global args
    args = parser.parse_args()
    # init_distributed_mode(args)
    args.rank,args.local_rank,args.world_size = dist_init(args.port)
    fix_random_seeds(args.seed)
    logger, training_stats = initialize_exp(args, "epoch", "loss")

    # build data
    train_dataset = Imagenet(
        mode="train",
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
    model = resnet50()
    if args.use_swav:
        model = Phead_with_pseudo(
            net=model, 
            dim_in=2048, 
            dim_feat=args.feat_dim,
            hidden_mlp=args.hidden_mlp, 
            dim_out=1000, 
            pseudo = args.K
            )
    else:
        model = Phead(
            net=model, 
            dim_in=2048, 
            dim_feat=args.feat_dim,
            hidden_mlp=args.hidden_mlp, 
            dim_out=1000, 
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
        # device_ids=[args.gpu_to_work_on]
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



    if args.use_swav:
        # build the queue
        # queue is a feature bank, 
        # but I can't figure out why a feature bank can work when the model is changing every batch
        # 我知道了，论文不在一开始就使用feature bank 而是在模型相对稳定之后再使用

        queue = None
        queue_path = os.path.join(args.dump_path, "queue" + str(args.rank) + ".pth")
        if os.path.isfile(queue_path):
            queue = torch.load(queue_path)["queue"]
        # the queue needs to be divisible by the batch size
        args.queue_length -= args.queue_length % (args.batch_size * args.world_size)


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

        if args.use_swav:
            # optionally starts a queue 
            # 当模型训练一段时间，稳定后可以开启feature bank，当到了epoch_queue_start时且queue=None时初始化queue
            # 因此queue_length和start_epoch不挂钩
            if args.queue_length > 0 and epoch >= args.epoch_queue_starts and queue is None:
                queue = torch.zeros(
                    
                    args.queue_length // args.world_size,
                    args.feat_dim,
                ).cuda()

        # train the network

        scores, queue = train(train_loader, model, optimizer, epoch, queue, scaler)
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
        if args.use_swav:
            if queue is not None:
                torch.save({"queue": queue}, queue_path)

def test(loader,model,epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
                                                                                               
    loss_fn = nn.CrossEntropyLoss()

    end = time.time()

    model.eval()
    for it, (img, label) in enumerate(loader): # 注意这里是enumerate遍历，下面的init里是直接遍历
        with torch.no_grad():
            # measure data loading time

            data_time.update(time.time() - end)

            # ============ multi-res forward passes ... ============
            if args.use_swav:
                emb, output, poutput = model(img)
            else:
                output = model(img)
            label = label.cuda(non_blocking=True)
            acc1, acc5 = accuracy(output, label, topk=(1, 5))
            loss = loss_fn(output, label)
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

def train(loader, model, optimizer, epoch, queue, scaler):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    conlosses = AverageMeter()
    suplosses = AverageMeter()
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    use_the_queue = False

    end = time.time()
    for it, (img, label) in enumerate(loader):
        label = label.cuda(args.local_rank, non_blocking=True)
        img = img.cuda(args.local_rank, non_blocking=True)
        # img2 = img2.cuda(args.local_rank, non_blocking=True)
        bs = img.size(0)
        
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

        if args.use_swav:
            # normalize the prototypes
            with torch.no_grad():
                w = model.module.pfc.weight.data.clone()
                w = nn.functional.normalize(w, dim=1, p=2)
                model.module.pfc.weight.copy_(w)

            # ============ multi-res forward passes ... ============

            emb, y, py = model(img)
            emb = emb.detach().float()

            # ============ swav loss ... ============
            loss = 0

            with torch.no_grad():
                sample = py.detach()
                
                if queue is not None:
                    if use_the_queue or not torch.all(queue[-1, :] == 0):
                        use_the_queue = True
                        # use feature bank 
                        sample = torch.cat((torch.mm(queue,model.module.pfc.weight.t()),py))
                    
                    # feature bank enqueue and dequeue
                    queue[bs:] = queue[-bs].clone()
                    queue[:bs] = emb

                q = distributed_sinkhorn(sample)[-bs:]
            x = F.log_softmax(py/args.temperature,dim=1)
            conloss = torch.mean(torch.sum(q*x,dim=1))

        else:
            # ============ baseline forward  ... ============
            y = model(img)
            # ============ baseline loss  ... ============
            conloss = 0
        suploss = loss_fn(y,label)
        loss = suploss+args.alpha*conloss

        # ============ backward and optim step ... ============
        optimizer.zero_grad()
        scaler.scale(loss).backward()

        # cancel gradients for the prototypes

        if args.use_swav:
            if iteration < args.freeze_prototypes_niters:
                # 只在第一个epoch冻结聚类中心
                for name, p in model.named_parameters():
                    if "pfc" in name:
                        p.grad = None

        scaler.step(optimizer)
        scaler.update()

        # ============ misc ... ============
        losses.update(loss.item(), img.size(0))
        conlosses.update(conloss.item(), img.size(0))
        suplosses.update(suploss.item(), img.size(0))

        batch_time.update(time.time() - end)
        end = time.time()
        if args.rank ==0 and it % 50 == 0:
            logger.info(
                "Epoch: [{0}][{1}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Data {data_time.val:.3f} ({data_time.avg:.3f})\t" 
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "ConLoss {conloss.val:.4f} ({conloss.avg:.4f})\t"
                "SupLoss {suploss.val:.4f} ({suploss.avg:.4f})\t"
                "Lr: {lr:.4f}".format(
                    epoch,
                    it,
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    conloss=conlosses,
                    suploss=suplosses,
                    lr=optimizer.param_groups[0]["lr"],
                )
            )
    return (epoch, losses.avg), queue


@torch.no_grad()
def distributed_sinkhorn(out, label):
    Q = torch.exp(out / args.epsilon).t() # Q is K-by-B for consistency with notations from our paper
    B = Q.shape[1] * args.world_size # number of samples to assign
    K = Q.shape[0] # how many prototypes

    # make the matrix sums to 1
    sum_Q = torch.sum(Q)
    dist.all_reduce(sum_Q)
    Q /= sum_Q
    # 
    for it in range(args.sinkhorn_iterations):
        # 相当于是，有B张图片和K个类别，以及每张图片属于某个类别的“真实分布”，如何保证满足图片和类别都满足其先验分布的基础上尽可能拟合这一“真实分布”
        # 先验假设：存储量和需求量满足均匀分布，即，在数据空间中一张随机图片属于任意pseudo类的概率相当，每个图片在数据空间中的概率相当
        # K可以和class挂钩，可以假设大类内的小类概率相同。如果总样本有N个，第i大类的样本有N_i个，每个大类有K个小类，那么属于第i大类的小类的先验概率为N_i/N*K
        # 问题在于这个“总样本”应该是数据集中的样本还是Batch中的样本
        # B仍然平均，1/B
        # normalize each row: total weight per prototype must be 1/K 
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        dist.all_reduce(sum_of_rows)
        Q /= sum_of_rows
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    Q *= B # the colomns must sum to 1 so that Q is an assignment
    return Q.t()


if __name__ == "__main__":
    main()
