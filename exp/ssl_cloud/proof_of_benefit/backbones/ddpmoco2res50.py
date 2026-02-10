import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from models.moco import loader
from models.moco import builder
import pdb
import sys

from torch.utils.tensorboard import SummaryWriter

#from datasets.BigEarthNet.bigearthnet_dataset_seco import Bigearthnet
#from datasets.BigEarthNet.bigearthnet_dataset_seco_lmdb import LMDBDataset
from datasets.SSL4EO.ssl4eo_dataset_lmdb import LMDBDataset_moco_res50
#from models.rs_transforms_uint8 import RandomChannelDrop,RandomBrightness,RandomContrast,ToGray
#from cvtorchvision import cvtransforms
from opencv_transforms import transforms
#from torchsat.transforms import transforms_cls

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


#TODO reset the saved epochs
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR',
                    help='path to dataset',
                    default='/data/Phd_data/Projects/SSL/data/ssl4eo/lmdb_mm/', 
                    type=str) # TODO
parser.add_argument('--checkpoints', metavar='DIR',
                    help='path to checkpoints', # TODO
                    default='/data/Phd_data/Projects/SSL/output/ssl4eo/ddpmoco2res50')
parser.add_argument('--save_path', metavar='DIR', # TODO
                    help='path to save trained model',
                    default='/data/Phd_data/Projects/SSL/output/ssl4eo/ddpmoco2res50')
parser.add_argument('--bands', type=str, default='MM',
                    help='bands to process') #TODO B12                    
parser.add_argument('--lmdb', action='store_true', default=True,
                    help='use lmdb dataset') 
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
#TODO make sure that data loader sets slurm id to true in ssl4eodatasetlmdb.py file
#TODO also set in train_loader = torch.utils.data.DataLoader( pin memorz to true
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)') # TODO
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run') # TODO 100
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
#TODO make sure that 65536 modulo (batchsize divided by number of gpus) is 0 
# this works for 256 and 4 gpus
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', # TODO 256
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='/data/Phd_data/Projects/SSL/output/ssl4eo/ddpmoco2res50/checkpoint_0069.pth.tar', type=str, metavar='PATH', #/data/Phd_data/Projects/SSL/output/ssl4eo/ddpmoco2res50/checkpoint_0013.pth.tar
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training') # TODO -1
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default="file:///code/ddd/ddpmoco2res50", type=str,
                    help='url used to set up distributed training') #TODO 'tcp://224.66.41.62:23456'
parser.add_argument('--dist-backend', default='nccl', type=str, #TODO nccl
                    help='distributed backend')
parser.add_argument('--seed', default=42, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training', default=True) #TODO 

# moco specific configs:
parser.add_argument('--moco-dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--moco-k', default=65536, type=int,
                    help='queue size; number of negative keys (default: 65536)')
parser.add_argument('--moco-m', default=0.999, type=float,
                    help='moco momentum of updating key encoder (default: 0.999)')
parser.add_argument('--moco-t', default=0.2, type=float,
                    help='softmax temperature (default: 0.07)') #0.02

# options for moco v2
parser.add_argument('--mlp', action='store_true', default=True,
                    help='use mlp head')
parser.add_argument('--aug-plus', action='store_true', default=True,
                    help='use moco v2 data augmentation')
parser.add_argument('--cos', action='store_true', default=True,
                    help='use cosine lr schedule')

parser.add_argument('--normalize', action='store_true', default=False)
parser.add_argument('--mode', nargs='*', default=['s1','s2a']) #['s1','s2a'] #TODO ['s2c']
parser.add_argument('--dtype', type=str, default='uint8')
parser.add_argument('--season', type=str, default='augment')

parser.add_argument('--in_size', type=int, default=224)
parser.add_argument("--is_slurm_job", action='store_true', help="running in slurm")

class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, season='fixed'):
        self.base_transform = base_transform
        self.season = season

    def __call__(self, x):

        if self.season=='augment':
            season1 = np.random.choice([0,1,2,3])
            season2 = np.random.choice([0,1,2,3])
        elif self.season=='fixed':
            np.random.seed(42)
            season1 = np.random.choice([0,1,2,3])
            season2 = season1
        elif self.season=='random':
            season1 = np.random.choice([0,1,2,3])
            season2 = season1

        x1 = np.transpose(x[season1,:,:,:],(1,2,0))
        x2 = np.transpose(x[season2,:,:,:],(1,2,0))

        q = self.base_transform(x1)
        k = self.base_transform(x2)

        return [q, k]
    
def init_dist(args):
    if args.world_size == -1:
        args.world_size = torch.cuda.device_count()
    if args.rank == -1:
        args.rank = 0
    if args.dist_url == "env://":
        args.dist_url = "file:///code/ddd/sharedtest"
    shared_file = Path("/code/ddd/sharedtest")
    if shared_file.is_file():
            Path.unlink(shared_file)

def main():
    args = parser.parse_args()

    init_dist(args)
    '''
    if args.rank==0 and not os.path.isdir(args.checkpoints):
        os.makedirs(args.checkpoints,exist_ok=True)
    if args.rank==0:
        tb_writer = SummaryWriter(os.path.join(args.checkpoints,'log'))
    '''

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        '''
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
        '''
    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    #TODO
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    
    ### add slurm option ###
    args.is_slurm_job = "SLURM_JOB_ID" in os.environ
    if args.is_slurm_job:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.world_size = int(os.environ["SLURM_NNODES"]) * int(
            os.environ["SLURM_TASKS_PER_NODE"][0]
        )
    
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    
    ngpus_per_node = torch.cuda.device_count()

    if args.multiprocessing_distributed: # TODO deactivate parallel
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        #args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu  # gpu corresponds to process ID
    args.rank = args.gpu
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    # suppress printing if not master

    if args.multiprocessing_distributed and args.gpu != 0 or (args.is_slurm_job and args.rank != 0):
        def print_pass(*args):
            pass
        builtins.print = print_pass

    

    if args.distributed:
        #TODO
        #if args.dist_url == "env://" and args.rank == -1:
        #    args.rank = int(os.environ["RANK"])
        #if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
        #    args.rank = args.rank * ngpus_per_node + gpu
        #dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
        #                        world_size=args.world_size, rank=args.rank)
        #print(f"Init distributed process {args.gpu} with {args.dist_backend}, {args.dist_url}, {args.world_size}, {args.gpu}")
        # only the 0 changes
        #Init distributed process 0 with nccl, file:///code/ddd/sharedtest, 3, 0
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.gpu)
    
    # create tb_writer
    if args.rank==0 and not os.path.isdir(args.checkpoints):
        os.makedirs(args.checkpoints, exist_ok=True)
    if args.rank==0:
        tb_writer = SummaryWriter(os.path.join(args.checkpoints,'log'))
  
    # create model
    print("=> creating model '{}'".format(args.arch))
    model = builder.MoCo(
        models.__dict__[args.arch],
        args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp, bands=args.bands)
    
    print('model created.')

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.

        ### add slurm option ###
        if args.is_slurm_job:         
            args.gpu_to_work_on = args.rank % torch.cuda.device_count()
            torch.cuda.set_device(args.gpu_to_work_on)
            model.cuda()
            model = nn.parallel.DistributedDataParallel(model,device_ids=[args.gpu_to_work_on])   
            print('model distributed.')          
        elif args.gpu is not None: # we go in here
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], output_device=args.gpu)
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        device = torch.device('cuda')
        model.to(device)
        print("Running single GPU without data parallelism")
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    
    
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    ### load dataset
    
    lmdb = args.lmdb
    
    if args.bands == 'B13':    
        n_channels = 13
    elif args.bands == 'B12':
        n_channels = 12
    elif args.bands == 'MM':
        n_channels = 14

    if args.dtype=='uint8':
        from models.rs_transforms_uint8 import RandomChannelDrop,RandomBrightness,RandomContrast,ToGray
    else:
        from models.rs_transforms_float32 import RandomChannelDrop,RandomBrightness,RandomContrast,ToGray
        
    train_transforms = transforms.Compose([
        #cvtransforms.Resize(128),
        transforms.RandomResizedCrop(args.in_size, scale=(0.2, 1.)),
        transforms.RandomApply([
            RandomBrightness(0.4),
            RandomContrast(0.4)
        ], p=0.8),
        transforms.RandomApply([ToGray(n_channels)], p=0.2),
        transforms.RandomApply([loader.GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),       
        transforms.ToTensor()
        #cvtransforms.RandomApply([RandomChannelDrop(min_n_drop=1, max_n_drop=6)], p=0.5),        
        ])
    
    train_dataset = LMDBDataset_moco_res50(
        lmdb_file=args.data,
        s12a_transform=TwoCropsTransform(train_transforms,season=args.season),
        normalize=args.normalize,
        dtype=args.dtype,
        mode=args.mode
    )   

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    if args.distributed:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)
    
    print('start training...')
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)
        
        # train for one epoch
        loss,top1,top5 = train(train_loader, model, criterion, optimizer, epoch, args)

        if args.rank==0:    
            tb_writer.add_scalar('loss',loss,global_step=epoch,walltime=None)
            tb_writer.add_scalar('acc1',top1,global_step=epoch,walltime=None)
            tb_writer.add_scalar('acc5',top5,global_step=epoch,walltime=None)

        if epoch%2==1:
            if args.distributed:
                if args.rank==0:
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'arch': args.arch,
                        'state_dict': model.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                    }, is_best=False, filename=os.path.join(args.checkpoints,'checkpoint_{:04d}.pth.tar'.format(epoch)))
            else:
               save_checkpoint({
                        'epoch': epoch + 1,
                        'arch': args.arch,
                        'state_dict': model.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                    }, is_best=False, filename=os.path.join(args.checkpoints,'checkpoint_{:04d}.pth.tar'.format(epoch)))
 

    print('Training finished.')
    if args.rank==0:
        tb_writer.close()

def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))
    
    # switch to train mode
    model.train()
    
    end = time.time()
    
    for i, s2c in enumerate(train_loader):
        images = s2c
        # measure data loading time
        data_time.update(time.time() - end)
        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)

        # compute output
        output, target = model(im_q=images[0].cuda(), im_k=images[1].cuda())
        loss = criterion(output, target)

        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images[0].size(0))
        top1.update(acc1[0], images[0].size(0))
        top5.update(acc5[0], images[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
    '''
    if args.rank==0:    
        tb_writer.add_scalar('loss',losses.avg,global_step=epoch,walltime=None)
        tb_writer.add_scalar('acc1',top1.avg,global_step=epoch,walltime=None)
        tb_writer.add_scalar('acc5',top5.avg,global_step=epoch,walltime=None)
    '''
    return losses.avg, top1.avg, top5.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'
    
def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    
if __name__ == '__main__':
    ss_time = time.time()
    
    # moco-v2
    #args.mlp = True
    #args.moco_t = 0.2
    #args.aug_plus = True
    #args.cos = True
    main()
    print('total time: %s.' % (time.time()-ss_time))