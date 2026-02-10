"""
This file applies cloudfree downstream tasks on ssl4eo backbones. This acquires performance of SSL methods 
trained on cloudfree data and cloudfree downstream tasks. Downstream tasks with cloudy data should
perform worse than the benchmark values reported here to proof the benefit.
"""

import torch 
import sys
#pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
#pip install opencv-contrib-python
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models

## change01 ##
#sys.path.append("Z:/Playground/install/opencv_transforms_torchvision-master/cvtorchvision")
#from cvtorchvision import cvtransforms
from opencv_transforms import transforms as cvtransforms
import time
import os
import math
import pdb
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score
import numpy as np
import argparse
import builtins

sys.path.append("Z:\\Projects\\ssl\\src")
sys.path.append("/work/Projects/ssl/src")
from exp.ssl_cloud.proof_of_benefit.benchmark_data_BEN import Bigearthnet

def main():
    pretrained = "Z:\\Phd_data\\Projects\\SSL\\weights\\ssl4eo\\moco_res50\\B13_rn50_moco_0099.pth" # location to ckpt file
    data_dir = "Z:\\Phd_data\\Projects\\SSL\\data\\BigEarthNet" # Path to Big Earthnet

    #pretrained = "/work/Phd_data/Projects/SSL/weights/ssl4eo/moco_res50/B13_rn50_moco_0099.pth" # location to ckpt file
    #data_dir = "/work/Phd_data/Projects/SSL/data/BigEarthNet" # Path to Big Earthnet

    bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']
    num_labels = 19

    net = models.resnet50(pretrained=False)
    net.fc = torch.nn.Linear(2048,19)
    net.conv1 = torch.nn.Conv2d(13, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    
    print("=> loading checkpoint '{}'".format(pretrained))
    checkpoint = torch.load(pretrained, map_location="cpu")
    state_dict = checkpoint['state_dict']
    sys.exit()
    for k in list(state_dict.keys()):
        # retain only encoder up to before the embedding layer
        if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
            #pdb.set_trace()
            # remove prefix
            state_dict[k[len("module.encoder_q."):]] = state_dict[k]
            # delete renamed or unused k
        del state_dict[k]
    msg = net.load_state_dict(state_dict, strict=False)
    assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
    print("=> loaded pre-trained model '{}'".format(pretrained))

    train_transforms = cvtransforms.Compose([
            cvtransforms.RandomResizedCrop(224,scale=(0.8,1.0)), # multilabel, avoid cropping out labels
            cvtransforms.RandomHorizontalFlip(),
            cvtransforms.ToTensor()])

    val_transforms = cvtransforms.Compose([
            cvtransforms.Resize(256),
            cvtransforms.CenterCrop(224),
            cvtransforms.ToTensor(),
            ])
    
    train_dataset = Bigearthnet(
            root=data_dir,
            split='train',
            bands=bands,
            use_new_labels = True,
            transform=train_transforms
        )
        
    val_dataset = Bigearthnet(
        root=data_dir,
        split='val',
        bands=bands,
        use_new_labels = True,
        transform=train_transforms
        )

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

if __name__ == "__main__":
    main()
