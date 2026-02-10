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
from torch.utils.tensorboard import SummaryWriter

sys.path.append("Z:\\Projects\\ssl\\src")
sys.path.append("/work/Projects/ssl/src")
sys.path.append("C:\\Users\\m_str\\Documents\\PhD_omu_laptop\\Projects\\ssl\\src")
from exp.ssl_cloud.proof_of_benefit.benchmark_data_BEN_mm import Bigearthnet

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='/mnt/d/codes/SSL_examples/datasets/BigEarthNet')
parser.add_argument('--lmdb_dir', type=str, default='/mnt/d/codes/SSL_examples/datasets/BigEarthNet/dataload_op1_lmdb')
parser.add_argument('--checkpoints_dir', type=str, default='C:\\Users\\m_str\\Documents\\Phd_data\\Projects\\SSL\\checkpoints\\ssl4eo\\resnet')
parser.add_argument('--resume', type=str, default='')
parser.add_argument('--save_path', type=str, default='./checkpoints/bigearthnet_s2_B12_100_no_pretrain_resnet50.pt')

parser.add_argument('--bands', type=str, default='all', choices=['all','RGB'], help='bands to process')  
parser.add_argument('--train_frac', type=float, default=1.0)
parser.add_argument('--backbone', type=str, default='resnet50')
parser.add_argument('--batchsize', type=int, default=16)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--lr', type=float, default=0.05)
parser.add_argument('--schedule', default=[60, 80], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--cos', action='store_true', help='use cosine lr schedule')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--pretrained', default='', type=str, help='path to moco pretrained checkpoint')

### distributed running ###
parser.add_argument('--dist_url', default='env://', type=str)
parser.add_argument("--world_size", default=-1, type=int, help="""
                    number of processes: it is set automatically and
                    should not be passed as argument""")
parser.add_argument("--rank", default=0, type=int, help="""rank of this process:
                    it is set automatically and should not be passed as argument""")
parser.add_argument("--local_rank", default=0, type=int,
                    help="this argument is not used and should be ignored")

parser.add_argument('--normalize',action='store_true',default=False)
parser.add_argument('--linear',action='store_true',default=False)

def main():

    #TODO recalculate mean and std in benchmark_data_BEN.py
    #TODO check how to freeze weights
    #TODO include radar images

    global args
    args = parser.parse_args()
    fix_random_seeds(args.seed)

    pretrained = "Z:\\Phd_data\\Projects\\SSL\\weights\\ssl4eo\\moco_res50\\B13_rn50_moco_0099.pth" # location to ckpt file
    data_dir = "Z:\\Phd_data\\Projects\\SSL\\data\\BigEarthNet" # Path to Big Earthnet

    #pretrained = "/work/Phd_data/Projects/SSL/weights/ssl4eo/moco_res50/B13_rn50_moco_0099.pth" # location to ckpt file
    #data_dir = "/work/Phd_data/Projects/SSL/data/BigEarthNet" # Path to Big Earthnet

    pretrained = "C:\\Users\\m_str\\Documents\\Phd_data\\Projects\\SSL\\weights\\ssl4eo\\moco_res50\\B13_rn50_moco_0099.pth" # location to ckpt file
    data_dir = "C:\\Users\\m_str\\Documents\\Phd_data\\Projects\\SSL\\data\\BigEarthNet" # Path to Big Earthnet

    bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12', 'VV', 'VH']
    num_labels = 19

    net = models.resnet50(pretrained=False)
    net.fc = torch.nn.Linear(2048,19)
    net.conv1 = torch.nn.Conv2d(14, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    
    tb_writer = SummaryWriter(os.path.join(args.checkpoints_dir,'log'))
    batch_size = args.batchsize
    num_workers = args.num_workers
    checkpoints_dir = args.checkpoints_dir

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
        split='validation',
        bands=bands,
        use_new_labels = True,
        transform=train_transforms
        )
    
    test_dataset = Bigearthnet(
        root=data_dir,
        split='test',
        bands=bands,
        use_new_labels = True,
        transform=train_transforms
        )

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              sampler = None, #TODO adapted
                              #shuffle=True,
                              num_workers=num_workers,
                              pin_memory=False, #TODO adapted
                              drop_last=True
                              
                              )
    
    val_loader = DataLoader(val_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=num_workers,
                              pin_memory=False, # improve a little when using lmdb dataset
                              drop_last=True
                              
                              )

    print("=> loading checkpoint '{}'".format(pretrained))
    checkpoint = torch.load(pretrained, map_location="cpu")
    state_dict = checkpoint['state_dict']
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
    net.cuda()
    criterion = torch.nn.MultiLabelSoftMarginLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)
    
    last_epoch = 0
    epochs = args.epochs
    print(f'Start training for {epochs} epochs...')
    for epoch in range(last_epoch,epochs):
        print(f"Training at epoch {epoch}")
        net.train()
        adjust_learning_rate(optimizer, epoch, args)

        #train_loader.sampler.set_epoch(epoch)
        running_loss = 0.0
        running_acc = 0.0
        
        running_loss_epoch = 0.0
        running_acc_epoch = 0.0
        
        start_time = time.time()
        end = time.time()
        sum_bt = 0.0
        sum_dt = 0.0
        sum_tt = 0.0
        sum_st = 0.0
        for i, data in enumerate(train_loader, 0):
            print(f"Minibatch number {i}")
            data_time = time.time()-end
            #inputs, labels = data
            b_zeros = torch.zeros((data[0].shape[0],1,data[0].shape[2],data[0].shape[3]),dtype=torch.float32)
            images = torch.cat((data[0][:,:10,:,:],b_zeros,data[0][:,10:,:,:]),dim=1)
            #inputs, labels = data[0].cuda(), data[1].cuda()
            inputs, labels = images.cuda(), data[1].cuda()
            
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            #pdb.set_trace()
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()
            train_time = time.time()-end-data_time
            if epoch%5==4:
                score = torch.sigmoid(outputs).detach().cpu()
                average_precision = average_precision_score(labels.cpu(), score, average='micro') * 100.0
            else:
                average_precision = 0
            score_time = time.time()-end-data_time-train_time
            
            # print statistics
            running_loss += loss.item()
            #running_acc += average_precision
            batch_time = time.time() - end
            end = time.time()        
            sum_bt += batch_time
            sum_dt += data_time
            sum_tt += train_time
            sum_st += score_time
            
            if i % 20 == 19:    # print every 20 mini-batches

                print('[%d, %5d] loss: %.3f acc: %.3f batch_time: %.3f data_time: %.3f train_time: %.3f score_time: %.3f' %
                      (epoch + 1, i + 1, running_loss / 20, running_acc / 20, sum_bt/20, sum_dt/20, sum_tt/20, sum_st/20))
                
                #train_iter =  i*args.batch_size / len(train_dataset)
                #tb_writer.add_scalar('train_loss', running_loss/20, global_step=(epoch+1+train_iter) )
                running_loss_epoch = running_loss/20
                running_acc_epoch = running_acc/20
                
                running_loss = 0.0
                running_acc = 0.0
                sum_bt = 0.0
                sum_dt = 0.0
                sum_tt = 0.0
                sum_st = 0.0

        if epoch % 5 == 4:
            running_loss_val = 0.0
            running_acc_val = 0.0
            count_val = 0
            net.eval()
            with torch.no_grad():
                for j, data_val in enumerate(val_loader, 0):

                    b_zeros = torch.zeros((data_val[0].shape[0],1,data_val[0].shape[2],data_val[0].shape[3]),dtype=torch.float32)
                    images = torch.cat((data_val[0][:,:10,:,:],b_zeros,data_val[0][:,10:,:,:]),dim=1)

                    #inputs_val, labels_val = data_val[0].cuda(), data_val[1].cuda()
                    inputs_val, labels_val = images.cuda(), data_val[1].cuda()

                    outputs_val = net(inputs_val)
                    loss_val = criterion(outputs_val, labels_val.long())
                    score_val = torch.sigmoid(outputs_val).detach().cpu()
                    average_precision_val = average_precision_score(labels_val.cpu(), score_val, average='micro') * 100.0   

                    count_val += 1
                    running_loss_val += loss_val.item()
                    running_acc_val += average_precision_val        

            print('Epoch %d val_loss: %.3f val_acc: %.3f time: %s seconds.' % (epoch+1, running_loss_val/count_val, running_acc_val/count_val, time.time()-start_time))

            if args.rank == 0:
                losses = {'train': running_loss_epoch,
                          'val': running_loss_val/count_val}
                accs = {'train': running_acc_epoch,
                        'val': running_acc_val/count_val}        
                tb_writer.add_scalars('loss', losses, global_step=epoch+1, walltime=None)
                tb_writer.add_scalars('acc', accs, global_step=epoch+1, walltime=None)
        
            
            
        if args.rank==0 and epoch % 10 == 9:
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': net.state_dict(),
                        'optimizer_state_dict':optimizer.state_dict(),
                        'loss':loss,
                        }, os.path.join(checkpoints_dir,'checkpoint_{:04d}.pth.tar'.format(epoch)))
        
    #if args.rank==0:
    #    torch.save(net.state_dict(), save_path)
        
    print('Training finished.')

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

def fix_random_seeds(seed=42):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

if __name__ == "__main__":
    main()
