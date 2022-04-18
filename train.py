import sys
import os
import random
import torch
import argparse
import numpy as np
import shutil
import json
import my_optim
import torch.optim as optim
from models import rca
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.autograd import Variable
from utils import AverageMeter
from utils.LoadData import train_data_loader
from tqdm import trange, tqdm

ROOT_DIR = '/'.join(os.getcwd().split('/')[:-1])
print('Project Root Dir:', ROOT_DIR)

def get_arguments():
    parser = argparse.ArgumentParser(description='-------------------RCA----------------')
    parser.add_argument("--root_dir", type=str, default=ROOT_DIR)
    parser.add_argument("--img_dir", type=str, default='data/JPEGImages')
    parser.add_argument("--train_list", type=str, default='data/train_cls.txt')
    parser.add_argument("--test_list", type=str, default='data/val_cls.txt')
    parser.add_argument("--num_classes", type=int, default=20) 
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--input_size", type=int, default=368)
    parser.add_argument("--crop_size", type=int, default=368)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--dataset", type=str, default='pascal_voc')
    parser.add_argument("--decay_points", type=str, default='5,10')
    parser.add_argument("--snapshot_dir", type=str, default='runs/pascal/model')
    parser.add_argument("--att_dir", type=str, default='runs/pascal/feat')
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--disp_interval", type=int, default=1)  
    parser.add_argument("--resume", type=str, default=False)
    parser.add_argument("--global_counter", type=int, default=0)
    parser.add_argument("--epoch", type=int, default=30)
    parser.add_argument("--current_epoch", type=int, default=0)

    return parser.parse_args()

def save_checkpoint(args, state, is_best, filename='checkpoint.pth.tar'):
    savepath = os.path.join(args.snapshot_dir, filename)
    torch.save(state, savepath)
    if is_best:
        shutil.copyfile(savepath, os.path.join(args.snapshot_dir, 'model_best.pth.tar'))

def get_model(args):                                                        
    model = rca.RCA_VGG(pretrained=True, num_classes=args.num_classes,att_dir=args.att_dir, training_epoch=args.epoch)
    print(model)
    device = torch.device(0)	
    model = torch.nn.DataParallel(model).cuda()
    model.to(device)
    param_groups = model.module.get_parameter_groups()
    optimizer = optim.SGD([
        {'params': param_groups[0], 'lr': args.lr},
        {'params': param_groups[1], 'lr': 2*args.lr},
        {'params': param_groups[2], 'lr': 10*args.lr},
        {'params': param_groups[3], 'lr': 20*args.lr}], momentum=0.9, weight_decay=args.weight_decay, nesterov=True)
    return  model, optimizer

def train(args):
    losses = AverageMeter()
    losses_cl_mix = AverageMeter()
    
    total_epoch = args.epoch
    global_counter = args.global_counter
    current_epoch = args.current_epoch

    train_loader, val_loader = train_data_loader(args)
    max_step = total_epoch*len(train_loader)
    args.max_step = max_step 
    print('Max step:', max_step)
    
    model, optimizer = get_model(args)
    model.train()

    while current_epoch < total_epoch:
        model.train()
        losses.reset()
        losses_cl_mix.reset()
        res = my_optim.reduce_lr(args, optimizer, current_epoch)
        steps_per_epoch = len(train_loader)

        index = 0  
        for idx, dat in enumerate(train_loader):
            img_name, img, label = dat
            label = label.cuda(non_blocking=True)           
            
            xss, xs, loss_cl_mix = model(img, label, current_epoch, index)
            index += args.batch_size

            if(current_epoch < 2):
                loss_train = F.multilabel_soft_margin_loss(xs, label) + 0.4 * F.multilabel_soft_margin_loss(xss, label)
                
                optimizer.zero_grad()
                loss_train.backward()
                optimizer.step()

                losses.update(loss_train.data.item(), img.size()[0])
            else:
                loss_train = (F.multilabel_soft_margin_loss(xs, label) + 0.4 * F.multilabel_soft_margin_loss(xss, label)
                                + 0.01 * loss_cl_mix.mean())
                
                optimizer.zero_grad()
                loss_train.backward()
                optimizer.step()

                losses.update(loss_train.data.item(), img.size()[0])
                losses_cl_mix.update(loss_cl_mix.mean().data.item(), img.size()[0])
            
            global_counter += 1
            if global_counter % 1000 == 0:
                losses.reset()
                losses_cl_mix.reset()

            if global_counter % args.disp_interval == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'LR: {:.5f}\t' 
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Loss_cl_mix {loss_cl_mix.val:.4f} ({loss_cl_mix.avg:.4f})\t'.format(
                        current_epoch, global_counter%len(train_loader), len(train_loader), 
                        optimizer.param_groups[0]['lr'], loss=losses,
                        loss_cl_mix = losses_cl_mix))

        if current_epoch == args.epoch-1:
            save_checkpoint(args,
                        {
                            'epoch': current_epoch,
                            'global_counter': global_counter,
                            'state_dict':model.state_dict(),
                            'optimizer':optimizer.state_dict()
                        }, is_best=False,
                        filename='%s_epoch_%d.pth' %(args.dataset, current_epoch))
        current_epoch += 1

if __name__ == '__main__':
    args = get_arguments()
    print('Running parameters:\n', args)
    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)
    train(args)
