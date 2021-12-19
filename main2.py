#from __future__ import print_function
import argparse
import os
import random
import shutil
import time
import warnings
import h5py
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

import matplotlib.pyplot as plt

import sys
import torchvision.transforms as transforms
import datasets
import models
from utils import *

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data','-d', metavar='DATA', default='cub',
                    help='dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--backbone', default='resnet18', help='backbone')
parser.add_argument('--save_path', '-s', metavar='SAVE', default='',
                    help='saving path')
parser.add_argument('-j', '--workers', default=3, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr1', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr2', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--epoch_decay', default=30, type=int,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--is_fix', dest='is_fix', action='store_true',
                    help='is_fix.')
''' data proc '''
parser.add_argument('--flippingtest', dest='flippingtest', action='store_true',
                    help='flipping test.')
''' loss '''
parser.add_argument('--sigma', dest='sigma', default=0.5, type=float,
                    help='sigma.')
                               
best_prec1 = 0
iterNum = 1
odrLoss,zsrLoss,auxLoss,valAcc,seenAcc=[],[],[],[],[]


def plotCurve(x_vals, y_vals, 
                x_label, y_label, 
                x2_vals, y2_vals,
                x3_vals,y3_vals,
                x4_vals,y4_vals,
                x5_vals,y5_vals,
                legend,
                figsize=(3.5, 2.5)):
    for i in range(5):
        plt.xlabel(x_label)
        if i >= 3:
            y_label='acc'
        plt.ylabel(y_label)
        if i == 0:
            plt.plot(x_vals, y_vals)
        elif i == 1:
            plt.plot(x2_vals, y2_vals)
        elif i == 2:
            plt.plot(x3_vals, y3_vals)
        elif i == 3:
            plt.plot(x4_vals, y4_vals)
        elif i == 4:
            plt.plot(x5_vals, y5_vals)
        fileName = legend[i] + ".png"
        plt.savefig(fileName)
        plt.close()

def main():
    global args, best_prec1
    args = parser.parse_args()
    print(args)

    ''' save path '''
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    ''' random seed '''
    if args.seed is not None:
        random.seed(args.seed)
    else:
        args.seed = random.randint(1, 10000)
        
    torch.manual_seed(args.seed)
    cudnn.deterministic = True
    print('==> random seed:',args.seed)
    

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    args.distributed = args.world_size > 1

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)

    ''' data load info '''
    data_info = h5py.File(os.path.join('./data',args.data,'data_info.h5'), 'r')
    nc = data_info['all_att'][...].shape[0]
    sf_size = data_info['all_att'][...].shape[1]
    semantic_data = {'seen_class':data_info['seen_class'][...],
                     'unseen_class': data_info['unseen_class'][...],
                     'all_class':np.arange(nc),
                     'all_att': data_info['all_att'][...]}

    ''' my data info '''
    # 类别与表情种类的对应关系：
    # 0 neutral
    # 1 happy / joy
    # 2 sad
    # 3 surprise
    # 4 fear
    # 5 disgust
    # 6 anger
    # 暂定1，2，4，6为可见类，0，3，5为不可见类
#     nc = 7  # 7个类别
#     sf_size = 6  # 6个属性
#     semantic_data = {}
#     semantic_data['seen_class'] = np.array([1, 2, 4, 6])
#     semantic_data['unseen_class'] = np.array([0, 3, 5])
#     semantic_data['all_class'] = np.array([0, 1, 2, 3, 4, 5, 6])
#     semantic_data['all_att'] = np.array([
#         [0.21, -0.06, -0.25, -0.21, -0.04, -0.23],
#         [0.77, 0.09, 1.00, 0.00, -0.47, -0.30],
#         [0.17, -0.04, -0.37, -0.09, -0.02, -0.27],
#         [-0.10, 0.60, -0.49, -0.13, 0.58, 0.10],
#         [-0.11, 0.20, -0.60, -0.17, 0.28, -0.02],
#         [0.91, 0.13, 0.88, 0.00, -0.32, -0.39],
#         [0.30, -0.07, 0.06, 0.04, -0.19, -0.34]
#     ],dtype=np.float32)

    ''' load semantic data '''
    args.num_classes = nc
    args.sf_size = sf_size
    args.sf = semantic_data['all_att']
    args.sf = args.sf.astype(np.float32)
    # adj
    adj = adj_matrix(nc)
    args.adj = adj

    ''' model building '''
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        best_prec1=0
        model,criterion = models.__dict__[args.arch](pretrained=True,args=args)
    else:
        print("=> creating model '{}'".format(args.arch))
        model,criterion = models.__dict__[args.arch](args=args)
    print("=> is the backbone fixed: '{}'".format(args.is_fix))

    if args.gpu is not None:
        model = model.cuda(args.gpu)
    elif args.distributed:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
    criterion = criterion.cuda(args.gpu)


    ''' optimizer '''
    odr_params = [v for k, v in model.named_parameters() if 'odr_' in k]
    zsr_params = [v for k, v in model.named_parameters() if 'zsr_' in k]

    odr_optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, odr_params),
                     args.lr1, momentum=args.momentum, weight_decay=args.weight_decay)
    zsr_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, zsr_params), args.lr2,
                                betas=(0.5,0.999),weight_decay=args.weight_decay)
                                
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                     args.lr1, momentum=args.momentum, weight_decay=args.weight_decay)

    ''' optionally resume from a checkpoint'''
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            if(best_prec1==0):
                best_prec1 = checkpoint['best_prec1']
            print('=> pretrained acc {:.4F}'.format(best_prec1))
            model.load_state_dict(checkpoint['state_dict'])
#             optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    if args.data.lower()=='cub':
        img_path = '../CUB/CUB_200_2011/images/'
    elif args.data.lower()=='affectnet':
        img_path = '../affectnet/images/'
    elif args.data.lower()=='raf':
        img_path = '../RAF/'
    elif args.data.lower()=='awa2':
        img_path = '~~~/Animals_with_Attributes2/JPEGImages/'
    elif args.data.lower()=='sun':
        img_path = 'xxxx'
    elif args.data.lower()=='apy':
        img_path = 'xxxx'
        
    traindir = os.path.join('./data',args.data,'train.list')
    valdir1 = os.path.join('./data',args.data,'test_seen.list')
    valdir2 = os.path.join('./data',args.data,'test_unseen.list')

    train_transforms, val_transforms = preprocess_strategy(args.data,args)

    train_dataset = datasets.ImageFolder(img_path,traindir,train_transforms)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader1 = torch.utils.data.DataLoader(
        datasets.ImageFolder(img_path,valdir1, val_transforms),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
        
    val_loader2 = torch.utils.data.DataLoader(
        datasets.ImageFolder(img_path,valdir2, val_transforms),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
        
    for epoch in range(args.start_epoch, args.epochs):
#         print("Attention!!!:epoch&start_epoch")
#         print(epoch)
#         print(args.start_epoch)
#         print("for valAcc:")
#         print(epoch+1-args.start_epoch+1)
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, odr_optimizer, zsr_optimizer, epoch, args)

        # train for one epoch
        train(train_loader,semantic_data, model, criterion, optimizer, odr_optimizer, zsr_optimizer, epoch,is_fix=args.is_fix)
        
        # evaluate on validation set
        prec1,seenPrec = validate(val_loader1, val_loader2, semantic_data, model, criterion)
        valAcc.append(prec1)
        seenAcc.append(seenPrec)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        # save model
        if args.is_fix:
            save_path = os.path.join(args.save_path,'fix.model')
        else:
            save_path = os.path.join(args.save_path,args.arch+('_{:.4f}.model').format(best_prec1))
        if is_best:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                #'optimizer' : optimizer.state_dict(),
            },filename=save_path)
            print('saving!!!!')
    
#         print("Attention!!!:epoch start_epoch")
#         print(epoch+" "+args.start_epoch)
#         print("for valAcc:")
#         print(epoch+1-args.start_epoch+1)
        plotCurve(range(1, iterNum), odrLoss,
                  "iterNum", "loss",
                  range(1, iterNum), zsrLoss,
                  range(1, iterNum),auxLoss,
                  range(1,epoch+1-args.start_epoch+1),valAcc,
                  range(1,epoch+1-args.start_epoch+1),seenAcc,
                  ["L_odr", "L_zsr","L_aux","val_acc","seen_acc"])


def train(train_loader, semantic_data, model, criterion, optimizer, odr_optimizer, zsr_optimizer, epoch,is_fix):    
    # switch to train mode
    model.train()
    if(is_fix):
        freeze_bn(model) 

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        logits,feats = model(input)
        total_loss,L_odr,L_zsr, L_aux = criterion(target,logits)

        # compute gradient and do SGD step
        if args.pretrained:
            odr_optimizer.zero_grad()
            L_odr.backward()
            odr_optimizer.step()
        
            zsr_optimizer.zero_grad()
            (L_zsr+L_aux).backward()
            zsr_optimizer.step()
        else:
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}] loss:'.format
                   (epoch, i, len(train_loader)),end='')
            print('L_odr {:.4f} L_zsl {:.4f} L_aux {:.4f}'.format(L_odr.item(),L_zsr.item(),L_aux.item()))
            global iterNum
            iterNum=iterNum+1
            odrLoss.append(L_odr.item())
            zsrLoss.append(L_zsr.item())
            auxLoss.append(L_aux.item())
            

def validate(val_loader1, val_loader2, semantic_data, model, criterion):

    ''' load semantic data'''
    seen_c = semantic_data['seen_class']
    unseen_c = semantic_data['unseen_class']
    all_c = semantic_data['all_class']
    
    # switch to evaluate mode
    model.eval()
    
    if args.flippingtest:#flipping test
        test_flip = True
    else:
        test_flip = False

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader1):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
            
            if test_flip:                
                [N,M,C,H,W] = input.size()
                input = input.view(N*M,C,H,W)   #flipping test
            
            # inference
            logits,feats = model(input)
            
            if test_flip: 
                odr_logit = F.softmax(logits[0],dim=1).view(N,M,-1).mean(dim=1).cpu().numpy()
                zsl_logit = F.softmax(logits[1],dim=1).view(N,M,-1).mean(dim=1).cpu().numpy()
            else:
                odr_logit = logits[0].cpu().numpy()
                zsl_logit = logits[1].cpu().numpy()
            zsl_logit_s = zsl_logit.copy();zsl_logit_s[:,unseen_c]=-1
            zsl_logit_t = zsl_logit.copy();zsl_logit_t[:,seen_c]=-1
			
            # evaluation
            if(i==0):
                gt_s = target.cpu().numpy()
                odr_pre_s = np.argmax(odr_logit, axis=1)
                if test_flip: 
                    odr_prob_s = odr_logit
                else:
                    odr_prob_s = softmax(odr_logit)
                zsl_pre_sA = np.argmax(zsl_logit, axis=1)
                zsl_pre_sS = np.argmax(zsl_logit_s, axis=1)
                if test_flip:
                    zsl_prob_s = zsl_logit_t
                else: 
                    zsl_prob_s = softmax(zsl_logit_t)
            else:
                gt_s = np.hstack([gt_s,target.cpu().numpy()])
                odr_pre_s = np.hstack([odr_pre_s,np.argmax(odr_logit, axis=1)])
                if test_flip:
                    odr_prob_s = np.vstack([odr_prob_s,odr_logit])
                else:
                    odr_prob_s = np.vstack([odr_prob_s,softmax(odr_logit)])
                zsl_pre_sA = np.hstack([zsl_pre_sA,np.argmax(zsl_logit, axis=1)])
                zsl_pre_sS = np.hstack([zsl_pre_sS,np.argmax(zsl_logit_s, axis=1)])
                if test_flip:
                    zsl_prob_s = np.vstack([zsl_prob_s,zsl_logit_t])
                else:
                    zsl_prob_s = np.vstack([zsl_prob_s,softmax(zsl_logit_t)])

        for i, (input, target) in enumerate(val_loader2):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
            
            if test_flip:                
                [N,M,C,H,W] = input.size()
                input = input.view(N*M,C,H,W)   #flipping test
                
            # inference
            logits,feats = model(input)           
            
            if test_flip: 
                odr_logit = F.softmax(logits[0],dim=1).view(N,M,-1).mean(dim=1).cpu().numpy()
                zsl_logit = F.softmax(logits[1],dim=1).view(N,M,-1).mean(dim=1).cpu().numpy()
            else:
                odr_logit = logits[0].cpu().numpy()
                zsl_logit = logits[1].cpu().numpy()
            zsl_logit_s = zsl_logit.copy();zsl_logit_s[:,unseen_c]=-1
            zsl_logit_t = zsl_logit.copy();zsl_logit_t[:,seen_c]=-1
                
            if(i==0):
                gt_t = target.cpu().numpy()
                odr_pre_t = np.argmax(odr_logit, axis=1)
                if test_flip: 
                    odr_prob_t = odr_logit
                else:
                    odr_prob_t = softmax(odr_logit)
                zsl_pre_tA = np.argmax(zsl_logit, axis=1)
                zsl_pre_tT = np.argmax(zsl_logit_t, axis=1)
                if test_flip: 
                    zsl_prob_t = zsl_logit_t
                else:
                    zsl_prob_t = softmax(zsl_logit_t)
            else:
                gt_t = np.hstack([gt_t,target.cpu().numpy()])
                odr_pre_t = np.hstack([odr_pre_t,np.argmax(odr_logit, axis=1)])
                if test_flip: 
                    odr_prob_t = np.vstack([odr_prob_t,odr_logit])
                else:
                    odr_prob_t = np.vstack([odr_prob_t,softmax(odr_logit)])
                zsl_pre_tA = np.hstack([zsl_pre_tA,np.argmax(zsl_logit, axis=1)])
                zsl_pre_tT = np.hstack([zsl_pre_tT,np.argmax(zsl_logit_t, axis=1)])
                if test_flip: 
                    zsl_prob_t = np.vstack([zsl_prob_t,zsl_logit_t])
                else:
                    zsl_prob_t = np.vstack([zsl_prob_t,softmax(zsl_logit_t)])
                
        odr_prob = np.vstack([odr_prob_s,odr_prob_t])
        zsl_prob = np.vstack([zsl_prob_s,zsl_prob_t])
        gt = np.hstack([gt_s,gt_t])
    
        SS = compute_class_accuracy_total(gt_s, zsl_pre_sS,seen_c)
        UU = compute_class_accuracy_total(gt_t, zsl_pre_tT,unseen_c)
        ST = compute_class_accuracy_total(gt_s, zsl_pre_sA,seen_c)
        UT = compute_class_accuracy_total(gt_t, zsl_pre_tA,unseen_c)
        H = 2*ST*UT/(ST+UT) 
        CLS = compute_class_accuracy_total(gt_s, odr_pre_s,seen_c)
        
        H_opt,S_opt,U_opt,Ds_opt,Du_opt,tau = post_process(odr_prob, zsl_prob, gt, gt_s.shape[0], seen_c,unseen_c, args.data)
        
        print(' SS: {:.4f} UU: {:.4f} ST: {:.4f} UT: {:.4f} H: {:.4f}'
              .format(SS,UU,ST,UT,H))
        print('CLS {:.4f} S_opt: {:.4f} U_opt {:.4f} H_opt {:.4f} Ds_opt {:.4f} Du_opt {:.4f} tau {:.4f}'
              .format(CLS, S_opt, U_opt,H_opt,Ds_opt,Du_opt,tau))
              
        H = max(H,H_opt)
    return H,CLS


if __name__ == '__main__':
    main()
