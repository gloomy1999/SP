import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
from models.MPNCOV import MPNCOV
import torch.nn.functional as F
import models.resnet
import models.densenet
import models.senet
from models.operations import *

import re

__all__ = ['seenDvbe']

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(torch.nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

def LoadParameter(_structure, _parameterDir):

    checkpoint = torch.load(_parameterDir)
    pretrained_state_dict = checkpoint['state_dict']
    model_state_dict = _structure.state_dict()

    for key in pretrained_state_dict:
        if ((key == 'module.fc.weight') | (key == 'module.fc.bias') | (key == 'module.feature.weight') | (key == 'module.feature.bias')):

            pass
        else:
            model_state_dict[key.replace('module.', '')] = pretrained_state_dict[key]

    _structure.load_state_dict(model_state_dict)
    # for name, param in _structure.named_parameters():
    #     print(name)
    #model = torch.nn.DataParallel(_structure).cuda()

    return _structure

class Model(nn.Module):
    def __init__(self, pretrained=True, args=None):
        self.inplanes = 64
        num_classes = args.num_classes
        is_fix = args.is_fix
        sf_size = args.sf_size
        self.arch = args.backbone
        self.adj = args.adj
        self.sf =  torch.from_numpy(args.sf).cuda()
        
        super(Model, self).__init__()

        ''' backbone net'''
        block = BasicBlock
        layers = [2, 2, 2, 2]
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # self.avgpool = nn.AdaptiveAvgPool2d(1)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

        if is_fix:
            for p in self.parameters():
                p.requires_grad = False

        feat_dim = 512

        ''' Open-Domain Recognition Module '''
        self.odr_proj1 =  nn.Sequential(
            nn.Conv2d(feat_dim, 256, kernel_size=1, stride=1, padding=0,bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.odr_proj2 =  nn.Sequential(
            nn.Conv2d(feat_dim, 256, kernel_size=1, stride=1, padding=0,bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.odr_spatial =  nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1,bias=False),
            nn.ReLU(inplace=True),   
            nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0,bias=False),
            nn.Sigmoid(),        
        )
        self.odr_channel =  nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(256, int(256/16), kernel_size=1, stride=1, padding=0,bias=False),
            nn.ReLU(inplace=True),   
            nn.Conv2d(int(256/16), 256, kernel_size=1, stride=1, padding=0,bias=False),
            nn.Sigmoid(),        
        )
        self.odr_classifier = nn.Linear(int(256*(256+1)/2), num_classes)

        ''' Zero-Shot Recognition Module '''
        self.zsr_proj = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, kernel_size=1, stride=1, padding=0,bias=False),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.zsr_sem = nn.Sequential(
            nn.Linear(sf_size,1024),
            nn.LeakyReLU(),
            #GraphConv(sf_size,1024,self.adj),
            #GraphConv(1024,feat_dim,self.adj),
            nn.Linear(1024,feat_dim),
            nn.LeakyReLU(),
        )
        self.zsr_aux = nn.Linear(feat_dim, num_classes)
        
        ''' params ini '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # backbone
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        last_conv = x

        ''' ODR Module '''
        x1 = self.odr_proj1(last_conv)
        x2 = x1#self.odr_proj1(last_conv)
        # att gen
        att1 = self.odr_spatial(x1)
        att2 = self.odr_channel(x2)
        # att1
        x1 = att2*x1+x1
        x1 = x1.view(x1.size(0),x1.size(1),-1)
        # att2
        x2 = att1*x2+x2
        x2 = x2.view(x2.size(0),x2.size(1),-1)
        # covariance pooling
        x1 = x1 - torch.mean(x1,dim=2,keepdim=True)
        x2 = x2 - torch.mean(x2,dim=2,keepdim=True)
        A = 1./x1.size(2)*x1.bmm(x2.transpose(1,2))
        # norm
        x = MPNCOV.SqrtmLayer(A, 5)
        x = MPNCOV.TriuvecLayer(x)
        odr_x = x.view(x.size(0), -1)
        # cls
        odr_logit = self.odr_classifier(odr_x)
        
        ''' ZSR Module '''
        zsr_x = self.zsr_proj(last_conv).view(last_conv.size(0),-1)
        zsr_classifier = self.zsr_sem(self.sf)
        w_norm = F.normalize(zsr_classifier, p=2, dim=1)
        x_norm = F.normalize(zsr_x, p=2, dim=1)
        zsr_logit = x_norm.mm(w_norm.permute(1,0))
        zsr_logit_aux = self.zsr_aux(zsr_x)
        
        return (odr_logit,zsr_logit,zsr_logit_aux),(odr_x,zsr_x)

class Loss(nn.Module):
    def __init__(self, args):
        super(Loss, self).__init__()
        
        self.cls_loss = nn.CrossEntropyLoss()#reduce=False)
        self.sigma = args.sigma
    def forward(self, label, logits):
        odr_logit = logits[0]
        zsr_logit = logits[1]
        zsr_logit_aux = logits[2]
        
        ''' ODR Loss '''
        prob = F.softmax(odr_logit,dim=1).detach()
#         print("prob : "+prob)
        y = prob[torch.arange(prob.size(0)).long(),label]
#         print("y : "+y)
        mw = torch.exp(-(y-1.0)**2/self.sigma)
#         print("mw : "+mw)
        one_hot = torch.zeros_like(odr_logit)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        odr_logit = odr_logit*(1-one_hot*mw.view(mw.size(0),1))
#         print("odr_logit : "+odr_logit)
        L_odr = self.cls_loss(odr_logit,label)
        
        ''' ZSL Loss '''
        idx = torch.arange(zsr_logit.size(0)).long()
        L_zsr = (1-zsr_logit[idx,label]).mean()
        
        L_aux = self.cls_loss(zsr_logit_aux,label)
        
        total_loss = L_odr + L_zsr + L_aux
        
        return total_loss,L_odr,L_zsr, L_aux

def LoadParameter(_structure, _parameterDir):

    checkpoint = torch.load(_parameterDir)
    pretrained_state_dict = checkpoint['state_dict']
    model_state_dict = _structure.state_dict()

    for key in pretrained_state_dict:
        if (key == 'module.fc.weight') | (key == 'module.fc.bias') | (key == 'module.feature.weight') | (key == 'module.feature.bias'):
            pass
        else:
            model_state_dict[key.replace('module.', '')] = pretrained_state_dict[key]

    _structure.load_state_dict(model_state_dict)
    # for name, param in _structure.named_parameters():
    #     print(name)
    #model = torch.nn.DataParallel(_structure).cuda()

    return _structure

def seenDvbe(pretrained=False, loss_params=None, args=None):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on FER
    """
    model = Model(pretrained,args)
    loss_model = Loss(args)
    if pretrained:
        model = LoadParameter(model,'../data/models/Resnet18_FER+_pytorch.pth.tar')
#         model = model.cuda(0)

    return model,loss_model