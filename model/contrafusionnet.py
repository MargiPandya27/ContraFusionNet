import os
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import cv2
import logging
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm.notebook import tqdm
from torchvision import models
from sklearn.metrics import confusion_matrix
from torchvision.datasets import Cityscapes
import torchvision.transforms


class ResNet(nn.Module):
    def __init__(self, in_channel =3, out_channel = 64):
        super(ResNet, self).__init__()

        self.resnet = models.resnet50(pretrained=True)

        self.relu = nn.ReLU(inplace = True)


    def forward(self, x):
        x = self.relu(self.resnet.bn1(self.resnet.conv1(x)))
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        return x
    


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=256):
        super(ASPP, self).__init__()


        self.relu = nn.ReLU(inplace = True)

        self.conv1 = nn.Conv2d(in_channels = in_channels,
                               out_channels = out_channels,
                               kernel_size = 1,
                               stride = 1,
                               padding = 0,
                               dilation = 1,
                               bias = False)
        
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(in_channels = in_channels,
                               out_channels = out_channels,
                               kernel_size = 3,
                               stride = 1,
                               padding = 6,
                               dilation = 6,
                               bias = False)
        
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(in_channels = in_channels,
                               out_channels = out_channels,
                               kernel_size = 3,
                               stride = 1,
                               padding = 12,
                               dilation = 12,
                               bias = False)
        
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.conv4 = nn.Conv2d(in_channels = in_channels,
                               out_channels = out_channels,
                               kernel_size = 3,
                               stride = 1,
                               padding = 18,
                               dilation = 18,
                               bias = False)
        
        self.bn4 = nn.BatchNorm2d(out_channels)

        self.conv5 = nn.Conv2d(in_channels = in_channels,
                               out_channels = out_channels,
                               kernel_size = 1,
                               stride = 1,
                               padding = 0,
                               dilation = 1,
                               bias = False)
        
        self.bn5 = nn.BatchNorm2d(out_channels)

        self.convf = nn.Conv2d(in_channels = out_channels*5,
                               out_channels = out_channels,
                               kernel_size = 1,
                               stride = 1,
                               padding = 0,
                               dilation = 1,
                               bias = False)
        
        self.bnf = nn.BatchNorm2d(out_channels)

        self.adapool = nn.AdaptiveAvgPool2d(1)

    
    def forward(self, x):
        #print(x.shape)
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        #print('X1',x1.shape)

        x2 = self.conv2(x)
        x2 = self.bn2(x2)
        x2 = self.relu(x2)
        #print('X2', x2.shape)

        x3 = self.conv3(x)
        x3 = self.bn3(x3)
        x3 = self.relu(x3)
        #print('X3', x3.shape)

        x4 = self.conv4(x)
        x4 = self.bn4(x4)
        x4 = self.relu(x4)
        #print('X4',x4.shape)

        x5 = self.adapool(x)
        #print('X5', x5.shape)
        x5 = self.conv5(x5)
        #print('X5', x5.shape)
        #print('X5', x5.shape)
        x5 = self.relu(x5)
        #print('X5', x5.shape)
        x5 = F.interpolate(x5, size = tuple(x4.shape[-2:]), mode='bilinear')
        

        x = torch.cat([x1,x2,x3,x4,x5], dim=1)
        #print('X',x.shape)
        x = self.convf(x)
        x = self.bnf(x)
        x = self.relu(x)
        #print(x.shape)

        return x


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=256):
        super(ASPP, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               dilation=1,
                               bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=6,
                               dilation=6,
                               bias=False)

        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=12,
                               dilation=12,
                               bias=False)

        self.bn3 = nn.BatchNorm2d(out_channels)

        self.conv4 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=18,
                               dilation=18,
                               bias=False)

        self.bn4 = nn.BatchNorm2d(out_channels)

        self.conv5 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               dilation=1,
                               bias=False)

        self.bn5 = nn.BatchNorm2d(out_channels)

        self.convf = nn.Conv2d(in_channels=out_channels * 5,
                               out_channels=out_channels,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               dilation=1,
                               bias=False)

        self.bnf = nn.BatchNorm2d(out_channels)

        self.adapool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)

        x2 = self.conv2(x)
        x2 = self.bn2(x2)
        x2 = self.relu(x2)

        x3 = self.conv3(x)
        x3 = self.bn3(x3)
        x3 = self.relu(x3)

        x4 = self.conv4(x)
        x4 = self.bn4(x4)
        x4 = self.relu(x4)

        x5 = self.adapool(x)
        x5 = self.conv5(x5)
        x5 = self.relu(x5)
        x5 = F.interpolate(x5, size=x4.shape[-2:], mode='bilinear')

        # Upsample x5 to match the spatial dimensions of x4
        x5 = F.interpolate(x5, size=x4.shape[-2:], mode='bilinear')

        # Concatenate the features along the channel dimension
        x = torch.cat([x1, x2, x3, x4, x5], dim=1)
        x = self.convf(x)
        x = self.bnf(x)
        x = self.relu(x)
        return x


class DeepLabv3(nn.Module):
    def __init__(self, classes):
        super(DeepLabv3, self).__init__()

        self.classes = classes

        self.resnet = ResNet()

        self.assp = ASPP(in_channels = 1024)

        self.conv = nn.Conv2d(in_channels = 256, out_channels = self.classes,
                              kernel_size = 1,
                              stride = 1,
                              padding = 0)

    def forward(self,x):
        _, _, h, w = x.shape
        x = self.resnet(x)
        #print(x.shape)
        x = self.assp(x)
        x = self.conv(x)
        x = F.interpolate(x, (h,w), mode = 'bilinear')

        return x


