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
    def __init__(self, in_channel = 3, out_channel = 64):
        super(ResNet, self).__init__()

        self.resnet = models.resnet50(pretrained=True)

        self.relu = nn.ReLU(inplace = True)


    def forward(self, x):
        x = self.relu(self.resnet.bn1(self.resnet.conv1(x)))
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        return x


class ConvBlock(nn.Module):
  def __init__(self,in_channels, out_channels, kernel_size=1, stride=1, dilation=1,padding=0, bais=False):
    super(ConvBlock, self).__init__()

    padding = (kernel_size + (kernel_size-1)*(dilation-1))//2

    self.conv = nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels,kernel_size=kernel_size, stride = stride, padding = padding, dilation = dilation),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace = True)
    )


  def forward(self, x):
    out = self.conv(x)
    return out



def upsample(input, size=None, scale_factor=None, align_corners=False):
    out = F.interpolate(input, size=size, scale_factor=scale_factor, mode='bilinear', align_corners=align_corners)
    return out


class PyramidalPooling(nn.Module):
  def __init__(self, in_channels, out_channels = 512):
    super(PyramidalPooling,self).__init__()
    self.pooling = [2,3,4,6]
    #self.pooling = [1,2,3,6]

    self.pool1 = nn.Sequential(nn.AdaptiveAvgPool2d(self.pooling[0]),
                               ConvBlock(in_channels = in_channels, out_channels = out_channels, kernel_size = 1))


    self.pool2 = nn.Sequential(nn.AdaptiveAvgPool2d(self.pooling[1]),
                ConvBlock(in_channels = in_channels, out_channels = out_channels, kernel_size = 1))

    self.pool3 = nn.Sequential(nn.AdaptiveAvgPool2d(self.pooling[2]),
                ConvBlock(in_channels = in_channels, out_channels = out_channels, kernel_size = 1))

    self.pool4 = nn.Sequential(nn.AdaptiveAvgPool2d(self.pooling[3]),
                  ConvBlock(in_channels = in_channels, out_channels = out_channels, kernel_size = 1))

  def forward(self, x):
    out1 = self.pool1(x)
    out1 = upsample(out1, size = x.size()[-2:])

    out2 =self.pool2(x) 
    out2 = upsample(out2, size = x.size()[-2:])

    out3 = self.pool3(x)
    out3 = upsample(out3, size = x.size()[-2:])

    out4 = self.pool4(x)
    out4 = upsample(out4, size = x.size()[-2:])

    return torch.cat([x, out1, out2, out3, out4],dim=1)



class PSPNet(nn.Module):
  def __init__(self, n_classes=21):
    super(PSPNet, self).__init__()

    self.in_channels = 2048
    self.depth = self.in_channels//4

    self.resnet = ResNet()

    self.Pyramid = PyramidalPooling(in_channels = self.in_channels)

    self.decoder = nn.Sequential(
            ConvBlock(self.in_channels * 2, self.depth, kernel_size=3),
            nn.Dropout(0.1),
            nn.Conv2d(self.depth, n_classes, kernel_size=1),
        )

  def forward(self, x):
    x1 = self.resnet(x)
    x2 = self.Pyramid(x1)
    x3 = self.decoder(x2)

    out = upsample(x3, size = x.size()[-2:],align_corners=True)
    
    return out

