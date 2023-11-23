import torch
from einops import rearrange
from torch import nn
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
from typing import List



class LayerNorm2d(nn.LayerNorm):
  '''
  Layer Normalization is applied while preserving the resolution.
  '''
  def forward(self, x):
      x = rearrange(x, "b c h w -> b h w c")
      x = super().forward(x)
      x = rearrange(x, "b h w c -> b c h w")
      return x
  
class OverlapPatchMerging(nn.Sequential):
  '''
  Original input image is divided into patches with patch_size != stride
  Preserves local continuity.
  '''
  def __init__(self, in_channels: int, out_channels: int, patch_size: int, overlap_size: int):
    super().__init__()
    self.conv = nn.Conv2d(
              in_channels,
              out_channels,
              kernel_size=patch_size,
              stride=overlap_size,
              padding=patch_size // 2,
              bias=False
          )

    self.norm = LayerNorm2d(out_channels)

  def forward(self,x):
    x = self.conv(x)
    x = self.norm(x)
    return x
  
class EfficientMultiHeadAttention(nn.Module):
    '''
    Converts the input resolution to NxC/R
    Apply multi-head attention.
    '''
    def __init__(self, channels: int, reduction_ratio: int = 1, num_heads: int = 8):
        super().__init__()
        self.reducer = nn.Sequential(
            nn.Conv2d(
                channels, channels, kernel_size=reduction_ratio, stride=reduction_ratio
            ),
            LayerNorm2d(channels),
        )
        self.att = nn.MultiheadAttention(
            channels, num_heads=num_heads, batch_first=True
        )

    def forward(self, x):
        _, _, h, w = x.shape
        #print('Original X', x.shape)
        reduced_x = self.reducer(x)
        #print('Reduced X', reduced_x.shape)
        # attention needs tensor of shape (batch, sequence_length, channels)
        reduced_x = rearrange(reduced_x, "b c h w -> b (h w) c")       # [batch_size, query_length, embed_dim]
        #print('Rearrange Reduced X', reduced_x.shape)
        x = rearrange(x, "b c h w -> b (h w) c")
        #print('Rearranged X', x.shape)
        out = self.att(x, reduced_x, reduced_x)[0]      # self.att(query, key, value), Dimension of output is same after nn.MultiheadAttention
        #print('Attention Weights', out.shape)
        # reshape it back to (batch, channels, height, width)
        out = rearrange(out, "b (h w) c -> b c h w", h=h, w=w)
        #print(out.shape)
        return out

class MixMLP(nn.Sequential):
    '''
    Depthwise convolution is performed to increase computation efficiency
    '''
    def __init__(self, channels: int, expansion: int = 4):
        super().__init__(
            # dense layer
            nn.Conv2d(channels, channels, kernel_size=1),
            # depth wise conv
            nn.Conv2d(
                channels,
                channels * expansion,
                kernel_size=3,
                groups=channels,
                padding=1,
            ),
            nn.GELU(),        # Guassian Error Linear Unit Activation
            # dense layer
            nn.Conv2d(channels * expansion, channels, kernel_size=1),
        )


from torchvision.ops import StochasticDepth

class ResidualAdd(nn.Module):
    """
    Just an util layer
    out = f(x) + x
    """
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        out = self.fn(x, **kwargs)
        x = x + out
        return x

class SegFormerEncoderBlock(nn.Sequential):
    def __init__(
        self,
        channels: int,
        reduction_ratio: int = 1,
        num_heads: int = 8,
        mlp_expansion: int = 4,
        drop_path_prob: float = .0
    ):
        super().__init__(
            ResidualAdd(
                nn.Sequential(
                    LayerNorm2d(channels),
                    EfficientMultiHeadAttention(channels, reduction_ratio, num_heads),
                )
            ),
            ResidualAdd(
                nn.Sequential(
                    LayerNorm2d(channels),
                    MixMLP(channels, expansion=mlp_expansion),
                    StochasticDepth(p=drop_path_prob, mode="batch")
                )
            ),
        )


class SegFormerEncoderStage(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: int,
        overlap_size: int,
        drop_probs: List[int],
        depth: int = 2,
        reduction_ratio: int = 1,
        num_heads: int = 8,
        mlp_expansion: int = 4,
    ):
        super().__init__()
        self.overlap_patch_merge = OverlapPatchMerging(
            in_channels, out_channels, patch_size, overlap_size,
        )
        self.blocks = nn.Sequential(
            *[
                SegFormerEncoderBlock(
                    out_channels, reduction_ratio, num_heads, mlp_expansion, drop_probs[i]
                )
                for i in range(depth)
            ]
        )
        self.norm = LayerNorm2d(out_channels)


from typing import Iterable

def chunks(data: Iterable, sizes: List[int]):
    """
    Given an iterable, returns slices using sizes as indices
    """
    curr = 0
    for size in sizes:
        chunk = data[curr: curr + size]
        curr += size
        yield chunk

class SegFormerEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        widths: List[int],
        depths: List[int],
        all_num_heads: List[int],
        patch_sizes: List[int],
        overlap_sizes: List[int],
        reduction_ratios: List[int],
        mlp_expansions: List[int],
        drop_prob: float = .0
    ):
        super().__init__()
        # create drop paths probabilities (one for each stage's block)
        drop_probs =  [x.item() for x in torch.linspace(0, drop_prob, sum(depths))]
        self.stages = nn.ModuleList(
            [
                SegFormerEncoderStage(*args)
                for args in zip(
                    [in_channels, *widths],
                    widths,
                    patch_sizes,
                    overlap_sizes,
                    chunks(drop_probs, sizes=depths),
                    depths,
                    reduction_ratios,
                    all_num_heads,
                    mlp_expansions
                )
            ]
        )

    def forward(self, x):
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features
    


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
    #print(out1.shape)
    out1 = upsample(out1, size = x.size()[-2:])
    #print(out1.shape)

    out2 =self.pool2(x)
    #print(out2.shape)
    out2 = upsample(out2, size = x.size()[-2:])
    #print(out2.shape)

    out3 = self.pool3(x)
    #print(out3.shape)
    out3 = upsample(out3, size = x.size()[-2:])
    #print(out3.shape)

    out4 = self.pool4(x)
    #print(out4.shape)
    out4 = upsample(out4, size = x.size()[-2:])
    #print(out4.shape)

    return torch.cat([x, out1, out2, out3, out4],dim=1)

class PSPNet(nn.Module):
  def __init__(self, n_classes=21):
    super(PSPNet, self).__init__()

    self.in_channels = 2048
    self.depth = self.in_channels//32

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
    out = upsample(x3, size = [64,64] ,align_corners=True)
    return x2


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


#import torch.nn as nn
#
## Assume model_ is an instance of PSPNet
#
## Get the layers of the model
#layers = list(model_.children())
#
## Remove the Conv2d layer at index 2
#layers[-1].pop(2)
#
## Reconstruct the modified model
#modified_model = nn.Sequential(*layers)


class SegFormerDecoderBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, scale_factor: int = 2):
        super().__init__(
            nn.UpsamplingBilinear2d(scale_factor=scale_factor),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
        )


class SegFormerDecoder(nn.Module):
    def __init__(self, out_channels: int, widths: List[int], scale_factors: List[int]):
        super().__init__()
        self.stages = nn.ModuleList(
            [
                SegFormerDecoderBlock(in_channels, out_channels, scale_factor)
                for in_channels, scale_factor in zip(widths, scale_factors)
            ]
        )

    def forward(self, features):
        new_features = []
        for feature, stage in zip(features,self.stages):
            x = stage(feature)
            new_features.append(x)
        return new_features
    

class SegFormerSegmentationHead(nn.Module):
    def __init__(self, channels: int, num_classes: int, num_features: int = 4):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * num_features, channels, kernel_size=1, bias=False),
            nn.ReLU(), # why relu? Who knows
            nn.BatchNorm2d(channels) # why batchnorm and not layer norm? Idk
        )
        self.predict = nn.Conv2d(channels, num_classes, kernel_size=1)

    def forward(self, features):
        x = torch.cat(features, dim=1)
        x = self.fuse(x)
        x = self.predict(x)
        return torch.softmax(x, dim=1)
    

from typing import List, Tuple

class ModelDecoder(nn.Module):
  def __init__(self,
              widths: List[int],
              decoder_channels:int,
              scale_factors: List[int],
              num_classes: int,
              size: Tuple[int, int]):
    super().__init__()
    self.decoder = SegFormerDecoder(out_channels = decoder_channels, widths = widths[::-1], scale_factors=scale_factors)
    self.head = SegFormerSegmentationHead(channels = decoder_channels, num_classes = num_classes, num_features = len(widths))
    self.size = size

  def forward(self, features):
    #print(features)
    features = self.decoder(features[::-1])
    #print(features)
    out = head(features)
    out = F.interpolate(out, size=self.size, mode='bilinear', align_corners=False)
    return out