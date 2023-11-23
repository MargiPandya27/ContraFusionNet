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
from model import SegFormerEncoder, ModelDecoder, PSPNet
from preprocessing import Preprocess
from utils import calculate_confusion_metrics, calculate_dice, calculate_iou, one_hot, decode_segmap, contrastive_loss
from torchvision.transforms import Compose, ToTensor, Resize

transform = Compose([
    ToTensor(),
])

target_transform = Compose([
    ToTensor(),
])

#train_data=Preprocess("/home/stud1/margi/U_Net/cityscapes_data/cityscape_original", split = "train", 
#                   mode="fine", target_type='semantic', transform = transform,target_transform=target_transform)
#val_data=Preprocess("/home/stud1/margi/U_Net/cityscapes_data/cityscape_original", split = "val", 
#                 mode="fine", target_type='semantic', transform = transform, target_transform=target_transform)

# Create a DataLoader for the train and validation dataset
#train_dataloader = DataLoader(train_data, batch_size=1, shuffle=True)
#val_dataloader = DataLoader(val_data, batch_size=1, shuffle=True)


device = "cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(device)

in_channels=3
widths=[64, 128, 256, 512]
depths=[3, 4, 6, 3]
all_num_heads=[1, 2, 4, 8]
patch_sizes=[7, 3, 3, 3]
overlap_sizes=[4, 2, 2, 2]
reduction_ratios=[8, 4, 2, 1]
mlp_expansions=[4, 4, 4, 4]
decoder_channels=256
scale_factors=[8, 4, 2, 1]
num_classes=20
drop_prob = 0.2

segformer_encoder = SegFormerEncoder(
                    in_channels,
                    widths,
                    depths,
                    all_num_heads,
                    patch_sizes,
                    overlap_sizes,
                    reduction_ratios,
                    mlp_expansions,
                    drop_prob
                  ).to(device)

#model_path = "/home/stud1/margi/Final_Models/PSPNet.pth"
model_ = PSPNet(n_classes=20).to(device)
#model_.load_state_dict(torch.load(model_path))


# Assume model_ is an instance of PSPNet

# Get the layers of the model
layers = list(model_.children())

# Remove the Conv2d layer at index 2
layers[-1].pop(2)

# Reconstruct the modified model
modified_model = nn.Sequential(*layers)

X, Y = next(iter(train_dataloader))

pspnet_encoder = PSPNet(n_classes = 20).to(device)
segformer_decoder = ModelDecoder(widths,
                                 decoder_channels,
                                 scale_factors,
                                 num_classes, 
                                 size = X.shape[-2:])

lr = 1e-4
criterion = torch.nn.CrossEntropyLoss()

lambda1 = 1e-6
lambda2 = 0
i=0
epochs=100

epoch_losses = []
epoch_dices = []
epoch_ious = []
epoch_accs = []

for epoch in range(epochs):
  epoch_loss = 0
  epoch_dice = 0
  epoch_iou = 0
  epoch_acc = 0
  for X, Y in tqdm(dataloader):

    X = X.to(device)

    optimizer_encoder = optim.Adam(segformer_encoder.parameters(), lr=1e-4)
    optimizer_decoder = optim.Adam(segformer_decoder.parameters(), lr=1e-4)

    optimizer_encoder.zero_grad()
    optimizer_decoder.zero_grad()

    # Ouptuts
    seg_encoder = segformer_encoder(X)
    psp_encoder = modified_model(X).cpu()
    seg_encoder = [feature.cpu() for feature in seg_encoder]
    seg_decoder = segformer_decoder(seg_encoder)
    #print(seg_encoder[3].shape, psp_encoder.shape)
    flatten_seg_encoder = torch.flatten(seg_encoder[3], start_dim=1)
    flatten_psp_encoder = torch.flatten(psp_encoder, start_dim=1)
    #print(flatten_psp_encoder.shape,  flatten_seg_encoder.shape)
    Y_encode = one_hot(Y, classes=20)
    #print(Y.shape, seg_decoder.shape)
  

    # Losses
    dice = calculate_dice(seg_decoder, Y_encode)
    loss_decoder = criterion(seg_decoder.permute(0, 2, 3, 1).reshape(-1, 20).float(), Y.view(-1))
    contrastive_loss_value = contrastive_loss(flatten_seg_encoder, flatten_psp_encoder, tau)
    loss = loss_decoder + (1 - dice) * lambda1 + lambda2 * contrastive_loss_value
    iou = calculate_iou(Y_encode, seg_decoder)
    #print(Y_encode.dtype, seg_decoder.dtype)
    #print(iou)
    dice = calculate_dice(Y_encode, seg_decoder)
    accuracy, precision, recall, specificity, f1_score = calculate_confusion_metrics(Y_encode, seg_decoder)

    loss.backward()

    optimizer_encoder.step()
    optimizer_decoder.step()

    epoch_loss += loss.item()
    epoch_dice += dice.item()
    epoch_iou += iou.item()
    epoch_acc += accuracy.item()

  if epoch%40==0:
      model_name = f"DFNet_encoder{epoch+1}.pth"
      torch.save(segformer_encoder.state_dict(), model_name)
      model_name = f"DFNet_decoder{epoch+1}.pth"
      torch.save(segformer_decoder.state_dict(), model_name)
      #visualize(seg_decoder[0], Y_encode [0])


  #if i%50==0:
  label_img = torch.squeeze(Y,dim=1)[0]
  output_img = torch.argmax(seg_decoder, dim=1)[0]
  #Display label and output images
  fig, axes = plt.subplots(1, 2,figsize = (10,10))
  #axes[0].imshow(label_img, cmap='jet')
  axes[0].imshow(decode_segmap(label_img))
  axes[0].set_title('Label')
  axes[0].set_axis_off()
  #axes[1].imshow(output_img, cmap='jet')
  axes[1].imshow(decode_segmap(output_img))
  axes[1].set_title('Output')
  axes[1].set_axis_off()
  plt.show()
  plt.close()
  #i = i + 1

  epoch_losses.append(epoch_loss/len(dataloader))
  epoch_dices.append(epoch_dice/len(dataloader))
  epoch_ious.append(epoch_iou/len(dataloader))
  epoch_accs.append(epoch_acc/len(dataloader))
  print(f'Training: Epoch {epoch+1}/{epochs} | Loss: {epoch_losses[epoch]} | Dice Co-Efficient: {epoch_dices[epoch]} | IOU: {epoch_ious[epoch]} | Accuracy: {epoch_accs[epoch]}')




# Increase plot width, font size, and line weight
matplotlib.rcParams['figure.figsize'] = (20, 5)
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['lines.linewidth'] = 2.5

# Plotting code
fig, axes = plt.subplots(1, 4)
axes[0].plot(epoch_losses, linewidth=2.5)
axes[0].set_title('Training Loss', fontweight='bold')
axes[0].set_xlabel('Epochs', fontweight='bold')
axes[0].set_ylabel('Cross_Entropy_loss', fontweight='bold')
axes[0].tick_params(axis='both', which='both')

axes[1].plot(epoch_dices, linewidth=2.5)
axes[1].set_title('Training Dice Coefficient', fontweight='bold')
axes[1].set_xlabel('Epochs', fontweight='bold')
axes[1].set_ylabel('Dice_Coefficient', fontweight='bold')
axes[1].tick_params(axis='both', which='both')

axes[2].plot(epoch_ious, linewidth=2.5)
axes[2].set_title('Training IOU', fontweight='bold')
axes[2].set_xlabel('Epochs', fontweight='bold')
axes[2].set_ylabel('mIOU', fontweight='bold')
axes[2].tick_params(axis='both', which='both')

axes[3].plot(epoch_accs, linewidth=2.5)
axes[3].set_title('Training Accuracy', fontweight='bold')
axes[3].set_xlabel('Epochs', fontweight='bold')
axes[3].set_ylabel('Accuracy', fontweight='bold')
axes[3].tick_params(axis='both', which='both')

plt.tight_layout()  # Adjust the spacing between subplots
plt.show()
