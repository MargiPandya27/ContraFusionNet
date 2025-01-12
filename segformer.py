# the train file for segformer model

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
from torchvision import transforms, models
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from torchvision.datasets import Cityscapes
from torchvision.transforms import Compose, ToTensor, Resize
from model.segformer import SegFormer
from preprocess.preprocessing import Preprocess
from utils.utils import calculate_confusion_metrics, calculate_dice, calculate_iou, one_hot, decode_segmap
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


model = SegFormer(
    in_channels=3,
    widths=[64, 128, 256, 512],
    depths=[3, 4, 6, 3],
    all_num_heads=[1, 2, 4, 8],
    patch_sizes=[7, 3, 3, 3],
    overlap_sizes=[4, 2, 2, 2],
    reduction_ratios=[8, 4, 2, 1],
    mlp_expansions=[4, 4, 4, 4],
    decoder_channels=256,
    scale_factors=[8, 4, 2, 1],
    num_classes=20,
).to(device)

lr = 1e-4
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)


epochs = 100
step_losses = []
epoch_losses = []
epoch_dices = []
epoch_ious = []
epoch_accs = []
for epoch in range(epochs):
    epoch_loss = 0
    epoch_dice = 0
    epoch_iou = 0
    epoch_acc = 0
    for X, Y in tqdm(train_dataloader):
        X = X.to(device)
        Y=torch.squeeze(Y, dim=1)*255
        #print(Y.view(-1).long().shape)
        optimizer.zero_grad()
        Y_pred = model(X).cpu()
        #print(Y_pred.permute(0, 2, 3, 1).reshape(-1, 20).float().shape)
        Y_encode = one_hot(Y, classes=20)
        loss = criterion(Y_pred.permute(0, 2, 3, 1).reshape(-1, 20).float(), Y.view(-1).long())
        iou = calculate_iou(Y_encode, (Y_pred>0.5))
        dice = calculate_dice(Y_encode, (Y_pred>0.5))
        acc, precision, recall, specificity, f1_score = calculate_confusion_metrics((Y_encode > 0.5), (Y_pred > 0.5))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_dice += dice.item()
        epoch_iou += iou.item()
        epoch_acc += acc.item()
        step_losses.append(loss.item())
    #if epoch%5==0:
    #    model_name = f"DeepLabv3_{epoch+1}.pth"
    #    torch.save(model.state_dict(), model_name)

    # Visualize label and output images
    label_img = torch.squeeze(Y,dim=1)[0]
    output_img = torch.argmax(Y_pred, dim=1)[0]
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

    epoch_losses.append(epoch_loss/len(train_data))
    epoch_dices.append(epoch_dice/len(train_data))
    epoch_ious.append(epoch_iou/len(train_data))
    epoch_accs.append(epoch_acc/len(train_data))
    print(f'Training: Epoch {epoch+1}/{epochs} | Loss: {epoch_losses[epoch]} | Dice Co-Efficient: {epoch_dices[epoch]} | IOU: {epoch_ious[epoch]} | Accuracy: {epoch_accs[epoch]}')
    #print(f'Epoch {epoch+1}/{epochs} | Loss: {epoch_losses[epoch]}')




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










