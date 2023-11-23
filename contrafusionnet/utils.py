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



import torch
import torch.nn.functional as F

def contrastive_loss(q_ob, q_txt, tau):
    """
    Calculates the contrastive loss (LQ→Qtext and LQtext→Q).

    Args:
        q_ob (torch.Tensor): Object query tensor of shape (B, B).
        q_txt (torch.Tensor): Text query tensor of shape (B,).
        tau (float): Temperature parameter.

    Returns:
        torch.Tensor: Total contrastive loss.
    """
    B = q_ob.size(0)  # Batch size

    # Object-to-text contrastive loss (LQ→Qtext)
    logits_qob_txt = torch.matmul(q_ob, q_txt.T) / tau
    numerator_qob_txt = torch.exp(logits_qob_txt)
    denominator_qob_txt = torch.sum(torch.exp(torch.matmul(q_ob, q_txt.T) / tau), dim=0, keepdim=True)
    probabilities_qob_txt = numerator_qob_txt / denominator_qob_txt
    loss_qob_txt = -torch.mean(torch.log(probabilities_qob_txt))

    # Text-to-object contrastive loss (LQtext→Q)
    logits_qtxt_ob = torch.matmul(q_txt, q_ob.T) / tau
    numerator_qtxt_ob = torch.exp(logits_qtxt_ob)
    denominator_qtxt_ob = torch.sum(torch.exp(torch.matmul(q_ob, q_txt.T) / tau), dim=0, keepdim=True)
    probabilities_qtxt_ob = numerator_qtxt_ob / denominator_qtxt_ob
    loss_qtxt_ob = -torch.mean(torch.log(probabilities_qtxt_ob))

    # Total contrastive loss
    total_loss = loss_qob_txt + loss_qtxt_ob
    return total_loss


#def calculate_iou(y_true, y_pred):
#    y_true = y_true.reshape(-1)
#    y_pred = y_pred.reshape(-1)
#    intersection = torch.sum(y_pred*y_true)
#    union = torch.sum(y_true) + torch.sum(y_pred) - intersection
#    #print('Intersecyion', intersection)
#    #print('Union', union)
#    iou = torch.mean(intersection/union)
#    return iou




def calculate_iou(y_true, y_pred):
    # Calculates the IOU

    num_classes = y_true.shape[1]  # Assuming y_true and y_pred are one-hot encoded with shape (batch_size, num_classes, height, width)
    iou = torch.zeros(num_classes)
    
    for i in range(num_classes):
        smooth=1
        y_true_i = y_true[:, i].reshape(-1).long()
        y_pred_i = y_pred[:, i].reshape(-1).long()
        intersection = torch.sum(y_pred_i * y_true_i)
        union = torch.sum(y_true_i) + torch.sum(y_pred_i) - intersection
        iou[i] = (intersection + smooth)/ (union + + smooth)
    
    m_iou = torch.mean(iou)
    return m_iou


def calculate_dice(y_true, y_pred, smooth=1):
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    intersection = torch.sum(y_pred*y_true)
    union = union = torch.sum(y_true) + torch.sum(y_pred)
    #print(union)
    dice = torch.mean((2. * intersection + smooth) / (union + smooth), dim=0)
    return dice


def calculate_confusion_metrics(true_labels, predicted_labels):
    predicted_labels = predicted_labels.reshape(-1)
    true_labels = true_labels.reshape(-1)
    true_positives = torch.sum((predicted_labels == 1) & (true_labels == 1))
    true_negatives = torch.sum((predicted_labels == 0) & (true_labels == 0))
    false_positives = torch.sum((predicted_labels == 1) & (true_labels == 0))
    false_negatives = torch.sum((predicted_labels == 0) & (true_labels == 1))

    true_observations = true_positives + true_negatives
    total_observations = true_positives + true_negatives + false_positives + false_negatives
    accuracy = true_observations / total_observations
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    specificity = true_negatives / (true_negatives + false_positives)
    f1_score = 2 * (precision * recall) / (precision + recall)

    return accuracy, precision, recall, specificity, f1_score

def plot_confusion_matrix(true_labels, predicted_labels, classes):
    true_labels = true_labels.cpu().detach().numpy()
    predicted_labels = predicted_labels.cpu().detach().numpy()
    cm = confusion_matrix(true_labels.reshape(-1), predicted_labels.reshape(-1))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.show()


def one_hot(label_seg, classes):
    # Convert labelled mask to binary label for each class
    label_one_hot = torch.zeros(label_seg.shape[0], classes, label_seg.shape[1], label_seg.shape[2])
    for i in range(classes):
        label_one_hot[:, i, :, :] = (label_seg == i).float()
    return label_one_hot


def decode_segmap(temp):
    #convert gray scale to color
    temp=temp.numpy()
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(0, n_classes):
        r[temp == l] = label_colours[l][0]
        g[temp == l] = label_colours[l][1]
        b[temp == l] = label_colours[l][2]

    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    return rgb