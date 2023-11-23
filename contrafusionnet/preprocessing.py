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
#from torchviz import make_dot


device = "cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(device)


#dataset = Cityscapes("/home/stud1/margi/U_Net/cityscapes_data/cityscape_original", 
#                     split = "train", mode="fine", target_type='semantic')


class Preprocess(Cityscapes):
    '''DataPipeline

    Args:
        root: Path to the directory
        split: 'train' or 'val'
        mode: 'fine' or 'coarse'
        transform: Define 'torchvision.transforms' to apply any transform on the input image
        target_transform: Define 'torchvision.transforms' to apply any transform on the target

    Outputs:    
        Transformed image and segmenataion mask tensor with required size 

    '''
    def __init__(self, root, split='train', mode='fine', target_type='semantic', transform=None, target_transform=None):
        super().__init__(root, split=split, mode=mode, target_type=target_type, transform=transform, target_transform=target_transform)
        
    def __getitem__(self, index: int):
        image = Image.open(self.images[index]).convert('RGB')
        resized_image = image.resize((512, 512), resample=Image.LANCZOS)
        
        targets = []
        for i, t in enumerate(self.target_type):
            if t == 'polygon':
                target = self._load_json(self.targets[index][i])
            else:
                target = Image.open(self.targets[index][i])
                resized_target = target.resize((512, 512), resample=Image.NEAREST)  # Resize the target image

            targets.append(self.encode_segmap(np.array(resized_target)))  # Apply segmentation encoding to the resized target

        target = tuple(targets) if len(targets) > 1 else targets[0]

        if self.transform is not None:
            image = self.transform(resized_image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target
    
    def encode_segmap(self, mask):
        # Remove unwanted classes and rectify the labels of wanted classes
        ignore_index = 255
        void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        valid_classes = [ignore_index, 7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
        class_map = {ignore_index:0, 7: 1, 8: 2, 11: 3, 12: 4, 13: 5, 17: 6, 19: 7, 20: 8, 21: 9, 22: 10, 23: 11, 24: 12, 25: 13, 
                    26: 14, 27: 15, 28: 16, 31: 17, 32: 18, 33: 19}
        for _voidc in void_classes:
            mask[mask == _voidc] = ignore_index
        for _validc in valid_classes:
            if _validc in class_map:
                mask[mask == _validc] = class_map[_validc]
            else:
                mask[mask == _validc] = ignore_index
        return mask


### For testing
#dataset = MyClass("/home/stud1/margi/U_Net/cityscapes_data/cityscape_original", 
#                  split = "train", mode="fine", target_type='semantic', transform = transform)
#
## Lower batch size reduces the computational load on the GPU
#dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
#
#X, Y = next(iter(dataloader))