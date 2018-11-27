"""
File: dataset.py
Created by: Qiqi Xiao
Email: xiaoqiqi177<at>gmail<dot>com
"""


import numpy as np
from torchvision import datasets, models, transforms
import torchvision
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import cv2

class IDRIDDataset(Dataset):

    def __init__(self, image_paths, mask_paths=None, class_id=0, transform=None):
        """
        Args:
            image_paths: paths to the original images []
            mask_paths: paths to the mask images, [[]]
            class_id: id of lesions, 0:ex, 1:he, 2:ma, 3:se
        """
        assert len(image_paths) == len(mask_paths)
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.images = [self.pil_loader(image_path) for image_path in image_paths]
        w, h = self.images[0].size
        self.masks = []
        if self.mask_paths is not None:
            for mask_path4 in mask_paths:
                mask_path = mask_path4[class_id]
                if mask_path is None:
                    self.masks.append(Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8)))
                else:
                    self.masks.append(self.pil_loader(mask_path))
        
        self.class_id = class_id
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def pil_loader(self, image_path):
        with open(image_path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def __getitem__(self, idx):
        
        info = [self.images[idx]]
        if self.mask_paths:
            info.append(self.masks[idx])
        if self.transform:
            info = self.transform(info)
        inputs = np.array(info[0])
        if inputs.shape[2] == 3:
            inputs = np.transpose(np.array(info[0]), (2, 0, 1))
            inputs = inputs / 255.
        
        if len(info) > 1:
            mask = np.array(np.array(info[1]))[:, :, 0] / 255.0
            empty_mask = 1 - mask
            masks = np.array([empty_mask, mask])

            return inputs, masks
        else:
            return inputs
