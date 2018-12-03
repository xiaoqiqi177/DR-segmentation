"""
File: input_viewer.py
Created by: Qiqi Xiao
Email: xiaoqiqi177<at>gmail<dot>com
"""

import sys
import os
import cv2
import numpy as np
from utils import get_images
from dataset import IDRIDDataset
from transform.transforms_group import *
from torch.utils.data import DataLoader
from optparse import OptionParser

parser = OptionParser()
parser.add_option('-n', '--net-name', dest='netname', default='unet',
                    type='str', help='net name:unet/hednet')
(args, _) = parser.parse_args()
net_name = args.netname

image_dir = '/home/qiqix/SegmentationSub1'
train_image_paths, train_mask_paths = get_images(image_dir, 'train')
eval_image_paths, eval_mask_paths = get_images(image_dir, 'eval')

if True:
    if net_name == 'unet':
        train_dataset = IDRIDDataset(train_image_paths, train_mask_paths, 1, transform=
                                Compose([
                                RandomRotation(20),
                                #ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                                RandomCrop(512),
                    ]))
        eval_dataset = IDRIDDataset(eval_image_paths, eval_mask_paths, 1, transform=
                                Compose([
                    ]))
    elif net_name == 'hednet':
        train_dataset = IDRIDDataset(train_image_paths, train_mask_paths, 1, transform=
                                Compose([
                                RandomRotation(20),
                                RandomCrop(512),
                                #ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ]))
        eval_dataset = IDRIDDataset(eval_image_paths, eval_mask_paths, 1, transform=
                                Compose([
                                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ]))

train_loader = DataLoader(train_dataset, 1, shuffle=True)
eval_loader = DataLoader(eval_dataset, 1, shuffle=False)

for inputs, true_masks in eval_loader:
    if net_name == 'unet':
        input_img = np.uint8(np.transpose(inputs[0], (1, 2, 0)) * 255.)[:,:,::-1]
    else:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        input_img = np.uint8((np.array(np.transpose(inputs[0], (1, 2, 0))) * std[None, None, :] + mean[None, None, :])*255.)[:,:,::-1]
    h, w = input_img.shape[:2]
    showimg = np.zeros((h, w*2, 3), dtype=np.uint8)
    showimg[:, :w, :] = input_img
    for i in range(1, 2):
        mask_img = np.uint8(true_masks[0, i] * 255.)
        for j in range(3):
            showimg[:, i*w:(i+1)*w, j] = mask_img
    cv2.imshow('img', showimg)
    cv2.waitKey(0)
