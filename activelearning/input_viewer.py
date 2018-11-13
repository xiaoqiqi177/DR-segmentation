import sys
import os
import cv2
import numpy as np
from utils import get_images
from utils_diaret import  get_images_diaretAL
from dataset import IDRIDDataset, DiaretALDataset, MixedDataset
from torchvision import datasets, models, transforms
from transform.transforms_group import *
from torch.utils.data import DataLoader
from optparse import OptionParser

parser = OptionParser()
parser.add_option('-n', '--net-name', dest='netname', default='unet',
                    type='str', help='net name:unet/hednet')
parser.add_option('-g', '--preprocess', dest='preprocess', action='store_true',
                      default=False, help='preprocess input images')
(args, _) = parser.parse_args()
net_name = args.netname

image_dir2 = '/home/qiqix/Diaretdb1/resources/images/'
predicted_dir = './output'

train_image_paths2, train_mask_paths2, train_predicted_mask_paths2 = get_images_diaretAL(image_dir2, predicted_dir, args.preprocess, phase='train')

rotation_angle = 20
if True:
    if net_name == 'unet':
        # will not happen.
        pass
    elif net_name == 'hednet':
        train_dataset = DiaretALDataset(train_image_paths2, train_mask_paths2, train_predicted_mask_paths2, 4, transform=
                                Compose([
                                RandomRotation(rotation_angle),
                                RandomCrop(512),
                                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ]))
train_loader = DataLoader(train_dataset, 1, shuffle=True)

for inputs, true_masks in train_loader:
    if net_name == 'unet':
        input_img = np.uint8(np.transpose(inputs[0], (1, 2, 0)) * 255.)[:,:,::-1]
    else:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        input_img = np.uint8((np.array(np.transpose(inputs[0], (1, 2, 0))) * std[None, None, :] + mean[None, None, :])*255.)[:,:,::-1]
    h, w = input_img.shape[:2]
    showimg = np.zeros((h, w*6, 3), dtype=np.uint8)
    showimg[:, :w, :] = input_img
    for i in range(1, 6):
        mask_img = np.uint8(true_masks[0, i] * 255.)
        for j in range(3):
            showimg[:, i*w:(i+1)*w, j] = mask_img
    cv2.imshow('img', showimg)
    cv2.waitKey(0)
