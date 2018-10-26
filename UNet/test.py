import sys
import os
from optparse import OptionParser
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from torch.optim import lr_scheduler
from dice_loss import dice_coeff
from unet import UNet
from utils import get_images
from dataset import IDRIDDataset
from torchvision import datasets, models, transforms
from transform.transforms_group import *
from torch.utils.data import DataLoader, Dataset
import copy
from logger import Logger
import os
import argparse

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dir_checkpoint = './models'
def test_model(model, test_loader):
    model.to(device = device)
    model.eval()
    cropsize = 512
    for inputs in test_loader:
        h, w = inputs.shape[-2:]
        h_number = (h - 1) // cropsize + 1
        w_number = (w - 1) // cropsize + 1
        pad_inputs = torch.zeros((inputs.shape[0], inputs.shape[1], h_number * cropsize, w_number*cropsize))
        pad_inputs[:,:,:h,:w] = inputs
        whole_mask_indices = np.zeros((1, h_number*cropsize, w_number*cropsize))
        for i in range(h_number):
            for j in range(w_number):
                input_grid = pad_inputs[:, :, i*cropsize:(i+1)*cropsize, j*cropsize:(j+1)*cropsize]
                input_grid = input_grid.to(device=device, dtype=torch.float)
                masks_pred = model(input_grid)
                _, mask_indices = torch.max(masks_pred, 1)
                whole_mask_indices[:,i*cropsize:(i+1)*cropsize, j*cropsize:(j+1)*cropsize] = np.array(mask_indices)
        whole_mask_indices = whole_mask_indices[:, :h, :w]

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default='MODEL.pth',
                        metavar='FILE',
                        help="Specify the file in which is stored the model"
                             " (default : 'MODEL.pth')")
    parser.add_argument('--no-crf', '-r', action='store_true',
                        help="Do not use dense CRF postprocessing",
                        default=False)
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    model = UNet(n_channels=3, n_classes=5)
    print("Loading model {}".format(args.model))
    model.load_state_dict(torch.load(args.model))
    
    image_dir = '/media/hdd1/qiqix/IDRID/Sub1'
    test_image_paths, eval_mask_paths = get_images(image_dir, 'test')
    test_dataset = IDRIDDataset(test_image_paths, eval_mask_paths, 4)
    test_loader = DataLoader(test_dataset, 1, shuffle=False)
    test_model(model, test_loader)
