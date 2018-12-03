
"""
File: test_npy.py
Created by: Qiqi Xiao
Email: xiaoqiqi177<at>gmail<dot>com
"""


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
from unet import UNet
from hednet import HNNNet
from utils import get_images
from dataset import IDRIDDataset
from torchvision import datasets, models, transforms
from transform.transforms_group import *
from torch.utils.data import DataLoader, Dataset
import copy
from logger import Logger
import os
from dice_loss import dice_loss, dice_coeff
from tqdm import tqdm
import matplotlib.pyplot as plt
import config_test as config

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

logdir = config.TEST_OUTPUT_DIR
if config.SAVE_OUTPUT_IMAGES: 
    logger = Logger('./logs', logdir)
if not os.path.exists(logdir):
    os.mkdir(logdir)

net_name = config.NET_NAME
image_size = config.IMAGE_SIZE
image_dir = config.IMAGE_DIR

softmax = nn.Softmax(1)
def eval_model(model, eval_loader):
    model.to(device=device)
    model.eval()
    eval_tot = len(eval_loader)
    vis_images = []
    
    with torch.set_grad_enabled(False):
        batch_id = 0
        for inputs, true_masks in tqdm(eval_loader):
            inputs = inputs.to(device=device, dtype=torch.float)
            true_masks = true_masks.to(device=device, dtype=torch.float)
            bs, _, h, w = inputs.shape
            h_size = (h-1) // image_size + 1
            w_size = (w-1) // image_size + 1
            masks_pred = torch.zeros(true_masks.shape).to(dtype=torch.float)
            for i in range(h_size):
                for j in range(w_size):
                    h_max = min(h, (i+1)*image_size)
                    w_max = min(w, (j+1)*image_size)
                    inputs_part = inputs[:,:, i*image_size:h_max, j*image_size:w_max]
                    if net_name == 'unet':
                        masks_pred[:, :, i*image_size:h_max, j*image_size:w_max] = model(inputs_part).to("cpu")
                    elif net_name == 'hednet':
                        masks_pred[:, :, i*image_size:h_max, j*image_size:w_max] = model(inputs_part)[-1].to("cpu")
        
            masks_pred_softmax = softmax(masks_pred)
            masks_max, _ = torch.max(masks_pred_softmax, 1)
            masks_soft = masks_pred_softmax[:, 1:, :, :]
            np.save(os.path.join(logdir, 'mask_soft_'+str(batch_id)+'.npy'), masks_soft.numpy())
            np.save(os.path.join(logdir, 'mask_true_'+str(batch_id)+'.npy'), true_masks[:,1:].cpu().numpy())
            masks_hard = (masks_pred_softmax == masks_max[:, None, :, :]).to(dtype=torch.float)[:, 1:, :, :]
            images_batch = generate_log_images_full(inputs, true_masks[:, 1:], masks_soft, masks_hard) 
            images_batch = images_batch.to("cpu").numpy()
            vis_images.extend(images_batch)     
            batch_id += 1
    return vis_images

def denormalize(inputs):
    if net_name == 'unet':
        return (inputs * 255.).to(device=device, dtype=torch.uint8)
    else:
        mean = torch.FloatTensor([0.485, 0.456, 0.406]).to(device)
        std = torch.FloatTensor([0.229, 0.224, 0.225]).to(device)
        return ((inputs * std[None, :, None, None] + mean[None, :, None, None])*255.).to(device=device, dtype=torch.uint8)

def generate_log_images_full(inputs, true_masks, masks_soft, masks_hard):
    true_masks = (true_masks * 255.).to(device=device, dtype=torch.uint8)
    masks_soft = (masks_soft * 255.).to(device=device, dtype=torch.uint8)
    masks_hard = (masks_hard * 255.).to(device=device, dtype=torch.uint8)
    inputs = denormalize(inputs)
    bs, _, h, w = inputs.shape
    pad_size = 5
    images_batch = (torch.ones((bs, 3, h*2+pad_size, w*2+pad_size)) * 255.).to(device=device, dtype=torch.uint8)
    
    images_batch[:, :, :h, :w] = inputs
    
    images_batch[:, :, :h, w+pad_size:] = 0
    images_batch[:, 0, :h, w+pad_size:] = true_masks[:, 0, :, :] 
    
    images_batch[:, :, h+pad_size:, :w] = 0
    images_batch[:, 0, h+pad_size:, :w] = masks_soft[:, 0, :, :]
    
    images_batch[:, :, h+pad_size:, w+pad_size:] = 0
    images_batch[:, 0, h+pad_size:, w+pad_size:] = masks_hard[:, 0, :, :]
    return images_batch

if __name__ == '__main__':

    if net_name == 'unet': 
        model = UNet(n_channels=3, n_classes=2)
    else:
        model = HNNNet(pretrained=True, class_number=2)
    
    test_model = config.TEST_MODEL
    if os.path.isfile(test_model):
        print("=> loading checkpoint '{}'".format(test_model))
        checkpoint = torch.load(test_model)
        try:
            model.load_state_dict(checkpoint['state_dict'])
        except:
            model.load_state_dict(checkpoint['g_state_dict'])
        print('Model loaded from {}'.format(test_model))
    else:
        print("=> no checkpoint found at '{}'".format(test_model))
        sys.exit(0)

    eval_image_paths, eval_mask_paths = get_images(image_dir, config.PREPROCESS, phase='test')

    if net_name == 'unet':
        eval_dataset = IDRIDDataset(eval_image_paths, eval_mask_paths, config.CLASS_ID, transform=
                                Compose([
                    ]))
    elif net_name == 'hednet':
        eval_dataset = IDRIDDataset(eval_image_paths, eval_mask_paths, config.CLASS_ID, transform=
                                Compose([
                                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ]))
    eval_loader = DataLoader(eval_dataset, config.TEST_BATCH_SIZE, shuffle=False)
                                
    vis_images = eval_model(model, eval_loader)
    logger.image_summary('eval_images', vis_images, step=0)
