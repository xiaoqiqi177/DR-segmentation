"""
File: train.py
Created by: Qiqi Xiao
Email: xiaoqiqi177<at>gmail<dot>com
"""

import sys
from torch.autograd import Variable
import os
from optparse import OptionParser
import numpy as np
from tqdm import tqdm
import copy

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim import lr_scheduler

import config_gan as config
from unet import UNet
from hednet import HNNNet
from dnet import DNet
from utils import get_images
from dataset import IDRIDDataset
from torchvision import datasets, models, transforms
from transform.transforms_group import *
from torch.utils.data import DataLoader, Dataset
from logger import Logger

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

logger = Logger('./logs', config.LOG_DIR)
dir_checkpoint = config.MODELS_DIR
net_name = config.NET_NAME
lesions = [config.LESION_NAME]
rotation_angle = config.ROTATION_ANGEL
image_size = config.IMAGE_SIZE
image_dir = config.IMAGE_DIR
batchsize = config.TRAIN_BATCH_SIZE
try:
    gan_weight = config.D_WEIGHT
except:
    gan_weight = 0.

softmax = nn.Softmax(1)
g_start = False
def eval_model(model, eval_loader, criterion):
    model.eval()
    eval_tot = len(eval_loader)
    eval_loss_ce = 0.

    with torch.set_grad_enabled(False):
        for inputs, true_masks in eval_loader:
            inputs = inputs.to(device=device, dtype=torch.float)
            true_masks = true_masks.to(device=device, dtype=torch.float)
            if net_name == 'unet':
                masks_pred = model(inputs)
            elif net_name == 'hednet':
                masks_pred = model(inputs)[-1]
        
            masks_pred_transpose = masks_pred.permute(0, 2, 3, 1)
            masks_pred_flat = masks_pred_transpose.reshape(-1, masks_pred_transpose.shape[-1])
            true_masks_indices = torch.argmax(true_masks, 1)
            true_masks_flat = true_masks_indices.reshape(-1)
            loss_ce = criterion(masks_pred_flat, true_masks_flat.long())
            eval_loss_ce += loss_ce

            masks_pred_softmax = softmax(masks_pred) 
        return eval_loss_ce / eval_tot

def denormalize(inputs):
    if net_name == 'unet':
        return (inputs * 255.).to(device=device, dtype=torch.uint8)
    else:
        mean = torch.FloatTensor([0.485, 0.456, 0.406]).to(device)
        std = torch.FloatTensor([0.229, 0.224, 0.225]).to(device)
        return ((inputs * std[None, :, None, None] + mean[None, :, None, None])*255.).to(device=device, dtype=torch.uint8)

    
def generate_log_images(inputs_t, true_masks_t, masks_pred_softmax_t):
    true_masks = (true_masks_t * 255.).to(device=device, dtype=torch.uint8)
    masks_pred_softmax = (masks_pred_softmax_t.detach() * 255.).to(device=device, dtype=torch.uint8)
    inputs = denormalize(inputs_t)
    bs, _, h, w = inputs.shape
    pad_size = 5
    images_batch = (torch.ones((bs, 3, h, w*3+pad_size*2)) * 255.).to(device=device, dtype=torch.uint8)
    
    images_batch[:, :, :, :w] = inputs
    
    images_batch[:, :, :, w+pad_size:w*2+pad_size] = 0
    images_batch[:, 0, :, w+pad_size:w*2+pad_size] = true_masks[:, 1, :, :]
    
    images_batch[:, :, :, w*2+pad_size*2:] = 0
    images_batch[:, 0, :, w*2+pad_size*2:] = masks_pred_softmax[:, 1, :, :]
    return images_batch
  
def image_to_patch(image, patch_size):
    bs, channel, h, w = image.shape
    return (image.reshape((bs, channel, h//patch_size, patch_size, w//patch_size, patch_size))
            .permute(2, 4, 0, 1, 3, 5)
            .reshape((-1, channel, patch_size, patch_size)))


def train_model(model, dnet, train_loader, eval_loader, criterion, g_optimizer, g_scheduler, d_optimizer, d_scheduler, batch_size, num_epochs=5, start_epoch=0, start_step=0):
    model.to(device=device)
    dnet.to(device=device)
    tot_step_count = start_step
    for epoch in range(start_epoch, start_epoch+num_epochs):
        print('Starting epoch {}/{}.'.format(epoch + 1, start_epoch+num_epochs))
        g_scheduler.step()
        d_scheduler.step()
        model.train()
        dnet.train()
        epoch_loss_ce = 0
        epoch_loss_d = 0
        epoch_loss_gan = 0
        N_train = len(train_dataset)
        batch_step_count = 0
        vis_images = []
        for inputs, true_masks in tqdm(train_loader):
            inputs = inputs.to(device=device, dtype=torch.float)
            true_masks = true_masks.to(device=device, dtype=torch.float)

            if net_name == 'unet':
                masks_pred = model(inputs)
            elif net_name == 'hednet':
                masks_pred = model(inputs)[-1]

            masks_pred_transpose = masks_pred.permute(0, 2, 3, 1)
            masks_pred_flat = masks_pred_transpose.reshape(-1, masks_pred_transpose.shape[-1])
            true_masks_indices = torch.argmax(true_masks, 1)
            true_masks_flat = true_masks_indices.reshape(-1)
            loss_ce = criterion(masks_pred_flat, true_masks_flat.long())
            masks_pred_softmax = softmax(masks_pred) 
            
            # Save images
            if (epoch + 1) % 20 == 0:
                images_batch = generate_log_images(inputs, true_masks, masks_pred_softmax) 
                vis_images.extend(images_batch)

            ce_weight = 1.
            g_loss = loss_ce * ce_weight
            
            epoch_loss_ce += loss_ce.item() * ce_weight
            epoch_loss_tot = epoch_loss_ce * ce_weight

            # add descriminator loss
            dnet.train()
            if config.D_MULTIPLY:
                input_real = torch.matmul(inputs, true_masks[:, 1:, :, :])
                input_real = image_to_patch(input_real, config.PATCH_SIZE)
                input_fake = torch.matmul(inputs, masks_pred_softmax[:, 1:, :, :])
                input_fake = image_to_patch(input_fake, config.PATCH_SIZE)
            else:
                input_real = torch.cat((inputs, true_masks[:, 1:, :, :]), 1)
                input_real = image_to_patch(input_real, config.PATCH_SIZE)
                input_fake = torch.cat((inputs, masks_pred_softmax[:, 1:, :, :]), 1)
                input_fake = image_to_patch(input_fake, config.PATCH_SIZE)
            d_real = dnet(input_real)
            d_fake = dnet(input_fake.detach()) #do not backward to generator
            d_real_loss = -torch.mean(d_real)
            d_fake_loss = torch.mean(d_fake)
            
            #update d loss
            loss_d = d_real_loss + d_fake_loss
            print('loss_d: ', loss_d.item())
            epoch_loss_d += loss_d.item()
            d_optimizer.zero_grad()
            loss_d.backward()
            d_optimizer.step()
            
            #updage g loss
            d_fake = dnet(input_fake) #do backward to generator
            loss_gan = -torch.mean(d_fake)
            epoch_loss_gan += loss_gan.item() * gan_weight
            epoch_loss_tot += epoch_loss_gan
            g_loss += loss_gan * gan_weight

            print('loss_ce: ', loss_ce.item())
            print('loss_gan: ', loss_gan.item())
            print('g_loss: ', g_loss.item())
            
            if loss_ce.item() < -0.1:
                g_start = True

            if g_start is True:
                g_optimizer.zero_grad()
                g_loss.backward()
                g_optimizer.step()
            
            batch_step_count += 1
            tot_step_count += 1
        
        # Traning logs
        logger.scalar_summary('train_loss_ce', epoch_loss_ce / batch_step_count, step=tot_step_count)
        logger.scalar_summary('train_loss_d', epoch_loss_d / batch_step_count, step=tot_step_count)
        logger.scalar_summary('train_loss_tot', epoch_loss_tot / batch_step_count, step=tot_step_count)
        
        # Validation logs
        eval_loss_ce = eval_model(model, eval_loader, criterion)
        logger.scalar_summary('eval_loss_ce', eval_loss_ce, step=tot_step_count)
        if not os.path.exists(dir_checkpoint):
            os.mkdir(dir_checkpoint)
        if (epoch + 1) % 20 == 0:
            if dnet:
                state = {
                    'epoch': epoch,
                    'step': tot_step_count,
                    'g_state_dict': model.state_dict(),
                    'd_state_dict': dnet.state_dict(),
                    'g_optimizer': g_optimizer.state_dict(),
                    'd_optimizer': d_optimizer.state_dict(),
                    }
            else:
                state = {
                    'epoch': epoch,
                    'step': tot_step_count,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                    }
            torch.save(state,
                        os.path.join(dir_checkpoint, 'model_{}.pth.tar'.format(epoch + 1)))
            print('Checkpoint {} saved !'.format(epoch + 1))
            logger.image_summary('train_images', [vis_image.cpu().numpy() for vis_image in vis_images], step=tot_step_count)


if __name__ == '__main__':

    if net_name == 'unet': 
        model = UNet(n_channels=3, n_classes=2)
    else:
        model = HNNNet(pretrained=True, class_number=2)
   
    if config.D_MULTIPLY:
        dnet = DNet(input_dim=3, output_dim=1, input_size=config.PATCH_SIZE)
    else:
        dnet = DNet(input_dim=4, output_dim=1, input_size=config.PATCH_SIZE)

    g_optimizer = optim.SGD(model.parameters(),
                              lr=config.G_LEARNING_RATE,
                              momentum=0.9,
                              weight_decay=0.0005)
    d_optimizer = optim.SGD(dnet.parameters(),
                              lr=config.D_LEARNING_RATE,
                              momentum=0.9,
                              weight_decay=0.0005)
    resume = config.RESUME_MODEL
    if resume:
        if os.path.isfile(resume):
            print("=> loading checkpoint '{}'".format(resume))
            checkpoint = torch.load(resume)
            start_epoch = checkpoint['epoch']+1
            start_step = checkpoint['step']
            try:
                model.load_state_dict(checkpoint['state_dict'])
                g_optimizer.load_state_dict(checkpoint['optimizer'])
            except:
                model.load_state_dict(checkpoint['g_state_dict'])
                dnet.load_state_dict(checkpoint['d_state_dict'])
                g_optimizer.load_state_dict(checkpoint['g_optimizer'])
                d_optimizer.load_state_dict(checkpoint['d_optimizer'])
            print('Model loaded from {}'.format(resume))
        else:
            print("=> no checkpoint found at '{}'".format(resume))
    else:
        start_epoch = 0
        start_step = 0

    train_image_paths, train_mask_paths = get_images(image_dir, config.PREPROCESS, phase='train')
    eval_image_paths, eval_mask_paths = get_images(image_dir, config.PREPROCESS, phase='eval')

    if net_name == 'unet':
        train_dataset = IDRIDDataset(train_image_paths, train_mask_paths, config.CLASS_ID, transform=
                                Compose([
                                RandomRotation(rotation_angle),
                                #ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                                RandomCrop(image_size),
                    ]))
        eval_dataset = IDRIDDataset(eval_image_paths, eval_mask_paths, config.CLASS_ID, transform=
                                Compose([
                                RandomCrop(image_size),
                    ]))
    elif net_name == 'hednet':
        train_dataset = IDRIDDataset(train_image_paths, train_mask_paths, config.CLASS_ID, transform=
                                Compose([
                                RandomRotation(rotation_angle),
                                RandomCrop(image_size),
                                #ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ]))
        eval_dataset = IDRIDDataset(eval_image_paths, eval_mask_paths, config.CLASS_ID, transform=
                                Compose([
                                RandomCrop(image_size),
                                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ]))
    train_loader = DataLoader(train_dataset, batchsize, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batchsize, shuffle=False)

    g_scheduler = lr_scheduler.StepLR(g_optimizer, step_size=200, gamma=0.9)
    d_scheduler = lr_scheduler.StepLR(d_optimizer, step_size=100, gamma=0.9)
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(config.CROSSENTROPY_WEIGHTS).to(device))
    
    train_model(model, dnet, train_loader, eval_loader, criterion, g_optimizer, g_scheduler, d_optimizer, d_scheduler, batchsize, \
            num_epochs=config.EPOCHES, start_epoch=start_epoch, start_step=start_step)
