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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger = Logger('./logs', 'drlog')
dir_checkpoint = './models'
def eval_model(model, eval_loader):
    model.eval()
    tot = 0
    for inputs, true_masks in eval_loader:
        inputs = inputs.to(device=device, dtype=torch.float)
        true_masks = true_masks.to(device=device)
        masks_pred = model(inputs)
        _, mask_indices = torch.max(masks_pred, 1)
        mask_size = true_masks.shape[0] * true_masks.shape[1] * true_masks.shape[2]
        correct = torch.sum(mask_indices == true_masks).float() / mask_size 
        tot += correct.item()
    return tot / len(eval_dataset)

disp_interval = 1
def train_model(model, train_loader, eval_loader, criterion, optimizer, scheduler, batch_size, num_epochs=5):
    best_model = copy.deepcopy(model.state_dict())
    best_acc = 0.
    model.to(device=device)
    batch_step_count = 0
    for epoch in range(num_epochs):
        print('Starting epoch {}/{}.'.format(epoch + 1, num_epochs))
        scheduler.step()
        model.train()
        epoch_loss = 0
        N_train = len(train_dataset)
        for inputs, true_masks in train_loader:
            inputs = inputs.to(device=device, dtype=torch.float)
            true_masks = true_masks.to(device=device)

            masks_pred = model(inputs)

            masks_pred_transpose = masks_pred.permute(0, 2, 3, 1)
            masks_pred_flat = masks_pred_transpose.reshape(-1, masks_pred_transpose.shape[-1])
            true_masks_flat = true_masks.reshape(-1)
            loss = criterion(masks_pred_flat, true_masks_flat.long())
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            batch_step_count += 1
            
            if batch_step_count % disp_interval == 0:
                logger.scalar_summary('train_loss', epoch_loss / batch_step_count, step=batch_step_count)

        val_dice = eval_model(model, eval_loader)

        print('Validation Dice Coeff: {}'.format(val_dice))
        logger.scalar_summary('val_dice', val_dice, step=batch_step_count)

        if not os.path.exists(dir_checkpoint):
            os.mkdir(dir_checkpoint)
        torch.save(model.state_dict(),
                       os.path.join(dir_checkpoint, 'model_{}.pth'.format(epoch + 1)))
        print('Checkpoint {} saved !'.format(epoch + 1))

def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=5, type='int',
                      help='number of epochs')
    parser.add_option('-b', '--batch-size', dest='batchsize', default=1,
                      type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.1,
                      type='float', help='learning rate')
    parser.add_option('-c', '--load', dest='load',
                      default=False, help='load file model')

    (options, args) = parser.parse_args()
    return options

if __name__ == '__main__':
    args = get_args()

    model = UNet(n_channels=3, n_classes=5)
    
    if args.load:
        model.load_state_dict(torch.load(args.load))
        print('Model loaded from {}'.format(args.load))
    
    image_dir = '/media/hdd1/qiqix/IDRID/Sub1'
    train_image_paths, train_mask_paths = get_images(image_dir, 'train')
    eval_image_paths, eval_mask_paths = get_images(image_dir, 'eval')

    train_dataset = IDRIDDataset(train_image_paths, train_mask_paths, 4, transform=
                                Compose([
                                RandomRotation(20),
                                RandomCrop(512),
                    ]))
    eval_dataset = IDRIDDataset(eval_image_paths, eval_mask_paths, 4, transform=
                                Compose([
                                RandomCrop(512),
                    ]))

    train_loader = DataLoader(train_dataset, args.batchsize, shuffle=True)
    eval_loader = DataLoader(eval_dataset, args.batchsize, shuffle=False)

    optimizer = optim.SGD(model.parameters(),
                              lr=args.lr,
                              momentum=0.9,
                              weight_decay=0.0005)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    criterion = nn.CrossEntropyLoss()
    
    train_model(model, train_loader, eval_loader, criterion, optimizer, scheduler, args.batchsize, num_epochs=args.epochs)
