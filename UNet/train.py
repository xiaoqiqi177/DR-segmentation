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
from utils import get_images
from dataset import IDRIDDataset
from torchvision import datasets, models, transforms
from transform.transforms_group import *
from torch.utils.data import DataLoader, Dataset
import copy
from logger import Logger
import os
from dice_loss import dice_loss, dice_coeff

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger = Logger('./logs', 'drlog')
dir_checkpoint = './models'
def eval_model(model, eval_loader):
    model.eval()
    fg_tot = 0
    bg_tot = 0
    eval_tot = len(eval_loader)
    dice_coeffs = np.zeros(4)
    for inputs, true_masks in eval_loader:
        if len(inputs) == 0:
            eval_tot -= 1
            continue

        inputs = inputs.to(device=device, dtype=torch.float)
        true_masks = true_masks.to(device=device, dtype=torch.float)
        
        masks_pred = model(inputs)
        
        _, mask_indices = torch.max(masks_pred, 1)
        _, true_masks_indices= torch.max(true_masks, 1)
        mask_size_bg = torch.sum(true_masks_indices == 0)
        mask_size_fg = true_masks_indices.shape[0] * true_masks_indices.shape[1] * true_masks_indices.shape[2] - torch.sum(true_masks_indices == 0)
        if mask_size_fg > 0: 
            correct_fg = (torch.sum((mask_indices == true_masks_indices) *(true_masks_indices > 0)).float()) / (mask_size_fg.float())
            fg_tot += correct_fg.item()
        if mask_size_bg > 0:
            correct_bg = (torch.sum((mask_indices == true_masks_indices) *(true_masks_indices == 0)).float()) / (mask_size_bg.float())
            bg_tot += correct_bg.item()
        
        dice_coeffs += dice_coeff(masks_pred[:, 1:, :, :], true_masks[:, 1:, :, :])
    return bg_tot / eval_tot, fg_tot / eval_tot, dice_coeffs / eval_tot

lesions = ['ex', 'he', 'ma', 'se']
softmax = nn.Softmax(1)
def train_model(model, train_loader, eval_loader, criterion, optimizer, scheduler, batch_size, num_epochs=5):
    best_model = copy.deepcopy(model.state_dict())
    best_acc = 0.
    model.to(device=device)
    tot_step_count = 0
    for epoch in range(num_epochs):
        print('Starting epoch {}/{}.'.format(epoch + 1, num_epochs))
        scheduler.step()
        model.train()
        epoch_loss_ce = 0
        epoch_losses_dice = [0, 0, 0, 0]
        N_train = len(train_dataset)
        batch_step_count = 0
        for inputs, true_masks in train_loader:
            if len(inputs) == 0:
                continue
            inputs = inputs.to(device=device, dtype=torch.float)
            true_masks = true_masks.to(device=device, dtype=torch.float)

            masks_pred = model(inputs)
            masks_pred = softmax(masks_pred) 
            losses_dice = dice_loss(masks_pred[:, 1:, :, :], true_masks[:, 1:, :, :])

            masks_pred_transpose = masks_pred.permute(0, 2, 3, 1)
            masks_pred_flat = masks_pred_transpose.reshape(-1, masks_pred_transpose.shape[-1])
            true_masks_indices = torch.argmax(true_masks, 1)
            true_masks_flat = true_masks_indices.reshape(-1)
            loss_ce = criterion(masks_pred_flat, true_masks_flat.long())
            
            ce_weight = 1.
            #lesion_dice_weights = [0., 0., 0., 0.]
            lesion_dice_weights = [1., 1., 1., 1.]
            loss = loss_ce * ce_weight
            for lesion_dice_weight, loss_dice in zip(lesion_dice_weights, losses_dice):
                loss += loss_dice * lesion_dice_weight
            
            epoch_loss_ce += loss_ce.item()
            epoch_loss_tot = epoch_loss_ce
            for i, loss_dice in enumerate(losses_dice):
                epoch_losses_dice[i] += losses_dice[i].item()
                epoch_loss_tot += epoch_losses_dice[i]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            batch_step_count += 1
            tot_step_count += 1
        
        logger.scalar_summary('train_loss_ce', epoch_loss_ce / batch_step_count, step=tot_step_count)
        for lesion, epoch_loss_dice in zip(lesions, epoch_losses_dice):
            logger.scalar_summary('train_loss_dice_'+lesion, epoch_loss_dice / batch_step_count, step=tot_step_count)
        logger.scalar_summary('train_loss_tot', epoch_loss_tot / batch_step_count, step=tot_step_count)

        bg_acc, fg_acc, dice_coeffs = eval_model(model, eval_loader)
        print('fg_acc, bg_acc: {}, {}'.format(fg_acc, bg_acc))
        logger.scalar_summary('bg_acc', bg_acc, step=tot_step_count)
        logger.scalar_summary('fg_acc', fg_acc, step=tot_step_count)
        for lesion, coeff in zip(lesions, dice_coeffs):
            logger.scalar_summary('dice_coeff_'+lesion, coeff, step=tot_step_count)

        if not os.path.exists(dir_checkpoint):
            os.mkdir(dir_checkpoint)
        if (epoch + 1) % 10 == 0:
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
                                RandomCrop(1024),
                    ]))
    eval_dataset = IDRIDDataset(eval_image_paths, eval_mask_paths, 4, transform=
                                Compose([
                                RandomCrop(1024),
                    ]))

    train_loader = DataLoader(train_dataset, args.batchsize, shuffle=True)
    eval_loader = DataLoader(eval_dataset, args.batchsize, shuffle=False)

    optimizer = optim.SGD(model.parameters(),
                              lr=args.lr,
                              momentum=0.9,
                              weight_decay=0.0005)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    #bg, ex, he, ma, se
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([0.1, 1., 1., 2., 1.]).to(device))
    
    train_model(model, train_loader, eval_loader, criterion, optimizer, scheduler, args.batchsize, num_epochs=args.epochs)
