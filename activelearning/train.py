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
from utils_diaret import  get_images_diaretAL
from dataset import IDRIDDataset, DiaretALDataset, MixedDataset
from torchvision import datasets, models, transforms
from transform.transforms_group import *
from torch.utils.data import DataLoader, Dataset
import copy
from logger import Logger
import os
from dice_loss import dice_loss, dice_coeff
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = OptionParser()
parser.add_option('-e', '--epochs', dest='epochs', default=5, type='int',
                      help='number of epochs')
parser.add_option('-b', '--batch-size', dest='batchsize', default=2,
                      type='int', help='batch size')
parser.add_option('-l', '--learning-rate', dest='lr', default=0.1,
                      type='float', help='learning rate')
parser.add_option('-r', '--resume', dest='resume',
                      default=False, help='resume file model')
parser.add_option('-p', '--log-dir', dest='logdir', default='drlog',
                    type='str', help='tensorboard log')
parser.add_option('-m', '--model-dir', dest='modeldir', default='./models',
                    type='str', help='models stored')
parser.add_option('-n', '--net-name', dest='netname', default='unet',
                    type='str', help='net name, unet or hednet')
parser.add_option('-g', '--preprocess', dest='preprocess', action='store_true',
                      default=False, help='preprocess input images')
parser.add_option('-i', '--healthy-included', dest='healthyincluded', action='store_true',
                      default=False, help='include healthy images')
parser.add_option('-a', '--active-learning', dest='al', action='store_true',
                      default=False, help='whether to use active learning')

(args, _) = parser.parse_args()

logger = Logger('./logs', args.logdir)
dir_checkpoint = args.modeldir
net_name = args.netname
lesion_dice_weights = [0., 0., 0., 0.]
lesions = ['ex', 'he', 'ma', 'se']
rotation_angle = 20
image_size = 512
image_dir = '/home/qiqix/Sub1'
image_dir = '/home/qiqix/Diaretdb1/resources/images/'
prediced_dir = './output'

softmax = nn.Softmax(1)
def eval_model(model, eval_loader, criterion):
    model.eval()
    fg_tot = 0
    bg_tot = 0
    eval_tot = len(eval_loader)
    dice_coeffs = np.zeros(4)
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

            _, mask_indices = torch.max(masks_pred, 1)
            _, true_masks_indices = torch.max(true_masks, 1)
            mask_size_bg = torch.sum(true_masks_indices == 0)
            mask_size_fg = true_masks_indices.shape[0] * true_masks_indices.shape[1] * true_masks_indices.shape[2] - torch.sum(true_masks_indices == 0) - torch.sum(true_masks_indices==5)
            if mask_size_fg > 0: 
                correct_fg = (torch.sum((mask_indices == true_masks_indices) *(true_masks_indices > 0) * (true_masks_indices<5)).float()) / (mask_size_fg.float())
                fg_tot += correct_fg.item()
            if mask_size_bg > 0:
                correct_bg = (torch.sum((mask_indices == true_masks_indices) *(true_masks_indices == 0)).float()) / (mask_size_bg.float())
                bg_tot += correct_bg.item()
        
            masks_pred_softmax = softmax(masks_pred) 
            dice_coeffs += dice_coeff(masks_pred_softmax[:, 1:-1, :, :], true_masks[:, 1:-1, :, :])
        return bg_tot / eval_tot, fg_tot / eval_tot, dice_coeffs / eval_tot, eval_loss_ce / eval_tot

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
    images_batch = (torch.ones((bs, 3, h*3+pad_size*2, w*4+pad_size*3)) * 255.).to(device=device, dtype=torch.uint8)
    
    images_batch[:, :, :h, :w] = inputs
    
    images_batch[:, :, h+pad_size:h*2+pad_size, :w] = 0
    images_batch[:, 0, h+pad_size:h*2+pad_size, :w] = true_masks[:, 1, :, :]
    
    images_batch[:, :, h+pad_size:h*2+pad_size, w+pad_size:w*2+pad_size] = 0
    images_batch[:, 1, h+pad_size:h*2+pad_size, w+pad_size:w*2+pad_size] = true_masks[:, 2, :, :]
    
    images_batch[:, :, h+pad_size:h*2+pad_size, w*2+pad_size*2:w*3+pad_size*2] = 0
    images_batch[:, 2, h+pad_size:h*2+pad_size, w*2+pad_size*2:w*3+pad_size*2] = true_masks[:, 3, :, :]
    
    images_batch[:, :, h+pad_size:h*2+pad_size, w*3+pad_size*3:] = true_masks[:, 4, :, :][:, None, :, :]
    
  
    images_batch[:, :, h*2+pad_size*2:, :w] = 0
    images_batch[:, 0, h*2+pad_size*2:, :w] = masks_pred_softmax[:, 1, :, :]
    
    images_batch[:, :, h*2+pad_size*2:, w+pad_size:w*2+pad_size] = 0
    images_batch[:, 1, h*2+pad_size*2:, w+pad_size:w*2+pad_size] = masks_pred_softmax[:, 2, :, :]
    
    images_batch[:, :, h*2+pad_size*2:, w*2+pad_size*2:w*3+pad_size*2] = 0
    images_batch[:, 2, h*2+pad_size*2:, w*2+pad_size*2:w*3+pad_size*2] = masks_pred_softmax[:, 3, :, :]
    
    images_batch[:, :, h*2+pad_size*2:, w*3+pad_size*3:] = masks_pred_softmax[:, 4, :, :][:, None, :, :]
    
    return images_batch

def train_model(model, train_loader, eval_loader, criterion, optimizer, scheduler, batch_size, num_epochs=5, start_epoch=0, start_step=0):
    model.to(device=device)
    tot_step_count = start_step
    for epoch in range(start_epoch, start_epoch+num_epochs):
        print('Starting epoch {}/{}.'.format(epoch + 1, start_epoch+num_epochs))
        scheduler.step()
        model.train()
        epoch_loss_ce = 0
        epoch_losses_dice = [0, 0, 0, 0]
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
            losses_dice = dice_loss(masks_pred_softmax[:, 1:-1, :, :], true_masks[:, 1:-1, :, :])
           
            
            # Save images
            if (epoch + 1) % 20 == 0:
                images_batch = generate_log_images(inputs, true_masks, masks_pred_softmax) 
                vis_images.extend(images_batch)

            ce_weight = 1.
            loss = loss_ce * ce_weight
            for lesion_dice_weight, loss_dice in zip(lesion_dice_weights, losses_dice):
                loss += loss_dice * lesion_dice_weight
            
            epoch_loss_ce += loss_ce.item()
            epoch_loss_tot = epoch_loss_ce * ce_weight
            for i, loss_dice in enumerate(losses_dice):
                epoch_losses_dice[i] += losses_dice[i].item() * lesion_dice_weights[i]
                epoch_loss_tot += epoch_losses_dice[i]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            batch_step_count += 1
            tot_step_count += 1
        
        # Traning logs
        logger.scalar_summary('train_loss_ce', epoch_loss_ce / batch_step_count, step=tot_step_count)
        for lesion, epoch_loss_dice in zip(lesions, epoch_losses_dice):
            logger.scalar_summary('train_loss_dice_'+lesion, epoch_loss_dice / batch_step_count, step=tot_step_count)
        logger.scalar_summary('train_loss_tot', epoch_loss_tot / batch_step_count, step=tot_step_count)
        
        # Validation logs
        bg_acc, fg_acc, dice_coeffs, eval_loss_ce = eval_model(model, eval_loader, criterion)
        print('eval_fg_acc, eval_bg_acc: {}, {}'.format(fg_acc, bg_acc))
        logger.scalar_summary('eval_loss_ce', eval_loss_ce, step=tot_step_count)
        logger.scalar_summary('bg_acc', bg_acc, step=tot_step_count)
        logger.scalar_summary('fg_acc', fg_acc, step=tot_step_count)
        for lesion, coeff in zip(lesions, dice_coeffs):
            logger.scalar_summary('dice_coeff_'+lesion, coeff, step=tot_step_count)

        if not os.path.exists(dir_checkpoint):
            os.mkdir(dir_checkpoint)
        if (epoch + 1) % 20 == 0:
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
        model = UNet(n_channels=3, n_classes=6)
    else:
        model = HNNNet(pretrained=True, class_number=6)
   
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']+1
            start_step = checkpoint['step']
            model.load_state_dict(checkpoint['state_dict'])
            print('Model loaded from {}'.format(args.resume))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        start_epoch = 0
        start_step = 0

    train_image_paths, train_mask_paths = get_images(image_dir, args.preprocess, phase='train', healthy_included=args.healthyincluded)
    train_image_paths2, train_mask_paths2, train_predicted_mask_paths2 = get_images_diaretAL(image_dir2, predicted_dir, args.preprocess, phase='train')
    eval_image_paths, eval_mask_paths = get_images(image_dir, args.preprocess, phase='eval', healthy_included=args.healthyincluded)

    if net_name == 'unet':
        # not using unet, deprecated yet.
        train_dataset = IDRIDDataset(train_image_paths, train_mask_paths, 4, transform=
                                Compose([
                                RandomRotation(rotation_angle),
                                RandomCrop(image_size),
                    ]))
        eval_dataset = IDRIDDataset(eval_image_paths, eval_mask_paths, 4, transform=
                                Compose([
                                RandomCrop(image_size),
                    ]))
    elif net_name == 'hednet':
        train_dataset1 = IDRIDDataset(train_image_paths, train_mask_paths, 4, transform=
                                Compose([
                                RandomRotation(rotation_angle),
                                RandomCrop(image_size),
                                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ]))
        train_dataset2 = IDRIDDataset(train_image_paths2, train_mask_paths2, train_predicted_mask_paths2, 4, transform=
                                Compose([
                                RandomRotation(rotation_angle),
                                RandomCrop(image_size),
                                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ]))
        train_dataset = MixedDataset(train_dataset1, train_dataset2)
        eval_dataset = IDRIDDataset(eval_image_paths, eval_mask_paths, 4, transform=
                                Compose([
                                RandomCrop(image_size),
                                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ]))

    train_loader = DataLoader(train_dataset, args.batchsize, shuffle=True)
    eval_loader = DataLoader(eval_dataset, args.batchsize, shuffle=False)

    optimizer = optim.SGD(model.parameters(),
                              lr=args.lr,
                              momentum=0.9,
                              weight_decay=0.0005)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
    #bg, ex, he, ma, se
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([0.1, 1., 2., 2., 4., 0.1]).to(device))
    
    train_model(model, train_loader, eval_loader, criterion, optimizer, scheduler, args.batchsize, num_epochs=args.epochs, start_epoch=start_epoch, start_step=start_step)
