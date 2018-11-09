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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
parser = OptionParser()
parser.add_option('-b', '--batch-size', dest='batchsize', default=2,
                      type='int', help='batch size')
parser.add_option('-p', '--log-dir', dest='logdir', default='eval',
                    type='str', help='tensorboard log')
parser.add_option('-m', '--model', dest='model', default='MODEL.pth.tar',
                    type='str', help='models stored')
parser.add_option('-n', '--net-name', dest='netname', default='unet',
                    type='str', help='net name, unet or hednet')
parser.add_option('-g', '--preprocess', dest='preprocess', action='store_true',
                      default=False, help='preprocess input images')
parser.add_option('-i', '--healthy-included', dest='healthyincluded', action='store_true',
                      default=False, help='include healthy images')

(args, _) = parser.parse_args()

logger = Logger('./logs', args.logdir)
net_name = args.netname
lesions = ['ex', 'he', 'ma', 'se']
image_size = 512
image_dir = '/home/qiqix/Sub1'

logdir = args.logdir
if not os.path.exists(logdir):
    os.mkdir(logdir)

softmax = nn.Softmax(1)
def eval_model(model, eval_loader):
    model.to(device=device)
    model.eval()
    eval_tot = len(eval_loader)
    dice_coeffs_soft = np.zeros(4)
    dice_coeffs_hard = np.zeros(4)
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
            masks_soft = masks_pred_softmax[:, 1:-1, :, :]
            np.save(os.path.join(logdir, 'mask_soft_'+str(batch_id)+'.npy'), masks_soft.numpy())
            np.save(os.path.join(logdir, 'mask_true_'+str(batch_id)+'.npy'), true_masks[:,1:-1].cpu().numpy())
            masks_hard = (masks_pred_softmax == masks_max[:, None, :, :]).to(dtype=torch.float)[:, 1:-1, :, :]
            dice_coeffs_soft += dice_coeff(masks_soft, true_masks[:, 1:-1, :, :].to("cpu"))
            dice_coeffs_hard += dice_coeff(masks_hard, true_masks[:, 1:-1, :, :].to("cpu"))
            images_batch = generate_log_images_full(inputs, true_masks[:, 1:-1], masks_soft, masks_hard) 
            images_batch = images_batch.to("cpu").numpy()
            vis_images.extend(images_batch)     
            batch_id += 1
    return dice_coeffs_soft / eval_tot, dice_coeffs_hard / eval_tot, vis_images

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
    
    images_batch[:, :, :h, w+pad_size:] = true_masks[:, 3, :, :][:, None, :, :]
    images_batch[:, 0, :h, w+pad_size:] += true_masks[:, 0, :, :] 
    images_batch[:, 1, :h, w+pad_size:] += true_masks[:, 1, :, :] 
    images_batch[:, 2, :h, w+pad_size:] += true_masks[:, 2, :, :] 
    
    images_batch[:, :, h+pad_size:, :w] = masks_soft[:, 3, :, :][:, None, :, :]
    images_batch[:, 0, h+pad_size:, :w] += masks_soft[:, 0, :, :]
    images_batch[:, 1, h+pad_size:, :w] += masks_soft[:, 1, :, :]
    images_batch[:, 2, h+pad_size:, :w] += masks_soft[:, 2, :, :]
    
    images_batch[:, :, h+pad_size:, w+pad_size:] = masks_hard[:, 3, :, :][:, None, :, :]
    images_batch[:, 0, h+pad_size:, w+pad_size:] += masks_hard[:, 0, :, :]
    images_batch[:, 1, h+pad_size:, w+pad_size:] += masks_hard[:, 1, :, :]
    images_batch[:, 2, h+pad_size:, w+pad_size:] += masks_hard[:, 2, :, :]
    return images_batch

if __name__ == '__main__':

    if net_name == 'unet': 
        model = UNet(n_channels=3, n_classes=6)
    else:
        model = HNNNet(pretrained=True, class_number=6)
    
    if os.path.isfile(args.model):
        print("=> loading checkpoint '{}'".format(args.model))
        checkpoint = torch.load(args.model)
        model.load_state_dict(checkpoint['state_dict'])
        print('Model loaded from {}'.format(args.model))
    else:
        print("=> no checkpoint found at '{}'".format(args.model))
        sys.exit(0)

    eval_image_paths, eval_mask_paths = get_images(image_dir, args.preprocess, phase='eval', healthy_included=args.healthyincluded)

    if net_name == 'unet':
        eval_dataset = IDRIDDataset(eval_image_paths, eval_mask_paths, 4, transform=
                                Compose([
                    ]))
    elif net_name == 'hednet':
        eval_dataset = IDRIDDataset(eval_image_paths, eval_mask_paths, 4, transform=
                                Compose([
                                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ]))
    eval_loader = DataLoader(eval_dataset, args.batchsize, shuffle=False)
                                
    dice_coeffs_soft, dice_coeffs_hard, vis_images = eval_model(model, eval_loader)
    print(dice_coeffs_soft, dice_coeffs_hard)
    #logger.image_summary('eval_images', vis_images, step=0)
