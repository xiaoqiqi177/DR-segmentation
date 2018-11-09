import sys
import os
from optparse import OptionParser
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from unet import UNet
from hednet import HNNNet
from utils_diaret import get_images_diaretdb
from dataset import IDRIDDataset
from torchvision import datasets, models, transforms
from transform.transforms_group import *
from torch.utils.data import DataLoader, Dataset
import copy
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

net_name = args.netname
lesions = ['ex', 'he', 'ma', 'se']
image_size = 512
image_dir = '/home/qiqix/Diaretdb1/resources/images/'

softmax = nn.Softmax(1)
def eval_model(model, eval_loader):
    model.to(device=device)
    model.eval()
    vis_images = []
    
    with torch.set_grad_enabled(False):
        batch_id = 0
        for inputs in tqdm(eval_loader):
            inputs = inputs.to(device=device, dtype=torch.float)
            bs, _, h, w = inputs.shape
            h_size = (h-1) // image_size + 1
            w_size = (w-1) // image_size + 1
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
            masks_soft = masks_pred_softmax[:, 1:-1, :, :]
            masks_soft = (masks_soft * 255.).to(device=device, dtype=torch.uint8)
            batch_id += 1
    return vis_images


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

    eval_image_paths = get_images_diaretdb(image_dir, args.preprocess, phase='eval', healthy_included=args.healthyincluded)

    if net_name == 'unet':
        eval_dataset = DiaretDataset(eval_image_paths, 4, transform=
                                Compose([
                    ]))
    elif net_name == 'hednet':
        eval_dataset = DiaretDataset(eval_image_paths, 4, transform=
                                Compose([
                                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ]))
    eval_loader = DataLoader(eval_dataset, args.batchsize, shuffle=False)
                                
    vis_images = eval_model(model, eval_loader)
