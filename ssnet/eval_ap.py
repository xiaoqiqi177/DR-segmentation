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
from sklearn.metrics import precision_recall_curve, average_precision_score

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

def plot_precision_recall(precisions, recalls, title, savefile):
    lt.figure()
    plt.step(recalls, precisions, color='b', alpha=0.2,
         where='post')
    plt.fill_between(recalls, precisions, step='post', alpha=0.2,
                 color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('{}'.format(title))
    #plt.title('{}: AP={}'.format(title, average_precision))
    plt.savefig(savefile)

def get_ap(recalls, precisions):
    idx = 0
    precision_sum = 0.
    assert len(recalls) == len(precisions)
    idx_number = len(recalls)
    for r in range(11):
        recall_val = r / 10.
        while idx < idx_number and recall_val > recalls[idx]:
            idx += 1
        # indicates the following precisions are all zero.
        if idx == idx_number:
            break
        if idx == 0:
            precision_sum += precisions[idx]
        else:
            recall1 = recalls[idx-1]
            recall2 = recalls[idx]
            ratio1 = (recall_val - recall1) / (recall2 - recall1)
            ratio2 = 1 - ratio1
            precision_sum += precisions[idx-1] * ratio1 + precisions[idx] * ratio2
    return precision_sum / 10.

def plot_precision_recall_all(precisions_all, recalls_all, titles, savefile):
    colors = ['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal']
    plt.figure(figsize=(7, 8))
    f_scores = np.linspace(0.2, 0.8, num=4)
    lines = []
    labels = []
    
    n_number = len(precisions_all)
    for i in range(n_number):
        l, = plt.plot(recalls_all[i], precisions_all[i], color=colors[i], lw=2)
        ap = get_ap(recalls_all[i], precisions_all[i])
        lines.append(l)
        labels.append('Precision-recall for {}; AP = {}'.format(titles[i], ap))
    
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))
        lines.append(l)
    labels.append('iso-f1 curves')

    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curves of different experiments')
    plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))
    plt.savefig(savefile)

def get_precision_recall(pred_masks, true_masks):
    precisions = []
    recalls = []
    batch_size = pred_masks.shape[0]
    class_no = pred_masks.shape[1]
    INTERNALS = 20
    delta = 0.001
    for threshold in range(INTERNALS+1):
        threshold = threshold / INTERNALS
        pred_masks_hard = (pred_masks > threshold).to(dtype=torch.float)
        pred_flat = pred_masks_hard.view(batch_size, class_no, -1)
        true_flat = true_masks.view(batch_size, class_no, -1)
        tp = torch.sum(pred_flat * true_flat, 2)
        predp = torch.sum(pred_flat, 2)
        truep = torch.sum(true_flat, 2)
        precisions.append(np.array((tp+delta) / (truep+delta)))
        recalls.append(np.array((tp+delta) / (predp+delta)))
    precisions = np.transpose(np.array(precisions), (1, 2, 0))
    recalls = np.transpose(np.array(recalls), (1, 2, 0))
    return precisions, recalls

softmax = nn.Softmax(1)
def eval_model(model, eval_loader):
    model.to(device=device)
    model.eval()
    eval_tot = len(eval_loader)
    dice_coeffs_soft = np.zeros(4)
    dice_coeffs_hard = np.zeros(4)
    vis_images = []
    precision_all = []
    recall_all = []
    with torch.set_grad_enabled(False):
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
            masks_hard = (masks_pred_softmax == masks_max[:, None, :, :]).to(dtype=torch.float)[:, 1:-1, :, :]
            dice_coeffs_soft += dice_coeff(masks_soft, true_masks[:, 1:-1, :, :].to("cpu"))
            dice_coeffs_hard += dice_coeff(masks_hard, true_masks[:, 1:-1, :, :].to("cpu"))
            precisions, recalls = get_precision_recall(masks_soft, true_masks[:, 1:-1, :, :].to("cpu"))
            precision_all.extend(precisions)
            recall_all.extend(recalls)
            images_batch = generate_log_images_full(inputs, true_masks[:, 1:-1], masks_soft, masks_hard) 
            images_batch = images_batch.to("cpu").numpy()
            vis_images.extend(images_batch)     
    return dice_coeffs_soft / eval_tot, dice_coeffs_hard / eval_tot, vis_images, np.mean(np.array(precision_all), 0), np.mean(np.array(recall_all), 0)

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
        model.load_state_dict(checkpoint['g_state_dict'])
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
                                
    dice_coeffs_soft, dice_coeffs_hard, vis_images, precisions, recalls = eval_model(model, eval_loader)
    print(dice_coeffs_soft, dice_coeffs_hard)
    print(precisions)
    print(recalls)
    plot_precision_recall_all(precisions, recalls, lesions, './recall_precision.png')
    #logger.image_summary('eval_images', vis_images, step=0)
