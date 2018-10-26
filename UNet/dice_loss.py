import torch
from torch.autograd import Function, Variable

def dice_loss(input, target):
    smooth = 1.
    class_no = input.shape[1]
    iflat = input.view(class_no, -1)
    tflat = target.view(class_no, -1)
    intersection = torch.sum(iflat * tflat, 1)
    return 1 - ((2. * intersection + smooth) / (torch.sum(iflat, 1) + torch.sum(tflat, 1) + smooth))

def dice_coeff(input, target):
    eps = 0.0001
    class_no = input.shape[1]
    inter = torch.dot(input.view(class_no, -1), target.view(class_no, -1))
    union = torch.sum(input, 1) + torch.sum(target, 1) + eps
    t = (2 * inter.float() + eps) / union.float()
    return t
