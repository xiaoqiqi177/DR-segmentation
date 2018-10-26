import torch
from torch.autograd import Function, Variable

def dice_loss(input, target):
    assert input.shape == target.shape
    eps = 0.0001
    class_no = input.shape[1]
    input_flat = input.view(class_no, -1)
    target_flat = target.view(class_no, -1)
    inter = torch.sum(input_flat * target_flat, 1)
    union = torch.sum(input_flat, 1) + torch.sum(target_flat, 1) + eps
    t = (2 * inter + eps) / union
    return 1-t

def dice_coeff(input, target):
    assert input.shape == target.shape
    eps = 0.0001
    class_no = input.shape[1]
    input_flat = input.view(class_no, -1)
    target_flat = target.view(class_no, -1)
    inter = torch.sum(input_flat * target_flat, 1)
    union = torch.sum(input_flat, 1) + torch.sum(target_flat, 1) + eps
    t = (2 * inter.float() + eps) / union.float()
    return t.cpu().detach().numpy()
