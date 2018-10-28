import torch
from torch.autograd import Function, Variable

def dice_loss(input, target):
    assert input.shape == target.shape
    eps = 1.
    batch_size = input.shape[0]
    class_no = input.shape[1]
    input_flat = input.view(batch_size, class_no, -1)
    target_flat = target.view(batch_size, class_no, -1)
    inter = torch.sum(input_flat * target_flat, (0, 2))
    union = torch.sum(input_flat, (0, 2)) + torch.sum(target_flat, (0, 2)) + eps
    t = (2 * inter + eps) / union
    return 1-t

def dice_coeff(input, target):
    assert input.shape == target.shape
    eps = 1.
    batch_size = input.shape[0]
    class_no = input.shape[1]
    input_flat = input.view(batch_size, class_no, -1)
    target_flat = target.view(batch_size, class_no, -1)
    inter = torch.sum(input_flat * target_flat, (0, 2))
    union = torch.sum(input_flat, (0, 2)) + torch.sum(target_flat, (0, 2)) + eps
    t = (2 * inter.float() + eps) / union.float()
    return t.cpu().detach().numpy()
