import os
import glob
from preprocess import clahe_gridsize
import cv2
import torch.nn as nn

def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()


train_ratio = 0.7
eval_ratio = 0.3
test_ratio = 0.

def get_images(image_dir, preprocess=False, phase='train', healthy_included=True):
    if preprocess:
        limit = 2
        grid_size = 8
        if not os.path.exists(os.path.join(image_dir, 'ApparentRetinopathy_CLAHE')):
            os.mkdir(os.path.join(image_dir, 'ApparentRetinopathy_CLAHE'))
            os.mkdir(os.path.join(image_dir, 'NoApparentRetinopathy_CLAHE'))
            apparent_ori = glob.glob(os.path.join(image_dir, 'ApparentRetinopathy/*.jpg'))
            noapparent_ori = glob.glob(os.path.join(image_dir, 'NoApparentRetinopathy/*.jpg'))
            apparent_ori.sort()
            noapparent_ori.sort()
            
            # mean brightness.
            meanbright = 0.
            for img_path in apparent_ori + noapparent_ori:
                img_name = os.path.split(img_path)[-1].split('.')[0]
                mask_path = os.path.join(image_dir, 'GroundTruth', 'MASK', img_name+'_MASK.tif')
                gray = cv2.imread(img_path, 0)
                mask_img = cv2.imread(mask_path, 0)
                brightness = gray.sum() / (mask_img.shape[0] * mask_img.shape[1] - mask_img.sum() / 255.)
                meanbright += brightness
            meanbright /= len(apparent_ori + noapparent_ori)
            
            # preprocess for apparent.
            for img_path in apparent_ori:
                img_name = os.path.split(img_path)[-1].split('.')[0]
                mask_path = os.path.join(image_dir, 'GroundTruth', 'MASK', img_name+'_MASK.tif')
                clahe_img = clahe_gridsize(img_path, mask_path, denoise=True, verbose=False, brightnessbalance=meanbright, cliplimit=limit, gridsize=grid_size)
                cv2.imwrite(os.path.join(image_dir, 'ApparentRetinopathy_CLAHE', os.path.split(img_path)[-1]), clahe_img)
            
            # preprocess for noapparent.
            for img_path in noapparent_ori:
                img_name = os.path.split(img_path)[-1].split('.')[0]
                mask_path = os.path.join(image_dir, 'GroundTruth', 'MASK', img_name+'_MASK.tif')
                clahe_img = clahe_gridsize(img_path, mask_path, denoise=True, verbose=False, brightnessbalance=meanbright, cliplimit=limit, gridsize=grid_size)
                cv2.imwrite(os.path.join(image_dir, 'NoApparentRetinopathy_CLAHE', os.path.split(img_path)[-1]), clahe_img)
        apparent = glob.glob(os.path.join(image_dir, 'ApparentRetinopathy_CLAHE/*.jpg'))
        noapparent = glob.glob(os.path.join(image_dir, 'NoApparentRetinopathy_CLAHE/*.jpg'))
    else:
        apparent = glob.glob(os.path.join(image_dir, 'ApparentRetinopathy/*.jpg'))
        noapparent = glob.glob(os.path.join(image_dir, 'NoApparentRetinopathy/*.jpg'))

    apparent.sort()     
    noapparent.sort()
    image_paths = []
    mask_paths = []
    if healthy_included:
        imgset = [apparent, noapparent]
    else:
        imgset = [apparent]
    for each in imgset:
        train_number = int(len(each) * train_ratio)
        eval_number = int(len(each) * eval_ratio)
        if phase == 'train':
            #image_paths.extend(each[:train_number])
            image_paths.extend(each[eval_number:])
        elif phase == 'eval':
            #image_paths.extend(each[train_number:train_number+eval_number])
            image_paths.extend(each[:eval_number])
        else:
            image_paths.extend(each[train_number+eval_number:])
    mask_path= os.path.join(image_dir, 'GroundTruth')
    lesions = ['EX', 'HE', 'MA', 'SE', 'MASK']
    for image_path in image_paths:
        paths = []
        name = os.path.split(image_path)[1].split('.')[0]
        for lesion in lesions:
            candidate_path = os.path.join(mask_path, lesion, name+'_'+lesion+'.tif')
            if os.path.exists(candidate_path):
                paths.append(candidate_path)
            else:
                paths.append(None)
        mask_paths.append(paths)
    
    return image_paths, mask_paths
