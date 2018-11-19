import os
import glob
from preprocess import clahe_gridsize
import cv2

train_ratio = 0.9
eval_ratio = 0.1

def get_images(image_dir, preprocess=False, phase='train', healthy_included=True):
    if phase == 'train' or phase == 'eval':
        setname = 'TrainingSet'
    elif phase == 'test':
        setname = 'TestingSet' 
    if preprocess:
        limit = 2
        grid_size = 8
        if not os.path.exists(os.path.join(image_dir, 'Images_CLAHE')):
            os.mkdir(os.path.join(image_dir, 'Images_CLAHE'))
        if not os.path.exists(os.path.join(image_dir, 'Images_CLAHE', setname)):
            os.mkdir(os.path.join(image_dir, 'Images_CLAHE', setname))
            
            # compute mean brightess
            meanbright = 0.
            images_number = 0
            for tempsetname in ['TrainingSet', 'TestingSet']:
                imgs_ori = glob.glob(os.path.join(image_dir, 'OriginalImages/'+tempsetname+'/*.jpg'))
                imgs_ori.sort()
                images_number += len(imgs_ori)
                # mean brightness.
                for img_path in imgs_ori:
                    img_name = os.path.split(img_path)[-1].split('.')[0]
                    mask_path = os.path.join(image_dir, 'Groundtruths', tempsetname, 'Mask', img_name+'_MASK.tif')
                    gray = cv2.imread(img_path, 0)
                    mask_img = cv2.imread(mask_path, 0)
                    brightness = gray.sum() / (mask_img.shape[0] * mask_img.shape[1] - mask_img.sum() / 255.)
                    meanbright += brightness
            meanbright /= images_number
            
            imgs_ori = glob.glob(os.path.join(image_dir, 'OriginalImages/'+setname+'/*.jpg'))
            # preprocess for apparent.
            for img_path in imgs_ori:
                img_name = os.path.split(img_path)[-1].split('.')[0]
                mask_path = os.path.join(image_dir, 'Groundtruths', setname, 'Mask', img_name+'_MASK.tif')
                clahe_img = clahe_gridsize(img_path, mask_path, denoise=True, verbose=False, brightnessbalance=meanbright, cliplimit=limit, gridsize=grid_size)
                cv2.imwrite(os.path.join(image_dir, 'Images_CLAHE', setname, os.path.split(img_path)[-1]), clahe_img)
            
        imgs = glob.glob(os.path.join(image_dir, 'Images_CLAHE', setname, '*.jpg'))
    else:
        imgs = glob.glob(os.path.join(image_dir, 'OriginalImages', setname, '*.jpg'))

    imgs.sort()     
    mask_paths = []
    train_number = int(len(imgs) * train_ratio)
    eval_number = int(len(imgs) * eval_ratio)
    if phase == 'train':
        image_paths = imgs[:train_number]
    elif phase == 'eval':
        image_paths = imgs[train_number:]
    else:
        image_paths = imgs
    mask_path = os.path.join(image_dir, 'Groundtruths', setname)
    lesions = ['HardExudates', 'Haemorrhages', 'Microaneurysms', 'SoftExudates', 'Mask']
    lesion_abbvs = ['EX', 'HE', 'MA', 'SE', 'MASK']
    for image_path in image_paths:
        paths = []
        name = os.path.split(image_path)[1].split('.')[0]
        for lesion, lesion_abbv in zip(lesions, lesion_abbvs):
            candidate_path = os.path.join(mask_path, lesion, name+'_'+lesion_abbv+'.tif')
            if os.path.exists(candidate_path):
                paths.append(candidate_path)
            else:
                paths.append(None)
        mask_paths.append(paths)
    return image_paths, mask_paths
