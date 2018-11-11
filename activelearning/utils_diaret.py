import os
import glob
from preprocess import clahe_gridsize
import cv2

def get_images_diaretdb(image_dir, preprocess=False):
    if preprocess:
        limit = 2
        grid_size = 8
        if not os.path.exists(os.path.join(image_dir, 'ddb1_fundusimages_CLAHE')):
            os.mkdir(os.path.join(image_dir, 'ddb1_fundusimages_CLAHE'))
            images_ori = glob.glob(os.path.join(image_dir, 'ddb1_fundusimages/*.png'))
            images_ori.sort()
            
            # mean brightness.
            meanbright = 0.
            for img_path in images_ori:
                img_name = os.path.split(img_path)[-1].split('.')[0]
                mask_path = os.path.join(image_dir, 'ddb1_fundusmask', 'fmask1.tif')
                gray = cv2.imread(img_path, 0)
                mask_img = cv2.imread(mask_path, 0)
                brightness = gray.sum() / (mask_img.shape[0] * mask_img.shape[1] - mask_img.sum() / 255.)
                meanbright += brightness
            meanbright /= len(images_ori)
            
            # preprocess for images.
            for img_path in images_ori:
                img_name = os.path.split(img_path)[-1]
                mask_path = os.path.join(image_dir, 'ddb1_fundusmask', 'fmask1.tif')
                clahe_img = clahe_gridsize(img_path, mask_path, denoise=True, verbose=False, brightnessbalance=meanbright, cliplimit=limit, gridsize=grid_size)
                cv2.imwrite(os.path.join(image_dir, 'ddb1_fundusimages_CLAHE', os.path.split(img_path)[-1]), clahe_img)
            
        image_paths = glob.glob(os.path.join(image_dir, 'ddb1_fundusimages_CLAHE/*.png'))
    else:
        image_paths = glob.glob(os.path.join(image_dir, 'ddb1_fundusimages/*.png'))

    image_paths.sort()     
    return image_paths

def get_images_diaretAL(image_dir, predicted_mask_dir, preprocess=False, phase='train'):
    image_paths = get_image_diaretdb(image_dir, preprocess)
    mask_paths = []
    predicted_mask_paths = []
    mask_dir = os.path.join(image_dir, 'ddb1_groundtruth')
    lesions = ['ex', 'he', 'ma', 'se']
    diaret_lesions = ['hardexudates', 'hemorrhages', 'redsmalldots', 'softexudates' ]
    for image_path in image_paths:
        # append masks.
        paths = []
        name = os.path.split(image_path)[-1]
        for lesion in diaret_lesions:
            candidate_path = os.path.join(mask_dir, lesion, name)
            if os.path.exists(candidate_path):
                paths.append(candidate_path)
            else:
                paths.append(None)
        mask_paths.append(paths)
        # append predicted masks.
        paths = []
        for lesion in lesions:
            candidate_path = os.path.join(predicted_mask_dir, lesion, name)
            if os.path.exists(candidate_path):
                paths.append(candidate_path)
            else:
                paths.append(None)
        predicted_mask_paths.append(paths)
    
    return image_paths, mask_paths, predicted_mask_paths
