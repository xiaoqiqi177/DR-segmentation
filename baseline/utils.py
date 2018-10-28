import os
import glob
train_ratio = 0.7
eval_ratio = 0.2
test_ratio = 0.1

def get_images(image_dir, phase='train'):
    apparent = glob.glob(os.path.join(image_dir, 'ApparentRetinopathy/*.jpg'))
    noapparent = glob.glob(os.path.join(image_dir, 'NoApparentRetinopathy/*.jpg'))
    apparent.sort()
    noapparent.sort()
    image_paths = []
    mask_paths = []
    for each in [apparent, noapparent]:
        train_number = int(len(each) * train_ratio)
        eval_number = int(len(each) * eval_ratio)
        if phase == 'train':
            image_paths.extend(each[:train_number])
        elif phase == 'eval':
            image_paths.extend(each[train_number:train_number+eval_number])
        else:
            image_paths.extend(each[train_number+eval_number:])
    mask_path= os.path.join(image_dir, 'GroundTruth')
    lesions = ['EX', 'HE', 'MA', 'SE']
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
