import numpy as np
from torchvision import datasets, models, transforms
import torchvision
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import cv2

class IDRIDDataset(Dataset):

    def __init__(self, image_paths, mask_paths=None, class_number=0, transform=None):
        """
        Args:
            image_paths: paths to the original images []
            mask_paths: paths to the mask images, [[]]
        """
        assert len(image_paths) == len(mask_paths)
        self.image_paths = image_paths
        if mask_paths is not None:
            self.mask_paths = mask_paths
        self.class_number = class_number
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def pil_loader(self, image_path):
        with open(image_path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def loader(self, image_path):
        with Image.open(path) as img:
            return img

    def padding(self, length, k):
        return length//k * k

    def cv2_loader(self, image_path):
        #read image, and transfrom from bgr to rgb
        image = cv2.imread(image_path)[:,:,::-1]
        h, w = image.shape[:2]
        h, w = h//4, w//4
        h, w = self.padding(h, 16), self.padding(w, 16)
        image = cv2.resize(image, (w, h))
        return image

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path4 = self.mask_paths[idx]
        #item = self.pil_loader(image_path)
        item = self.cv2_loader(image_path)
        if self.mask_paths is not None:
            h, w = item.shape[:2]
            info = np.zeros((h, w, 3 + self.class_number))
            info[:, :, :3] = item 
            for i, mask_path in enumerate(mask_path4):
                if mask_path is None:
                    continue
                else:
                    #info[:, :, 3+i] = self.pil_loader(mask_path)
                    info[:, :, 3+i] = self.cv2_loader(mask_path)[:, :, 0]
            #if self.transfrom:
            #    info = self.transform(info)
            info = np.transpose(info, [2, 0, 1])
            return info[:3, :, :], info[3:, :, :]
        else:
            return item
