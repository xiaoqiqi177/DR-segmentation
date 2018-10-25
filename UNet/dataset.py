import numpy as np
from torchvision import datasets, models, transforms
import torchvision
from torch.utils.data import DataLoader, Dataset
from PIL import Image

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

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path4 = self.mask_paths[idx]
        item = self.pil_loader(image_path)
        if mask_paths is not None:
            h, w = item.shape[:2]
            info = np.zeors(h, w, 3 + class_number)
            info[:, :, :3] = item 
            for i, mask_path in enumerate(mask_path4):
                if mask_path is None:
                    continue
                else:
                    info[:, :, 3+i] = self.pil_loader(mask_path)
            if self.transfrom:
                info = self.transform(info)
            return info[:, :, :3], info[3:]
        else:
            return item
