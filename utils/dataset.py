import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2
from torchvision.transforms.functional import InterpolationMode
from scipy.ndimage import binary_dilation

from matplotlib import pyplot as plt
from matplotlib import colors as mcolors

class BasicDataset(Dataset):
    """
    This class is used to create a custom dataset from an h5 file.
    """

    def __init__(self, filepath, augment=False, calc_edge_map=False):
        """
        Args:
            filepath (str): path to h5 file
            augment (bool): whether to apply data augmentation
            calc_edge_map (bool): whether to calculate the edge map, if False then returned edge_map will be all zeros
        """
        self.filepath = filepath
        self.augment = augment
        self.calc_edge_map = calc_edge_map
        
        if self.augment:
            print("Using data augmentation")

        with h5py.File(self.filepath, 'r') as hf:
            self.length = hf['data'].shape[0]
        
        mean = [0.87770971, 0.80533165, 0.8622918]
        std = [0.10774853, 0.15939812, 0.10747936]
        self.transforms = v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.RandomRotation((90, 90)),
            v2.ElasticTransform(alpha=50),
            # v2.RandomAffine(degrees=10, translate=(0.1, 0.1), interpolation=InterpolationMode.BILINEAR),
            # v2.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            # v2.RandomResizedCrop(size=(256, 256), scale=(0.8, 1.0)),
            # v2.ToPILImage(),
            # v2.ToTensor(),
            v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            # v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            v2.Normalize(mean=mean, std=std)
        ])

        self.normalise_only = v2.Compose([
            v2.Normalize(mean=mean, std=std)
        ])


    def __len__(self):
        return self.length
    

    def __getitem__(self, idx):
        """
        Returns:
            img: torch.Tensor (float) (C H W)
            mask: torch.Tensor (long) (H W)
            edge_map: torch.Tensor (long) (H W)
        """

        img, mask = self.get_data(idx) # img: Tensor (float) (H W C), mask: Tensor (long) (H W)
        img = img.permute(2, 0, 1) # H W C -> C H W

        edge_map = self.get_edge_map(mask) # edge_map: Tensor (long) (H W)

        img, mask, edge_map = self.apply_transform(img, mask, edge_map)

        return img, mask, edge_map

    
    def get_data(self, idx):
        """
        Returns:
            img: torch.Tensor (float) (H W C)
            mask: torch.Tensor (long) (H W)
        """
        with h5py.File(self.filepath, 'r') as hf:
            img = hf['data'][idx]
            mask = hf['labels'][idx]

        img = torch.from_numpy(img).float() / 255.0 # [0-255] -> [0-1]
        mask = torch.from_numpy(mask).long()

        return img, mask
    

    def apply_transform(self, img, mask, edge_map):
        # img = img / 255.0
        img = tv_tensors.Image(img)
        mask = tv_tensors.Mask(mask)
        edge_map = tv_tensors.Mask(edge_map)

        if self.augment:
            img, mask, edge_map = self.transforms(img, mask, edge_map)
            return img, mask, edge_map
        
        else:
            img, mask, edge_map = self.normalise_only(img, mask, edge_map)
            return img, mask, edge_map
        

    def get_edge_map(self, mask):
        '''
        Calculate the edge weight map

        Args:
            mask: torch.Tensor (int64) (H W)

        Returns:
            edge_map: torch.Tensor (long) (H W). (Will be all zeros if calc_edge_map is False)
        '''
        mask_np = mask.numpy()
        edge_map = np.zeros(mask_np.shape, dtype=mask_np.dtype)

        if self.calc_edge_map:
            for i in [1, 2, 3, 4]:
                edge_map += binary_dilation(mask_np == i, iterations=5) & ~(mask_np == i)

        return torch.from_numpy(edge_map).long()
    

if __name__ == "__main__":
    dataset = BasicDataset("/vol/bitbucket/dks20/renal_ssn/labelbox_download/train_data.h5", augment=True, calc_edge_map=True)
    for i in range(50, 51):
        dataset.__getitem__(i)
