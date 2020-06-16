import os

import torch
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader

use_gpu = torch.cuda.is_available()


class CustomImageFolder(ImageFolder):
    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader, is_valid_file=None, sample_size=None):
        super(CustomImageFolder, self).__init__(root, transform=transform, target_transform=target_transform,
                                                loader=loader, is_valid_file=is_valid_file)
        self.imgs = self.samples
        try:
            if int(sample_size):
                self.imgs = self.imgs[:sample_size]
                self.samples = self.samples[:sample_size]
                self.targets = self.targets[:sample_size]
        except:
            pass
        self.ori_labels = self.targets

    def set_pseudo_labels(self, pseudo_labels):
        self.targets = pseudo_labels

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        target = self.targets[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        filename = os.path.basename(path).split('.')[0].replace('_', '')
        return sample, int(target), filename
