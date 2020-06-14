import os

import hydra
import numpy as np
from torchvision.datasets import CIFAR10


def download(image_root_folder, label_file):
    train = CIFAR10('/tmp', train=True, transform=None, target_transform=None, download=True)
    test = CIFAR10('/tmp', train=False, transform=None, target_transform=None, download=True)

    os.makedirs(image_root_folder, exist_ok=True)
    os.makedirs(os.path.dirname(label_file), exist_ok=True)
    d = dict()
    for i, (img, label) in enumerate(train + test):
        total_label = np.zeros(10)
        total_label[label] = 1
        d[i] = total_label
        img.save(os.path.join(image_root_folder, '%s.jpg' % i))

    np.save(label_file, d)


@hydra.main(config_path="conf/config.yaml")
def main(cfg):
    download(cfg.dataset.image_root_folder, cfg.dataset.label_file)


if __name__ == '__main__':
    main()
