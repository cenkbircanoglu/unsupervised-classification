import os

import hydra
import numpy as np
from torchvision.datasets import FashionMNIST


def download(dataset_cfg):
    train = FashionMNIST('/tmp', train=True, transform=None, target_transform=None, download=True)
    test = FashionMNIST('/tmp', train=False, transform=None, target_transform=None, download=True)

    img_root = os.path.join(dataset_cfg.image_root_folder, 'images')
    os.makedirs(img_root, exist_ok=True)
    os.makedirs(os.path.dirname(dataset_cfg.label_file), exist_ok=True)
    d = dict()
    for i, (img, label) in enumerate(train + test):
        total_label = np.zeros(10)
        total_label[label] = 1
        d[i] = total_label
        img.save(os.path.join(img_root, '%s.jpg' % i))

    np.save(dataset_cfg.label_file, d)


@hydra.main(config_path="conf/config.yaml")
def main(cfg):
    download(cfg.dataset)


if __name__ == '__main__':
    main()
