import os

import hydra
import numpy as np
from torchvision.datasets import MNIST


def download(dataset_cfg):
    train = MNIST('/tmp', train=True, transform=None, target_transform=None, download=True)
    test = MNIST('/tmp', train=False, transform=None, target_transform=None, download=True)

    img_root = dataset_cfg.image_root_folder
    os.makedirs(img_root, exist_ok=True)
    os.makedirs(os.path.dirname(dataset_cfg.label_file), exist_ok=True)
    label_set = set(train.train_labels.tolist())
    for label in label_set:
        os.makedirs(os.path.join(img_root, str(label)), exist_ok=True)
    d = dict()
    for i, (img, label) in enumerate(train + test):
        total_label = np.zeros(len(label_set))
        total_label[label] = 1
        d[i] = total_label
        img.save(os.path.join(img_root, str(label), '%s.jpg' % i))

    np.save(dataset_cfg.label_file, d)


@hydra.main(config_path="conf/config.yaml")
def main(cfg):
    download(cfg.dataset)


if __name__ == '__main__':
    main()
