import os
import os.path
import shutil
import tempfile

import hydra

from src.utils.download_url import download_url
from src.utils.tar_utils import untar_file

CAT_LIST = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
    'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

urls = {
    'trval':
        'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar',
    'test':
        'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar'
}


def parse_labels(dataset_cfg):
    root_folder = dataset_cfg.root_folder
    labels_root = os.path.join(root_folder, 'VOCdevkit', 'VOC2007', 'ImageSets', 'Main')
    images_root = os.path.join(root_folder, 'VOCdevkit', 'VOC2007', 'JPEGImages')
    destination_root = dataset_cfg.image_root_folder
    for category_name in CAT_LIST:
        category_folder = os.path.join(destination_root, str(CAT_LIST.index(category_name)))
        os.makedirs(category_folder, exist_ok=True)
        for split in ['trainval', 'train', 'val', 'test']:
            path = os.path.join(labels_root, '%s_%s.txt' % (category_name, split))
            with open(path, mode='r') as f:
                for line in f.readlines():
                    filename = line.strip().split(' ')[0]
                    cls = line.strip()[-2:].strip()
                    cls = int(cls)
                    if cls == 1:
                        src_img_path = os.path.join(images_root, '%s.jpg' % filename)
                        dst_img_path = os.path.join(category_folder, '%s.jpg' % filename)
                        shutil.copyfile(src_img_path, dst_img_path)


def download(dataset_cfg):
    root_folder = dataset_cfg.root_folder
    os.makedirs(root_folder, exist_ok=True)

    for data_type in ['trval', 'test']:
        tmp_file = tempfile.mktemp()
        download_url(urls[data_type], tmp_file)
        untar_file(tmp_file, root_folder)

    parse_labels(dataset_cfg)


@hydra.main(config_path="conf/config.yaml")
def main(cfg):
    download(cfg.dataset)


if __name__ == '__main__':
    main()
