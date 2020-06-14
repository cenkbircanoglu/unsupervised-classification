import os

import hydra
import numpy as np
import torch
from hydra import utils
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from src.clusterers.deep_kmeans import DeepKmeans
from src.datasets.custom_image_folder import CustomImageFolder
from src.deep_clusterers import models
from src.deep_clusterers.pseudo_labels import reassign_labels
from src.utils import checkpoint_utils
from src.utils.pyutils import AverageMeter
from src.utils.uni_sampler import UnifLabelSampler

use_gpu = torch.cuda.is_available()
seed = 7
torch.manual_seed(7)
np.random.seed(7)

if use_gpu:
    torch.cuda.manual_seed_all(7)


def train(dataset_cfg, model_cfg, training_cfg, debug_root=None):
    image_root_folder = os.path.join(utils.get_original_cwd(), dataset_cfg.image_root_folder)
    groundtruth_label_file = os.path.join(utils.get_original_cwd(), dataset_cfg.groundtruth_label_file)

    img_transform = transforms.Compose([
        transforms.Resize((training_cfg.img_size, training_cfg.img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = CustomImageFolder(image_root_folder, transform=img_transform,
                                sample_size=dataset_cfg.sample_size)

    model = models.__dict__[model_cfg.name](num_classes=training_cfg.n_clusters)

    model, already_trained_epoch = checkpoint_utils.load_latest_checkpoint(model, training_cfg.checkpoint, use_gpu)
    if use_gpu:
        model = model.cuda()
    deep_kmeans = DeepKmeans(groundtruth_label_file, n_clusters=training_cfg.n_clusters,
                             debug_root=debug_root, assign=training_cfg.assign)

    model.train()
    criterion = nn.CrossEntropyLoss()
    if training_cfg.optimizer.name == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=training_cfg.optimizer.lr,
            momentum=training_cfg.optimizer.momentum,
            weight_decay=10 ** training_cfg.optimizer.wd
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=training_cfg.optimizer.lr,
                                     weight_decay=10 ** training_cfg.optimizer.wd
                                     )
    losses = AverageMeter()
    os.makedirs(os.path.dirname(training_cfg.log_file), exist_ok=True)
    os.makedirs(debug_root, exist_ok=True)
    for epoch in range(already_trained_epoch + 1, training_cfg.num_epochs):
        dataset, kmeans_loss, acc, informational_acc = reassign_labels(model, dataset, deep_kmeans,
                                                                       debug_root=debug_root, epoch=epoch,
                                                                       batch_size=training_cfg.batch_size)
        model.train()
        sampler = UnifLabelSampler(N=int(len(dataset) * training_cfg.reassign), images_lists=dataset.targets,
                                   cluster_size=training_cfg.n_clusters)
        dataloader = DataLoader(dataset, batch_size=training_cfg.batch_size, shuffle=False, num_workers=4,
                                drop_last=True, sampler=sampler)
        print('epoch [{}/{}] started'.format(epoch, training_cfg.num_epochs))
        for data in tqdm(dataloader, total=int(len(dataset) * training_cfg.reassign / training_cfg.batch_size)):
            img, y, _ = data
            if use_gpu:
                img = Variable(img).cuda(non_blocking=True)
                y = y.cuda(non_blocking=True)
            # ===================forward=====================
            y_hat = model(img)
            loss = criterion(y_hat, y)
            # record loss
            losses.add({'loss_%s' % epoch: loss.item() / img.size(0)})
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # ===================log========================
        correct = 0
        total = 0
        with torch.no_grad():
            dataloader = DataLoader(dataset, batch_size=training_cfg.batch_size, shuffle=False, num_workers=4,
                                    drop_last=True)
            for data in dataloader:
                img, y, _ = data
                if use_gpu:
                    img = Variable(img).cuda(non_blocking=True)
                    y = y.cuda(non_blocking=True)
                y_hat = model(img)
                _, predicted = torch.max(y_hat.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()

        log = 'Epoch [%s/%s],\tLoss:%s,\tKmeans loss:%s\t' \
              'Acc:%s\tInformational acc:%s\tNetwork Acc:%s' % (epoch, training_cfg.num_epochs,
                                                                losses.get('loss_%s' % epoch),
                                                                kmeans_loss, acc, informational_acc,
                                                                (100 * correct / total))
        print(log)
        with open(training_cfg.log_file, mode='a') as f:
            f.write(log)
        if epoch % 5 == 0:
            checkpoint_utils.save_checkpoint(model, training_cfg.checkpoint, epoch)
    if use_gpu:
        torch.cuda.empty_cache()


@hydra.main(config_path="conf/train.yaml")
def main(cfg):
    print('Training Starting')
    train(cfg.dataset, cfg.model, cfg.training, debug_root=cfg.debug_root)
    print('Training Finished')


if __name__ == '__main__':
    main()
