import os

import hydra
import numpy as np
import torch
from hydra import utils
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from src.clusterers.calculate_accuracy import calculate_accuracy
from src.clusterers.deep_kmeans import DeepKmeans
from src.datasets.custom_image_folder import CustomImageFolder
from src.deep_clusterers import models
from src.deep_clusterers.extract_features import extract_features
from src.utils import checkpoint_utils
from src.utils.pyutils import AverageMeter
from src.utils.uni_sampler import UnifLabelSampler

use_gpu = torch.cuda.is_available()
seed = 7
torch.manual_seed(7)
np.random.seed(7)

if use_gpu:
    torch.cuda.manual_seed_all(7)


def apply_dimensionality_reduction(features, pca_components=None):
    if pca_components is int:
        pca = PCA(n_components=pca_components, whiten=True)
        features = pca.fit_transform(features)
    return features


def train(dataset_cfg, model_cfg, training_cfg, debug_root=None):
    image_root_folder = os.path.join(utils.get_original_cwd(), dataset_cfg.image_root_folder)

    img_transform = transforms.Compose([
        transforms.Resize((training_cfg.img_size, training_cfg.img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    dataset = CustomImageFolder(image_root_folder, transform=img_transform,
                                sample_size=dataset_cfg.sample_size)

    model = models.__dict__[model_cfg.name](num_classes=training_cfg.n_clusters, initialize=model_cfg.initialize)

    model, already_trained_epoch = checkpoint_utils.load_latest_checkpoint(model, training_cfg.checkpoint, use_gpu)
    if use_gpu:
        model = model.cuda()
    deep_kmeans = DeepKmeans(n_clusters=training_cfg.n_clusters)

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

    for epoch in range(already_trained_epoch, training_cfg.num_epochs):
        features = extract_features(model, dataset, batch_size=training_cfg.batch_size)
        features = apply_dimensionality_reduction(features, pca_components=training_cfg.pca.component_size)

        pseudo_labels, kmeans_loss = deep_kmeans.cluster(features)
        if not training_cfg.use_original_labels:
            dataset.set_pseudo_labels(pseudo_labels)
        nmi = normalized_mutual_info_score(
            dataset.ori_labels, dataset.targets
        )
        print('NMI against original assignment: {0:.3f}'.format(nmi))
        acc, informational_acc, category_mapping = calculate_accuracy(dataset.ori_labels, dataset.targets)
        print('Classification Acc:%s\tInformational Acc:%s\n' % (acc, informational_acc))
        if training_cfg.reinitialize:
            model.reinitialize_fc()
        optimizer_tl = torch.optim.SGD(
            model.fc.parameters(),
            lr=training_cfg.optimizer.lr,
            weight_decay=10 ** training_cfg.optimizer.wd,
        )
        if use_gpu:
            model.cuda()
        model.train()
        sampler = UnifLabelSampler(N=int(len(dataset) * training_cfg.reassign), images_lists=dataset.targets,
                                   cluster_size=training_cfg.n_clusters)
        dataloader = DataLoader(dataset, batch_size=training_cfg.batch_size, shuffle=False, num_workers=4,
                                drop_last=False, sampler=sampler)
        print('Epoch [{}/{}] started'.format(epoch, training_cfg.num_epochs))
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
            optimizer_tl.zero_grad()
            loss.backward()
            optimizer.step()
            optimizer_tl.step()
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
              'Acc:%s\tInformational acc:%s\tNetwork Acc:%s\tNMI:%s\n' % (epoch, training_cfg.num_epochs,
                                                                          losses.get('loss_%s' % epoch),
                                                                          kmeans_loss, acc, informational_acc,
                                                                          (100 * correct / total), nmi)
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
