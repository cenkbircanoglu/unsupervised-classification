import os

import numpy as np
import torch
from sklearn.decomposition import PCA

from src.deep_clusterers.extract_features import extract_features

use_gpu = torch.cuda.is_available()


def reassign_labels(model, dataset, deep_kmeans, pca_components=None, debug_root=None, epoch=None,
                    batch_size=128):
    print('Features Creating Started')
    feature_filename_dict = extract_features(model, dataset, debug_root=debug_root, epoch=epoch, batch_size=batch_size)
    features = np.array(feature_filename_dict['features'])
    filenames = feature_filename_dict['filenames']
    if pca_components:
        pca = PCA(n_components=pca_components, whiten=True)
        features = pca.fit_transform(features)
        if debug_root:
            pca_path = os.path.join(debug_root, 'pca_%s.npy' % epoch)
            np.save(pca_path, {
                'features': features,
                'filenames': filenames
            })
    labels, loss, acc, informational_acc = deep_kmeans.cluster(features, filenames, epoch=epoch)
    dataset.targets = labels
    print('Features Creating Finished')
    return dataset, loss, acc, informational_acc
