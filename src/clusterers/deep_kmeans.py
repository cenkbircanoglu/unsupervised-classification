import os

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import euclidean_distances

from src.clusterers.calculate_accuracy import calculate_accuracy

use_gpu = torch.cuda.is_available()


class DeepKmeans(object):
    def __init__(self, groundtruth_path, n_clusters=100, debug_root=None, assign=False):
        self.current_cluster_centers_ = []
        self.previous_cluster_centers_ = []
        self.n_clusters = n_clusters
        self.groundtruth_path = groundtruth_path
        self.debug_root = debug_root
        self.assign = assign

    @staticmethod
    def l2_normalization(X):
        row_sums = np.linalg.norm(X, axis=1)
        X = X / row_sums[:, np.newaxis]
        return X

    def assign_labels_according_to_previous_centroids(self, labels, epoch=None):
        if len(self.previous_cluster_centers_) == 0:
            return labels
        dist = euclidean_distances(self.previous_cluster_centers_, self.current_cluster_centers_)
        mapping = []
        while not np.all(np.isinf(dist)):
            mapp = np.where(dist == np.amin(dist))
            i = mapp[0][0]
            j = mapp[1][0]
            mapping.append((i, j))
            dist[i] = np.inf
            dist[:, j] = np.inf
        new_labels = labels.copy()

        for previous_l, new_l in mapping:
            indices = np.where(labels == new_l)
            new_labels[indices] = previous_l
        if self.debug_root:
            labels_path = os.path.join(self.debug_root, 'labels_%s.npy' % epoch)
            np.save(labels_path, mapping)
        return new_labels

    def cluster(self, X, filenames, epoch=None):
        print('KMeans Clustering Started')
        self.previous_cluster_centers_ = self.current_cluster_centers_
        X = self.l2_normalization(X)
        clusterer = KMeans(n_clusters=self.n_clusters, max_iter=1000)
        clusterer = clusterer.fit(X)
        self.current_cluster_centers_ = clusterer.cluster_centers_
        labels = clusterer.labels_
        assign_labels = self.assign_labels_according_to_previous_centroids(labels, epoch=epoch)
        if self.assign:
            df = pd.DataFrame({'img_name': filenames, 'prediction': assign_labels, 'kmeans_labels': labels})
        else:
            df = pd.DataFrame({'img_name': filenames, 'prediction': labels, 'assign_labels': assign_labels})
        acc, informational_acc, prediction_df, _ = calculate_accuracy(df, self.groundtruth_path,
                                                                      category_size=self.n_clusters,
                                                                      debug_root=self.debug_root, epoch=epoch)
        loss = clusterer.inertia_
        if self.debug_root:
            prediction_path = os.path.join(self.debug_root, 'predictions_%s.json' % epoch)
            df.to_json(prediction_path, orient='records')
            predictions_after_calculation_path = os.path.join(self.debug_root,
                                                              'predictions_after_calculation_%s.json' % epoch)
            prediction_df.to_json(predictions_after_calculation_path, orient='records')
            cluster_centers_path = os.path.join(self.debug_root, 'cluster_centers_%s.npy' % epoch)
            np.save(cluster_centers_path, self.current_cluster_centers_)
        if self.assign:
            return assign_labels, loss, acc, informational_acc
        return labels, loss, acc, informational_acc
