import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score

use_gpu = torch.cuda.is_available()


class DeepKmeans(object):
    def __init__(self, n_clusters=100):
        self.n_clusters = n_clusters
        self.previous_labels = None

    @staticmethod
    def l2_normalization(X):
        row_sums = np.linalg.norm(X, axis=1)
        X = X / row_sums[:, np.newaxis]
        return X

    def cluster(self, X):
        X = self.l2_normalization(X)
        clusterer = KMeans(n_clusters=self.n_clusters, max_iter=1000)
        clusterer = clusterer.fit(X)

        labels = clusterer.labels_
        nmi = 0
        if self.previous_labels is not None:
            nmi = normalized_mutual_info_score(
                self.previous_labels,
                labels
            )
            print('NMI against previous assignment: {0:.3f}'.format(nmi))
        self.previous_labels = labels
        loss = clusterer.inertia_

        return labels, loss, nmi
