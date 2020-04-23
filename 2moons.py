from sklearn.datasets import make_moons
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Clustering.algorithms.KMeans import KMeans
from Clustering.utils.functions import KL_divergence, euclidean_distance


X, y = make_moons(n_samples=1000, noise=0.05)
# y_pred = KMeans(pd.DataFrame(X), initialization="random", distance=KL_divergence, k=4).H.to_numpy()
#
# print(y_pred)
# plt.scatter(X[:, 0], X[:, 1], c=y_pred[:, 0])
# plt.show()

true_cluster_num = 2


def KCC(basic_partition_set, num_clusters):
    """Kmeans based consensus clustering
    Input:
    basic_partition_set: the set of basic partitionings
    num_clusters: the number of clusters
    Output:
    the consensus partitioning
    the optimal consensus-function value"""
    X_b = np.concatenate(basic_partition_set, axis=1)
    consensus_partition = KMeans(pd.DataFrame(X_b), initialization="random", distance=KL_divergence, k=num_clusters)
    return consensus_partition.H.to_numpy()


labels_2 = KCC([KMeans(pd.DataFrame(X), initialization="random", distance=KL_divergence, k=num).H.to_numpy()
                for num in range(2, 2 * true_cluster_num + 1)], true_cluster_num)


plt.scatter(X[:, 0], X[:, 1], c=labels_2[:, 0])
plt.show()

