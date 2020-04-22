from sklearn.datasets import make_moons
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import numpy as np



X, y = make_moons(n_samples=1000, noise=0.05)
y_pred = KMeans(n_clusters=2).fit(X)
labels = y_pred.labels_.reshape(-1, 1)

# plt.scatter(X[:, 0], X[:, 1], c=labels)
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
    consensus_partition = KMeans(n_clusters=num_clusters).fit(X_b)
    return consensus_partition.labels_


labels_2 = KCC([OneHotEncoder().fit_transform(KMeans(n_clusters=num).fit(X).labels_.reshape(-1, 1)).toarray()
                for num in range(2, 2 * true_cluster_num + 1)], true_cluster_num)


plt.scatter(X[:, 0], X[:, 1], c=labels_2)
plt.show()

