import pandas as pd
import numpy as np
from numpy.linalg import norm
from Clustering.utils.functions import distance


class KMeans:
    def __init__(self, X, initialization="random", iteration_type="lloyd", k=5):
        """perform k-means clustering on the data in X, where X is a pandas dataframe of records data
         X is an n by d matrix, where d is the number of features and n is the number of data points
         M is a k by d  matrix of centroids, where each row is a centroid, and there are k classes
         H is a n by k matrix, where each row corresponds to a datapoint, and each column to a class
         there is an entry of 1 if the datapoint belongs to class, and 0 otherwise
         n datapoints, d dimensions, k centroids"""
        self.X = X
        if initialization == "random":
            # Arbitrarily choose k initial centers
            self.M = X.sample(k)
            self.H = None

        elif initialization == "kmeans++":
            X.sample(weights=self.new_centroid_probability_vector())
            # update centroid matrix with https://pandas.pydata.org/docs/user_guide/indexing.html#setting-with-enlargement
        elif initialization == "global":
            pass
        else:
            raise ValueError(f"{initialization} is not a defined type of initialization")

        if iteration_type == "lloyd":
            self.lloyd_iteration()
        elif iteration_type == "hartigan":
            self.hartigan_iteration()
        else:
            raise ValueError(f"{iteration_type} is not a defined type of iteration")

    def new_centroid_probability_vector(self):
        """returns a n-vector of distances of each point from its closest centroid """

        def closest_centroid(x):
            """takes x, a datapoint
            returns the closest centroid to x"""
            return self.closest_centroid_indicator_vector(x, self.M) * self.M

        np.array(distance(x, closest_centroid(x)) ** 2 for x in self.X.iterrows())

    def closest_centroid_indicator_vector(self, x, y):
        """x is a datapoint, y is the matrix of centroids
        returns vector b such that b * y is the closest centroid"""
        minimum = np.zeros(y.shape[0])
        distances = pd.DataFrame(distance(x, z) for _, z in y.iterrows())
        min_index = distances.idxmin().iat[0]
        minimum[min_index] = 1
        return minimum

    def lloyd_iteration(self):
        while True:
            # recompute cluster membership
            new_H = pd.DataFrame(self.closest_centroid_indicator_vector(x, self.M) for _, x in self.X.iterrows())
            # check if H has changed
            if new_H.equals(self.H):
                break
            self.H = new_H
            # move centroids
            self.M = pd.DataFrame(self.H.loc[:, col].dot(self.X) / self.H.loc[:, col].sum() for col in self.H)

    def hartigan_iteration(self):
        pass
