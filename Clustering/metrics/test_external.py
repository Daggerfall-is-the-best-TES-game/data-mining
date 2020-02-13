from unittest import TestCase
import numpy as np
from Clustering.metrics.external import accuracy, normalized_mutual_information, normalized_rand_index


class Test(TestCase):
    def setUp(self):
        self.table_3_clustering_2 = np.array(
            [[27, 0, 0, 2, 0], [0, 3, 0, 0, 0], [0, 0, 6, 0, 0], [3, 0, 0, 8, 0], [0, 0, 0, 0, 2]])
        self.table_10_clustering_2 = np.array([[0, 7, 12], [11, 0, 12], [12, 12, 0]])

    def test_accuracy(self):
        self.assertAlmostEqual(accuracy(self.table_3_clustering_2), 0.9, delta=0.002)

    def test_normalized_mutual_information(self):
        self.assertAlmostEqual(normalized_mutual_information(self.table_10_clustering_2), 0.62, delta=0.01)

    def test_normalized_rand_index(self):
        self.assertAlmostEqual(normalized_rand_index(self.table_10_clustering_2), 0.24, delta=0.01)
