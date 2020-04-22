from unittest import TestCase
import numpy as np
from Clustering.utils.functions import submatrix, euclidean_distance


class Test(TestCase):
    def test_submatrix(self):
        a = np.array(range(9)).reshape(3, 3)
        np.testing.assert_array_equal(submatrix(a, (2, 1)), np.array([[0, 2], [3, 5]]))

    def test_distance(self):
        a = np.array([1, 0])
        b = np.array([0, 1])
        np.testing.assert_array_equal(euclidean_distance(a, b), np.sqrt(2))
