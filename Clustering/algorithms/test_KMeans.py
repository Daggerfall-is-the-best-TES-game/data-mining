from unittest import TestCase
import numpy as np
import pandas as pd
from Clustering.algorithms.KMeans import KMeans


class TestKMeans(TestCase):
    def setUp(self):
        self.data = pd.DataFrame(np.reshape(np.array(range(30)), (10,3)))
        self.km = KMeans(self.data, initialization="random", k=10)

    def test_closest_centroid_indicator_vector(self):
        x = np.array([6, 7, 8])

        np.testing.assert_array_equal(self.km.closest_centroid_indicator_vector(x, self.data), np.array([x == 2 for x in range(10)]))

    def test_it(self):
        a = self.km.H
        b = self.km.X
        c = self.km.M
        print(a, b, c, a.dot(b))
        assert 2 == 1
