import numpy as np
from scipy.special import rel_entr


def submatrix(array, indices):
    """
    array is a numpy array
    index is an iterable of indices
    returns a copy of array with the elements specified by indices removed
    """
    a = np.copy(array)
    for axis, array_index in enumerate(indices):
        a = np.delete(a, array_index, axis)
    return a


def euclidean_distance(a, b):
    """computes the squared euclidean distance between two vectors"""
    return np.linalg.norm(a - b).item()

def KL_divergence(a, b):
    """computes the Kullback-Leibler divergence between two vectors"""
    return sum(rel_entr(a, b))
