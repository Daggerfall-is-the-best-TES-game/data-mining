import numpy as np


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


def distance(a, b):
    """computes the squared euclidean distance between two vectors"""
    return np.linalg.norm(a - b).item()
