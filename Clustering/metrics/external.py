import numpy as np
from Clustering.utils.functions import submatrix
from math import log2 as log
from scipy.special import comb


def accuracy(contingency_matrix):
    # algorithm. find max entry, bind row to column and cross out. repeat accuracy = sum of max entries divided by sum of all entries
    c = np.copy(contingency_matrix)
    correct_classifications = 0
    while c.shape[0] > 1:
        max_value_index = np.unravel_index(np.argmax(c), c.shape)
        correct_classifications += c[max_value_index]
        c = submatrix(c, max_value_index)
    correct_classifications += c[0,0]
    return correct_classifications / contingency_matrix.sum()


def normalized_mutual_information(contingency_matrix):
    # Note that the normalized variation of information is equivalent
    # to the normalized mutual information.
    n = contingency_matrix.sum()
    c = contingency_matrix
    dim = range(c.shape[0])

    def p(member):
        """input: slice or element of contingency matrix
        returns the probability that a datapoint is in that slice or element of the contingency matrix"""
        if isinstance(member, np.ndarray):
            return member.sum() / n
        return member / n

    part1 = sum(p(c[i, j]) * log(p(c[i, j]) / (p(c[i]) * p(c[:, j]))) if p(c[i, j]) else 0 for i in dim for j in dim)
    part2 = sum(p(c[i]) * log(p(c[i])) if p(c[i]) else 0 for i in dim) + \
        sum(p(c[:, j]) * log(p(c[:, j])) if p(c[:, j]) else 0 for j in dim)
    return 1 + 2 * part1 / part2


def normalized_rand_index(contingency_matrix):
    c = contingency_matrix
    dim = range(c.shape[0])111
    m = sum(comb(c[i, j], 2) for i in dim for j in dim)
    m1 = sum(comb(c[i, :].sum(), 2) for i in dim)
    m2 = sum(comb(c[:, j].sum(), 2) for j in dim)
    M = comb(c.sum(), 2)

    part1 = m - m1 * m2 / M
    part2 = m1 / 2 + m2 / 2 - m1 * m2 / M
    return part1 / part2

