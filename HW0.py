from numpy.linalg import pinv, inv, matrix_power
from math import sqrt
from scipy.spatial.distance import pdist, squareform
import numpy as np
import pandas as pd


#problem 4
A = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(f"A pseudoinverse:{pinv(A)}")
print(f"A^100:{matrix_power(A, 100)}")

#problem 7
text_data = """Jupiter 778000 71492 1.90e27
Saturn 1429000 60268 5.69e26
Uranus 2870990 25559 8.69e25
Neptune 4504300 24764 1.02e26
Earth 149600 6378 5.98e24
Venus 108200 6052 4.87e24
Mars 227940 3398 6.42e23
Mercury 57910 2439 3.30e23
Pluto 5913520 1160 1.32e22""".split() #split at spaces and newlines
data = pd.DataFrame(np.array(text_data).reshape(-1, 4), columns=["Planet", "Distance", "Radius", "Mass"])
data = data.set_index("Planet")
data = data.astype("float64")


def s(p1_index, p2_index):
    """similarity function between two planets , p1 and p2 are integers that index planets in data"""
    parameters = np.array([3.5e-7, 1.6e-5, 1.1e-27])
    p1 = data.iloc[p1_index].to_numpy()
    p2 = data.iloc[p2_index].to_numpy()
    features = (p1 - p2) ** 2
    return sqrt(np.sum(features * parameters))


similarity_matrix = np.empty(shape=(len(data), len(data)))
for x in range(len(data)):
    for y in range(len(data)):
        similarity_matrix[x, y] = s(x, y)
similarity_matrix = np.around(similarity_matrix, 3)
print(f"similarity matrix:{similarity_matrix}")
