import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.io import loadmat, savemat

def solve_upper(U, rhs):
    x = np.zeros(rhs.shape)
    for i in range(len(x) - 1, -1, -1):
        x[i] = rhs[i]
        for j in range(i + 1, len(x)):
            x[i] -= U[i, j] * x[j]
        x[i] /= U[i, i]
    return x

U = np.array([[1, 2, 3], [0, 4, 5], [0, 0, 6]])
b = np.array([[1], [2], [3]])

x_ref = np.matmul(np.linalg.inv(U), b)
x = solve_upper(U, b)
print(x-x_ref)

def solve_lower(L, rhs):
    x = np.zeros(rhs.shape)
    for i in range(len(x)):
        x[i] = rhs[i]
        for j in range(i):
            x[i] -= L[i, j] * x[j]
        x[i] /= L[i, i]
    return x

L = np.array([[1, 0, 0], [2, 3, 0], [4, 5, 6]])
b = np.array([[1], [2], [3]])
x = solve_lower(L, b)

x_ref = np.matmul(np.linalg.inv(L), b)
print(x-x_ref)
