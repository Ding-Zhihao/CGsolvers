import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import bicgstab
import matplotlib.pyplot as plt


R = np.array([[4, 2, 0, 1],
              [3, 0, 0, 2],
              [0, 1, 1, 1],
              [0, 2, 1, 0]])
A = csc_matrix(R)
b = np.array([-1, -0.5, -1, 2])
x, exit_code = bicgstab(A, b)

R=np.loadtxt('../data/Amatrix.txt')
A = csc_matrix(R)
b=np.loadtxt('../data/bc_vector.txt')
x, exit_code = bicgstab(A, b)
x_ref=np.loadtxt('../data/Result.txt')