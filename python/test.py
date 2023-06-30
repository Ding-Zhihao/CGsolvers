import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import bicgstab
import matplotlib.pyplot as plt

# load data
R=np.loadtxt('../data/Amatrix.txt')
A = csc_matrix(R)
b=np.loadtxt('../data/bc_vector.txt')
x_ref=np.loadtxt('../data/Result.txt')

# computation
x, exit_code = bicgstab(A, b, tol=1e-7)

#post
plt.figure()
plt.scatter(x_ref, x)
plt.plot([0,1], [0,1], 'k--')
plt.xlabel('Jacobi')
plt.ylabel('BiCGStab')
plt.grid()
