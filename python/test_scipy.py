import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import bicgstab
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse import linalg

class bicgstab_counter(object):
    def __init__(self, disp=True):
        self._disp = disp
        self.niter = 0
    def __call__(self, xk=None):
        self.niter += 1
        if self._disp:
            print('iter %3i' % (self.niter))



# load data
R = np.loadtxt("../data/Amatrix.txt")
A = csc_matrix(R)
b = np.loadtxt("../data/bc_vector.txt")
x_ref = np.loadtxt("../data/Result.txt")

# computation
counter1 = bicgstab_counter()
x, exit_code = bicgstab(A, b, tol=1e-100, callback=counter1)

# post
plt.figure()
plt.scatter(x_ref, x)
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("Jacobi")
plt.ylabel("BiCGStab")
plt.grid()

counter2 = bicgstab_counter()
sA = sparse.csc_matrix(A)
sA_iLU = sparse.linalg.spilu(sA)
M = sparse.linalg.LinearOperator((918, 918), sA_iLU.solve)
x_precond, exit_code = bicgstab(A, b, tol=1e-100, M=M, callback=counter2)

# post
plt.figure()
plt.scatter(x_ref, x_precond)
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("Jacobi")
plt.ylabel("BiCGStab iLU")
plt.grid()


