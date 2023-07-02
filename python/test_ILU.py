import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import bicgstab
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse import linalg
from scipy.io import loadmat, savemat


class bicgstab_counter(object):
    def __init__(self, disp=True):
        self._disp = disp
        self.niter = 0

    def __call__(self, xk=None):
        self.niter += 1
        if self._disp:
            print("iter %3i" % (self.niter))


# load data
A = np.loadtxt("../data/Amatrix.txt")
b = np.loadtxt("../data/bc_vector.txt")
x_ref = np.loadtxt("../data/Result.txt")
data = loadmat("../data/ILU_ref.mat")
L_ref = data["L"].toarray()
U_ref = data["U"].toarray()

# sA = sparse.csc_matrix(A)
# sA_iLU = sparse.linalg.spilu(sA)
# L_ref2=sA_iLU.L.toarray()
# U_ref2=sA_iLU.U.toarray()

L = np.zeros(A.shape)
U = np.zeros(A.shape)
N = len(x_ref)

U[0,0]=A[0,0]

for i in range(N):
    L[i,i]=1
    
for i in range(1, N):
    U[i, :] = A[i, :]
    for k in range(i):
        if A[i, k] != 0:
            L[i, k] = U[i, k] / U[k, k]
            for j in range(k, N):
                if A[i, j] != 0:
                    U[i, j] = U[i, j] - L[i, k] * U[k, j]


plt.figure()
plt.scatter(U_ref, U)
plt.figure()
plt.scatter(L_ref, L)