import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import bicgstab
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse import linalg
from scipy.io import loadmat, savemat

# load data
A = np.loadtxt("../data/A.txt")
L_solver = np.loadtxt("../data/L2.txt")
L_ref = loadmat("../data/ichol_L_solver.mat")['L'].toarray()
L_ref = np.linalg.cholesky(A)

L = np.zeros(A.shape)
N = np.size(A,axis=1)

# L[0, 0] = np.sqrt(A[0, 0])
for i in range(N):
    for k in range(i):
        L[i, k] = A[i, k] / L[k, k]
        for j in range(k):
            L[i, k] -= L[i, j] * L[k, j]

        L[i, i] -= L[i, k] ** 2
    L[i, i] = (L[i, i] + A[i, i]) ** 0.5


plt.figure()
plt.scatter(L_ref, L)

# plt.figure()
# plt.scatter(L_ref, L_solver)