import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import bicgstab
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse import linalg, csr_array
from scipy.io import loadmat, savemat

# load data
A = loadmat("../data/input")["A"]
b = loadmat("../data/input")["b"]
# x_ref = loadmat("../data/pcg")["x"]
data = loadmat("../data/ichol_L.mat")
L_ref = data["L"].toarray()

L = np.zeros(A.shape)
N = np.size(A,axis=1)
L[0, 0] = np.sqrt(A[0, 0])
for i in range(1, N):
    for k in range(i):
        if A[i, k] != 0:
            L[i, k] = A[i, k] / L[k, k]
            for j in range(k):
                L[i, k] -= L[i, j] * L[k, j]

        L[i, i] -= L[i, k] ** 2
    L[i, i] = (L[i, i] + A[i, i]) ** 0.5

plt.figure()
plt.scatter(L_ref, L)

plt.figure()
plt.spy(L, markersize=0.1)