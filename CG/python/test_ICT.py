import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import bicgstab
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse import linalg
from scipy.io import loadmat, savemat

def ICT(A, tau):
    L = np.zeros(A.shape)
    N = np.size(A, axis=0)
    L[0, 0] = np.sqrt(A[0, 0])
    for i in range(1, N):
        for k in range(i):
            L[i, k] = A[i, k]
            for j in range(k):
                L[i, k] -= L[i, j] * L[k, j]
            norm_of_row_i = np.linalg.norm(A[k, k:], ord=1)    
            if abs(L[i, k]) < tau * norm_of_row_i:
                L[i, k] = 0
            else:
                L[i, k] = L[i, k] / L[k, k]
            L[i, i] -= L[i, k] ** 2
        L[i, i] = np.sqrt(L[i, i] + A[i, i])
    return L

# load data
A = np.loadtxt("../data/A.txt")
L_ref = loadmat('../data/ICT.mat')['L6'].toarray()

L = ICT(A, 1e-4)

plt.figure()
plt.scatter(L_ref, L)
