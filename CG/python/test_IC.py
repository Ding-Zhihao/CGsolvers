import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import bicgstab
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse import linalg, csr_array
from scipy.io import loadmat, savemat
from csr_lib import *


def IC(A):
    L = np.zeros(A.shape)
    N = np.size(A, axis=1)
    L[0, 0] = np.sqrt(A[0, 0])
    for i in range(1, N):
        for k in range(i):
            if A[i, k] != 0:
                L[i, k] = A[i, k] / L[k, k]
                for j in range(k):
                    L[i, k] -= L[i, j] * L[k, j]

            L[i, i] -= L[i, k] ** 2
        L[i, i] = (L[i, i] + A[i, i]) ** 0.5
    return L


if __name__ == "__main__":
    # load data
    A = loadmat("../data/input")["A"]
    data = loadmat("../data/ichol_L.mat")
    L_ref = loadmat("../data/ichol_L.mat")["L"].toarray()

    L = IC(A)

    L_ref = dense_to_csr(L_ref)
    L = dense_to_csr(L)
    print(np.allclose(L.data, L_ref.data))
    assert np.array_equal(L.indices, L_ref.indices)
    assert np.array_equal(L.indptr, L_ref.indptr)
    assert np.array_equal(L.nnz, L_ref.nnz)
