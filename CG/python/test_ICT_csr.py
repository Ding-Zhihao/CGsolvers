import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import bicgstab
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse import linalg
from scipy.io import loadmat, savemat
from csr_lib import *
import time


def ICT_csr(A_csr, tau):
    # initialization
    num_rows = A_csr.shape[0]
    nnz = 0
    data = np.array([])
    indices = np.array([], dtype="int32")
    indptr = np.zeros(1, dtype="int32")
    L = csr_matrix_(data, indices, indptr, (0, 0))

    for i in range(0, num_rows):
        L.indptr = np.append(L.indptr, L.indptr[i])
        L.shape = (i + 1, i + 1)
        if i == 0:
            L.data = np.append(L.data, np.sqrt(A_csr.data[0]))
            L.indices = np.append(L.indices, 0)
            L.indptr[i + 1] += 1
            L.nnz += 1
        else:
            Lii = 0.0
            nnz_row = A_csr.indptr[i + 1] - A_csr.indptr[i]
            row_vals = np.array([])
            row_indices = np.array([])
            for i_nnz in range(nnz_row):
                index = A_csr.indices[A_csr.indptr[i] + i_nnz]
                if index < i:
                    row_vals = np.append(row_vals, A_csr.data[A_csr.indptr[i] + i_nnz])
                    row_indices = np.append(row_indices, index)
                if index == i:
                    Aii = A_csr.data[A_csr.indptr[i] + i_nnz]
            for k in range(i):
                Lik = 0.0
                for kk in range(len(row_vals)):
                    if row_indices[kk] == k:
                        Lik = row_vals[kk]

                L_row_i = get_csr_row(L, i)
                L_row_k = get_csr_row(L, k)
                Lik -= vec_dot_vec_slice_inf(L_row_i, L_row_k, k)

                A_row_k = get_csr_row(A_csr, k)
                A_row_k_slice_data = [
                    np.abs(A_row_k.data[i])
                    for i in range(A_row_k.nnz)
                    if A_row_k.indices[i] >= k
                ]
                norm_of_row_i = np.sum(np.array(A_row_k_slice_data))
                if np.abs(Lik) < tau * norm_of_row_i:
                    Lik = 0.0
                else:
                    Lik /= L_row_k.data[-1]
                    L.data = np.append(L.data, Lik)
                    L.indices = np.append(L.indices, k)
                    L.indptr[i + 1] += 1
                    L.nnz += 1
                Lii -= Lik**2

            Lii = np.sqrt(Aii + Lii)

            L.data = np.append(L.data, Lii)
            L.indices = np.append(L.indices, i)
            L.indptr[i + 1] += 1
            L.nnz += 1

    return L


if __name__ == "__main__":
    # load data
    A = np.loadtxt("../data/A.txt")
    A_csr = dense_to_csr(A)
    L_ref = loadmat("../data/ICT.mat")["L5"]

    tau = 1e-3
    L = ICT_csr(A_csr, tau)
    L_ref = dense_to_csr(L_ref)
    print(np.allclose(L.data, L_ref.data))
    assert np.array_equal(L.indices, L_ref.indices)
    assert np.array_equal(L.indptr, L_ref.indptr)
    assert np.array_equal(L.nnz, L_ref.nnz)

    for i in range(5):
        tau = 10 ** (-i)

        t1 = time.time()
        L = ICT_csr(A_csr, tau)
        t2 = time.time()
        print(t2 - t1, tau)
