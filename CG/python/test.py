import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.io import loadmat, savemat
from csr_lib import *
from test_ICT_csr import *

A = loadmat("../data/input")["A"]
b = loadmat("../data/input")["b"]
A_csr_ref = dense_to_csr(A)
tau = 1e-3
L_csr_ref = ICT_csr(A_csr_ref, tau)

with open("../data/A.dat","r") as f:
    data=[x.split() for x in f.readlines()]
    indptr=np.array(list(map(int, data[0])), dtype="int32")
    indices=np.array(list(map(int, data[1])), dtype="int32")
    data=np.array(list(map(float, data[2])))
    A_csr = csr_matrix_(data, indices, indptr, (len(indptr)-1, len(indptr)-1))
    
b_new=np.loadtxt("../data/b.dat").reshape(-1,1)
z=np.loadtxt("../data/z.dat").reshape(-1,1)
z_init=np.loadtxt("../data/z_init.dat").reshape(-1,1)
r=np.loadtxt("../data/r.dat").reshape(-1,1)

    
with open("../data/L.dat","r") as f:
    data=[x.split() for x in f.readlines()]
    indptr=np.array(list(map(int, data[0])), dtype="int32")
    indices=np.array(list(map(int, data[1])), dtype="int32")
    data=np.array(list(map(float, data[2])))
    L_csr = csr_matrix_(data, indices, indptr, (len(indptr)-1, len(indptr)-1))

with open("../data/L_T.dat","r") as f:
    data=[x.split() for x in f.readlines()]
    indptr=np.array(list(map(int, data[0])), dtype="int32")
    indices=np.array(list(map(int, data[1])), dtype="int32")
    data=np.array(list(map(float, data[2])))
    L_T_csr = csr_matrix_(data, indices, indptr, (len(indptr)-1, len(indptr)-1))

# L_csr_T_ref = L_csr.transpose()
# # assert np.array_equal(A_csr_ref.data, A_csr.data)
# # assert np.array_equal(A_csr_ref.indices, A_csr.indices)
# # assert np.array_equal(A_csr_ref.indptr, A_csr.indptr)
# # assert np.array_equal(A_csr_ref.nnz, A_csr.nnz)

# tau = 1e-3
# L_csr_new = ICT_csr(A_csr, tau)

# # assert np.array_equal(L_csr.data, L_csr_new.data)
# print(np.allclose(L_T_csr.data, L_csr_T_ref.data))
# assert np.array_equal(L_T_csr.indices, L_csr_T_ref.indices)
# assert np.array_equal(L_T_csr.indptr, L_csr_T_ref.indptr)
# assert np.array_equal(L_T_csr.nnz, L_csr_T_ref.nnz)

def solve_lower_csr(L, rhs):
    x = np.zeros(rhs.shape)
    for i in range(len(x)):
        x[i] = rhs[i]
        row_csr = get_csr_row(L, i)
        for j in range(row_csr.nnz - 1):
            x[i] -= row_csr.data[j] * x[row_csr.indices[j]]
        x[i] /= row_csr.data[-1]
    return x


def solve_upper_csr(U, rhs):
    x = np.zeros(rhs.shape)
    for i in range(len(x) - 1, -1, -1):
        x[i] = rhs[i]
        row_csr = get_csr_row(U, i)
        for j in range(1, row_csr.nnz):
            x[i] -= row_csr.data[j] * x[row_csr.indices[j]]
        x[i] /= row_csr.data[0]
    return x


x = np.zeros(b_new.shape)
r = b_new - mat_dot_vec(A_csr, x)

z_init_ref = solve_lower_csr(L_csr, r)
z_ref = solve_upper_csr(L_T_csr, z_init_ref)
