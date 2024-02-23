import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import bicgstab
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse import linalg
from scipy.io import loadmat, savemat
from csr_lib import *
import time


def DIC(A):
    M = np.zeros(A.shape)
    N = np.size(A, axis=1)
    for i in range(N):
        M[i,i] = A[i,i]
    
    for i in range(N):
        for j in range(i+1, N):
            M[j, j]-=(A[i, j]*A[i, j])/M[i, i]
            
    for i in range(N):
        M[i,i] = 1/M[i,i]
        
    return M

def DIC_csr(A_csr):
    # initialization
    num_rows = A_csr.shape[0]
    nnz = 0
    data = np.array([])
    indices = np.array([], dtype="int32")
    indptr = np.zeros(1, dtype="int32")
    M = csr_matrix_(data, indices, indptr, (0, 0))
    
    for i in range(num_rows):
        M.indptr = np.append(M.indptr, M.indptr[i])
        M.shape = (i + 1, i + 1)
        A_row_i = get_csr_row(A_csr, i)
        for j in range(A_row_i.nnz):
            if A_row_i.indices[j] == i:
                break
        M.data = np.append(M.data, A_row_i.data[j])
        M.indices = np.append(M.indices, i)
        M.indptr[i + 1] += 1
        M.nnz += 1
    
    for i in range(num_rows):
        A_row_i = get_csr_row(A_csr, i)
        for k in range(A_row_i.nnz):
            if A_row_i.indices[k]>i:           
                M.data[A_row_i.indices[k]] -= (A_row_i.data[k]*A_row_i.data[k])/M.data[i]
                            
    for i in range(num_rows):
        M.data[i] = 1.0/M.data[i]
        
    return M


if __name__ == "__main__":
    # load data
    A = loadmat("/mnt/d/work/code/solver/CG/data/input")["A"]
    A_csr = dense_to_csr(A)
    M = DIC_csr(A_csr)
    
    M_ref = DIC(A)
    M_ref = dense_to_csr(M_ref)
    
    assert np.array_equal(M.data, M_ref.data)
    assert np.array_equal(M.indices, M_ref.indices)
    assert np.array_equal(M.indptr, M_ref.indptr)
    assert np.array_equal(M.nnz, M_ref.nnz)
