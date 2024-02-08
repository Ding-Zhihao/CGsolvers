import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.io import loadmat, savemat
from csr_convert import *


def get_csr_row(A_csr, row):
    data = A_csr.data[A_csr.indptr[row] : A_csr.indptr[row + 1]]
    indices = A_csr.indices[A_csr.indptr[row] : A_csr.indptr[row + 1]]
    indptr = A_csr.indptr[row : row + 2]
    indptr = indptr - indptr[0]
    shape = (1, A_csr.shape[1])
    row_csr = csr_matrix_(data, indices, indptr, shape)
    return row_csr


if __name__ == "__main__":
    rng = np.random.default_rng()
    A = sparse.random(100, 50, density=0.01, random_state=rng)
    B = sparse.random(50, 100, density=0.01, random_state=rng)
    out_ref = A * B
    A_csr = dense_to_csr(A.toarray())
    B_csr = dense_to_csr(B.toarray())

    row_data_ref = sparse.csr_matrix(A.toarray()[0, :])
    row_data = get_csr_row(A_csr, 0)

    assert np.array_equal(row_data.data, row_data_ref.data)
    assert np.array_equal(row_data.indices, row_data_ref.indices)
    assert np.array_equal(row_data.indptr, row_data_ref.indptr)
    assert np.array_equal(row_data.shape, row_data_ref.shape)

    B_csr_tr = B_csr.transpose()
    
    
    nnz = 0
    data = []
    indices = []
    indptr = [0]
    for i in range(A_csr.shape[0]):
        indptr.append(indptr[i])
        row_A = get_csr_row(A_csr, i)
        if row_A.nnz > 0:
            for j in range(B_csr_tr.shape[0]):
                row_B = get_csr_row(B_csr_tr, j)
                if row_B.nnz > 0:
                    sum = 0.0
                    for k in range(row_A.nnz):
                        for m in range(row_B.nnz):
                            if row_A.indices[k] == row_B.indices[m]:
                                sum+=row_A.data[k]*row_B.data[m]
                    
                    if sum != 0.0:
                        data.append(sum)
                        indices.append(j)
                        indptr[i + 1] += 1
                        nnz += 1
                        
    out_csr = csr_matrix_(np.array(data), np.array(indices), np.array(indptr), (A_csr.shape[0], B_csr_tr.shape[0]))

    assert np.array_equal(out_ref.data, out_csr.data)
    assert np.array_equal(out_ref.indices, out_csr.indices)
    assert np.array_equal(out_ref.indptr, out_csr.indptr)
    assert np.array_equal(out_ref.shape, out_csr.shape)