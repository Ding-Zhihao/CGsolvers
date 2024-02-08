import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.io import loadmat, savemat


class csr_matrix_:
    def __init__(self, data, indices, indptr, shape):
        self.data = data
        self.indices = indices
        self.indptr = indptr
        self.nnz = len(self.data)
        self.shape = shape

    def transpose(self):
        data = self.data
        indices = self.indices
        indptr = self.indptr
        shape = self.shape
        nnz = self.nnz

        counter = np.zeros(shape[1], dtype='int32')
        for k in range(nnz):
            counter[indices[k]] += 1

        new_indptr = np.zeros(shape[1]+1, dtype='int32')
        for i in range(1, len(new_indptr)):
            new_indptr[i]=np.sum(counter[:i])
            
        
        new_indices = np.zeros(nnz)
        new_data = np.zeros(nnz)
        for i in range(shape[0]):
            for j in range(indptr[i], indptr[i+1]):
                index = int(new_indptr[indices[j]])
                new_data[index] = data[j]
                new_indices[index] = i
                new_indptr[indices[j]]+=1
        
        new_indptr2=np.zeros(shape[1]+1, dtype='int32')
        new_indptr2[1:]=new_indptr[:-1]
        shape=(shape[1], shape[0])
        
        tr = csr_matrix_(np.array(new_data), np.array(new_indices), np.array(new_indptr2), shape)
        return tr

def dense_to_csr(A):
    nnz = 0
    data = []
    indices = []
    indptr = [0]
    for i in range(A.shape[0]):
        indptr.append(indptr[i])
        for j in range(A.shape[1]):
            if A[i, j] != 0.0:
                data.append(A[i, j])
                indices.append(j)
                indptr[i + 1] += 1
                nnz += 1
    A_csr = csr_matrix_(np.array(data), np.array(indices), np.array(indptr), A.shape)
    return A_csr


if __name__ == "__main__":
    A = loadmat("../data/input")["A"]
    # A=np.random.rand(1000,1000)
    A_csr_ref = sparse.csr_matrix(A)
    A_csr_tr_ref = sparse.csr_matrix(A.transpose())
    A_csr = dense_to_csr(A)
    A_csr_tr = A_csr.transpose()
    

    assert np.array_equal(A_csr.data, A_csr_ref.data)
    assert np.array_equal(A_csr.indices, A_csr_ref.indices)
    assert np.array_equal(A_csr.indptr, A_csr_ref.indptr)
    assert np.array_equal(A_csr.shape, A_csr_ref.shape)
    
    assert np.array_equal(A_csr_tr.data, A_csr_tr_ref.data)
    assert np.array_equal(A_csr_tr.indices, A_csr_tr_ref.indices)
    assert np.array_equal(A_csr_tr.indptr, A_csr_tr_ref.indptr)
    assert np.array_equal(A_csr_tr.shape, A_csr_tr_ref.shape)


    