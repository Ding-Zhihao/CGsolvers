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

        counter = np.zeros(shape[1], dtype="int32")
        for k in range(nnz):
            counter[indices[k]] += 1

        new_indptr = np.zeros(shape[1] + 1, dtype="int32")
        for i in range(1, len(new_indptr)):
            new_indptr[i] = np.sum(counter[:i])

        new_indices = np.zeros(nnz, dtype="int32")
        new_data = np.zeros(nnz)
        for i in range(shape[0]):
            for j in range(indptr[i], indptr[i + 1]):
                index = int(new_indptr[indices[j]])
                new_data[index] = data[j]
                new_indices[index] = i
                new_indptr[indices[j]] += 1

        new_indptr2 = np.zeros(shape[1] + 1, dtype="int32")
        new_indptr2[1:] = new_indptr[:-1]
        shape = (shape[1], shape[0])

        tr = csr_matrix_(
            np.array(new_data), np.array(new_indices), np.array(new_indptr2), shape
        )
        return tr

    def toarray(self):
        array = np.zeros(self.shape).astype(self.data.dtype)
        num_rows = self.shape[0]
        for i in range(num_rows):
            num_vals = self.indptr[i + 1] - self.indptr[i]
            for k in range(num_vals):
                val = self.data[self.indptr[i] + k]
                j = self.indices[self.indptr[i] + k]
                array[i][j] = val
        return array


def dense_to_csr(A):
    nnz = 0
    data = np.array([])
    indices = np.array([], dtype="int32")
    indptr = np.zeros(1, dtype="int32")
    
    for i in range(A.shape[0]):
        indptr = np.append(indptr, indptr[i])
        for j in range(A.shape[1]):
            if A[i, j] != 0.0:
                data=np.append(data, A[i, j])
                indices=np.append(indices, j)
                indptr[i + 1] += 1
                nnz += 1
    A_csr = csr_matrix_(data, indices, indptr, A.shape)
    return A_csr


def get_csr_row(A_csr, row):
    data = A_csr.data[A_csr.indptr[row] : A_csr.indptr[row + 1]]
    indices = A_csr.indices[A_csr.indptr[row] : A_csr.indptr[row + 1]]
    indptr = A_csr.indptr[row : row + 2]
    indptr = indptr - indptr[0]
    shape = (1, A_csr.shape[1])
    row_csr = csr_matrix_(data, indices, indptr, shape)
    return row_csr


def intersect_indices(x, y):
    data = []
    ind1_list = []
    ind2_list = []
    ind1 = 0
    ind2 = 0
    while ind1 < len(x) and ind2 < len(y):
        if x[ind1] == y[ind2]:
            data.append(x[ind1])
            ind1_list.append(ind1)
            ind2_list.append(ind2)
            ind1 += 1
            ind2 += 1
        elif x[ind1] < y[ind2]:
            ind1 += 1
        elif x[ind1] > y[ind2]:
            ind2 += 1
    return (
        np.array(data, dtype="int32"),
        np.array(ind1_list, dtype="int"),
        np.array(ind2_list, dtype="int"),
    )


def vec_dot_vec(row_A, row_B):
    sum = 0.0
    if row_A.nnz > 0 and row_B.nnz > 0:
        tmp, index1, index2 = intersect_indices(row_A.indices, row_B.indices)
        if len(tmp):
            for k in range(len(tmp)):
                sum += row_A.data[int(index1[k])] * row_B.data[int(index2[k])]
    return sum


def vec_dot_vec_slice_inf(row_A, row_B, indice):
    sum = 0.0
    if row_A.nnz > 0 and row_B.nnz > 0:
        tmp, index1, index2 = intersect_indices(row_A.indices, row_B.indices)
        if len(tmp):
            for k in range(len(tmp)):
                if tmp[k] < indice:
                    sum += row_A.data[int(index1[k])] * row_B.data[int(index2[k])]
    return sum


def mat_dot_vec(mat, vec):
    assert mat.shape[1] == vec.shape[0]
    out = np.zeros(vec.shape)
    for i in range(mat.shape[0]):
        row = get_csr_row(mat, i)
        for j in range(row.nnz):
            out[i] += row.data[j] * vec[row.indices[j]]
    return out


def mat_dot_mat(A_csr, B_csr):
    assert A_csr.shape[1] == B_csr.shape[0]
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
                    sum = vec_dot_vec(row_A, row_B)
                    if sum != 0.0:
                        data.append(sum)
                        indices.append(j)
                        indptr[i + 1] += 1
                        nnz += 1

    out_csr = csr_matrix_(
        np.array(data),
        np.array(indices),
        np.array(indptr),
        (A_csr.shape[0], B_csr_tr.shape[0]),
    )
    return out_csr


if __name__ == "__main__":
    rng = np.random.default_rng()
    A = sparse.random(918, 918, density=0.005, random_state=rng)

    A_csr_ref = sparse.csr_matrix(A.toarray())
    A_csr_tr_ref = sparse.csr_matrix(A.transpose())
    A_csr = dense_to_csr(A.toarray())
    A_csr_tr = A_csr.transpose()

    assert np.array_equal(A_csr.data, A_csr_ref.data)
    assert np.array_equal(A_csr.indices, A_csr_ref.indices)
    assert np.array_equal(A_csr.indptr, A_csr_ref.indptr)
    assert np.array_equal(A_csr.shape, A_csr_ref.shape)

    assert np.array_equal(A_csr_tr.data, A_csr_tr_ref.data)
    assert np.array_equal(A_csr_tr.indices, A_csr_tr_ref.indices)
    assert np.array_equal(A_csr_tr.indptr, A_csr_tr_ref.indptr)
    assert np.array_equal(A_csr_tr.shape, A_csr_tr_ref.shape)

    A_array = A_csr.toarray()
    assert np.array_equal(A.toarray(), A_csr.toarray())

    B = sparse.random(918, 918, density=0.005, random_state=rng)
    out_ref = A.dot(B)
    out_ref.sort_indices()

    A_csr = dense_to_csr(A.toarray())
    B_csr = dense_to_csr(B.toarray())

    row_data_ref = sparse.csr_matrix(A.toarray()[0, :])
    row_data = get_csr_row(A_csr, 0)
    assert np.array_equal(row_data.data, row_data_ref.data)
    assert np.array_equal(row_data.indices, row_data_ref.indices)
    assert np.array_equal(row_data.indptr, row_data_ref.indptr)
    assert np.array_equal(row_data.shape, row_data_ref.shape)

    out_csr = mat_dot_mat(A_csr, B_csr)
    assert np.array_equal(out_ref.data, out_csr.data)
    assert np.array_equal(out_ref.indices, out_csr.indices)
    assert np.array_equal(out_ref.indptr, out_csr.indptr)
    assert np.array_equal(out_ref.shape, out_csr.shape)
