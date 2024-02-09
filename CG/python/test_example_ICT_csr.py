import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.io import loadmat, savemat
from csr_lib import *
from test_ICT_csr import *


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


if __name__ == "__main__":
    A = loadmat("../data/input")["A"]
    b = loadmat("../data/input")["b"]
    x_ref = loadmat("../data/ichol_L")["x1"]
    n = len(x_ref)
    A = dense_to_csr(A)

    plt.figure()
    for pow in range(6):
        print(pow)
        tau = 10 ** (-pow)
        t1 = time.time()
        L = ICT_csr(A, tau)
        converged = False
        x = np.zeros((n, 1))
        r = b - mat_dot_vec(A, x)
        z = solve_upper_csr(L.transpose(), solve_lower_csr(L, r))
        p = z.copy()
        tol = 1.0e-20
        limit = 100
        iters = 0
        tol_list = []

        while iters < limit:
            iters = iters + 1
            tol_current = np.linalg.norm(r) / np.linalg.norm(b)
            tol_list.append(tol_current)
            print(tol_current)
            converged = tol_current < tol
            if converged == True:
                break

            u = mat_dot_vec(A, p)
            alpha = np.dot(np.transpose(r), z) / np.dot(np.transpose(p), u)
            x_new = x + p * alpha
            r_new = r - u * alpha
            z_new = solve_upper_csr(L.transpose(), solve_lower_csr(L, r_new))
            beta = np.dot(np.transpose(r_new), z_new) / np.dot(np.transpose(r), z)
            p = z_new + p * beta
            x = x_new
            r = r_new
            z = z_new

        t2 = time.time()

        print("number of iterations", iters)
        print("residual", tol_current)
        print("tol " + str(tau) + " time used", t2 - t1, " s")

        plt.semilogy(range(1, iters), tol_list[1:], label="1e-" + str(pow))
    plt.grid()
    plt.legend()
