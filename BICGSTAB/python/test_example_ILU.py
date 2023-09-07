import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.io import loadmat, savemat

def ILU0(A):
    L = np.zeros(A.shape)
    U = np.zeros(A.shape)
    n = len(x_ref)

    U[0, 0] = A[0, 0]
    for i in range(n):
        L[i, i] = 1

    for i in range(1, n):
        U[i, :] = A[i, :]
        for k in range(i):
            if A[i, k] != 0:
                L[i, k] = U[i, k] / U[k, k]
                for j in range(k, n):
                    if A[i, j] != 0:
                        U[i, j] = U[i, j] - L[i, k] * U[k, j]
    return L, U


def solve_upper(U, rhs):
    x = np.zeros(rhs.shape)
    for i in range(len(x) - 1, -1, -1):
        x[i] = rhs[i]
        for j in range(i + 1, len(x)):
            x[i] -= U[i, j] * x[j]
        x[i] /= U[i, i]
    return x


def solve_lower(L, rhs):
    x = np.zeros(rhs.shape)
    for i in range(len(x)):
        x[i] = rhs[i]
        for j in range(i):
            x[i] -= L[i, j] * x[j]
        x[i] /= L[i, i]
    return x


# A = np.loadtxt("../data/Amatrix.txt")
# b = np.loadtxt("../data/bc_vector.txt").reshape(-1, 1)
# x_ref = np.loadtxt("../data/Result.txt").reshape(-1, 1)
# n = len(x_ref)

A = loadmat('../data/input')['A']
b = loadmat('../data/input')['b']
x_ref = loadmat('../data/pcg')['x']
n = len(x_ref)

# data = loadmat("../data/ILUTP_ref.mat")
# L = data["L"].toarray()
# U = data["U"].toarray()

L, U = ILU0(A)

converged = False
x = np.zeros((n, 1))
r = b - np.dot(A, x)
r0_hat = r.copy()
rho0 = 1.0
alpha = 1.0
w = 1.0
p = np.zeros((n, 1))
v = np.zeros((n, 1))

tol = 1.0e-100
limit = 100
iters = 0
tol_list = []
while iters < limit:
    iters = iters + 1
    tol_current = np.linalg.norm(r) / np.linalg.norm(b)
    tol_list.append(tol_current)
    converged = tol_current < tol
    if converged == True:
        break
    rho1 = np.dot(np.transpose(r0_hat), r)
    beta = rho1 / rho0 * alpha / w
    p = r + beta * (p - w * v)

    y = solve_upper(U, solve_lower(L, p))
    v = np.dot(A, y)

    alpha = rho1 / np.dot(np.transpose(r0_hat), v)
    s = r - alpha * v

    z = solve_upper(U, solve_lower(L, s))
    t = np.dot(A, z)
    w = np.dot(np.transpose(t), s) / np.dot(np.transpose(t), t)
    rho0 = rho1
    x += alpha * y + w * z
    r = s - w * t

print("number of iterations", iters)
print("residual", tol_current)

plt.figure()
plt.semilogy(range(1, iters), tol_list[1:])
plt.grid()

plt.figure()
plt.scatter(x_ref, x)
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("Jacobi")
plt.ylabel("BiCGStab+ILU")
plt.grid()
