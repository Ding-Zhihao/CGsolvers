import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.io import loadmat, savemat

A = np.loadtxt("../data/Amatrix.txt")
b = np.loadtxt("../data/bc_vector.txt").reshape(-1, 1)
x_ref = np.loadtxt("../data/Result.txt").reshape(-1, 1)
n = len(x_ref)

# sA = sparse.csc_matrix(A)
# sA_iLU = sparse.linalg.spilu(sA)
# L=sA_iLU.L.toarray()
# U=sA_iLU.U.toarray()

# data = loadmat("../data/ILU_ref.mat")
# L = data["L"].toarray()
# U = data["U"].toarray()

L = np.zeros(A.shape)
U = np.zeros(A.shape)
N = len(x_ref)

U[0,0]=A[0,0]

for i in range(n):
    L[i,i]=1
    
for i in range(1, n):
    U[i, :] = A[i, :]
    for k in range(i):
        if A[i, k] != 0:
            L[i, k] = U[i, k] / U[k, k]
            for j in range(k, n):
                if A[i, j] != 0:
                    U[i, j] = U[i, j] - L[i, k] * U[k, j]

M=np.matmul(L, U)
A=np.matmul(np.linalg.inv(M), A)
b=np.dot(np.linalg.inv(M), b)

converged = False
x = np.zeros((n, 1))
r = b - np.dot(A, x)
r0_hat = r.copy()
rho0 = 1.0
alpha = 1.0
w = 1.0
p = np.zeros((n, 1))
v = np.zeros((n, 1))

tol = 1.0e-20
limit = 100
iters = 0
tol_list=[]
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
    v = np.dot(A, p)
    alpha = rho1 / np.dot(np.transpose(r0_hat), v)
    s = r - alpha * v
    t = np.dot(A, s)
    w = np.dot(np.transpose(t), s) / np.dot(np.transpose(t), t)
    rho0 = rho1
    x += alpha * p + w * s
    r = s - w * t

print("number of iterations", iters)
print("residual", tol_current)

plt.figure()
plt.semilogy(range(1, iters), tol_list[1:])

plt.figure()
plt.scatter(x_ref, x)
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("Jacobi")
plt.ylabel("BiCGStab")
plt.grid()
