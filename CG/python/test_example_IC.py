import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.io import loadmat, savemat

def IC(A):
    L = np.zeros(A.shape)
    N = np.size(A, axis=0)
    L[0, 0] = np.sqrt(A[0, 0])
    for i in range(1, N):
        for k in range(i):
            # 按照Cholesky分解的定义，更新L的非对角线元素
            L[i, k] = A[i, k]
            for j in range(k):
                L[i, k] -= L[i, j] * L[k, j]

            # 使用范数筛选非零元素
            norm_of_row_i = np.linalg.norm(A[i, :k + 1], ord=1)    
            if abs(L[i, k]) <= 0.00001 * norm_of_row_i:
                L[i, k] = 0
            else:
                L[i, k] = L[i, k] / L[k, k]

            # 更新对角线元素
            L[i, i] -= L[i, k] ** 2

        # 更新当前行的对角线元素
        L[i, i] = np.sqrt(L[i, i] + A[i, i])
    return L

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


A = loadmat('../data/input')['A']
b = loadmat('../data/input')['b']
x_ref = loadmat('../data/pcg')['x']


A = np.loadtxt("../data/A.txt")
b = np.loadtxt("../data/b.txt").reshape(-1,1)
x_ref = np.loadtxt("../data/x_ref.txt")

# savemat('../data/test.mat',{'A':A, 'b':b, 'x_ref': x_ref})

n = len(x_ref)
L=IC(A)

converged = False
x = np.zeros((n, 1))
r = b - np.dot(A, x)
z=solve_upper(L.transpose(), solve_lower(L, r))
p = z.copy()
tol = 1.0e-11
limit = 100
iters = 0
tol_list=[]

while iters < limit:
    iters = iters + 1
    tol_current = np.linalg.norm(r) / np.linalg.norm(b)
    tol_list.append(tol_current)
    print(tol_current)
    converged = tol_current < tol
    if converged == True:
        break
    
    u=np.dot(A, p)
    alpha=np.dot(np.transpose(r), z)/np.dot(np.transpose(p), u)
    x_new=x+p*alpha
    r_new=r-u*alpha
    z_new=solve_upper(L.transpose(), solve_lower(L, r_new))
    beta=np.dot(np.transpose(r_new), z_new)/np.dot(np.transpose(r), z)
    p=z_new+p*beta
    x=x_new
    r=r_new
    z=z_new

print("number of iterations", iters)
print("residual", tol_current)

plt.figure()
plt.semilogy(range(1, iters), tol_list[1:])
plt.grid()

# plt.figure()
# plt.scatter(x_ref, x)
# plt.plot([0, 1], [0, 1], "k--")
# plt.xlabel("Matlab result")
# plt.ylabel("ICCG")
# plt.grid()
