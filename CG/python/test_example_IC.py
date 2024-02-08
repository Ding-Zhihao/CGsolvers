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
            if A[i, k] != 0:
                L[i, k] = A[i, k] / L[k, k]
                for j in range(k):
                    L[i, k] -= L[i, j] * L[k, j]
            L[i, i] -= L[i, k] ** 2
        L[i, i] = (L[i, i] + A[i, i]) ** 0.5
    return L

def ICT(A, tau):
    L = np.zeros(A.shape)
    N = np.size(A, axis=0)
    L[0, 0] = np.sqrt(A[0, 0])
    for i in range(1, N):
        for k in range(i):
            L[i, k] = A[i, k]
            for j in range(k):
                L[i, k] -= L[i, j] * L[k, j]
            norm_of_row_i = np.linalg.norm(A[k, k:], ord=1)    
            if abs(L[i, k]) < tau * norm_of_row_i:
                L[i, k] = 0
            else:
                L[i, k] = L[i, k] / L[k, k]
            L[i, i] -= L[i, k] ** 2
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
x_ref = loadmat('../data/ichol_L')['x1']
n = len(x_ref)

plt.figure()
for pow in range(6):
    print(pow)
    L=ICT(A, 10**(-pow))
    # L = loadmat('../data/ICT.mat')['L6'].toarray()
    # L = np.loadtxt('../data/L1.txt')

    converged = False
    x = np.zeros((n, 1))
    r = b - np.dot(A, x)
    z=solve_upper(L.transpose(), solve_lower(L, r))
    p = z.copy()
    tol = 1.0e-10
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

# plt.figure()
    plt.semilogy(range(1, iters), tol_list[1:], label='1e-'+str(pow))
    plt.grid()
plt.legend()
# plt.figure()
# plt.scatter(x_ref, x)
# plt.plot([0, 1], [0, 1], "k--")
# plt.xlabel("MATLAB result")
# plt.ylabel("ICCG")
# plt.grid()
