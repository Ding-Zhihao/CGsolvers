import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import bicgstab
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse import linalg
from scipy.io import loadmat, savemat
from scipy.sparse import find

# load data
A = np.loadtxt("../data/Amatrix.txt")
data = loadmat("../data/ILUTP_ref.mat")
L_ref = data["L"].toarray()
U_ref = data["U"].toarray()
tol=1e-6
p=9
n = A.shape[0]
L = np.zeros((n, n))
U = np.zeros((n, n))
w = np.zeros(n)  

U[0,0]=A[0,0]

for i in range(n):
    L[i,i]=1

for i in range(1, n):
    w = A[i, :].copy()
    for k in range(i):
        if A[i, k] != 0:
            w[k]=w[k]/A[k,k]
            if np.abs(w[k])<np.linalg.norm(A[i, :])*tol:
                w[k]=0
            if w[k]!=0:
                w[k+1:]=w[k+1:]-w[k]*U[k, k+1:]
    w[np.abs(w)<np.linalg.norm(A[i, :])*tol]=0
    
    # row_L=w[:i]
    # row_L[np.abs(row_L)<np.max(np.abs(row_L))]=0
    # row_U=w[i+1:]
    # if len(row_U)>=p:
    #     row_U[np.abs(row_U)<np.max(np.abs(row_U))]=0
    # w[:i]=row_L
    # w[i+1:]=row_U
    L[i,:i]=w[:i]    
    U[i,i:]=w[i:]     

plt.figure()
plt.scatter(U_ref, U)
plt.figure()
plt.scatter(L_ref, L)