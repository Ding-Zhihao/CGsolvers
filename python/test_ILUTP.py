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
b = np.loadtxt("../data/bc_vector.txt")
x_ref = np.loadtxt("../data/Result.txt")
data = loadmat("../data/ILUTP_ref.mat")
L_ref = data["L"].toarray()
U_ref = data["U"].toarray()
tol=1e-6

n = A.shape[0]
L = np.zeros((n, n))
U = np.zeros((n, n))
perm = np.arange(n)  



plt.figure()
plt.scatter(U_ref, U)
plt.figure()
plt.scatter(L_ref, L)