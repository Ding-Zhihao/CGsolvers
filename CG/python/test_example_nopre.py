import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat

A = loadmat("../data/input")["A"]
b = loadmat("../data/input")["b"]
x_ref = loadmat("../data/ichol_L")["x1"]
n = len(x_ref)

converged = False
x = np.zeros((n, 1))
r = b - np.dot(A, x)
p = r.copy()

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
    u = np.dot(A, p)
    alpha = np.dot(np.transpose(r), r) / np.dot(np.transpose(p), r)
    x_new = x + p * alpha
    r_new = r - alpha * u
    beta = np.dot(np.transpose(r_new), r_new) / np.dot(np.transpose(r), r)
    p = r_new + p * beta
    x = x_new
    r = r_new

print("number of iterations", iters)
print("residual", tol_current)

plt.figure()
plt.semilogy(range(1, iters), tol_list[1:])
plt.grid()

plt.figure()
plt.scatter(x_ref, x)
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("Matlab result")
plt.ylabel("CG")
plt.grid()
