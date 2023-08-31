import numpy as np
import math
import matplotlib.pyplot as plt

A = np.loadtxt("../data/Amatrix.txt")
b = np.loadtxt("../data/bc_vector.txt").reshape(-1, 1)
x_ref = np.loadtxt("../data/Result.txt").reshape(-1, 1)
n = len(x_ref)


for i in range(n):
    coef=A[i,i]
    A[i,]/=coef
    b[i]/=coef

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
