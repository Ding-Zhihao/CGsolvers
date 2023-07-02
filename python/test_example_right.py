import numpy as np
import math
import matplotlib.pyplot as plt

a = np.loadtxt("../data/Amatrix.txt")
b = np.loadtxt("../data/bc_vector.txt").reshape(-1, 1)
x_ref = np.loadtxt("../data/Result.txt").reshape(-1, 1)
n = len(x_ref)


converged=np.array([False])
precon=np.zeros((n,1))
v=np.zeros((n,1))
r=np.zeros((n,1))
r0_hat=np.zeros((n,1))
p=np.zeros((n,1))
p1=np.zeros((n,1))
s=np.zeros((n,1))
s1=np.zeros((n,1))
t=np.zeros((n,1))
x = np.zeros((n, 1))
tol = 1.0e-20
limit = 100

for i in range(1,n+1):
    precon[i-1,0]=1.0/a[i-1,i-1]
for i in range(1,n+1):
    a[i-1,:]=a[i-1,:]*precon[i-1,0]
b[:]=b[:]*precon[:]
r[:]=b[:]-np.dot(a,x)
r0_hat[:]=r[:]
x[:]=x[:]/precon[:]
rho0=1.0
alpha=1.0
w=1.0
v[:]=1.0
p[:]=1.0
rho1=np.dot(np.transpose(r0_hat),r)
iters=0
tol_list=[]

while iters < limit:
    iters = iters + 1
    tol_current = np.linalg.norm(r) / np.linalg.norm(b)
    tol_list.append(tol_current)
    converged = tol_current < tol
    if converged == True:
        break
    beta=(rho1/rho0)*(alpha/w)
    p[:,0]=r[:,0]+beta*(p[:,0]-w*v[:,0])
    p1[:,0]=p[:,0]*precon[:,0]
    v[:]=np.dot(a,p1)
    alpha=rho1/np.dot(np.transpose(r0_hat),v)
    s[:,0]=r[:,0]-alpha*v[:,0]
    s1[:,0]=s[:,0]*precon[:,0]
    t[:]=np.dot(a,s1)
    w=np.dot(np.transpose(t),s)/np.dot(np.transpose(t),t)
    rho0=rho1
    rho1=-w*np.dot(np.transpose(r0_hat),t)
    x[:,0]=x[:,0]+alpha*p[:,0]+w*s[:,0]
    r[:,0]=s[:,0]-w*t[:,0]
x[:,0]=x[:,0]*precon[:,0]


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
