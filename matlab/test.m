close all
clearvars
clc

A = readmatrix('../data/Amatrix.txt');
b = readmatrix('../data/bc_vector.txt');
x_ref = readmatrix('../data/Result.txt');

A=sparse(A);

tol=1e-20;
maxit=100;

[x,fl0,rr0,it0,rv0] = bicgstab(A,b, tol, maxit);

figure,
scatter(x, x_ref)
hold on
plot([0,1], [0,1], 'k')
xlabel("bicgstab")
ylabel("Jacobi")
grid on
hold off

% options = struct("type","ilutp","droptol",1e-6);
% [L,U] = ilu(A,options);
[L,U] = ilu(A);
[x_precond,fl1,rr1,it1,rv1] = bicgstab(A,b,tol,maxit,L,U);

figure,
scatter(x_precond, x_ref)
hold on
plot([0,1], [0,1], 'k')
xlabel("bicgstab + iLU")
ylabel("Jacobi")
grid on
hold off

figure,
semilogy((0:numel(rv0)-1)/2,rv0/norm(b),"-o")
hold on
semilogy((0:numel(rv1)-1)/2,rv1/norm(b),"-o")
yline(tol,"r--");
legend("No preconditioner","ILU preconditioner","Tolerance")
xlabel("Iteration number")
ylabel("Relative residual")

