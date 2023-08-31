close all
clearvars
clc

A = full(delsq(numgrid('S',30)));
b = ones(size(A,1),1);
tol = 1e-8;
maxit = 100;
save("../data/input.mat", 'A', 'b');
A = sparse(A);


[x,fl0,rr0,it0,rv0] = pcg(A,b,tol,maxit);
save("../data/pcg.mat", 'x');


L = ichol(A);
[x1,fl1,rr1,it1,rv1] = pcg(A,b,tol,maxit,L,L');
save("../data/ichol_L.mat", 'L', 'x1');

L = ichol(A,struct('michol','on'));
[x2,fl2,rr2,it2,rv2] = pcg(A,b,tol,maxit,L,L');
save("../data/michol_L.mat", 'L', 'x2');



figure,
semilogy(0:length(rv0)-1,rv0/norm(b),'-o')
hold on
semilogy(0:length(rv1)-1,rv1/norm(b),'-o')
semilogy(0:length(rv2)-1,rv2/norm(b),'-o')
yline(tol,'r--');
legend('No Preconditioner','Default ICHOL','Modified ICHOL','Tolerance','Location','East')
xlabel('Iteration number')
ylabel('Relative residual')