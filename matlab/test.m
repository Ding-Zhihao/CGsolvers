close all
clearvars
clc

A = readmatrix('../data/Amatrix.txt');
b = readmatrix('../data/bc_vector.txt');
x = bicgstab(A,b, 1e-8);
x_ref = readmatrix('../data/Result.txt');

figure,
scatter(x, x_ref)
hold on
plot([0,1], [0,1], 'k')
xlabel("bicgstab")
ylabel("Jacobi")
grid on