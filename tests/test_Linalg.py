################################################################################
#  Github: https://github.com/MaxInGaussian/GomPlex
#  Author: Max W. Y. Lam (maxingaussian@gmail.com)
################################################################################

from sys import path
path.append("../")
import numpy as np
from scipy import linalg
from GomPlex import *

N, M, D = 500, 100, 30
X = .5*np.random.rand(N, D)
C = linalg.circulant(np.random.rand(N)+1j*np.random.rand(N))
k = np.arange(M)+1
Omega = np.random.randn(D, M)
X_O = X.dot(Omega)
X_nfft = X_O/k[None, :]
x = X_nfft[0, :]
f = np.sin(-20*np.pi*x)
Phi = np.exp(-2j*np.pi*X_O)/np.sqrt(M)
alpha = np.random.randn(M, M)+1j*np.random.randn(M, M)

print()
print('test of nfft')
print('approx l0 error:',
    np.max(np.abs(ndft(x, f, M)-nfft(x, f, M)))/np.mean(np.absolute(f)))
print('approx l1 error:',
    np.mean(np.abs(ndft(x, f, M)-nfft(x, f, M)))/np.mean(np.absolute(f)))

print()
print('test of nfft_mat')
print('approx l0 error:',
    np.max(np.abs(ndft_mat(x, alpha, M)-nfft_mat(x, alpha, M)))/np.mean(np.absolute(alpha)))
print('approx l1 error:',
    np.mean(np.abs(ndft_mat(x, alpha, M)-nfft_mat(x, alpha, M)))/np.mean(np.absolute(alpha)))
Y = ndft_mat(x, alpha, M)
    
print()
print('test of adj_nfft')
print('approx l0 error:',
    np.max(np.abs(adj_ndft(x, f, M)-adj_nfft(x, f, M)))/np.mean(np.absolute(f)))
print('approx l1 error:',
    np.mean(np.abs(adj_ndft(x, f, M)-adj_nfft(x, f, M)))/np.mean(np.absolute(f)))

print()
print('test of adj_nfft_mat')
print('approx l0 error:',
    np.max(np.abs(adj_ndft_mat(x, Y, M)-adj_nfft_mat(x, Y, M)))/np.mean(np.absolute(Y)))
print('approx l1 error:',
    np.mean(np.abs(adj_ndft_mat(x, Y, M)-adj_nfft_mat(x, Y, M)))/np.mean(np.absolute(Y)))

print()
print('test of fast_multiply')
print('approx l0 error:',
    np.max(np.abs(C.dot(X)-fast_multiply(C, X))))
print('approx l1 error:',
    np.mean(np.abs(C.dot(X)-fast_multiply(C, X))))

print()
print('test of fast_multiply with conjugate transpose')
print('approx l0 error:',
    np.max(np.abs(C.conj().T.dot(X)-fast_multiply(C, X, True))))
print('approx l1 error:',
    np.mean(np.abs(C.conj().T.dot(X)-fast_multiply(C, X, True))))

print()
print('test of interp_Phi_by_basis')
phi_basis = np.exp(-2j*np.pi*np.median(X_O, 0))/np.sqrt(M)
Phi_basis = linalg.circulant(phi_basis)
W = interp_Phi_by_basis(Phi, phi_basis)
W_true = (linalg.solve(Phi_basis.T, Phi.T)).T
print('approx l0 error:',
    np.max(np.abs(get_Phi_by_basis(W, Phi_basis)-Phi))/np.mean(np.absolute(Phi)))
print('approx l1 error:',
    np.mean(np.abs(get_Phi_by_basis(W, Phi_basis)-Phi))/np.mean(np.absolute(Phi)))
    
print()
print('test of interp_Phi_by_FFT')
W_fft = interp_Phi_by_FFT(Phi)
print('approx l0 error:',
    np.max(np.abs(get_Phi_by_FFT(W_fft)-Phi))/np.mean(np.absolute(Phi)))
print('approx l1 error:',
    np.mean(np.abs(get_Phi_by_FFT(W_fft)-Phi))/np.mean(np.absolute(Phi)))

print()
print('test of fast_solve')
noise = 1e-1*(1+1j)
Q = W.conj().T.dot(W)+noise*np.eye(M)
A = Phi_basis.conj().T.dot(Q.dot(Phi_basis))
A_inv_Phi = linalg.solve(A, Phi.conj().T.dot(W)+noise*Phi_basis.conj().T)
A_inv_Phi_fast = fast_solve(Q, Phi_basis, np.eye(M), 30)
print('approx l0 error:',
    np.max(np.abs(A_inv_Phi-A_inv_Phi_fast))/np.mean(np.absolute(A_inv_Phi)))
print('approx l1 error:',
    np.mean(np.abs(A_inv_Phi-A_inv_Phi_fast))/np.mean(np.absolute(A_inv_Phi)))

