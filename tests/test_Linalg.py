################################################################################
#  Github: https://github.com/MaxInGaussian/GomPlex
#  Author: Max W. Y. Lam (maxingaussian@gmail.com)
################################################################################

from sys import path
path.append("../")
import numpy as np
from scipy import linalg
from GomPlex import fast_multiply, interpolate_Phi, approx_Phi, fast_solve

N, M, D = 500, 100, 30
X = .5*np.random.rand(N, D)+.5j*np.random.rand(N, D)
C = linalg.circulant(np.random.rand(N)+1j*np.random.rand(N))

print('test of fast_multiply')
print('all close:', np.allclose(C.dot(X), fast_multiply(C, X)))

print()
print('test of fast_multiply with conjugate transpose')
print('all close:', np.allclose(C.conj().T.dot(X), fast_multiply(C, X, True)))

print()
print('test of interpolate_Phi')
Omega = np.random.randn(D, M)
X_O = X.dot(Omega)
Phi = np.exp(-2j*np.pi*X_O)/np.sqrt(M)
phi_basis = np.exp(-2j*np.pi*np.median(X_O, 0))/np.sqrt(M)
Phi_basis = linalg.circulant(phi_basis)
W = interpolate_Phi(Phi, phi_basis)
W_true = (linalg.solve(Phi_basis.T, Phi.T)).T
print('approx l0 error:',
    np.max(np.abs(np.dot(W, Phi_basis)-Phi))/np.mean(np.absolute(Phi)))
print('approx l1 error:',
    np.mean(np.abs(np.dot(W, Phi_basis)-Phi))/np.mean(np.absolute(Phi)))

print()
print('test of approx_Phi')
print('approx l0 error:',
    np.max(np.abs(approx_Phi(W, Phi_basis)-Phi))/np.mean(np.absolute(Phi)))
print('approx l1 error:',
    np.mean(np.abs(approx_Phi(W, Phi_basis)-Phi))/np.mean(np.absolute(Phi)))

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

