################################################################################
#  Github: https://github.com/MaxInGaussian/GomPlex
#  Author: Max W. Y. Lam (maxingaussian@gmail.com)
################################################################################

from sys import path
path.append("../")
import numpy as np
from scipy import linalg
from GomPlex import fast_multiply, interpolate_Phi, approx_Phi, fast_solve

N, M = 100, 10
X = np.random.randn(N, N)+1j*np.random.randn(N, N)
C = linalg.circulant(np.random.rand(N)+1j*np.random.rand(N))

print('test of fast_multiply')
print('all close:', np.allclose(C.dot(X), fast_multiply(C, X)))

print()
print('test of fast_multiply with conjugate transpose')
print('all close:', np.allclose(C.conj().T.dot(X), fast_multiply(C, X, True)))

print()
print('test of interpolate_Phi')
phi_basis = np.random.rand(M)+1j*np.random.rand(M)
Phi_basis = linalg.circulant(phi_basis)
Phi = np.random.randn(N, M)+1j*np.random.randn(N, M)
W = interpolate_Phi(Phi, phi_basis)
W_true = (linalg.solve(Phi_basis.T, Phi.T)).T
print('approx l0 error:', np.max(np.abs(W-W_true)))
print('approx l1 error:', np.sum(np.abs(W-W_true))/(N*M))

print()
print('test of approx_Phi')
print('approx l0 error:', np.max(np.abs(approx_Phi(W, Phi_basis)-Phi)))
print('approx l1 error:', np.sum(np.abs(approx_Phi(W, Phi_basis)-Phi))/(N*M))

print()
print('test of fast_solve')
noise = 1e-1*(1+1j)
Q = W.conj().T.dot(W)+noise*np.eye(M)
A = Phi_basis.conj().T.dot(Q.dot(Phi_basis))
A_inv_Phi = linalg.solve(A, Phi.conj().T)
A_inv_Phi_fast = fast_solve(Q, Phi_basis, Phi.conj().T)
print('approx l0 error:', np.max(np.abs(A_inv_Phi-A_inv_Phi_fast)))
print('approx l1 error:', np.sum(np.abs(A_inv_Phi-A_inv_Phi_fast))/(N*M))

