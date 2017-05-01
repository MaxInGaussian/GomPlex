################################################################################
#  Github: https://github.com/MaxInGaussian/GomPlex
#  Author: Max W. Y. Lam (maxingaussian@gmail.com)
################################################################################

import numpy as np
from scipy import linalg

_all__ = [
    "approx_Phi",
    "interpolate_Phi",
    "fast_multiply",
    "fast_solve"
]

def approx_Phi(W, Phi_basis):
    return fast_multiply(Phi_basis, W.conj().T, True).conj().T

def interpolate_Phi(Phi, phi_basis):
    W_H = np.zeros((phi_basis.shape[0], Phi.shape[0]))+0j
    H_basis = np.concatenate(([phi_basis[0]], phi_basis[1:][::-1])).conj()
    fft_basis = np.fft.fft(H_basis)
    Phi_basis = linalg.circulant(phi_basis)
    for i in range(Phi.shape[0]):
        W_H[:, i] = np.fft.ifft(np.fft.fft(Phi[i, :].conj())/fft_basis)
    return W_H.conj().T

def fast_multiply(C, X, conj_trans=False):
    CX = np.zeros((C.shape[0], X.shape[1]))+0j
    fft_cir = np.fft.fft(C[0, :].conj() if conj_trans else C[:, 0])
    for i in range(X.shape[1]):
        CX[:, i] = np.fft.ifft(fft_cir*np.fft.fft(X[:, i]))
    return CX

def fast_solve(Q, Phi_basis, X, L=20):
    f = np.zeros((Phi_basis.shape[0], X.shape[1]))+0j
    _r = X-fast_multiply(Phi_basis, f)
    p = fast_multiply(Phi_basis, _r, True)
    for _ in range(L):
        a = np.sum(_r.conj()*_r, 0)/np.sum(p.conj()*(Q.dot(p)), 0)
        f += a*Q.dot(p)
        r = _r-a*fast_multiply(Phi_basis, Q.dot(p))
        b = np.sum(r.conj()*r, 0)/np.sum(_r.conj()*_r, 0)
        p = b*p+fast_multiply(Phi_basis, r, True)
        _r = r
    return f
        