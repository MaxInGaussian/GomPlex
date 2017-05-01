################################################################################
#  Github: https://github.com/MaxInGaussian/GomPlex
#  Author: Max W. Y. Lam (maxingaussian@gmail.com)
################################################################################

import numpy as np

__all__ = [
    "fast_multiply"
]

def fast_multiply(C, X, conj_trans=False):
    CX = np.zeros((C.shape[0], X.shape[1]))+0j
    fft_cir = np.fft.fft(C[0, :].conj() if conj_trans else C[:, 0])
    for i in range(X.shape[1]):
        CX[:, i] = np.fft.ifft(fft_cir*np.fft.fft(X[:, i]))
    return CX

def solve_A(W, Phi_k, y):
    _r = y-fast_multiply(Phi_k, y)
    _p = Phi_k.conj().T
    