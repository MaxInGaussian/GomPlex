################################################################################
#  Github:https://github.com/MaxInGaussian/GomPlex
#  Author:Max W. Y. Lam (maxingaussian@gmail.com)
################################################################################

import numpy as np
from scipy import linalg
from scipy.sparse import csr_matrix
from numpy.fft import fft, ifft, fftshift, ifftshift

_all__ = [
    "ndft",
    "nfft",
    "interp_Phi_by_basis",
    "get_Phi_by_basis",
    "interp_Phi_by_FFT",
    "get_Phi_by_FFT",
    "fast_multiply",
    "fast_solve"
]

def ndft(x, f_hat, M):
    k = -(M//2)+np.arange(M)
    return np.dot(np.exp(-2j*np.pi*k*x[:, None]), f_hat)
    
def nfft(x, f_hat, M, sigma=2, tol=1e-8):
    n = M*sigma
    m = np.ceil(-np.log(0.25*tol/M)/(np.pi*(1-1/(2*sigma-1))))
    shift = lambda x:-0.5+(x+0.5)%1
    k = -(M//2)+np.arange(M)
    b = (2*sigma*m)/((2*sigma-1)*np.pi)
    g_hat = f_hat/np.exp(-b*(np.pi*k/n)**2)
    g_hat_n = np.concatenate([g_hat[M//2:], np.zeros(n-M), g_hat[:M//2]])
    g = fftshift(fft(g_hat_n))
    col_ind = np.floor(n*x[:, np.newaxis]).astype(int)+np.arange(-m, m)
    vals = np.exp(-(n*shift(x[:, None]-col_ind/n))**2/b)/np.sqrt(np.pi*b)
    col_ind = (col_ind+n//2)%n
    indptr = np.arange(len(x)+1)*col_ind.shape[1]
    mat = csr_matrix((vals.ravel(), col_ind.ravel(), indptr), shape=(len(x), n))
    f = mat.dot(g)
    return f

def ndft_mat(x, F_hat, M):
    k = -(M//2)+np.arange(M)
    return np.dot(np.exp(-2j*np.pi*k*x[:, None]), F_hat)
    
def nfft_mat(x, F_hat, M, sigma=2, tol=1e-8):
    F = np.zeros((x.shape[0], F_hat.shape[1]))+0j
    for j in range(F_hat.shape[1]):
        F[:, j] = nfft(x, F_hat[:, j], M, sigma, tol)
    return F

def adj_ndft(x, f, M):
    k = -(M//2)+np.arange(M)
    return np.dot(np.exp(2j*np.pi*x*k[:, None]), f)
    
def adj_nfft(x, f, M, sigma=2, tol=1e-8):
    n = M*sigma
    m = np.ceil(-np.log(0.25*tol/M)/(np.pi*(1-1/(2*sigma-1))))
    shift = lambda x:-0.5+(x+0.5)%1
    col_ind = np.floor(n*x[:, None]).astype(int)+np.arange(-m, m)
    b = (2*sigma*m)/((2*sigma-1)*np.pi)
    vals = np.exp(-(n*shift(x[:, None]-col_ind/n))**2/b)/np.sqrt(np.pi*b)
    col_ind = (col_ind+n//2)%n
    indptr = np.arange(len(x)+1)*col_ind.shape[1]
    mat = csr_matrix((vals.ravel(), col_ind.ravel(), indptr), shape=(len(x), n))
    g = mat.T.dot(f)
    k = -(M//2)+np.arange(M)
    g_hat_n = fftshift(ifft(ifftshift(g)))
    g_hat = n*g_hat_n[(n-M)//2:(n+M)//2]
    f_hat = g_hat/np.exp(-b*(np.pi*k/n)**2)
    return f_hat

def adj_ndft_mat(x, F, M):
    k = -(M//2)+np.arange(M)
    return np.dot(np.exp(2j*np.pi*x*k[:, None]), F)
    
def adj_nfft_mat(x, F, M, sigma=2, tol=1e-8):
    F_hat = np.zeros((x.shape[0], F.shape[1]))+0j
    for j in range(F.shape[1]):
        F_hat[:, j] = adj_nfft(x, F[:, j], M, sigma, tol)
    return F_hat

def Phi(X, spectral_freqs):
    X_sparse = self.X.dot(self.spectral_freqs)
    Phi_const = np.sqrt(self.kernel_scale/self.M)
    Phi = Phi_const*np.exp(2j*np.pi*X_sparse)
    return Phi

def Phi_nfft(X, spectral_freqs):
    X_sparse = X.dot(spectral_freqs)
    
    W_H = np.zeros((phi_basis.shape[0], Phi.shape[0]))+0j
    H_basis = np.concatenate(([phi_basis[0]], phi_basis[1:][::-1])).conj()
    fft_basis = np.fft.fft(H_basis)
    Phi_basis = linalg.circulant(phi_basis)
    for i in range(Phi.shape[0]):
        W_H[:, i] = np.fft.ifft(np.fft.fft(Phi[i, :].conj())/fft_basis)
    return W_H.conj().T

def interp_Phi_by_basis(Phi, phi_basis):
    W_H = np.zeros((phi_basis.shape[0], Phi.shape[0]))+0j
    H_basis = np.concatenate(([phi_basis[0]], phi_basis[1:][::-1])).conj()
    fft_basis = np.fft.fft(H_basis)
    Phi_basis = linalg.circulant(phi_basis)
    for i in range(Phi.shape[0]):
        W_H[:, i] = np.fft.ifft(np.fft.fft(Phi[i, :].conj())/fft_basis)
    return W_H.conj().T

def get_Phi_by_basis(W, Phi_basis):
    return fast_multiply(Phi_basis, W.conj().T, True).conj().T

def interp_Phi_by_FFT(Phi, vert=True):
    if(vert):
        return np.fft.fft(Phi.T).T
    else:
        return np.fft.fft(Phi)
    
def get_Phi_by_FFT(W, vert=True):
    if(vert):
        return np.fft.ifft(W.T).T
    else:
        return np.fft.ifft(W)
    
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
        