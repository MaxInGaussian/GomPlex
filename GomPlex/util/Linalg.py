################################################################################
#  Github:https://github.com/MaxInGaussian/GomPlex
#  Author:Max W. Y. Lam (maxingaussian@gmail.com)
################################################################################

import numpy as np
from scipy import linalg
from scipy.sparse import csr_matrix
from numpy.fft import fft, ifft, fftshift, ifftshift

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
    
def faster_nfft(n, q, mat, f_hat, M, sigma=2, tol=1e-8):
    g_hat = f_hat/q
    g_hat_n = np.concatenate([g_hat[M//2:], np.zeros(n-M), g_hat[:M//2]])
    g = fftshift(fft(g_hat_n))
    f = mat.dot(g)
    return f

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
    
def faster_adj_nfft(n, q, mat, f, M, sigma=2, tol=1e-8):
    g = mat.T.dot(f)
    g_hat_n = fftshift(ifft(ifftshift(g)))
    g_hat = n*g_hat_n[(n-M)//2:(n+M)//2]
    f_hat = g_hat/q
    return f_hat

def numpy_solve_Phi(y, x, M):
    k = -(M//2)+np.arange(M)
    Phi = np.exp(-2j*np.pi*k*x[:, None])
    return linalg.solve(Phi.conj().T.dot(Phi), Phi.conj().T.dot(y))

def solve_Phi_algo_1(y, x, M, sigma=2, tol=1e-8):
    f_hat = np.random.rand(M)*(1e-1+1e-1j)
    r = y-nfft(x, f_hat, M, sigma, tol)
    _z = adj_nfft(x, r, M, sigma, tol)
    p = _z.copy()
    n = M*sigma
    for _ in range(n):
        v = nfft(x, p, M, sigma, tol)
        a = _z.conj().T.dot(_z)/v.conj().T.dot(v)
        f_hat += a*p
        r -= a*v
        z = adj_nfft(x, r, M, sigma, tol)
        b = z.conj().T.dot(z)/_z.conj().T.dot(_z)
        p = b*p+z
        if(np.mean(np.absolute(z)) < tol):
            break
        _z = z
    return f_hat

def solve_Phi_algo_2(y, x, M, sigma=2, tol=1e-8):
    f_hat = np.random.rand(M)*(1e-1+1e-1j)
    n = M*sigma
    m = np.ceil(-np.log(0.25*tol/M)/(np.pi*(1-1/(2*sigma-1))))
    shift = lambda x:-0.5+(x+0.5)%1
    k = -(M//2)+np.arange(M)
    col_ind = np.floor(n*x[:, None]).astype(int)+np.arange(-m, m)
    b = (2*sigma*m)/((2*sigma-1)*np.pi)
    q = np.exp(-b*(np.pi*k/n)**2)
    vals = np.exp(-(n*shift(x[:, None]-col_ind/n))**2/b)/np.sqrt(np.pi*b)
    col_ind = (col_ind+n//2)%n
    indptr = np.arange(len(x)+1)*col_ind.shape[1]
    mat = csr_matrix((vals.ravel(), col_ind.ravel(), indptr), shape=(len(x), n))
    r = y-faster_nfft(n, q, mat, f_hat, M, sigma, tol)
    _z = faster_adj_nfft(n, q, mat, r, M, sigma, tol)
    p = _z.copy()
    for _ in range(n):
        v = faster_nfft(n, q, mat, p, M, sigma, tol)
        a = _z.conj().T.dot(_z)/v.conj().T.dot(v)
        f_hat += a*p
        r -= a*v
        z = faster_adj_nfft(n, q, mat, r, M, sigma, tol)
        b = z.conj().T.dot(z)/_z.conj().T.dot(_z)
        p = b*p+z
        if(np.mean(np.absolute(z)) < tol):
            break
        _z = z
    return f_hat

def solve_Phi_H_algo_1(y, x, M, sigma=2, tol=1e-8):
    f_hat = np.random.rand(x.shape[0])*(1e-1+1e-1j)
    r = y-adj_nfft(x, f_hat, M, sigma, tol)
    _z = nfft(x, r, M, sigma, tol)
    p = _z.copy()
    n = M*sigma
    for _ in range(n):
        v = adj_nfft(x, p, M, sigma, tol)
        a = _z.conj().T.dot(_z)/v.conj().T.dot(v)
        f_hat += a*p
        r -= a*v
        z = nfft(x, r, M, sigma, tol)
        b = z.conj().T.dot(z)/_z.conj().T.dot(_z)
        p = b*p+z
        if(np.mean(np.absolute(z)) < tol):
            break
        _z = z
    return f_hat

def solve_Phi_H_algo_2(y, x, M, sigma=2, tol=1e-8):
    f_hat = np.random.rand(x.shape[0])*(1e-1+1e-1j)
    n = M*sigma
    m = np.ceil(-np.log(0.25*tol/M)/(np.pi*(1-1/(2*sigma-1))))
    shift = lambda x:-0.5+(x+0.5)%1
    k = -(M//2)+np.arange(M)
    col_ind = np.floor(n*x[:, None]).astype(int)+np.arange(-m, m)
    b = (2*sigma*m)/((2*sigma-1)*np.pi)
    q = np.exp(-b*(np.pi*k/n)**2)
    vals = np.exp(-(n*shift(x[:, None]-col_ind/n))**2/b)/np.sqrt(np.pi*b)
    col_ind = (col_ind+n//2)%n
    indptr = np.arange(len(x)+1)*col_ind.shape[1]
    mat = csr_matrix((vals.ravel(), col_ind.ravel(), indptr), shape=(len(x), n))
    r = y-faster_adj_nfft(n, q, mat, f_hat, M, sigma, tol)
    _z = faster_nfft(n, q, mat, r, M, sigma, tol)
    p = _z.copy()
    for _ in range(n):
        v = faster_adj_nfft(n, q, mat, p, M, sigma, tol)
        a = _z.conj().T.dot(_z)/v.conj().T.dot(v)
        f_hat += a*p
        r -= a*v
        z = faster_nfft(n, q, mat, r, M, sigma, tol)
        b = z.conj().T.dot(z)/_z.conj().T.dot(_z)
        p = b*p+z
        if(np.mean(np.absolute(z)) < tol):
            break
        _z = z
    return f_hat

def numpy_solve_A_tilde(y, x, M, tol=1e-8):
    k = -(M//2)+np.arange(M)
    Phi = np.exp(-2j*np.pi*k*x[:, None])
    return linalg.solve(Phi.conj().T.dot(Phi), y)

def solve_A_tilde_algo_1(y, x, M, tol=1e-8):
    return solve_Phi_algo_1(solve_Phi_H_algo_1(y, x, M), x, M)

def solve_A_tilde_algo_2(y, x, M, tol=1e-8):
    return solve_Phi_algo_2(solve_Phi_H_algo_2(y, x, M), x, M)

def numpy_solve_A(y, x, M, noise, tol=1e-8):
    k = -(M//2)+np.arange(M)
    Phi = np.exp(-2j*np.pi*k*x[:, None])
    A_tilde = Phi.conj().T.dot(Phi)
    return linalg.solve(A_tilde+noise*np.eye(M), Phi.conj().T.dot(y))

def solve_A_algo_1(y, x, M, noise, tol=1e-8):
    f_hat = solve_Phi_algo_1(y, x, M)
    f_hat -= noise*solve_A_tilde_algo_1(f_hat, x, M)
    return f_hat

def solve_A_algo_2(y, x, M, noise, tol=1e-8):
    f_hat = solve_Phi_algo_2(y, x, M)
    f_hat -= noise*solve_A_tilde_algo_2(f_hat, x, M)
    return f_hat

def get_x_nfft(X, spectral_freqs, kernel_scale, M):
    K = spectral_freqs.shape[1]
    X_sparse = X.dot(spectral_freqs)
    X_sparse -= np.log(np.sqrt(kernel_scale/(M*K)))
    return X_sparse.ravel()

def get_y_nfft(y, K):
    return np.repeat(y.ravel(), K)

def interp_Phi_by_basis(Phi, phi_basis):
    W_H = np.zeros((phi_basis.shape[0], Phi.shape[0]))+0j
    H_basis = np.concatenate(([phi_basis[0]], phi_basis[1:][::-1])).conj()
    fft_basis = np.fft.fft(H_basis)
    Phi_basis = linalg.circulant(phi_basis)
    for i in range(Phi.shape[0]):
        W_H[:, i] = np.fft.ifft(np.fft.fft(Phi[i, :].conj())/fft_basis)
    return W_H.conj().T

def get_Phi_by_basis(W, Phi_basis):
    return circulant_mul(Phi_basis, W.conj().T, True).conj().T

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
    
def circulant_mul(C, X, conj_trans=False):
    CX = np.zeros((C.shape[0], X.shape[1]))+0j
    fft_cir = np.fft.fft(C[0, :].conj() if conj_trans else C[:, 0])
    for i in range(X.shape[1]):
        CX[:, i] = np.fft.ifft(fft_cir*np.fft.fft(X[:, i]))
    return CX

def fast_solve_circulant(Q, Phi_basis, X, tol=1e-8):
    f = np.zeros((Phi_basis.shape[0], X.shape[1]))+0j
    _r = X-circulant_mul(Phi_basis, f)
    p = circulant_mul(Phi_basis, _r, True)
    while(True):
        a = np.sum(_r.conj()*_r, 0)/np.sum(p.conj()*(Q.dot(p)), 0)
        f += a*Q.dot(p)
        r = _r-a*circulant_mul(Phi_basis, Q.dot(p))
        b = np.sum(r.conj()*r, 0)/np.sum(_r.conj()*_r, 0)
        p = b*p+circulant_mul(Phi_basis, r, True)
        if(r < tol):
            break
        _r = r
    return f
        