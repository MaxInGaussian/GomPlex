################################################################################
#  Github: https://github.com/MaxInGaussian/GomPlex
#  Author: Max W. Y. Lam (maxingaussian@gmail.com)
################################################################################

from sys import path
path.append("../")
import numpy as np
from scipy import linalg
from GomPlex import fast_multiply

N = 1000
X = np.random.randn(N, N)+1j*np.random.randn(N, N)
C = linalg.circulant(np.random.rand(N)+np.random.rand(N)*1j)

print('test of fast_multiply')
print('all close:', np.allclose(C.dot(X), fast_multiply(C, X)))

print('test of fast_multiply with conjugate transpose')
print('all close:', np.allclose(C.conj().T.dot(X), fast_multiply(C, X, True)))