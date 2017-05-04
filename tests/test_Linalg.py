################################################################################
#  Github: https://github.com/MaxInGaussian/GomPlex
#  Author: Max W. Y. Lam (maxingaussian@gmail.com)
################################################################################

from sys import path
path.append("../")
import numpy as np
from scipy import linalg
from timeit import Timer

from GomPlex import *

repeat = 1
N, M, K, D = 1000, 500, 10, 30
X = .5*np.random.rand(N, D)
C = linalg.circulant(np.random.rand(N)+1j*np.random.rand(N))
Omega = np.random.randn(D, K)
X_nfft = X.dot(Omega)
x = X_nfft.ravel()
f = np.repeat(np.sin(-20*np.pi*np.mean(X_nfft, 1)), K)+0j
f_hat = np.random.randn(M)+0j
Phi = np.exp(-2j*np.pi*X_nfft)/np.sqrt(M)
alpha = np.random.randn(M, N)+1j*np.random.randn(M, N)

# print()
# print('test of nfft')
# f_true = ndft(x, f_hat, M)
# f_fast = nfft(x, f_hat, M)
# print('approx l0 error:', np.max(np.abs(f_true-f_fast)))
# print('approx l1 error:', np.mean(np.abs(f_true-f_fast)))
# print('approx l2 error:', np.sqrt(np.mean(np.abs(f_true-f_fast)**2)))
# timer = Timer(lambda:ndft(x, f_hat, M))
# print('orig algo needs', timer.timeit(repeat)/repeat, 's')
# timer = Timer(lambda:nfft(x, f_hat, M))
# print('fast algo needs', timer.timeit(repeat)/repeat, 's')
# 
# print()
# print('test of adj_nfft')
# f_hat_true = adj_ndft(x, f, M)
# f_hat_fast = adj_nfft(x, f, M)
# print('approx l0 error:', np.max(np.abs(f_hat_true-f_hat_fast)))
# print('approx l1 error:', np.mean(np.abs(f_hat_true-f_hat_fast)))
# print('approx l2 error:', np.sqrt(np.mean(np.abs(f_hat_true-f_hat_fast)**2)))
# timer = Timer(lambda:adj_ndft(x, f, M))
# print('orig algo needs', timer.timeit(repeat)/repeat, 's')
# timer = Timer(lambda:adj_nfft(x, f, M))
# print('fast algo needs', timer.timeit(repeat)/repeat, 's')
# 
# print()
# print('test of fast_solve_Phi')
# pinv_true = numpy_solve_Phi(f, x, M)
# pinv_fast = fast_solve_Phi(f, x, M)
# print('approx l0 error:', np.max(np.abs(pinv_true-pinv_fast)))
# print('approx l1 error:', np.mean(np.abs(pinv_true-pinv_fast)))
# print('approx l2 error:', np.sqrt(np.mean(np.abs(pinv_true-pinv_fast)**2)))
# timer = Timer(lambda:numpy_solve_Phi(f, x, M))
# print('orig algo needs', timer.timeit(repeat)/repeat, 's')
# timer = Timer(lambda:fast_solve_Phi(f, x, M))
# print('fast algo needs', timer.timeit(repeat)/repeat, 's')
# 
# print()
# print('test of fast_solve_A_tilde')
# pinvA_true = numpy_solve_A_tilde(f_hat, x, M)
# pinvA_fast = fast_solve_A_tilde(f_hat, x, M)
# print('approx l0 error:', np.max(np.abs(pinvA_true-pinvA_fast)))
# print('approx l1 error:', np.mean(np.abs(pinvA_true-pinvA_fast)))
# print('approx l2 error:', np.sqrt(np.mean(np.abs(pinvA_true-pinvA_fast)**2)))
# timer = Timer(lambda:numpy_solve_A_tilde(f_hat, x, M))
# print('orig algo needs', timer.timeit(repeat)/repeat, 's')
# timer = Timer(lambda:fast_solve_A_tilde(f_hat, x, M))
# print('fast algo needs', timer.timeit(repeat)/repeat, 's')

print()
print('test of fast_solve_noisy_nfft')
noise = 1e-2
y = f+np.random.randn(N*K)*np.sqrt(noise)
sol_true = numpy_solve_noisy_nfft(y, x, M, noise)
sol_fast = fast_solve_noisy_nfft(y, x, M, noise)
print('approx l0 error:', np.max(np.abs(sol_true-sol_fast)))
print('approx l1 error:', np.mean(np.abs(sol_true-sol_fast)))
print('approx l2 error:', np.sqrt(np.mean(np.abs(sol_true-sol_fast)**2)))
timer = Timer(lambda:numpy_solve_noisy_nfft(y, x, M, noise))
print('orig algo needs', timer.timeit(repeat)/repeat, 's')
timer = Timer(lambda:fast_solve_noisy_nfft(y, x, M, noise))
print('fast algo needs', timer.timeit(repeat)/repeat, 's')








