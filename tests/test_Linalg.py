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

time_reps = 1
N, M, K, D = 1000, 500, 10, 30
X = .5*np.random.rand(N, D)
spectral_freqs = np.random.randn(D, K)
X_nfft = X.dot(spectral_freqs)
x = X_nfft.ravel()
f = np.repeat(np.sin(-20*np.pi*np.mean(X_nfft, 1)), K)+0j
f_hat = np.random.randn(M)+0j
noise = 1e-2
y = f+np.random.randn(N*K)*np.sqrt(noise)

print()
print('test of nfft')
f_true = ndft(x, f_hat, M)
f_fast = nfft(x, f_hat, M)
print('approx l0 error:', np.max(np.abs(f_true-f_fast)))
print('approx l1 error:', np.mean(np.abs(f_true-f_fast)))
print('approx l2 error:', np.sqrt(np.mean(np.abs(f_true-f_fast)**2)))
timer = Timer(lambda:ndft(x, f_hat, M))
print('numpy needs   ', timer.timeit(time_reps)/time_reps, 's')
timer = Timer(lambda:nfft(x, f_hat, M))
print('our algo needs', timer.timeit(time_reps)/time_reps, 's')

print()
print('test of adj_nfft')
f_hat_true = adj_ndft(x, f, M)
f_hat_fast = adj_nfft(x, f, M)
print('approx l0 error:', np.max(np.abs(f_hat_true-f_hat_fast)))
print('approx l1 error:', np.mean(np.abs(f_hat_true-f_hat_fast)))
print('approx l2 error:', np.sqrt(np.mean(np.abs(f_hat_true-f_hat_fast)**2)))
timer = Timer(lambda:adj_ndft(x, f, M))
print('numpy needs   ', timer.timeit(time_reps)/time_reps, 's')
timer = Timer(lambda:adj_nfft(x, f, M))
print('our algo needs', timer.timeit(time_reps)/time_reps, 's')

print()
print('test of solve_Phi_algo_1')
pinv_true = numpy_solve_Phi(f, x, M)
pinv_fast = solve_Phi_algo_1(f, x, M)
print('approx l0 error:', np.max(np.abs(pinv_true-pinv_fast)))
print('approx l1 error:', np.mean(np.abs(pinv_true-pinv_fast)))
print('approx l2 error:', np.sqrt(np.mean(np.abs(pinv_true-pinv_fast)**2)))
timer = Timer(lambda:numpy_solve_Phi(f, x, M))
print('numpy needs   ', timer.timeit(time_reps)/time_reps, 's')
timer = Timer(lambda:solve_Phi_algo_1(f, x, M))
print('our algo needs', timer.timeit(time_reps)/time_reps, 's')

print()
print('test of solve_Phi_algo_2')
pinv_true = numpy_solve_Phi(f, x, M)
pinv_fast = solve_Phi_algo_2(f, x, M)
print('approx l0 error:', np.max(np.abs(pinv_true-pinv_fast)))
print('approx l1 error:', np.mean(np.abs(pinv_true-pinv_fast)))
print('approx l2 error:', np.sqrt(np.mean(np.abs(pinv_true-pinv_fast)**2)))
timer = Timer(lambda:numpy_solve_Phi(f, x, M))
print('numpy needs   ', timer.timeit(time_reps)/time_reps, 's')
timer = Timer(lambda:solve_Phi_algo_2(f, x, M))
print('our algo needs', timer.timeit(time_reps)/time_reps, 's')

print()
print('test of solve_A_tilde_algo_1')
pinvA_true = numpy_solve_A_tilde(f_hat, x, M)
pinvA_fast = solve_A_tilde_algo_1(f_hat, x, M)
print('approx l0 error:', np.max(np.abs(pinvA_true-pinvA_fast)))
print('approx l1 error:', np.mean(np.abs(pinvA_true-pinvA_fast)))
print('approx l2 error:', np.sqrt(np.mean(np.abs(pinvA_true-pinvA_fast)**2)))
timer = Timer(lambda:numpy_solve_A_tilde(f_hat, x, M))
print('numpy needs   ', timer.timeit(time_reps)/time_reps, 's')
timer = Timer(lambda:solve_A_tilde_algo_1(f_hat, x, M))
print('our algo needs', timer.timeit(time_reps)/time_reps, 's')

print()
print('test of solve_A_tilde_algo_2')
pinvA_true = numpy_solve_A_tilde(f_hat, x, M)
pinvA_fast = solve_A_tilde_algo_2(f_hat, x, M)
print('approx l0 error:', np.max(np.abs(pinvA_true-pinvA_fast)))
print('approx l1 error:', np.mean(np.abs(pinvA_true-pinvA_fast)))
print('approx l2 error:', np.sqrt(np.mean(np.abs(pinvA_true-pinvA_fast)**2)))
timer = Timer(lambda:numpy_solve_A_tilde(f_hat, x, M))
print('numpy needs   ', timer.timeit(time_reps)/time_reps, 's')
timer = Timer(lambda:solve_A_tilde_algo_2(f_hat, x, M))
print('our algo needs', timer.timeit(time_reps)/time_reps, 's')

print()
print('test of solve_A_algo_1')
sol_true = numpy_solve_A(y, x, M, noise)
sol_fast = solve_A_algo_1(y, x, M, noise)
print('approx l0 error:', np.max(np.abs(sol_true-sol_fast)))
print('approx l1 error:', np.mean(np.abs(sol_true-sol_fast)))
print('approx l2 error:', np.sqrt(np.mean(np.abs(sol_true-sol_fast)**2)))
timer = Timer(lambda:numpy_solve_A(y, x, M, noise))
print('numpy needs   ', timer.timeit(time_reps)/time_reps, 's')
timer = Timer(lambda:solve_A_algo_1(y, x, M, noise))
print('our algo needs', timer.timeit(time_reps)/time_reps, 's')

print()
print('test of solve_A_algo_2')
sol_true = numpy_solve_A(y, x, M, noise)
sol_fast = solve_A_algo_2(y, x, M, noise)
print('approx l0 error:', np.max(np.abs(sol_true-sol_fast)))
print('approx l1 error:', np.mean(np.abs(sol_true-sol_fast)))
print('approx l2 error:', np.sqrt(np.mean(np.abs(sol_true-sol_fast)**2)))
timer = Timer(lambda:numpy_solve_A(y, x, M, noise))
print('numpy needs   ', timer.timeit(time_reps)/time_reps, 's')
timer = Timer(lambda:solve_A_algo_2(y, x, M, noise))
print('our algo needs', timer.timeit(time_reps)/time_reps, 's')








