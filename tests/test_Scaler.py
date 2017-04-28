################################################################################
#  Github: https://github.com/MaxInGaussian/GomPlex
#  Author: Max W. Y. Lam (maxingaussian@gmail.com)
################################################################################

from sys import path
path.append("../")
import numpy as np
from GomPlex import Scaler


matrix = np.random.randn(5000, 6)*10+100

print('test of minmax scaler')
scaler = Scaler('minmax', matrix)
scaled_matrix = scaler.eval(matrix)
print('\t scaled matrix min:', np.min(scaled_matrix))
print('\t scaled matrix max:', np.max(scaled_matrix))
inv_scaled_matrix = scaler.eval(scaled_matrix, inv=True)
print('\t inv scaled matrix = matrix:', np.allclose(inv_scaled_matrix, matrix))

print('test of normal scaler')
scaler = Scaler('normal', matrix)
scaled_matrix = scaler.eval(matrix)
print('\t scaled matrix mean:', np.mean(scaled_matrix))
print('\t scaled matrix std:', np.std(scaled_matrix))
inv_scaled_matrix = scaler.eval(scaled_matrix, inv=True)
print('\t inv scaled matrix = matrix:', np.allclose(inv_scaled_matrix, matrix))