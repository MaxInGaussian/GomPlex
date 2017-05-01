################################################################################
#  Github: https://github.com/MaxInGaussian/GomPlex
#  Author: Max W. Y. Lam (maxingaussian@gmail.com)
################################################################################

from sys import path
path.append("../")
import numpy as np
from GomPlex import Metric


target = np.array([2.1, 3.3, 2.2, 3.5, 5.1])[:, None]
print('target    =', target)

mu_pred = np.array([1.3, 2.1, 2.5, 3.3, 4.5])[:, None]
print('pred mu   =', mu_pred)

std_pred = np.array([1.1, 0.5, 0.5, 0.4, 0.5])[:, None]
print('pred std  =', std_pred)

print('residuals =', np.abs(target-mu_pred))

print('test of mse metric')
mse = Metric(None, 'mse').eval(target, mu_pred, std_pred)
print('\t mse =', mse)

print('test of nmse metric')
nmse = Metric(None, 'nmse').eval(target, mu_pred, std_pred)
print('\t nmse =', nmse)

print('test of nlpd metric')
nlpd = Metric(None, 'nlpd').eval(target, mu_pred, std_pred)
print('\t nlpd =', nlpd)

