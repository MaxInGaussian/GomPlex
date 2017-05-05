################################################################################
#  Github: https://github.com/MaxInGaussian/GomPlex
#  Author: Max W. Y. Lam (maxingaussian@gmail.com)
################################################################################

from sys import path
path.append("../")
import numpy as np
from GomPlex import GomPlex

fun1 = lambda x: x*np.sin(x)
fun2 = lambda x: np.sin(x)+x*np.cos(x)

x = np.linspace(-2*np.pi, 2*np.pi, 100)
y_r = fun1(x)+np.random.randn(*x.shape)*1.
y_i = fun2(x)+np.random.randn(*x.shape)*0.3
X, y = x[:, None], (y_r+y_i*1j)[:, None]

print('test of Visualizer in GomPlex')
gp = GomPlex(20)
gp.fit(X, y, opt_rate=1e-1, max_iter=500, iter_tol=30, cost_tol=1e-4, plot=True)