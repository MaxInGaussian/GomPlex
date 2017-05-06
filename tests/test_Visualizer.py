################################################################################
#  Github: https://github.com/MaxInGaussian/GomPlex
#  Author: Max W. Y. Lam (maxingaussian@gmail.com)
################################################################################

from sys import path
path.append("../")
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from GomPlex import GomPlex

fun1 = lambda x: x*np.sin(x)
fun2 = lambda x: np.sin(x)+x*np.cos(x)

x = np.linspace(-2*np.pi, 2*np.pi, 100)
y_r = fun1(x)+np.random.randn(*x.shape)*1.
y_i = fun2(x)+np.random.randn(*x.shape)*0.3
X, y = x[:, None], (y_r+y_i*1j)[:, None]

print('test of Visualizer in GomPlex')
gp = GomPlex(30, mean_only=True)
gp.fit(X, y, plot=True)