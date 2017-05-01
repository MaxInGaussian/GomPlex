################################################################################
#  Github: https://github.com/MaxInGaussian/GomPlex
#  Author: Max W. Y. Lam (maxingaussian@gmail.com)
################################################################################

from sys import path
path.append("../")
import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from GomPlex import GomPlex

fun1 = lambda x: x*np.sin(x)
fun2 = lambda x: np.sin(x)+x*np.cos(x)

x = np.linspace(-2*np.pi, 2*np.pi, 100)
y_r = fun1(x)+np.random.randn(*x.shape)*1.
y_i = fun2(x)+np.random.randn(*x.shape)*0.3
X, y = x[:, None], (y_r+y_i*1j)[:, None]

print('test of GomPlex')
gp = GomPlex(20)
gp.fit(X, y, opt_rate=1, max_iter=500, iter_tol=50, nlml_tol=1e-4)

mpl.rcParams['legend.fontsize'] = 10

fig = plt.figure()
ax = fig.gca(projection='3d')
x_plot = np.linspace(-2*np.pi, 2*np.pi, 300)
ax.plot(x_plot, fun1(x_plot), fun2(x_plot), 'b-', linewidth=2, label='true function')
ax.legend()
ax.set_xlabel('x')
ax.set_ylabel('Re{y}')
ax.set_zlabel('Im{y}')
ax.set_xlim([-2.5*np.pi, 2.5*np.pi])
ax.set_ylim([-2.5*np.pi, 2.5*np.pi])
ax.set_zlim([-2.5*np.pi, 2.5*np.pi])
fig.savefig('../plots/toy_1d_example_true_function.png')

fig = plt.figure()
ax = fig.gca(projection='3d')
x_plot = np.linspace(-2*np.pi, 2*np.pi, 300)
ax.scatter(x, y_r, y_i, marker='x', s=30, label='training data')
ax.legend()
ax.set_xlabel('x')
ax.set_ylabel('Re{y}')
ax.set_zlabel('Im{y}')
ax.set_xlim([-2.5*np.pi, 2.5*np.pi])
ax.set_ylim([-2.5*np.pi, 2.5*np.pi])
ax.set_zlim([-2.5*np.pi, 2.5*np.pi])
fig.savefig('../plots/toy_1d_example_synthesize_data.png')

fig = plt.figure()
ax = fig.gca(projection='3d')
x_plot = np.linspace(-2*np.pi, 2*np.pi, 300)
ax.scatter(x, y_r, y_i, marker='x', s=30, label='training data')
ax.legend()
mu, std = gp.predict(x_plot[:, None])
mu_r = mu.real.ravel()
mu_i = mu.imag.ravel()
ax.plot(x_plot, mu_r, mu_i, 'r-', linewidth=2, label='complex regression')
ax.legend()
ax.set_xlabel('x')
ax.set_ylabel('Re{y}')
ax.set_zlabel('Im{y}')
ax.set_xlim([-2.5*np.pi, 2.5*np.pi])
ax.set_ylim([-2.5*np.pi, 2.5*np.pi])
ax.set_zlim([-2.5*np.pi, 2.5*np.pi])
fig.savefig('../plots/toy_1d_example_regression.png')

plt.show()

