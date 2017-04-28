################################################################################
#  Github: https://github.com/MaxInGaussian/GomPlex
#  Author: Max W. Y. Lam (maxingaussian@gmail.com)
################################################################################

from sys import path
path.append("../../")
import numpy as np
import os, fnmatch
import pandas as pd
import matplotlib.pyplot as plt
from GomPlex import GomPlex, Scaler

DRAWING_RAW_DATA_PATH = 'drawing_raw_data.csv'
drawing_raw_data_df = pd.read_csv(DRAWING_RAW_DATA_PATH, index_col=0, header=0)

def get_drawing_data_as_complex_signal(subject_num):
    X, y = None, None
    for subject_id in drawing_raw_data_df.index[:subject_num]:
        d_X = np.array(list(map(float, drawing_raw_data_df['X'][subject_id].split('|'))))
        d_Y = np.array(list(map(float, drawing_raw_data_df['Y'][subject_id].split('|'))))
        d_W = np.array(list(map(float, drawing_raw_data_df['W'][subject_id].split('|'))))
        d_T = np.array(list(map(float, drawing_raw_data_df['T'][subject_id].split('|'))))
        d_X = d_X[np.where(d_W==0)]
        d_Y = d_Y[np.where(d_W==0)]
        d_T = d_T[np.where(d_W==0)]
        d_cT = np.cumsum(d_T)
        scaled_T = Scaler('minmax', d_cT).eval(d_cT)
        input = np.hstack((scaled_T[:-1, None], d_X[1:, None], d_Y[1:, None]))
        if(X is None):
            X = input
        else:
            X = np.vstack((X, input))
        complex_signal = d_X[1:]+1j*d_Y[1:]
        if(y is None):
            y = complex_signal[:, None]
        else:
            y = np.vstack((y, complex_signal[:, None]))
    return X, y
        
X, y = get_drawing_data_as_complex_signal(5)

gp = GomPlex(30)
gp.fit(X, y, opt_rate=1e-1, max_iter=500, iter_tol=30, nlml_tol=1e-4, plot=False)

fig = plt.figure()
ax = fig.gca(projection='3d')
x_plot = gp.X[:, 0]
ax.scatter(gp.X[:, 0], gp.y.real.ravel(), gp.y.imag.ravel(), marker='x', s=30, alpha=0.2, label='training data')
ax.legend()
mu, std = gp.predict(gp.X, scaled=False)
mu_r = mu.real.ravel()
mu_i = mu.imag.ravel()
ax.scatter(x_plot, mu_r, mu_i, marker='.', color='red', s=30, alpha=0.5, label='complex regression')
ax.legend()
ax.set_xlabel('x')
ax.set_ylabel('Re{y}')
ax.set_zlabel('Im{y}')

plt.show()