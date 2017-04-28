################################################################################
#  Github: https://github.com/MaxInGaussian/GomPlex
#  Author: Max W. Y. Lam (maxingaussian@gmail.com)
################################################################################

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Visualizer(object):
    
    def __init__(self, gp, metric='nmse', plot_limit=80):
        self.gp = gp
        self.metric = metric
        self.plot_limit = plot_limit
    
    def plot_training(self):
        self.fig = plt.figure(1)
        if(self.gp.D == 1):
            plt.axis('off')
            return self.plot_training_1d()
        return self.plot_training_general()
    
    def plot_training_1d(self):
        ax1 = self.fig.add_subplot(211)
        ax2 = self.fig.add_subplot(212)
        def animate(trainer):
            pts = 200
            X_plot = np.linspace(-0.5, 0.5, pts)
            mu, std = self.gp.predict(X_plot[:, None], scaled=False)
            ax1.cla()
            errors = [0.25, 0.39, 0.52, 0.67, 0.84, 1.04, 1.28, 1.64, 2.2]
            for er in errors:
                ax1.fill_between(X_plot,
                    (mu.real-er*std.real).ravel(),
                    (mu.real+er*std.real).ravel(),
                    alpha=((2.9-er)/6)**1.9, facecolor='orange',
                    linewidth=1e-3)
            ax1.plot(X_plot, mu.real.ravel(), 'r-',
                linewidth=2, label=self.gp.__str__())
            ax1.scatter(self.gp.X[:, 0], self.gp.y.real.ravel(),
                marker='x', s=30, label='training data')
            ax1.set_xlim([-.5, .5])
            ax1.set_ylim([-3.5, 3.5])
            ax1.set_ylabel('Re{y}')
            ax1.set_title('Real-time Plot of Complex Regression (Real Part)')
            ax2.cla()
            for er in errors:
                ax2.fill_between(X_plot,
                    (mu.imag-er*std.imag).ravel(),
                    (mu.imag+er*std.imag).ravel(),
                    alpha=((2.9-er)/6)**1.9, facecolor='orange',
                    linewidth=1e-3)
            ax2.plot(X_plot, mu.imag.ravel(), 'r-',
                linewidth=2, label=self.gp.__str__())
            ax2.scatter(self.gp.X[:, 0], self.gp.y.imag.ravel(),
                marker='x', s=30, label='training data')
            ax2.set_xlim([-.5, .5])
            ax2.set_ylim([-3.5, 3.5])
            ax2.set_ylabel('Im{y}')
            ax2.set_title('Real-time Plot of Complex Regression (Imaginary Part)')
            plt.xlabel('x', fontsize=13)
            self.fig.subplots_adjust(hspace=0.3)
            plt.pause(0.01)
        return animate

    def plot_training_general(self):
        self.fig.suptitle(self.gp.__str__(), fontsize=15)
        ax1 = self.fig.add_subplot(211)
        ax2 = self.fig.add_subplot(212)
        plt.xlabel('TIME(s)', fontsize=13)
        def animate(trainer):
            if(trainer.iter == 0):
                data_x1, data_y1, data_x2, data_y2 = [], [], [], []
            else:
                data_x1 = ax1.lines[0].get_xdata().tolist()
                data_y1 = ax1.lines[0].get_ydata().tolist()
                data_x2 = ax2.lines[0].get_xdata().tolist()
                data_y2 = ax2.lines[0].get_ydata().tolist()
            data_x1.append(self.gp.evals['time'][1][-1])
            obj = self.gp.evals['obj'][1][self.gp.evals_ind]
            data_y1.append(obj)
            ax1.cla()
            ax1.plot(data_x1[-self.plot_limit:], data_y1[-self.plot_limit:],
                color='r', linewidth=2.0, label='MIN OBJ')
            handles, labels = ax1.get_legend_handles_labels()
            ax1.legend(handles, labels, loc='upper center',
                bbox_to_anchor=(0.5, 1.05), ncol=1, fancybox=True)   
            data_x2.append(self.gp.evals['time'][1][-1])
            val = self.gp.evals[self.eval][1][self.gp.evals_ind]
            data_y2.append(val)          
            ax2.cla()
            ax2.plot(data_x2[-self.plot_limit:], data_y2[-self.plot_limit:],
                color='b', linewidth=2.0, label=self.eval.upper())
            handles, labels = ax2.get_legend_handles_labels()
            ax2.legend(handles, labels, loc='upper center',
                bbox_to_anchor=(0.5, 1.05), ncol=1, fancybox=True)
            plt.pause(0.01)
        return animate