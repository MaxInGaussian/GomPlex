################################################################################
#  Github: https://github.com/MaxInGaussian/GomPlex
#  Author: Max W. Y. Lam (maxingaussian@gmail.com)
################################################################################

from sys import path
path.append("../../")
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import numpy.random as npr
import os, fnmatch
import pandas as pd
import matplotlib.pyplot as plt
from bisect import bisect_left

from GomPlex import *

class FeatureLearner(object):
    
    df_drawing_data = None
    CENTIMETER_TO_PIXELS = 62.992126
    
    def __init__(self, in_centimeter=True, sampling_points=50, forecast_step=1,
        use_gender=False, use_age=True, use_edu_level=True, metric='nmse',
        stroke_size_tol=10, stroke_length_tol=1, plot_samples=False):
        self.in_centimeter = in_centimeter
        self.sampling_points = sampling_points
        self.forecast_step = forecast_step
        self.use_gender = use_gender
        self.use_age = use_age
        self.use_edu_level = use_edu_level
        self.stroke_size_tol = stroke_size_tol
        self.stroke_length_tol = stroke_length_tol
        self.metric = Metric(metric)
        self.plot_samples = plot_samples
        self.model_path = "feature_learner_"
        if(self.in_centimeter):
            self.model_path += "cm_"+"s%d_f%d_"%(sampling_points, forecast_step)
        else:
            self.model_path += "px_"+"s%d_f%d_"%(sampling_points, forecast_step)
        if(self.use_gender):
            self.model_path += "g"
        if(self.use_age):
            self.model_path += "a"
        if(self.use_edu_level):
            self.model_path += "e"
        self.model_path += ".pkl"
    
    def load_drawing_data(self, csv_path):
        self.df_drawing_data = pd.read_csv(csv_path, index_col=0, header=0)
        return self
    
    def train_inner_regressor(self, regression_method=None, ratio=0.3,
        cv_folds=3, plot_error=True):
        if(regression_method == "MLP"):
            from keras.models import Sequential
            from keras.layers import Dense
            from keras.wrappers.scikit_learn import KerasRegressor
            from sklearn.model_selection import cross_val_score
            from sklearn.model_selection import KFold
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import Pipeline
        else:
            # Default Regression Method - GomPlex
            print('# Preprocessing Raw Drawing Data')
            X_train, y_train, X_test, y_test = self.get_train_test_data(ratio)
            print('  Gathered %d Training Examples.'%(X_train.shape[0]))
            print('  Gathered %d Testing Examples.'%(X_test.shape[0]))
            print('  Done.')
            print('# Training GomPlex')
            gp = GomPlex(npr.randint(int(np.log(X_train.shape[0]))*2)+8, True)
            gp.fit(X_train, y_train, cv_folds=cv_folds, plot=plot_error)
            print('  Done.')
            print('# Choosing GomPlex Models')
            gp_score = metric.eval(y_test, *gp.predict(X_test))
            print(' new model     - %s %.6f'%(metric.metric, gp_score))
            if(not os.path.exists(self.model_path)):
                gp.save(self.model_path)
            else:
                best_gp = GomPlex().load(self.model_path).fit(X_train, y_train)
                best_gp_score = metric.eval(y_test, *best_gp.predict(X_test))
                print(' original best - %s %.6f'%(metric.metric, best_gp_score))
                if(gp_score < best_gp_score):
                    gp.save(self.model_path)
                    print('  Found Better Model!')
            print('  Done.')
    
    def get_train_test_data(self, ratio):
        X_train, y_train, X_test, y_test = None, None, None, None
        subjects_train, subjects_test = self.get_subjects_for_train_test(ratio)
        for subject_id in subjects_train:
            X_train_i, y_train_i = self.get_input_output_by_subject(subject_id)
            if(X_train_i is None and y_train_i is None):
                continue
            if(X_train is None and y_train is None):
                X_train = X_train_i.copy()
                y_train = y_train_i.copy()
                continue
            X_train = np.append(X_train, X_train_i, axis=0)
            y_train = np.append(y_train, y_train_i, axis=0)
        for subject_id in subjects_test:
            X_test_i, y_test_i = self.get_input_output_by_subject(subject_id)
            if(X_test_i is None and y_test_i is None):
                continue
            if(X_test is None and y_test is None):
                X_test = X_test_i.copy()
                y_test = y_test_i.copy()
                continue
            X_test = np.append(X_test, X_test_i, axis=0)
            y_test = np.append(y_test, y_test_i, axis=0)
        return X_train, y_train, X_test, y_test
    
    def get_subjects_for_train_test(self, ratio):
        ci_data = self.df_drawing_data['Cognitive Impairment']
        ci_subjects = self.df_drawing_data[ci_data==1].index
        num_ci = len(ci_subjects)
        test_ci = int(num_ci*ratio)
        rand_ci = npr.choice(range(num_ci), test_ci, replace=False)
        rand_ci = ci_subjects[rand_ci]
        rand_ci_train = list(set(ci_subjects).difference(rand_ci))
        nonci_subjects = self.df_drawing_data[ci_data==0].index
        num_nonci = len(nonci_subjects)
        test_nonci = int(num_nonci*ratio)
        rand_nonci = npr.choice(range(num_nonci), test_nonci, replace=False)
        rand_nonci = nonci_subjects[rand_nonci]
        rand_nonci_train = list(set(nonci_subjects).difference(rand_nonci))
        subjects_train = rand_ci_train+rand_nonci_train
        subjects_test = rand_ci.tolist()+rand_nonci.tolist()
        return subjects_train, subjects_test
    
    def get_input_output_by_subject(self, subject_id):
        decode = lambda str: np.array(list(map(float, str.split('|'))))
        d_X = decode(self.df_drawing_data['X'][subject_id])
        d_Y = decode(self.df_drawing_data['Y'][subject_id])
        d_W = decode(self.df_drawing_data['W'][subject_id])
        d_T = decode(self.df_drawing_data['T'][subject_id])
        if(len(d_X) < self.sampling_points):
            return None, None
        if(self.in_centimeter):
            d_X /= self.CENTIMETER_TO_PIXELS
            d_Y /= self.CENTIMETER_TO_PIXELS
        input, output = self.get_drawing_input_output_by_XYWT(d_X, d_Y, d_W, d_T)
        ci = np.array(self.df_drawing_data['Cognitive Impairment'][subject_id])
        input = np.hstack((ci*np.ones((input.shape[0], 1)), input))
        if(self.use_age):
            age = np.array(self.df_drawing_data['Age'][subject_id])
            input = np.hstack((input, age*np.ones((input.shape[0], 1))))
        if(self.use_gender):
            gender = np.array(self.df_drawing_data['Male'][subject_id])
            input = np.hstack((input, gender*np.ones((input.shape[0], 1))))
        if(self.use_edu_level):
            edu_levels = ['Uneducated', 'Primary', 'Secondary', 'University']
            edu_level = np.array(self.df_drawing_data[edu_levels].loc[subject_id])
            input = np.hstack((input, edu_level*np.ones((input.shape[0], 1))))
        return input, output
        
    def show_drawing_data(self, d_X, d_Y, variable, d_SI=None, label="Time"):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        if(d_SI is None):
            ax.scatter(variable, d_X, d_Y, marker='o', s=30)
        else:
            for d_I in d_SI:
                ax.plot(variable[d_I], d_X[d_I], d_Y[d_I], 'r-')
        ax.legend()
        ax.set_xlabel(label)
        ax.set_ylabel('X')
        ax.set_zlabel('Y')
    
    def get_drawing_features_by_XYWT(self, d_X, d_Y, d_W, d_T):
        length = lambda sx, sy, ex, ey: np.sqrt((sx-ex)**2+(sy-ey)**2)
        coordinates, strokes, coordinate_to_stroke = [], [], {}
        d_L, d_V, d_DI = [], [], []
        stop_points = np.where(d_W==1)[0]
        if(self.plot_samples):
            self.show_drawing_data(d_X, d_Y, np.cumsum(d_T))
            plt.title('raw drawing data')
        for s in range(len(stop_points)+1):
            st = stop_points[s-1] if s > 0 else 0
            ed = stop_points[s] if s < len(stop_points) else len(d_X)-1
            if(ed-st < self.stroke_size_tol):
                continue
            frag_lengths = []
            for i in range(st+1, ed):
                frag_lengths.append(length(d_X[i], d_Y[i], d_X[i+1], d_Y[i+1]))
            stroke_length = np.sum(frag_lengths)
            if(len(strokes) > 0 and
                stroke_length < min(self.stroke_length_tol, 0.5*strokes[-1])):
                continue
            d_L.extend(frag_lengths)
            for i in range(ed-st):
                coordinate_to_stroke[len(coordinates)+i] = len(strokes)
            coordinates.extend(list(range(st+1, ed)))
            strokes.append(stroke_length)
        d_X, d_Y, d_T = d_X[coordinates], d_Y[coordinates], d_T[coordinates]
        d_cT = np.cumsum(d_T)
        if(self.plot_samples):
            self.show_drawing_data(d_X, d_Y, d_cT)
            plt.title('drawing data after cleansing')
        d_cL = np.cumsum(d_L)
        if(len(d_cL) == 0):
            self.show_drawing_data(d_X, d_Y, d_cT)
            plt.show()
        sampled_SI, d_SI = [], [[]for _ in range(len(strokes))]
        for s in range(self.sampling_points-1):
            d_cl = (s+1)*d_cL[-1]/self.sampling_points
            sampled_i = bisect_left(d_cL, d_cl)
            if(sampled_i not in sampled_SI):
                stroke_order = coordinate_to_stroke[sampled_i]
                d_SI[stroke_order].append(len(sampled_SI))
                sampled_SI.append(sampled_i)
                if(len(sampled_SI) > 1):
                    last_i = sampled_SI[-2]
                    length_diff = d_cL[sampled_i]-d_cL[last_i]
                    time_diff = d_cT[sampled_i]-d_cT[last_i]
                    d_V.append(length_diff/time_diff)
                    diff_vector = 1j*(d_Y[sampled_i]-d_Y[last_i])               
                    diff_vector += d_X[sampled_i]-d_X[last_i]
                    di = np.angle(diff_vector, deg=True)
                    d_DI.append(di)
                else:
                    d_V.append(d_cL[sampled_i]/d_cT[sampled_i])
                    diff_vector = 1j*(d_Y[sampled_i]-d_Y[0])               
                    diff_vector += d_X[sampled_i]-d_X[0]
                    di = np.angle(diff_vector, deg=True)
                    d_DI.append(di)
        d_X, d_Y, d_cT = d_X[sampled_SI], d_Y[sampled_SI], d_cT[sampled_SI]
        d_V, d_DI = np.array(d_V), np.array(d_DI)
        if(self.plot_samples):
            self.show_drawing_data(d_X, d_Y, d_cT)
            plt.title('drawing data after discretizing')
            self.show_drawing_data(d_V, d_DI, d_cT)
            plt.title('drawing data velocity polar coordinate')
            plt.show()
        return d_X, d_Y, d_V, d_DI, d_SI
    
    def get_drawing_input_output_by_XYWT(self, d_X, d_Y, d_W, d_T):
        d_X, d_Y, d_V, d_DI, d_SI =\
            self.get_drawing_features_by_XYWT(d_X, d_Y, d_W, d_T)
        forecast_input, forecast_target = [], []
        for d_I in d_SI:
            if(len(d_I) <= self.forecast_step):
                continue
            for i in range(len(d_I)-self.forecast_step-1):
                input_coord = d_I[i+1]
                v, di = d_V[input_coord], d_DI[input_coord]
                v_diff = d_V[input_coord]-d_V[d_I[i]]
                di_diff = d_DI[input_coord]-d_DI[d_I[i]]
                forecast_coord = d_I[i+self.forecast_step+1]
                forecast_input.append([v, di, v_diff, di_diff])
                x_diff = d_X[forecast_coord]-d_X[input_coord]
                y_diff = d_Y[forecast_coord]-d_Y[input_coord]
                forecast_target.append([x_diff+1j*y_diff])
        return np.array(forecast_input), np.array(forecast_target)
    
    
    
    
    
    