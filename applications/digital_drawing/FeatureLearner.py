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
    
    def __init__(self, moca_cutoff=20, in_centimeter=True, seperate_length=0.2,
        use_gender=True, use_age=True, use_edu_level=True, metric='nmse',
        stroke_size_tol=10, stroke_length_tol=1, plot_samples=False):
        self.moca_cutoff = moca_cutoff
        self.in_centimeter = in_centimeter
        self.seperate_length = seperate_length
        self.use_gender = use_gender
        self.use_age = use_age
        self.use_edu_level = use_edu_level
        self.stroke_size_tol = stroke_size_tol
        self.stroke_length_tol = stroke_length_tol
        self.metric = Metric(metric)
        self.plot_samples = plot_samples
    
    def load_drawing_data(self, csv_path):
        self.df_drawing_data = pd.read_csv(csv_path, index_col=0, header=0)
        return self
    
    def get_regressor_path(self, regress_meth='GomPlex'):
        model_path = regress_meth+"_"
        if(self.in_centimeter):
            model_path += "cm_"+"s%.1f_"%(self.seperate_length)
        else:
            model_path += "px_"+"s%.1f_"%(self.seperate_length)
        if(self.use_gender):
            model_path += "g"
        if(self.use_age):
            model_path += "a"
        if(self.use_edu_level):
            model_path += "e"
        return model_path+".pkl"
    
    def load_regression(regress_meth='GomPlex'):
        model_path = self.get_regressor_path(regress_meth)
        if(regress_meth == 'GomPlex'):
            complex_regressor = GomPlex().load(model_path)
        return complex_regressor
    
    def learn_features_for_subjects(self, regress_meth='GomPlex'):
        subjects = self.df_drawing_data.index
        X_feat = {}
        for subject in subjects:
            X_feat[subject] =\
                self.learn_features_for_one_subject(subject, regress_meth)
        return X_feat
    
    def learn_features_for_one_subject(self, subject, regress_meth='GomPlex'):
        complex_regressor = self.load_regression(regress_meth)
        subjects = self.df_drawing_data.index
        subjects = list(set(subjects).difference([subject]))
        X_train, y_train = self.get_input_output_matrix_by_subjects(subjects)
        complex_regressor.fit(X_train, y_train)
        X_subject = self.get_input_output_matrix_by_one_subject(subject)[0]
        if(X_subject is None):
            return None
        X_subject_ci, X_subject_nonci = X_subject.copy(), X_subject.copy()
        y_predict_ci = complex_regressor.predict(X_subject_ci)[0]
        y_predict_nonci = complex_regressor.predict(X_subject_nonci)[0]
        return y_predict_ci.ravel()-y_predict_nonci.ravel()
    
    def train_regressor(self, regress_meth='GomPlex', ratio=0.3,
        cv_folds=3, plot_error=True):
        if(regress_meth == 'GomPlex'):
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
            new_score = self.metric.eval(y_test, *gp.predict(X_test))
            print('  new score - %s=%.6f'%(self.metric.metric, new_score))
            model_path = self.get_regressor_path(regress_meth)
            if(not os.path.exists(model_path)):
                gp.save(model_path)
            else:
                best_gp = GomPlex().load(model_path).fit(X_train, y_train)
                ori_score = self.metric.eval(y_test, *best_gp.predict(X_test))
                print('  ori score - %s=%.6f'%(self.metric.metric, ori_score))
                if(new_score < ori_score):
                    gp.save(model_path)
                    print('  Found Better Model!')
            print('  Done.')
        elif(regress_meth == "MLP"):
            from keras.models import Sequential
            from keras.layers import Dense
            from keras.wrappers.scikit_learn import KerasRegressor
            from sklearn.model_selection import cross_val_score
            from sklearn.model_selection import KFold
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import Pipeline
    
    def get_train_test_data(self, ratio):
        subjects_train, subjects_test = self.get_subjects_for_train_test(ratio)
        X_train, y_train = self.get_input_output_matrix_by_subjects(subjects_train)
        X_test, y_test = self.get_input_output_matrix_by_subjects(subjects_test)
        return X_train, y_train, X_test, y_test
    
    def get_subjects_for_train_test(self, ratio):
        moca_data = self.df_drawing_data['MoCA Total']
        ci_subjects = self.df_drawing_data[moca_data<self.moca_cutoff].index
        num_ci = len(ci_subjects)
        test_ci = int(num_ci*ratio)
        rand_ci = npr.choice(range(num_ci), test_ci, replace=False)
        rand_ci = ci_subjects[rand_ci]
        rand_ci_train = list(set(ci_subjects).difference(rand_ci))
        nonci_subjects = self.df_drawing_data[moca_data>=self.moca_cutoff].index
        num_nonci = len(nonci_subjects)
        test_nonci = int(num_nonci*ratio)
        rand_nonci = npr.choice(range(num_nonci), test_nonci, replace=False)
        rand_nonci = nonci_subjects[rand_nonci]
        rand_nonci_train = list(set(nonci_subjects).difference(rand_nonci))
        subjects_train = rand_ci_train+rand_nonci_train
        subjects_test = rand_ci.tolist()+rand_nonci.tolist()
        return subjects_train, subjects_test
    
    def get_input_output_matrix_by_subjects(self, subjects):
        input_mat, output_mat = None, None
        for subject in subjects:
            input, output = self.get_input_output_matrix_by_one_subject(subject)
            if(input is None and output is None):
                continue
            if(input_mat is None and output_mat is None):
                input_mat = input.copy()
                output_mat = output.copy()
                continue
            input_mat = np.append(input_mat, input, axis=0)
            output_mat = np.append(output_mat, output, axis=0)
        return input_mat, output_mat
    
    def get_input_output_matrix_by_one_subject(self, subject):
        decode = lambda str: np.array(list(map(float, str.split('|'))))
        d_X = decode(self.df_drawing_data['X'][subject])
        d_Y = decode(self.df_drawing_data['Y'][subject])
        d_W = decode(self.df_drawing_data['W'][subject])
        d_T = decode(self.df_drawing_data['T'][subject])/1000
        if(self.in_centimeter):
            d_X /= self.CENTIMETER_TO_PIXELS
            d_Y /= self.CENTIMETER_TO_PIXELS
        input, output = self.get_drawing_input_output_by_XYWT(d_X, d_Y, d_W, d_T)
        moca_ci = self.df_drawing_data['MoCA Total'] < self.moca_cutoff
        ci = moca_ci.loc[subject]
        input = np.hstack((ci*np.ones((input.shape[0], 1)), input))
        if(self.use_age):
            age = np.array(self.df_drawing_data['Age'][subject])
            input = np.hstack((input, age*np.ones((input.shape[0], 1))))
        if(self.use_gender):
            gender = np.array(self.df_drawing_data['Male'][subject])
            input = np.hstack((input, gender*np.ones((input.shape[0], 1))))
        if(self.use_edu_level):
            edu_levels = ['Uneducated', 'Primary', 'Secondary', 'University']
            edu_level = np.array(self.df_drawing_data[edu_levels].loc[subject])
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
        sample_points = int(d_cL[-1]/self.seperate_length)
        sampled_SI, d_SI = [], [[]for _ in range(len(strokes))]
        for s in range(sample_points-1):
            d_cl = (s+1)*d_cL[-1]/sample_points
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
                    d_DI.append(np.angle(diff_vector, deg=True))
                else:
                    d_V.append(d_cL[sampled_i]/d_cT[sampled_i])
                    diff_vector = 1j*(d_Y[sampled_i]-d_Y[0])               
                    diff_vector += d_X[sampled_i]-d_X[0]
                    d_DI.append(np.angle(diff_vector, deg=True))
        d_X, d_Y, d_cT = d_X[sampled_SI], d_Y[sampled_SI], d_cT[sampled_SI]
        d_V, d_DI = np.array(d_V), np.array(d_DI)
        if(self.plot_samples):
            self.show_drawing_data(d_X, d_Y, d_cT)
            plt.title('drawing data after discretizing')
            self.show_drawing_data(d_V, d_DI, d_cT)
            plt.title('drawing data velocity polar coordinate')
            plt.show()
        return d_X, d_Y, d_cT, d_V, d_DI, d_SI
    
    def get_drawing_input_output_by_XYWT(self, d_X, d_Y, d_W, d_T):
        d_X, d_Y, d_cT, d_V, d_DI, d_SI =\
            self.get_drawing_features_by_XYWT(d_X, d_Y, d_W, d_T)
        forecast_input, forecast_target = [], []
        for d_I in d_SI:
            for i in range(len(d_I)-1):
                input_coord = d_I[i]
                v_k, di_k = d_V[input_coord], d_DI[input_coord]
                x_k, y_k = d_X[input_coord], d_Y[input_coord]
                forecast_input.append([x_k, y_k, v_k, di_k])
                forecast_coord = d_I[i+1]
                x_f, y_f = d_X[forecast_coord], d_Y[forecast_coord]
                forecast_target.append([x_f+1j*y_f])
        return np.array(forecast_input), np.array(forecast_target)
    
    
    
    
    
    