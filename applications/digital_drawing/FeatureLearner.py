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
from sklearn import linear_model

from GomPlex import *

class FeatureLearner(object):
    
    df_drawing_data, complex_regressor = None, None
    CENTIMETER_TO_PIXELS = 62.992126
    
    def __init__(self, moca_cutoff=20, sample_time=100, use_past=10,
        use_gender=True, use_age=True, use_edu_level=True,
        stroke_size_tol=10, stroke_length_tol=1, centimeter=True, metric='nmse',
        show_training_drawings=False, show_predicted_drawings=False):
        self.moca_cutoff = moca_cutoff
        self.centimeter = centimeter
        self.sample_time = sample_time
        self.use_gender = use_gender
        self.use_age = use_age
        self.use_edu_level = use_edu_level
        self.stroke_size_tol = stroke_size_tol
        self.stroke_length_tol = stroke_length_tol
        self.use_past = use_past
        self.forecast_step = sample_time/5
        self.metric = Metric(metric)
        self.show_training_drawings = show_training_drawings
        self.show_predicted_drawings = show_predicted_drawings
    
    def load_drawing_data(self, csv_path):
        self.df_drawing_data = pd.read_csv(csv_path, index_col=0, header=0)
        return self
    
    def get_regressor_path(self, reg_meth='GomPlex'):
        model_path = reg_meth+"_s%d_p%d_"%(self.sample_time, self.use_past)
        if(self.use_gender):
            model_path += "g"
        if(self.use_age):
            model_path += "a"
        if(self.use_edu_level):
            model_path += "e"
        return model_path+".pkl"
    
    def load_regression(self, reg_meth='GomPlex'):
        model_path = self.get_regressor_path(reg_meth)
        if(reg_meth == 'GomPlex'):
            return GomPlex().load(model_path)
    
    def eval_features_for_subjects(self, model=None, reg_meth='GomPlex'):
        if(model is None):
            model = self.load_regression(reg_meth)
        subjects = self.df_drawing_data.index
        cfs_mat = np.zeros((2, 2))
        ci_p = (self.df_drawing_data['MoCA Total'] < self.moca_cutoff).mean()
        for subject in subjects:
            ci, y, mu_ci, std_ci, mu_nonci, std_nonci = self.learn_features_for_subject(subject, model, reg_meth)
            log_p_ci = -np.sum(((mu_ci-y)/std_ci).real**2)-0.5*np.log(std_ci[0, 0].real)-\
                np.sum(((mu_ci-y)/std_ci).imag**2)-0.5*np.log(std_ci[0, 0].imag)
            log_p_nonci = -np.sum(((mu_nonci-y)/std_nonci).real**2)-0.5*np.log(std_nonci[0, 0].real)-\
                np.sum(((mu_nonci-y)/std_nonci).imag**2)-0.5*np.log(std_nonci[0, 0].imag)
            pred_ci = np.exp(log_p_ci)/(np.exp(log_p_ci)+np.exp(log_p_nonci))
            print(np.exp(log_p_ci), np.exp(log_p_nonci))
            if(pred_ci >= 0.5 and ci == 1):
                cfs_mat[0, 0] += 1
            elif(pred_ci < 0.5 and ci == 1):
                cfs_mat[0, 1] += 1
            elif(pred_ci >= 0.5 and ci == 0):
                cfs_mat[1, 0] += 1
            elif(pred_ci < 0.5 and ci == 0):
                cfs_mat[1, 1] += 1
            accuracy = (cfs_mat[0, 0]+cfs_mat[1, 1])/np.sum(cfs_mat)
            precision = 0 if np.sum(cfs_mat[:, 0]) == 0 else\
                cfs_mat[0, 0]/np.sum(cfs_mat[:, 0])
            recall = 0 if np.sum(cfs_mat[0]) == 0 else\
                cfs_mat[0, 0]/np.sum(cfs_mat[0])
            specificity = 0 if np.sum(cfs_mat[1]) == 0 else\
                cfs_mat[1, 1]/np.sum(cfs_mat[1])
            F1 = 0 if precision+recall == 0 else\
                2*(precision*recall)/(precision+recall)
            adj_accuracy = 0 if np.sum(cfs_mat[1]) == 0 or\
                np.sum(cfs_mat[0]) == 0 else\
                    cfs_mat[0, 0]/np.sum(cfs_mat[0])/2+\
                        cfs_mat[1, 1]/np.sum(cfs_mat[1])/2
            print('  (%d|%.3f), accuracy=%.3f, adj_accuracy=%.3f'%(
                ci, pred_ci, accuracy, adj_accuracy))
            print('           TP=%d, FN=%d, FP=%d, TN=%d'%(
                cfs_mat[0, 0],cfs_mat[0, 1], cfs_mat[1, 0], cfs_mat[1, 1]))
        path = 'save_models/'+self.get_regressor_path()[:-4]
        path += '_%s_'%(model.hashed_name)
        if(adj_accuracy > 0.7):
            path += '%d_%d_%d_%d.pkl'%(
                cfs_mat[0, 0],cfs_mat[0, 1], cfs_mat[1, 0], cfs_mat[1, 1])
            model.save(path)
        return adj_accuracy, accuracy, precision, recall, F1
    
    def learn_features_for_subject(self, subject, model=None, reg_meth='GomPlex'):
        if(model is None):
            model = self.load_regression(reg_meth)
        # subjects = self.df_drawing_data.index
        # subjects = list(set(subjects).difference([subject]))
        # X_train, y_train = self.get_input_output_matrix_by_subjects(subjects)
        # model.fit(X_train, y_train)
        X, y = self.get_input_output_matrix_by_subject(subject)
        X_ci, X_nonci = X.copy(), X.copy()
        X_ci[:, 0], X_nonci[:, 0] = 1, 0
        mu_ci, std_ci = model.predict(X_ci)
        mu_nonci, std_nonci = model.predict(X_nonci)
        return X[0, 0], y, mu_ci, std_ci, mu_nonci, std_nonci
    
    def show_predicted_drawing(self, X, y, y_ci, y_nonci):
        fig = plt.figure()
        ax = fig.gca()
        ax.scatter(y.real, y.imag, color='black', marker='o', s=20,
            label="true drawing path (%d)"%(int(X[0, 0])))
        ax.scatter(y_ci.real, y_ci.imag, color='red', marker='o', s=20,
            label="CI predicted path")
        ax.scatter(y_nonci.real, y_nonci.imag, color='green', marker='o', s=20,
            label="non-CI predicted path")
        ax.legend()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_xlim([-1, 15])
        ax.set_ylim([-1, 9])
    
    def train_regressor(self, reg_meth='GomPlex', iter_tol=30, ratio=0.3,
        cv_folds=3, plot_error=False):
        if(reg_meth == 'GomPlex'):
            # Default Regression Method - GomPlex
            print('# Preprocessing Raw Drawing Data')
            X_train, y_train, X_test, y_test = self.get_train_test_data(ratio)
            print('  Gathered %d Training Examples.'%(X_train.shape[0]))
            print('  Gathered %d Testing Examples.'%(X_test.shape[0]))
            print('  Done.')
            print('# Training GomPlex')
            gp = GomPlex(npr.randint(int(np.log(X_train.shape[0]))*2)+8, True)
            gp.fit(X_train, y_train, iter_tol=iter_tol,
                cv_folds=cv_folds, plot=plot_error)
            print('  Done.')
            print('# Choosing GomPlex Models')
            new_score = self.eval_features_for_subjects(model=gp)[0]
            print('  new score - %s=%.6f'%(self.metric.metric, new_score))
            model_path = self.get_regressor_path(reg_meth)
            if(not os.path.exists(model_path)):
                gp.save(model_path)
            else:
                best_gp = GomPlex().load(model_path).fit(X_train, y_train)
                ori_score = self.eval_features_for_subjects()[0]
                print('  ori score - %s=%.6f'%(self.metric.metric, ori_score))
                if(new_score > ori_score):
                    gp.save(model_path)
                    print('  Found New Model!')
            print('  Done.')
        elif(reg_meth == "MLP"):
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
            input, output = self.get_input_output_matrix_by_subject(subject)
            if(input is None and output is None):
                continue
            if(input_mat is None and output_mat is None):
                input_mat = input.copy()
                output_mat = output.copy()
                continue
            input_mat = np.append(input_mat, input, axis=0)
            output_mat = np.append(output_mat, output, axis=0)
        return input_mat, output_mat
    
    def get_input_output_matrix_by_subject(self, subject):
        decode = lambda str: np.array(list(map(float, str.split('|'))))
        d_X = decode(self.df_drawing_data['X'][subject])
        d_Y = decode(self.df_drawing_data['Y'][subject])
        d_W = decode(self.df_drawing_data['W'][subject])
        d_W[-1] = 0
        d_T = decode(self.df_drawing_data['T'][subject])/1000
        drawing = np.vstack([d_X, d_Y, d_W, d_T]).T
        if(self.centimeter):
            drawing[:, :2] /= self.CENTIMETER_TO_PIXELS
        input, output = self.get_drawing_input_output_by_XYWT(drawing)
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
    
    def get_drawing_input_output_by_XYWT(self, drawing):
        drawing, d_cL, d_V, d_DI = self.get_drawing_features_by_XYWT(drawing)
        d_X, d_Y, d_W, d_T = drawing.T.tolist()
        d_cT = np.cumsum(d_T)
        rand_bound = d_cT[-1]-np.median(d_T)*self.forecast_step
        rand_bound_max = bisect_left(d_cT, rand_bound)
        rand_bound = np.median(d_T)*self.use_past*self.forecast_step
        rand_bound_min = max(self.use_past, bisect_left(d_cT, rand_bound))
        if(rand_bound_min>=rand_bound_max):
            print(d_cT)
        rand_range = range(rand_bound_min, rand_bound_max)
        input, target = [], []
        while(len(input) < self.sample_time):
            d_ci = npr.choice(rand_range)
            x, y = d_X[d_ci], d_Y[d_ci]
            v, di = d_V[d_ci], d_DI[d_ci]
            lp, tp = d_cL[d_ci], d_cT[d_ci]
            cur_info = [lp, tp, x, y]
            V, DI = [v], [di]
            I = [d_ci]
            for d_p in range(self.use_past+1):
                d_ptp = tp-(d_p+1)*self.forecast_step*np.median(d_T)
                d_pi = max(0, bisect_left(d_cT, d_ptp)-1)
                cur_info.extend([
                    d_V[d_pi] if d_pi==I[-1] else np.mean(d_V[d_pi:I[-1]]),
                    d_DI[d_pi] if d_pi==I[-1] else np.mean(d_DI[d_pi:I[-1]])])
                I.append(d_pi)
            if(np.any(np.isnan(cur_info))):
                print(I)
            d_ftp = tp+self.forecast_step*np.median(d_T)
            d_fi = bisect_left(d_cT, d_ftp)-1
            I.append(d_fi)
            if(min(I) < 0):
                continue
            if(self.show_training_drawings):
                print(I)
            input.append(cur_info)
            target.append([d_X[d_fi]+1j*(d_Y[d_fi])])
        return np.array(input), np.array(target)
    
    def get_drawing_features_by_XYWT(self, drawing):
        d_X, d_Y, d_W, d_T = drawing.T.tolist()
        for d_i, d_t in enumerate(d_T):
            if(d_W[d_i] == 0 and d_t > np.median(d_T)*1e3):
                drawing[d_i, 3] = np.median(d_T)*1e3
        length = lambda sx, sy, ex, ey: np.sqrt((sx-ex)**2+(sy-ey)**2)
        if(self.show_training_drawings):
            self.show_drawing_data(drawing)
            plt.title('original drawing data')
        d_I, strokes = [], []
        d_L, d_V, d_DI, d_ST = [], [], [], []
        stop_points = np.sort(np.where(drawing[:, 2]==1)[0])
        for s in range(len(stop_points)+1):
            # stroke -> st to ed (inclusive)
            st = stop_points[s-1]+1 if s > 0 else 0
            ed = stop_points[s] if s < len(stop_points) else drawing.shape[0]-1
            stroke_size = ed-st+1
            if(stroke_size < self.stroke_size_tol):
                continue
            frag_I = []
            stroke_frag_lengths = []
            immediate_velocities = []
            immediate_direcions = []
            for d_i in range(st+1, ed+1):
                frag_length = length(d_X[d_i], d_Y[d_i], d_X[d_i-1], d_Y[d_i-1])
                frag_direction = d_X[d_i]-d_X[d_i-1]+1j*(d_Y[d_i]-d_Y[d_i-1])               
                frag_direction = np.angle(frag_direction, deg=True)
                frag_velocity = frag_length/d_T[d_i]
                if(frag_velocity>20):
                    continue
                frag_I.append(d_i)
                stroke_frag_lengths.append(frag_length)
                immediate_velocities.append(frag_velocity)
                immediate_direcions.append(frag_direction)
            stroke_length = np.sum(stroke_frag_lengths)
            if(len(strokes) > 0 and
                stroke_length < min(self.stroke_length_tol, min(strokes)/2)):
                continue
            immediate_direcions[0] = immediate_direcions[1]
            d_L.extend(stroke_frag_lengths)
            d_V.extend(immediate_velocities)
            d_DI.extend(immediate_direcions)
            d_ST.append(len(d_I))
            d_I.extend(frag_I)
            strokes.append(stroke_length)
        drawing = drawing[d_I]
        d_cL, d_V, d_DI = np.cumsum(d_L), np.array(d_V), np.array(d_DI)
        if(self.show_training_drawings):
            self.show_drawing_data(drawing)
            plt.title('drawing data after cleansing')
            self.show_drawing_data(drawing, d_cL, 'Length')
            self.show_drawing_data(drawing, d_V, 'Velocity')
            self.show_drawing_data(drawing, d_DI, 'Direction')
            plt.show()
        return drawing, d_cL, d_V, d_DI
        
    def show_drawing_data(self, drawing, variable=None, label="Time"):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        if(variable is None):
            variable = np.cumsum(drawing[:, 3])
        stop_points = np.sort(np.where(drawing[:, 2]==1)[0])
        for s in range(len(stop_points)+1):
            st = stop_points[s-1]+1 if s > 0 else 0
            ed = stop_points[s] if s < len(stop_points) else drawing.shape[0]-1
            if(st >= ed):
                break
            d_S = list(range(st, ed+1))
            ax.plot(drawing[d_S, 0], drawing[d_S, 1], variable[d_S], 'k-')
            ax.scatter(drawing[st, 0], drawing[st, 1], variable[st], color='b')
            ax.scatter(drawing[ed, 0], drawing[ed, 1], variable[ed], color='r')
        ax.legend()
        ax.set_zlabel(label)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_xlim([-1, 10])
        ax.set_ylim([-1, 8])
    
    
    
    
    
    