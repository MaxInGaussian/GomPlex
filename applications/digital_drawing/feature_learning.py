################################################################################
#  Github: https://github.com/MaxInGaussian/GomPlex
#  Author: Max W. Y. Lam (maxingaussian@gmail.com)
################################################################################

from sys import path
path.append("../../")
import numpy as np
import numpy.random as npr
import os, fnmatch
import pandas as pd
import matplotlib.pyplot as plt
from bisect import bisect_left

from GomPlex import *

MODEL_PATH = 'feature_learner.pkl'
DRAWING_RAW_DATA_PATH = 'drawing_raw_data.csv'
drawing_raw_data_df = pd.read_csv(DRAWING_RAW_DATA_PATH, index_col=0, header=0)
PIXEL_CENTIMETER = 62.992126

in_centimeter = True
metric = Metric('nmse')
forecast_step = 1
sampling_points = 50
stroke_size_tol, stroke_length_tol = 10, 1
time_faction, gender, age, edu_level = True, True, True, True

length = lambda sx, sy, ex, ey: np.sqrt((sx-ex)**2+(sy-ey)**2)
    
def show_drawing_data(d_X, d_Y, spectrum, d_sI=None, label="Time"):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    if(d_sI is None):
        ax.scatter(spectrum, d_X, d_Y, marker='o', s=30)
    else:
        for d_s in d_sI:
            ax.plot(spectrum[d_s], d_X[d_s], d_Y[d_s], 'r-')
    ax.legend()
    ax.set_xlabel(label)
    ax.set_ylabel('X')
    ax.set_zlabel('Y')
    ax.set_ylim([-1.6, 900/PIXEL_CENTIMETER] if in_centimeter else [-100, 900])
    ax.set_zlim([-0.8, 450/PIXEL_CENTIMETER] if in_centimeter else [-50, 450])

def preprocessing(d_X, d_Y, d_W, d_T, plot=False):
    coordinates, strokes, coordinate_to_stroke = [], [], {}
    d_L, d_V = [], []
    stop_points = np.where(d_W==1)[0]
    if(plot):
        show_drawing_data(d_X, d_Y, np.cumsum(d_T))
        plt.title('raw drawing data')
    for s in range(len(stop_points)+1):
        st = stop_points[s-1] if s > 0 else 0
        ed = stop_points[s] if s < len(stop_points) else len(d_X)-1
        if(ed-st < stroke_size_tol):
            continue
        stroke_lengths = []
        for i in range(st+1, ed):
            stroke_lengths.append(length(d_X[i], d_Y[i], d_X[i+1], d_Y[i+1]))
        if(np.sum(stroke_lengths) < stroke_length_tol):
            continue
        d_L.extend(stroke_lengths)
        for i in range(ed-st):
            coordinate_to_stroke[len(coordinates)+i] = len(strokes)
        coordinates.extend(list(range(st+1, ed)))
        strokes.append(np.sum(stroke_lengths))
    d_X = d_X[coordinates]
    d_Y = d_Y[coordinates]
    d_T = d_T[coordinates]
    d_cT = np.cumsum(d_T)
    if(plot):
        show_drawing_data(d_X, d_Y, d_cT)
        plt.title('drawing data after cleansing')
    d_cL = np.cumsum(d_L)
    if(len(d_cL) == 0):
        show_drawing_data(d_X, d_Y, d_cT)
        plt.show()
    sampled_coordinates, d_sI = [], [[]for _ in range(len(strokes))]
    for s in range(sampling_points-1):
        d_cl = (s+1)*d_cL[-1]/sampling_points
        sampled_coordinate = bisect_left(d_cL, d_cl)
        if(sampled_coordinate not in sampled_coordinates):
            stroke_order = coordinate_to_stroke[sampled_coordinate]
            d_sI[stroke_order].append(len(sampled_coordinates))
            sampled_coordinates.append(sampled_coordinate)
            if(len(sampled_coordinates) > 1):
                last_coordinate = sampled_coordinates[-2]
                length_diff = d_cL[sampled_coordinate]-d_cL[last_coordinate]
                time_diff = d_cT[sampled_coordinate]-d_cT[last_coordinate]
                d_V.append(length_diff/time_diff)
            else:
                d_V.append(d_cL[sampled_coordinate]/d_cT[sampled_coordinate])
    d_X = d_X[sampled_coordinates]
    d_Y = d_Y[sampled_coordinates]
    d_cT = d_cT[sampled_coordinates]
    d_cL = d_cL[sampled_coordinates]
    d_V = np.array(d_V)
    if(plot):
        show_drawing_data(d_X, d_Y, d_cT)
        plt.title('drawing data after discretizing')
        show_drawing_data(d_X, d_Y, d_V, d_sI, "Velocity")
        plt.title('drawing data velocity spectrum')
        plt.show()
    return d_X, d_Y, d_cT, d_cL, d_V, d_sI

def get_steps_ahead_forecasting_data(d_X, d_Y, d_W, d_T):
    d_X, d_Y, d_cT, d_cL, d_V, d_sI = preprocessing(d_X, d_Y, d_W, d_T)
    if(time_faction):
        d_cT = Scaler('minmax', d_cT).eval(d_cT)
    forecast_input, forecast_target = [], []
    for s, d_s in enumerate(d_sI):
        if(len(d_s) <= forecast_step):
            continue
        for i in range(len(d_s)-forecast_step):
            input_coord = d_s[i]
            x, y, ct = d_X[input_coord], d_Y[input_coord], d_cT[input_coord]
            cl, v = d_cL[input_coord], d_V[input_coord]
            forecast_coord = d_s[i+forecast_step]
            forecast_input.append([x, y, ct, cl, v, s])
            forecast_target.append([d_X[forecast_coord]+1j*d_Y[forecast_coord]])
    return np.array(forecast_input), np.array(forecast_target)

def get_training_data(test_proportion=0.3):
    X_train, y_train, X_test, y_test = None, None, None, None
    decode = lambda str: np.array(list(map(float, str.split('|'))))
    ci_data = drawing_raw_data_df['Cognitive Impairment']
    ci_subjects = drawing_raw_data_df[ci_data==1].index
    num_ci = len(ci_subjects)
    test_ci = int(num_ci*test_proportion)
    rand_ci = npr.choice(list(range(num_ci)), test_ci, replace=False)
    rand_ci = ci_subjects[rand_ci]
    rand_ci_train = list(set(ci_subjects).difference(rand_ci))
    nonci_subjects = drawing_raw_data_df[ci_data==0].index
    num_nonci = len(nonci_subjects)
    test_nonci = int(num_nonci*test_proportion)
    rand_nonci = npr.choice(list(range(num_nonci)), test_nonci, replace=False)
    rand_nonci = nonci_subjects[rand_nonci]
    rand_nonci_train = list(set(ci_subjects).difference(rand_ci))
    subjects_for_testing = rand_ci.tolist()+rand_nonci.tolist()
    subjects_for_training = rand_ci_train+rand_nonci_train
    for subject_id in subjects_for_training:
        d_X = decode(drawing_raw_data_df['X'][subject_id])
        d_Y = decode(drawing_raw_data_df['Y'][subject_id])
        d_W = decode(drawing_raw_data_df['W'][subject_id])
        d_T = decode(drawing_raw_data_df['T'][subject_id])
        if(len(d_X) < sampling_points):
            continue
        if(in_centimeter):
            d_X /= PIXEL_CENTIMETER
            d_Y /= PIXEL_CENTIMETER
        input, target = get_steps_ahead_forecasting_data(d_X, d_Y, d_W, d_T)
        CI = np.array(ci_data[subject_id])
        input = np.hstack((CI*np.ones((input.shape[0], 1)), input))
        if(age):
            AGE = np.array(drawing_raw_data_df['Age'][subject_id])
            input = np.hstack((input, AGE*np.ones((input.shape[0], 1))))
        if(gender):
            GENDER = np.array(drawing_raw_data_df['Male'][subject_id])
            input = np.hstack((input, GENDER*np.ones((input.shape[0], 1))))
        if(edu_level):
            EDU_cols = ['Uneducated', 'Primary', 'Secondary', 'University']
            EDU = np.array(drawing_raw_data_df[EDU_cols].loc[subject_id])
            input = np.hstack((input, EDU*np.ones((input.shape[0], 1))))
        X_train = input if(X_train is None) else np.vstack((X_train, input))
        y_train = target if(y_train is None) else np.vstack((y_train, target))
    for subject_id in subjects_for_testing:
        d_X = decode(drawing_raw_data_df['X'][subject_id])
        d_Y = decode(drawing_raw_data_df['Y'][subject_id])
        d_W = decode(drawing_raw_data_df['W'][subject_id])
        d_T = decode(drawing_raw_data_df['T'][subject_id])
        if(len(d_X) < sampling_points):
            continue
        if(in_centimeter):
            d_X /= PIXEL_CENTIMETER
            d_Y /= PIXEL_CENTIMETER
        input, target = get_steps_ahead_forecasting_data(d_X, d_Y, d_W, d_T)
        CI = np.array(ci_data[subject_id])
        input = np.hstack((CI*np.ones((input.shape[0], 1)), input))
        if(age):
            AGE = np.array(drawing_raw_data_df['Age'][subject_id])
            input = np.hstack((input, AGE*np.ones((input.shape[0], 1))))
        if(gender):
            GENDER = np.array(drawing_raw_data_df['Male'][subject_id])
            input = np.hstack((input, GENDER*np.ones((input.shape[0], 1))))
        if(edu_level):
            EDU_cols = ['Uneducated', 'Primary', 'Secondary', 'University']
            EDU = np.array(drawing_raw_data_df[EDU_cols].loc[subject_id])
            input = np.hstack((input, EDU*np.ones((input.shape[0], 1))))
        X_test = input if(X_test is None) else np.vstack((X_test, input))
        y_test = target if(y_test is None) else np.vstack((y_test, target))
    return X_train, y_train, X_test, y_test

print('# Preprocessing Raw Drawing Data')
X_train, y_train, X_test, y_test = get_training_data()
print('  Gathered %d Training Samples.'%(X_train.shape[0]))
print('  Gathered %d Testing Samples.'%(X_test.shape[0]))
print('  Done.')

while(True):
    
    print('# Training GomPlex')
    gp = GomPlex(30)
    gp.fit(X_train, y_train, opt_rate=1, cost_tol=1, plot=True)
    print('  Done.')
    
    print('# Choosing GomPlex Models')
    if(not os.path.exists(MODEL_PATH)):
        gp.save(MODEL_PATH)
    else:
        best_gp = GomPlex().load(MODEL_PATH)
        best_gp_score = metric.eval(y_test, *best_gp.predict(X_test))
        new_gp_score = metric.eval(y_test, *gp.predict(X_test))
        if(new_gp_score < best_gp_score):
            gp.save(MODEL_PATH)
    print('  Found Better Model!')
    print('  Done.')

fig = plt.figure()
ax = fig.gca(projection='3d')
x_plot = gp.X[:, 0]
ax.scatter(gp.X[:, 0], gp.y.real.ravel(), gp.y.imag.ravel(),
    marker='x', s=30, alpha=0.2, label='training data')
ax.legend()
mu, std = gp.predict(gp.X, scaled=False)
mu_r = mu.real.ravel()
mu_i = mu.imag.ravel()
ax.scatter(x_plot, mu_r, mu_i, marker='.', color='red',
    s=30, alpha=0.5, label='complex regression')
ax.legend()
ax.set_xlabel('x')
ax.set_ylabel('Re{y}')
ax.set_zlabel('Im{y}')

plt.show()