################################################################################
#  Github: https://github.com/MaxInGaussian/GomPlex
#  Author: Max W. Y. Lam (maxingaussian@gmail.com)
################################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from DecisionSystem import DecisionSystem

from sys import path
path.append("../../")
from GomPlex import *

DRAWING_RAW_DATA_PATH = 'drawing_raw_data.csv'

model = DecisionSystem(sample_time=100, use_past=4,
    use_gender=True, use_age=True, use_edu_level=True,
    show_training_drawings=False, show_predicted_drawings=False)
model.load_drawing_data(DRAWING_RAW_DATA_PATH)
num_ci, num_nci = model.ci.sum(), len(model.ci)-model.ci.sum()

# model.show_velocity_graph('MS0045')
# model.show_direction_graph('MS0045')

def get_eval_from_fpr_tpr(fpr, tpr):
    cfs_mat = np.array([[tpr*num_ci, num_ci-tpr*num_ci],
        [fpr*num_nci, num_nci-fpr*num_nci]])
    accuracy = (cfs_mat[0, 0]+cfs_mat[1, 1])/np.sum(cfs_mat)
    precision = 0 if np.sum(cfs_mat[:, 0]) == 0 else\
        cfs_mat[0, 0]/np.sum(cfs_mat[:, 0])
    sensitivity = 0 if np.sum(cfs_mat[0]) == 0 else\
        cfs_mat[0, 0]/np.sum(cfs_mat[0])
    specificity = 0 if np.sum(cfs_mat[1]) == 0 else\
        cfs_mat[1, 1]/np.sum(cfs_mat[1])
    F1 = 0 if precision+sensitivity == 0 else\
        2*(precision*sensitivity)/(precision+sensitivity)
    print("Sensitivity =", sensitivity)
    print("Specificity =", specificity)
    print("Accuracy =", accuracy)
    print("Precision =", precision)
    print("F1 Score =", F1)
    
AUC, F1, cfs_mat, cis, pred_cis = model.eval_features_for_subjects()

fpr, tpr, thresholds = roc_curve(cis, pred_cis)
AUC = auc(fpr, tpr)
arg = np.argmax(tpr-fpr)
plt.plot(fpr, tpr, color='black', lw=2, linestyle='-', label='GPMC (AUC = %0.3f)' % AUC)
plt.scatter(fpr[arg], tpr[arg], s=50, color='k', marker='x')
print('GPMC:')
get_eval_from_fpr_tpr(fpr[arg], tpr[arg])

lr = ['Number of angles', 'Intersection', 'Closure', 'Rotation', 'Closing-in']
lr_mat = model.df_drawing_data[lr].as_matrix().tolist()
lr_y = model.ci.as_matrix().astype(np.int64)
X, y = [], []
for i, lr_vec in enumerate(lr_mat):
    avg_V = model.avg_V[model.df_drawing_data.index[i]]
    std_V = model.std_V[model.df_drawing_data.index[i]]
    avg_T = model.avg_T[model.df_drawing_data.index[i]]
    std_T = model.std_T[model.df_drawing_data.index[i]]
    if(not np.any(np.isnan([avg_V, std_V, avg_T, std_T]))):
        lr_vec.extend([avg_V, std_V, avg_T, std_T])
        X.append(lr_vec)
        y.append(lr_y[i])
lr_model = LogisticRegression().fit(X[:int(len(X)*0.65)], y[:int(len(X)*0.65)])
pred_cis_lr = lr_model.predict_proba(X[int(len(X)*0.65):])
fpr_lr, tpr_lr, thresholds_lr = roc_curve(y[int(len(X)*0.65):], pred_cis_lr[:, 1])
AUC_lr = auc(fpr_lr, tpr_lr)
arg_lr = np.argmax(tpr_lr-fpr_lr)
plt.plot(fpr_lr, tpr_lr, color='black', lw=2, linestyle='--', label='Logistic Regression (AUC = %0.3f)' % AUC_lr)
plt.scatter(fpr_lr[arg_lr], tpr_lr[arg_lr], s=50, color='k', marker='x')
print('Logistic Regression:')
get_eval_from_fpr_tpr(fpr_lr[arg_lr], tpr_lr[arg_lr])


caffarra = ['Number of angles', 'Intersection', 'Closure', 'Rotation', 'Closing-in']
caffarra_score = model.df_drawing_data[caffarra].sum(axis=1)
pred_cis_caff = np.array(caffarra_score).ravel()/13
fpr_caff, tpr_caff, thresholds_caff = roc_curve(cis, 1-pred_cis_caff)
AUC_caff = auc(fpr_caff, tpr_caff)
arg_caff = np.argmax(tpr_caff-fpr_caff)
plt.plot(fpr_caff, tpr_caff, color='black', lw=2, linestyle='-.', label='Caffarra\'s Method (AUC = %0.3f)' % AUC_caff)
plt.scatter(fpr_caff[arg_caff], tpr_caff[arg_caff], s=50, color='k', marker='x')
print('Caffarra\'s Method:')
get_eval_from_fpr_tpr(fpr_caff[arg_caff], tpr_caff[arg_caff])

mmse = ['Number of angles', 'Intersection']
mmse_score = np.array(model.df_drawing_data[mmse].sum(axis=1)==8)/1.
fpr_mmse, tpr_mmse, thresholds_mmse = roc_curve(cis, 1-mmse_score)
AUC_mmse = auc(fpr_mmse, tpr_mmse)
arg_mmse = np.argmax(tpr_mmse-fpr_mmse)
plt.plot(fpr_mmse, tpr_mmse, color='black', lw=2, linestyle=':', label='MMSE Method (AUC = %0.3f)' % AUC_mmse)
plt.scatter(fpr_mmse[arg_mmse], tpr_mmse[arg_mmse], s=50, color='k', marker='x')
print('MMSE\'s Method:')
get_eval_from_fpr_tpr(fpr_mmse[arg_mmse], tpr_mmse[arg_mmse])

plt.plot([0, 1], [0, 1], 'k-', label='Random Guessing (AUC = 0.5)', alpha=0.3)
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

def plot_confusion_matrix(cm, classes):
    normalize=False
    cmap=plt.cm.Blues
    plt.imshow(cfs_mat, interpolation='nearest', cmap=cmap)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = cfs_mat.max() / 2.
    for i, j in itertools.product(range(cfs_mat.shape[0]), range(cfs_mat.shape[1])):
        plt.text(j, i, cfs_mat[i, j], horizontalalignment="center",
            color="white" if cfs_mat[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')