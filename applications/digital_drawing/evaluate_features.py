################################################################################
#  Github: https://github.com/MaxInGaussian/GomPlex
#  Author: Max W. Y. Lam (maxingaussian@gmail.com)
################################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from DecisionSystem import DecisionSystem

from sys import path
path.append("../../")
from GomPlex import *

DRAWING_RAW_DATA_PATH = 'drawing_raw_data.csv'

model = DecisionSystem(sample_time=50, use_past=4,
    use_gender=True, use_age=True, use_edu_level=True,
    show_training_drawings=False, show_predicted_drawings=False)
model.load_drawing_data(DRAWING_RAW_DATA_PATH)

# model.show_velocity_graph('MS0045')
# model.show_direction_graph('MS0045')

def get_eval_from_fpr_tpr(fpr, tpr, num_ci, num_nci):
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
    return cfs_mat, sensitivity, specificity, accuracy, precision, F1

AUC, F1, cfs_mat, cis, pred_cis = model.eval_features_for_subjects()    
num_ci, num_nci = sum(cis), len(cis)-sum(cis)
fpr, tpr, thresholds = roc_curve(cis, pred_cis)
AUC = auc(fpr, tpr)
arg = np.argmax(tpr-fpr)
plt.plot(fpr, tpr, color='black', lw=2, linestyle='-', label='Our Method (AUC = %0.3f)' % AUC)
plt.scatter(fpr[arg], tpr[arg], s=50, color='red', marker='*')
# plt.text(fpr[arg]+0.005, tpr[arg]-0.04, '(%.3f, %.3f)'%(fpr[arg], tpr[arg]))

caffarra = ['Number of angles', 'Intersection', 'Closure', 'Rotation', 'Closing-in']
caffarra_score = model.df_drawing_data[caffarra].sum(axis=1)
pred_cis_caffarra = np.array(caffarra_score).ravel()/13
fpr_caffarra, tpr_caffarra, thresholds_caffarra = roc_curve(cis, 1-pred_cis_caffarra)
AUC_caffarra = auc(fpr_caffarra, tpr_caffarra)
arg_caffarra = np.argmax(tpr_caffarra-fpr_caffarra)
plt.plot(fpr_caffarra, tpr_caffarra, color='black', lw=2, linestyle='-.', label='Caffarra\'s Method (AUC = %0.3f)' % AUC_caffarra)
plt.scatter(fpr_caffarra[arg_caffarra], tpr_caffarra[arg_caffarra], s=50, color='red', marker='*')
# plt.text(fpr_caffarra[arg_caffarra]+0.005, tpr_caffarra[arg_caffarra]-0.02,
    # '(%.3f, %.3f)'%(fpr_caffarra[arg_caffarra], tpr_caffarra[arg_caffarra]))

mmse = ['Number of angles', 'Intersection']
mmse_score = np.array(model.df_drawing_data[mmse].sum(axis=1)==8)/1.
fpr_mmse, tpr_mmse, thresholds_mmse = roc_curve(cis, 1-mmse_score)
AUC_mmse = auc(fpr_mmse, tpr_mmse)
arg_mmse = np.argmax(tpr_mmse-fpr_mmse)
plt.plot(fpr_mmse, tpr_mmse, color='black', lw=2, linestyle=':', label='MMSE\'s Method (AUC = %0.3f)' % AUC_mmse)
plt.scatter(fpr_mmse[arg_mmse], tpr_mmse[arg_mmse], s=50, color='red', marker='*')
# plt.text(fpr_mmse[arg_mmse]+0.008, tpr_mmse[arg_mmse]-0.04,
    # '(%.3f, %.3f)'%(fpr_mmse[arg_mmse], tpr_mmse[arg_mmse]))

plt.plot([0, 1], [0, 1], 'k-', label='Random Guessing (AUC = 0.5)', alpha=0.3)
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('False Positive Rate (1 - Specifity)')
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