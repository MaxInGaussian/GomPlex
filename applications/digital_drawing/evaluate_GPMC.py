################################################################################
#  Github: https://github.com/MaxInGaussian/GomPlex
#  Author: Max W. Y. Lam (maxingaussian@gmail.com)
################################################################################

import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from DecisionSystem import DecisionSystem

from sys import path
path.append("../../")
from GomPlex import *

DRAWING_RAW_DATA_PATH = 'data/drawing_raw_data.csv'

model = DecisionSystem(sample_time=100, use_past=4,
    use_doctor_diag=True, use_gender=True, use_age=True, use_edu_level=True,
    show_training_drawings=False, show_predicted_drawings=False)
model.load_drawing_data(DRAWING_RAW_DATA_PATH)
num_ci, num_nci = model.ci.sum(), len(model.ci)-model.ci.sum()
print("%d:%d"%(num_ci, num_nci))

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
    return sensitivity, specificity, accuracy, precision, F1

trains_nums = [200] #[30*(i+1) for i in range(10)]
performance_log = []
#plt.figure()
#plt.show()
for n_trains in trains_nums:
    AUC, F1, cfs_mat, cis, pred_cis, age, gender, edu_lv =\
        model.eval_model_for_subjects(n_trains=n_trains)
    fpr, tpr, thresholds = roc_curve(cis, pred_cis)
    AUC = auc(fpr, tpr)
    arg = np.argmax(tpr-fpr)
    #plt.plot(fpr, tpr, linestyle='-', label='GPMC-%d (AUC = %0.3f)' % (n_trains, AUC))
    #plt.scatter(fpr[arg], tpr[arg], s=50, marker='x')
    print('GPMC-%d:'%(n_trains))
    print("AUC =", AUC)
    performance_log.append(list(get_eval_from_fpr_tpr(fpr[arg], tpr[arg])))
    performance_log[-1].append(AUC)
#plt.plot([0, 1], [0, 1], 'k-', label='Random Guessing (AUC = 0.5)', alpha=0.3)
#plt.xlim([0, 1])
#plt.ylim([0, 1])
#plt.xlabel('False Positive Rate (1 - Specificity)')
#plt.ylabel('True Positive Rate (Sensitivity)')
#plt.title('Receiver Operating Characteristic')
#plt.legend(loc="lower right")
#plt.tight_layout()

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
