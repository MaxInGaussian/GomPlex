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

# model.show_velocity_graph('HK1520')
# model.show_direction_graph('HK1520')

AUC, F1, cfs_mat, cis, pred_cis = model.eval_features_for_subjects()
fpr, tpr, thresholds = roc_curve(cis, pred_cis)
AUC = auc(fpr, tpr)
arg = np.argmax(tpr-fpr)
plt.plot(fpr, tpr, label='ROC curve (AUC = %0.3f)' % AUC)
plt.scatter(fpr[arg], tpr[arg], s=50, color='red', marker='x')
plt.text(fpr[arg]+0.005, tpr[arg]-0.04, '(%.3f, %.3f)'%(fpr[arg], tpr[arg]))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('False Positive Rate (1 - Specifity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
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