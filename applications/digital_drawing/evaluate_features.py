################################################################################
#  Github: https://github.com/MaxInGaussian/GomPlex
#  Author: Max W. Y. Lam (maxingaussian@gmail.com)
################################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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