################################################################################
#  Github: https://github.com/MaxInGaussian/GomPlex
#  Author: Max W. Y. Lam (maxingaussian@gmail.com)
################################################################################

import sys

from DecisionSystem import DecisionSystem

DRAWING_RAW_DATA_PATH = 'data/drawing_raw_data.csv'

def nonstop_train():
    model = DecisionSystem(sample_time=50, use_past=4,
        use_doctor_diag=True, use_gender=True, use_age=True, use_edu_level=True,
        show_training_drawings=False, show_predicted_drawings=False)
    model.load_drawing_data(DRAWING_RAW_DATA_PATH)
    while(True):
        model.train_regressor(iter_tol=30, ratio=0.3, cv_folds=2)

nonstop_train()
