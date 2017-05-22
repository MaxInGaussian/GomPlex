################################################################################
#  Github: https://github.com/MaxInGaussian/GomPlex
#  Author: Max W. Y. Lam (maxingaussian@gmail.com)
################################################################################

import sys

from FeatureLearner import FeatureLearner

DRAWING_RAW_DATA_PATH = 'drawing_raw_data.csv'

def nonstop_train():
    model = FeatureLearner(sample_time=200, use_past=10,
        use_gender=True, use_age=True, use_edu_level=True,
        show_training_drawings=False, show_predicted_drawings=False)
    model.load_drawing_data(DRAWING_RAW_DATA_PATH)
    while(True):
        model.train_regressor(ratio=0.3, cv_folds=3, plot_error=False)

nonstop_train()