################################################################################
#  Github: https://github.com/MaxInGaussian/GomPlex
#  Author: Max W. Y. Lam (maxingaussian@gmail.com)
################################################################################

import sys
import multiprocessing

from FeatureLearner import FeatureLearner

DRAWING_RAW_DATA_PATH = 'drawing_raw_data.csv'

model = FeatureLearner(use_gender=True, use_age=True, use_edu_level=True,
    show_training_drawings=False, show_predicted_drawings=False)
model.load_drawing_data(DRAWING_RAW_DATA_PATH)

def nonstop_train():
    while(True):
        model.train_regressor(ratio=0.3, cv_folds=3, plot_error=False)

pool = multiprocessing.Pool(min(1, multiprocessing.cpu_count()//2))
pool.apply_async(nonstop_train)
pool.close()
pool.join()