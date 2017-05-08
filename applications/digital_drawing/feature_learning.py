################################################################################
#  Github: https://github.com/MaxInGaussian/GomPlex
#  Author: Max W. Y. Lam (maxingaussian@gmail.com)
################################################################################

from FeatureLearner import FeatureLearner

DRAWING_RAW_DATA_PATH = 'drawing_raw_data.csv'

feat_learner = FeatureLearner()
feat_learner.load_drawing_data(DRAWING_RAW_DATA_PATH)

while(True):
    feat_learner.train_inner_regressor()