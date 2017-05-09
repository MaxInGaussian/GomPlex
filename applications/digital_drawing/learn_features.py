################################################################################
#  Github: https://github.com/MaxInGaussian/GomPlex
#  Author: Max W. Y. Lam (maxingaussian@gmail.com)
################################################################################

from FeatureLearner import FeatureLearner

DRAWING_RAW_DATA_PATH = 'drawing_raw_data.csv'

feat_learner = FeatureLearner()
feat_learner.load_drawing_data(DRAWING_RAW_DATA_PATH)
X_feat = feat_learner.learn_features_for_subjects()
X_mean_feat = np.mean(X_feat, 1)