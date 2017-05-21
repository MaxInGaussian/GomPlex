################################################################################
#  Github: https://github.com/MaxInGaussian/GomPlex
#  Author: Max W. Y. Lam (maxingaussian@gmail.com)
################################################################################

import numpy as np

from FeatureLearner import FeatureLearner

DRAWING_RAW_DATA_PATH = 'drawing_raw_data.csv'

model = FeatureLearner(moca_cutoff=20, forecast_step=0.02, sample_time=100,
        use_past=10, use_gender=True, use_age=True, use_edu_level=True,
        stroke_size_tol=10, stroke_length_tol=1, centimeter=True, metric='nmse',
        show_training_drawings=False, show_predicted_drawings=False)

model.load_drawing_data(DRAWING_RAW_DATA_PATH)

accuracy, X_feat = model.eval_features_for_subjects()
