################################################################################
#  Github: https://github.com/MaxInGaussian/GomPlex
#  Author: Max W. Y. Lam (maxingaussian@gmail.com)
################################################################################

import numpy as np

from FeatureLearner import FeatureLearner

DRAWING_RAW_DATA_PATH = 'drawing_raw_data.csv'

model = FeatureLearner(sample_time=30, use_past=5,
    use_gender=True, use_age=True, use_edu_level=True,
    show_training_drawings=True, show_predicted_drawings=False)
model.load_drawing_data(DRAWING_RAW_DATA_PATH)
model.get_input_output_matrix_by_subject('AD077')
# model.eval_features_for_subjects()
