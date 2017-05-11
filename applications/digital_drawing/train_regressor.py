################################################################################
#  Github: https://github.com/MaxInGaussian/GomPlex
#  Author: Max W. Y. Lam (maxingaussian@gmail.com)
################################################################################

from FeatureLearner import FeatureLearner

DRAWING_RAW_DATA_PATH = 'drawing_raw_data.csv'

model = FeatureLearner(moca_cutoff=20, forecast_step=0.01, sample_time=100,
        use_past=3, use_gender=True, use_age=True, use_edu_level=True,
        stroke_size_tol=10, stroke_length_tol=1, centimeter=True, metric='nmse',
        show_training_drawings=False, show_predicted_drawings=False)

model.load_drawing_data(DRAWING_RAW_DATA_PATH)

while(True):
    model.train_regressor(ratio=0.3, cv_folds=3, plot_error=True)
    model.eval_features_for_subjects()