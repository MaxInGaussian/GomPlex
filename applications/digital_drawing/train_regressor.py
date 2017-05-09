################################################################################
#  Github: https://github.com/MaxInGaussian/GomPlex
#  Author: Max W. Y. Lam (maxingaussian@gmail.com)
################################################################################

from FeatureLearner import FeatureLearner

DRAWING_RAW_DATA_PATH = 'drawing_raw_data.csv'

model = FeatureLearner(moca_cutoff=20, in_centimeter=True, seperate_length=0.2,
    use_gender=True, use_age=True, use_edu_level=True, metric='nmse',
    stroke_size_tol=10, stroke_length_tol=1, plot_samples=False)

model.load_drawing_data(DRAWING_RAW_DATA_PATH)

while(True):
    model.train_regressor()