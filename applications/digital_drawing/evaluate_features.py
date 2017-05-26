################################################################################
#  Github: https://github.com/MaxInGaussian/GomPlex
#  Author: Max W. Y. Lam (maxingaussian@gmail.com)
################################################################################

import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from FeatureLearner import FeatureLearner

DRAWING_RAW_DATA_PATH = 'drawing_raw_data.csv'

model = FeatureLearner(sample_time=50, use_past=4,
    use_gender=True, use_age=True, use_edu_level=True,
    show_training_drawings=False, show_predicted_drawings=False)
model.load_drawing_data(DRAWING_RAW_DATA_PATH)
# model.show_velocity_graph('HK1520')
# model.show_direction_graph('HK1520')
cfs_mat = model.eval_features_for_subjects()[-1]
df_cm = pd.DataFrame(cfs_mat, index = [i for i in ["CI", "non-CI"]],
                  columns = ["Screened as CI", "Screened as non-CI"])

sn.set(font_scale=1.4)
sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})
plt.show()