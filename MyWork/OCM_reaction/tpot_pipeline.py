import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from tpot.builtins import StackingEstimator
from xgboost import XGBRegressor
from tpot.export_utils import set_param_recursive

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('MyWork/datasets/df_tpot.csv', encoding= 'cp932', dtype=np.float64)
features = tpot_data.drop('C2 yield', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['C2 yield'], random_state=42)

# Average CV score on the training set was: -0.9710009985349023
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=XGBRegressor(learning_rate=0.5, max_depth=7, min_child_weight=19, n_estimators=100, n_jobs=1, objective="reg:squarederror", subsample=0.8500000000000001, verbosity=0)),
    MinMaxScaler(),
    PolynomialFeatures(degree=2, include_bias=False, interaction_only=False),
    RidgeCV()
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 42)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
