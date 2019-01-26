import os

import pandas as pd
import rampwf as rw
import numpy as np

from sklearn.model_selection import KFold





problem_title = 'Predicting the pollution of Bejing with the weather'


# -----------------------------------------------------------------------------
# Predictions type
# -----------------------------------------------------------------------------

_target_column_name = 'pm2.5'

Predictions = rw.prediction_types.make_regression(
    label_names=[_target_column_name])


# -----------------------------------------------------------------------------
# Worklow element
# -----------------------------------------------------------------------------


workflow = rw.workflows.FeatureExtractorRegressor()




# -----------------------------------------------------------------------------
# Score types
# -----------------------------------------------------------------------------


score_types = [
       rw.score_types.RMSE()
]






# -----------------------------------------------------------------------------
# Cross-validation scheme
# -----------------------------------------------------------------------------


def get_cv(X, y):
    cv = KFold(n_splits=8)
    return cv.split(X, y)




# -----------------------------------------------------------------------------
# Training / testing data reader
# -----------------------------------------------------------------------------


def _read_data(path, filename):
    df = pd.read_csv(os.path.join(path, 'data', filename), index_col = 0)
    df.cbwd = df.cbwd.astype("category")
    df.index = pd.to_datetime(df[["year", "month", "day", "hour"]])
    X, y = df.drop(_target_column_name, axis = 1), df[_target_column_name]
    return X, y

    


def get_train_data(path='.'):
    return _read_data(path, filename = 'train.csv')


def get_test_data(path='.'):
    return _read_data(path, filename = 'test.csv')