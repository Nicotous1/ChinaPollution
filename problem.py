import os

import pandas as pd
import rampwf as rw
import numpy as np

from rampwf.score_types.base import BaseScoreType

from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, explained_variance_score, accuracy_score

problem_title = 'Predicting the pollution of Bejing with the weather'

# -----------------------------------------------------------------------------
# Predictions type
# -----------------------------------------------------------------------------

_target_column_name = 'pm2.5'

Predictions = rw.prediction_types.make_regression()

# -----------------------------------------------------------------------------
# Worklow element
# -----------------------------------------------------------------------------

workflow = rw.workflows.FeatureExtractorRegressor()

# -----------------------------------------------------------------------------
# Score types
# -----------------------------------------------------------------------------

class R2(BaseScoreType):
    def __init__(self, name='r2', precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        score = r2_score(y_true, y_pred)
        return round(score, self.precision)
    
    
class AlertAccuracy(BaseScoreType):
    """
        The accuracy of the model for detecting a severe or critic event (pm2.5 above 200)
        When there is an event and the model predict it.
    """
    def __init__(self, name='alert_accuracy', precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        y_true = np.array(y_true) >= 200
        y_pred = np.array(y_pred) >= 200
        score = accuracy_score(y_true, y_pred)
        return round(score, self.precision)
    
    
class AlertFN(BaseScoreType):
    """
        Gives the amount of event not detected (false negative) for a severe or critic event. (pm2.5 above 200)
        When there is an event and the model do not predict it.
    """
    def __init__(self, name='alert_missed', precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        y_true = np.array(y_true) >= 200 # There is an event
        y_pred = np.array(y_pred) < 200 # Do not predict an event
        score = sum(y_true & y_pred)/len(y_true)
        return round(score, self.precision)


score_types = [
       rw.score_types.RMSE(),
       R2(),
       AlertAccuracy(),
       AlertFN(),
]

# -----------------------------------------------------------------------------
# Cross-validation scheme
# -----------------------------------------------------------------------------

def get_cv(X, y):
    # using 5 folds as default
    k = 5
    # up to 10 fold cross-validation based on 5 splits, using two parts for
    # testing in each fold
    n_splits = 5
    cv = KFold(n_splits=n_splits)
    splits = list(cv.split(X, y))
    # 5 folds, each point is in test set 4x
    # set k to a lower number if you want less folds
    pattern = [
        ([2, 3, 4], [0, 1]), ([0, 1, 4], [2, 3]), ([0, 2, 3], [1, 4]),
        ([0, 1, 3], [2, 4]), ([1, 2, 4], [0, 3]), ([0, 1, 2], [3, 4]),
        ([0, 2, 4], [1, 3]), ([1, 2, 3], [0, 4]), ([0, 3, 4], [1, 2]),
        ([1, 3, 4], [0, 2])
    ]
    for ps in pattern[:k]:
        yield (np.hstack([splits[p][1] for p in ps[0]]),
               np.hstack([splits[p][1] for p in ps[1]]))


# -----------------------------------------------------------------------------
# Training / testing data reader
# -----------------------------------------------------------------------------


def _read_data(path, filename):
    df = pd.read_csv(os.path.join(path, 'data', filename))
    df.cbwd = df.cbwd.astype("category")
    df.index = pd.to_datetime(df[["year", "month", "day", "hour"]])
    X, y = df.drop(_target_column_name, axis = 1), df[_target_column_name]
    return X, y.values

def get_train_data(path='.'):
    return _read_data(path, filename = 'train.csv')


def get_test_data(path='.'):
    return _read_data(path, filename = 'test.csv')
