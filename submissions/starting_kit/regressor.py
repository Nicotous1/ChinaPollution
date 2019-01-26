from sklearn.base import BaseEstimator
import numpy as np


class Regressor(BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.zeros((X.shape[0], 1))

    def predict_proba(self, X):
        return np.ones((X.shape[0], 1))
