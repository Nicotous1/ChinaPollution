from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
import numpy as np

class Regressor(BaseEstimator):
    def __init__(self):
        self.reg = make_pipeline(StandardScaler(), Ridge())

    def fit(self, X, y):
        self.reg.fit(X, y)
        return self

    def predict(self, X):
        return np.maximum(self.reg.predict(X), 0)