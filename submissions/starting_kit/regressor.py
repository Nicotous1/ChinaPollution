from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

class Regressor(BaseEstimator):
    def __init__(self):
        self.reg = make_pipeline(StandardScaler(), XGBRegressor())

    def fit(self, X, y):
        self.reg.fit(X, y)
        return self

    def predict(self, X):
        return np.maximum(self.reg.predict(X), 1e-10)
    
    def score(self, X, y):
        return np.sqrt(mean_squared_error(self.predict(X), y))
    
    def get_coef(self, index=None):
        return pd.Series(self.reg.steps[1][1].coef_, index=index)