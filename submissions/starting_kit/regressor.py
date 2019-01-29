from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

class Regressor(BaseEstimator):
    def __init__(self):
        # We used a Grid search to estimate good hyperparameters for the XGBRegressor
        xgb_regressor = XGBRegressor(max_depth=3, learning_rate=0.1, n_estimators=100, n_jobs=1)
        self.reg = make_pipeline(StandardScaler(), xgb_regressor)
        # The Grid search was useful to tune the hyperparameters but isn't used in the ramp workflow
        param_grid = {
            'learning_rate': [0.01, 0.1, 1],
            'max_depth': [3, 5, 10],
            'n_estimators': [10, 100],
        }
        grid_search_estimator = GridSearchCV(xgb_regressor, param_grid, scoring='r2', cv=5, verbose=10)
        self.reg_grid_search = make_pipeline(StandardScaler(), grid_search_estimator)

    def fit(self, X, y):
        self.reg.fit(X, y)
        return self

    def predict(self, X):
        # We set the minimum of the predictions to 0 because the level of PM2.5 is a positive number
        return np.maximum(self.reg.predict(X), 0)
    
    # The next methods were useful to choose a model and to tune the hyperparameters but aren't used in the rampworkflow

    def rmse(self, X, y):
        return np.sqrt(mean_squared_error(y, self.predict(X)))

    def r2(self, X, y):
        return r2_score(y, self.predict(X))

    def fit_grid_search(self, X, y):
        self.reg_grid_search.fit(X, y)
