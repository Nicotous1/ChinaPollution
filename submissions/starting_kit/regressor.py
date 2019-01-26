from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from sklearn import svm


class Regressor(BaseEstimator):
    def __init__(self):
        self.reg = make_pipeline(StandardScaler(), svm.SVR())

    def fit(self, X, y):
        self.reg.fit(X, y)
        return self

    def predict(self, X):
        return self.reg.predict(X)

    def predict_proba(self, X):
        return self.reg.predict_proba(X)
