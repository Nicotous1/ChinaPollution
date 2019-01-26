from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
import numpy as np

class FeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X_df, y):
        return self

    def transform(self, X_df):
        # get only the continuous variable
        return X_df[["DEWP", "TEMP", "PRES"]].astype(np.float).fillna(0)
