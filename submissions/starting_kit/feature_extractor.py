from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
import numpy as np
import pandas as pd

COLUMNS_ROLLING = ['DEWP', 'TEMP', 'PRES']
COLUMNS_DUMMY = ['cbwd']
COLUMNS_CYCLIC = ['month', 'day', 'hour']
ROLLING_WINDOWS = [4, 15]

class FeatureExtractor(BaseEstimator, TransformerMixin):
    # Mandatory method
    def fit(self, X_df, y):
        return self

    # Mandatory method
    def transform(self, X_df):
        X_df = pd.get_dummies(X_df, columns=COLUMNS_DUMMY, drop_first=True)
        X_df = self.compute_rolling(X_df)
        X_df = self.encode_cyclic_values(X_df)
        return X_df.astype(np.float).fillna(0)
    
    # We compute rolling means and deviations on the "temporal" variables
    def compute_rolling(self, X_df):
        for window in ROLLING_WINDOWS:
            for column in COLUMNS_ROLLING:
                X_df['rolling_mean_{}'.format(column)] = X_df[column].rolling(window).mean()
                X_df['rolling_std_{}'.format(column)] = X_df[column].rolling(window).std()
        return X_df
    
    # We encode cyclic values by using trigonometric functions
    def encode_cyclic_values(self, X_df):
        for column in COLUMNS_CYCLIC:
            X_df['cos_{}'.format(column)] = np.cos(X_df[column] * 2 * np.pi / X_df[column].max())
            X_df['sin_{}'.format(column)] = np.sin(X_df[column] * 2 * np.pi / X_df[column].max())
        return X_df
