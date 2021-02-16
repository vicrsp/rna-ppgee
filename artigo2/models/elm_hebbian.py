import numpy as np
import random
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, roc_auc_score


class ELMHebbianClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, p=5):
        self.base = ELMHebbianRegressor(p)

    def predict(self, X):
        return np.sign(self.base.predict(X))

    def fit(self, X, y):
        self.base.fit(X, y)
        return self

    def decision_function(self, X):
        return self.base.predict(X)

    def score(self, X, y):
        return roc_auc_score(y, self.predict(X))

class ELMHebbianRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, p=5):
        self.p = p

    def predict(self, X):
        # input validation
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, ['coef_', 'Z_', 'scaler_'])

        N, _ = X.shape
        x_aug = np.hstack((-np.ones((N, 1)), X))
        x_aug = self.scaler_.transform(x_aug)
        H = np.tanh(x_aug @ self.Z_)
        return H @ self.coef_

    def pre_process(self, X):
        self.scaler_ = StandardScaler().fit(X)
        return self.scaler_.transform(X)

    def calculate_weights_default(self, y, H):
        w0 = y @ H
        w = w0 / np.linalg.norm(w0)
        return w

    def fit(self, X, y):
        # check X, y consistency
        X, y = check_X_y(X, y, accept_sparse=True)
        # augument x
        N, n = X.shape
        x_aug = np.hstack((-np.ones((N, 1)), X))
        x_aug = self.pre_process(x_aug)

        # create initial Z matrix
        Z = np.random.uniform(-0.5, 0.5, (n+1, self.p))
        self.Z_ = Z
        # apply activation function: tanh
        H = np.tanh(x_aug @ Z)

        # calculate weights
        self.coef_ = self.calculate_weights_default(y, H)
        
        return self

    def score(self, X, y):
        return r2_score(y, self.predict(X))

