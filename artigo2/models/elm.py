import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class ELMClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, p=5, reg_factor=None):
        self.base = ELMRegressor(p, reg_factor)

    def predict(self, X):
        return np.sign(self.base.predict(X))

    def fit(self, X, y):
        self.base.fit(X, y)
        return self


class ELMRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, p=5, reg_factor=None):
        self.p = p
        self.reg_factor = reg_factor

    def predict(self, X):
        # input validation
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, ['coef_', 'Z_'])

        N, _ = X.shape
        x_aug = np.hstack((-np.ones((N, 1)), X))
        H = np.tanh(x_aug @ self.Z_)
        return H @ self.coef_

    def fit(self, X, y):
        # check X, y consistency
        X, y = check_X_y(X, y, accept_sparse=True)
        # augment X
        N, n = X.shape
        x_aug = np.hstack((-np.ones((N, 1)), X))
        # create initial Z matrix
        Z = np.random.uniform(-0.5, 0.5, (n+1, self.p))
        # apply activation function: tanh
        H = np.tanh(x_aug @ Z)
        # calculate the weights
        if(self.reg_factor == None):
            w = np.linalg.pinv(H) @ y
        else:
            A = H.T @ H + np.eye(self.p) * self.reg_factor
            w = np.linalg.inv(A) @ H.T @ y
        # store fitted data
        self.coef_ = w
        self.Z_ = Z

        return self
