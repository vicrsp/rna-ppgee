import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

class ELMClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, p=5):
        self.p = p

    def predict(self, X):
        # input validation
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, ['w_','Z_'])

        N, _ = X.shape
        x_aug = np.hstack((-np.ones((N, 1)), X))
        H = np.tanh(x_aug @ self.Z_)
        u = np.sign(H @ self.w_)
        return u

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
        w = np.linalg.pinv(H) @ y
        # store fitted data
        self.w_ = w
        self.Z_ = Z

        return self