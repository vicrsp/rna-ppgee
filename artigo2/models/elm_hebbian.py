import numpy as np
import random
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.preprocessing import StandardScaler


class ELMHebbianClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, p=5, hebbian_rule='oja'):
        self.base = ELMHebbianRegressor(p, hebbian_rule)

    def predict(self, X):
        return np.sign(self.base.predict(X))

    def fit(self, X, y):
        self.base.fit(X, y)
        return self

# http://persoal.citius.usc.es/manuel.fernandez.delgado/programs/dpp_two_class.c


class ELMHebbianRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, p=5, hebbian_rule='oja'):
        self.p = p
        self.hebbian_rule = hebbian_rule

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

    def calculate_weights_oja_rule(self, y, H):
        # calculate weights
        w = np.random.uniform(-0.5, 0.5, (self.p,))

        eta = 0.001
        tol = 1e-4
        delta = np.Inf

        iters = 0
        while delta > tol:
            dw = eta * y @ (H - y.reshape(-1, 1) @ w.reshape(-1, 1).T)
            w_old = w
            w = w + dw
            delta = np.linalg.norm(w - w_old)

            iters += 1
            if(iters > 1000):
                print('No convergence for hebbian perceptron!')
                break
        return w

    def calculate_weights_spp(self, y, H):
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
        if self.hebbian_rule == 'oja':
            self.coef_ = self.calculate_weights_oja_rule(y, H)
        elif self.hebbian_rule == 'spp':
            self.coef_ = self.calculate_weights_spp(y, H)
        else:
            raise ValueError('Invalid hebbian_rule!')

        return self
