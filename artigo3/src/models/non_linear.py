import numpy as np
from .kernel import KernelProjection
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

class KernelSVM(BaseEstimator, ClassifierMixin):
    def __init__(self, reg_factor=1.0):
        self.reg_factor = reg_factor

    def fit(self, X, y):
        # check X, y consistency
        X, y = check_X_y(X, y, accept_sparse=True)
        # find the optimal scale
        self.kernel_ = KernelProjection(kernel='gaussian').fit(X, y)
        # fit the model with the scale
        self.model_ = SVC(C=self.reg_factor, gamma=self.kernel_.kernel_opt_.width).fit(X, y)

        self.classes_ = unique_labels(y)
        return self

    def predict(self, X):
        return self.model_.predict(X)
        
    def decision_function(self, X):
        return self.model_.decision_function(X)

    def score(self, X, y):
        return roc_auc_score(y, self.predict(X))


class ELM(BaseEstimator, ClassifierMixin):
    def __init__(self, p=5, reg_factor=0.0):
        self.reg_factor = reg_factor
        self.p = p

    
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
        A = H.T @ H + np.eye(self.p) * self.reg_factor
        w = np.linalg.inv(A) @ H.T @ y
        # store fitted data
        self.coef_ = w
        self.Z_ = Z
        self.classes_ = unique_labels(y)
        return self
    
    def decision_function(self, X):
        # input validation
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, ['coef_', 'Z_'])

        N, _ = X.shape
        x_aug = np.hstack((-np.ones((N, 1)), X))
        H = np.tanh(x_aug @ self.Z_)
        return H @ self.coef_

    def predict(self, X):
        return np.sign(self.decision_function(X))

    def score(self, X, y):
        return roc_auc_score(y, self.predict(X))