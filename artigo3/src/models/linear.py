import numpy as np
from .kernel import KernelProjection
from sklearn.linear_model import Perceptron
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels


# class KernelPerceptron:
#     def __init__(self, reg_factor=0):
#         self.reg_factor = reg_factor

#     def fit(self, X, y):
#         # check X, y consistency
#         X, y = check_X_y(X, y, accept_sparse=True)
#         # calculate the likelihood space
#         self.kernel_ = KernelProjection(kernel='gaussian').fit(X, y)
#         # project X to likelihood space
#         X_proj = self.kernel_.H_
#         # fit the model in the projected space
#         self.model_ = Perceptron(penalty='l2',alpha=self.reg_factor).fit(X_proj, y)

#         self.classes_ = unique_labels(y)
#         return self

#     def predict(self, X):
#         # input validation
#         X = check_array(X, accept_sparse=True)
#         # calculate the projected X
#         X_proj = self.kernel_.transform(X)
#         return self.model_.predict(X_proj)
        
#     def decision_function(self, X):
#         return self.model_.decision_function(X)

#     def score(self, X, y):
#         return roc_auc_score(y, self.predict(X))

class KernelPerceptron:
    def __init__(self):
        pass

    def calculate_weights(self, y, H):
        return np.linalg.pinv(H) @ y

    def pre_process(self, X):
        self.scaler_ = StandardScaler().fit(X)
        return self.scaler_.transform(X)

    def fit(self, X, y):
        # check X, y consistency
        X, y = check_X_y(X, y, accept_sparse=True)
        # augument x
        N = X.shape[0]
        x_aug = np.hstack((-np.ones((N, 1)), X))
        # x_aug = self.pre_process(x_aug)
        # calculate the likelihood space
        self.kernel_ = KernelProjection(kernel='gaussian').fit(x_aug, y)
        # calculate the projected X
        H = self.kernel_.H_
        # fit the model 
        self.coef_ = self.calculate_weights(y, H)

        self.classes_ = unique_labels(y)
        return self

    def decision_function(self, X):
        # check X, y consistency
        X = check_array(X, accept_sparse=True)
        N, _ = X.shape
        x_aug = np.hstack((-np.ones((N, 1)), X))
        H = self.kernel_.transform(x_aug)
        return H @ self.coef_

    def predict(self, X):
        return np.sign(self.decision_function(X))

    def score(self, X, y):
        return roc_auc_score(y, self.predict(X))


class KernelHebbian:
    def __init__(self):
        pass

    def calculate_weights(self, y, H):
        w0 = y @ H
        w = w0 / np.linalg.norm(w0)
        return w

    def pre_process(self, X):
        self.scaler_ = StandardScaler().fit(X)
        return self.scaler_.transform(X)

    def fit(self, X, y):
        # check X, y consistency
        X, y = check_X_y(X, y, accept_sparse=True)
        # augument x
        N = X.shape[0]
        x_aug = np.hstack((-np.ones((N, 1)), X))
        # x_aug = self.pre_process(x_aug)
        # calculate the likelihood space
        self.kernel_ = KernelProjection(kernel='gaussian').fit(x_aug, y)
        # calculate the projected X
        H = self.kernel_.H_
        # fit the model 
        self.coef_ = self.calculate_weights(y, H)

        self.classes_ = unique_labels(y)
        return self

    def decision_function(self, X):
        # check X, y consistency
        X = check_array(X, accept_sparse=True)
        N, _ = X.shape
        x_aug = np.hstack((-np.ones((N, 1)), X))
        H = self.kernel_.transform(x_aug)
        return H @ self.coef_

    def predict(self, X):
        return np.sign(self.decision_function(X))

    def score(self, X, y):
        return roc_auc_score(y, self.predict(X))