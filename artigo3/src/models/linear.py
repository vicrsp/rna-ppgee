import numpy as np
from .kernel import KernelProjection

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import Perceptron
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from abc import ABC, abstractmethod 

class KernelLinearModel(BaseEstimator, ClassifierMixin, ABC):
    def __init__(self, kernel):
        self.kernel = kernel

    @abstractmethod
    def calculate_weights(self, y, H):
        pass

    def fit(self, X, y):
        # check X, y consistency
        X, y = check_X_y(X, y, accept_sparse=True)
        # augument x
        N = X.shape[0]
        x_aug = np.hstack((-np.ones((N, 1)), X))
        # x_aug = self.pre_process(x_aug)
        # calculate the likelihood space
        self.kernel_projection_ = KernelProjection(kernel=self.kernel).fit(x_aug, y)
        # calculate the projected X
        H = self.kernel_projection_.H_
        # fit the model 
        self.coef_ = self.calculate_weights(y, H)

        self.classes_ = unique_labels(y)
        return self

    def decision_function(self, X):
        # check X, y consistency
        X = check_array(X, accept_sparse=True)
        N, _ = X.shape
        x_aug = np.hstack((-np.ones((N, 1)), X))
        H = self.kernel_projection_.transform(x_aug)
        return H @ self.coef_

    def predict(self, X):
        return np.sign(self.decision_function(X))

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))


class KernelPerceptron(KernelLinearModel):
    def __init__(self, kernel='gaussian'):
        super().__init__(kernel)

    def calculate_weights(self, y, H):
        return np.linalg.pinv(H) @ y

class KernelHebbian(KernelLinearModel):
    def __init__(self, kernel='gaussian'):
        super().__init__(kernel)

    def calculate_weights(self, y, H):
        w0 = y @ H
        w = w0 / np.linalg.norm(w0)
        return w
