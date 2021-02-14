import numpy as np
import pandas as pd
import scipy as sp

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import roc_auc_score, mean_squared_error

class RBFClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, p=5):
        self.base = RBFRegressor(p)

    def predict(self, X):
        return np.sign(self.base.predict(X))

    def fit(self, X, y):
        self.base.fit(X, y)
        self.classes_ = unique_labels(y)
        return self

    def decision_function(self, X):
        return self.base.predict(X)

    def score(self, X, y):
        return roc_auc_score(y, self.predict(X))

    def set_regularization_factor(self, reg_factor):
        self.base.set_regularization_factor(reg_factor)


class RBFRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, p=5):
        self.p = p
        self.reg_factor = 0.0

    def apply_transformation(self, X):
        check_is_fitted(self, ['cov_', 'centers_'])
        N, n = X.shape
        H = np.zeros((N, self.p))
        for j in range(N):
            for i in range(self.p):
                mi = self.centers_[i, :]
                covi = self.cov_[i] + 0.001 * np.eye(n)
                H[j, i] = self.gaussian_kernel(X[j, :], mi, covi, n)
        return H

    def predict(self, X):
        # check X, y consistency
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, ['cov_', 'centers_', 'H_', 'coef_'])
        N, _ = X.shape

        H = self.apply_transformation(X)
        H_aug = np.hstack((np.ones((N, 1)), H))
        yhat = H_aug @ self.coef_

        return yhat

    def gaussian_kernel(self, X, m, K, n):
        center_distance = (X - m).reshape(-1, 1)
        normalization_factor = np.sqrt(((2*np.pi)**n)*sp.linalg.det(K))
        dist = float(
            np.exp(-0.5 * (center_distance.T @ (sp.linalg.inv(K)) @ center_distance)))
        return dist / normalization_factor

    def make_centers(self, X):
        kmeans = KMeans(n_clusters=self.p).fit(X)
        self.centers_ = kmeans.cluster_centers_
        # estimate covariance matrix for all centers
        clusters = kmeans.predict(X)
        covlist = []
        for i in range(self.p):
            xci = X[clusters == i, :]
            covi = np.cov(xci, rowvar=False)
            covlist.append(covi)
        self.cov_ = covlist

    def fit(self, X, y):
        # check X, y consistency
        X, y = check_X_y(X, y, accept_sparse=True)
        N, _ = X.shape

        # define centers
        self.make_centers(X)
        # calculate H matrix
        H = self.apply_transformation(X)
        H_aug = np.hstack((np.ones((N, 1)), H))

        # calculate the weight
        A = H_aug.T @ H_aug + np.eye(self.p + 1) * self.reg_factor
        w = np.linalg.inv(A) @ H_aug.T @ y

        self.coef_ = w
        self.H_ = H
        
        return self

    def set_regularization_factor(self, reg_factor):
        self.reg_factor = reg_factor

    def score(self, X, y):
        return mean_squared_error(y, self.predict(X))
