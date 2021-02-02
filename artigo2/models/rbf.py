import numpy as np
import pandas as pd
import scipy as sp

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score


class RBFClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, p, center_estimation='kmeans'):
        self.p = p
        self.center_estimation = center_estimation

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

        return np.sign(yhat)

    def gaussian_kernel(self, X, m, K, n):
        center_distance = (X - m).reshape(-1, 1)
        normalization_factor = np.sqrt(((2*np.pi)**n)*sp.linalg.det(K))
        dist = float(
            np.exp(-0.5 * (center_distance.T @ (sp.linalg.inv(K)) @ center_distance)))
        return dist / normalization_factor

    def make_centers(self, X):
        if(self.center_estimation == 'kmeans'):
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
        else:
            raise ValueError

    def fit(self, X, y):
        # check X, y consistency
        X, y = check_X_y(X, y, accept_sparse=True)
        N, _ = X.shape

        # define centers
        self.make_centers(X)
        # calculate H matrix
        H = self.apply_transformation(X)

        H_aug = np.hstack((np.ones((N, 1)), H))
        self.coef_ = (sp.linalg.inv(H_aug.T @ H_aug) @ H_aug.T) @ y
        self.H_ = H

        return self
