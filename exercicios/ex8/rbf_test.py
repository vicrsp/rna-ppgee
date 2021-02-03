import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp

from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import random


class RBF(BaseEstimator):
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

        return yhat

    def gaussian_kernel(self, X, m, K, n):
        if n == 1:
            r = np.sqrt(float(K))
            px = (1/(np.sqrt(2*np.pi*r*r)))*np.exp(-0.5 * (float(X-m)/r)**2)
            return px
        else:
            center_distance = (X - m).reshape(-1, 1)
            normalization_factor = np.sqrt(((2*np.pi)**n)*sp.linalg.det(K))
            dist = float(
                np.exp(-0.5 * (center_distance.T @ (sp.linalg.inv(K)) @ center_distance)))
            return dist / normalization_factor
            
    def make_centers(self, X):
        N, n = X.shape

        if(self.center_estimation == 'kmeans'):
            kmeans = KMeans(n_clusters=self.p).fit(X)
            self.centers_ = kmeans.cluster_centers_
            # estimate covariance matrix for all centers
            clusters = kmeans.predict(X)
            covlist = []
            for i in range(self.p):
                xci = X[clusters == i, :]
                covi = np.cov(xci, rowvar=False) if n > 1 else np.asarray(
                    np.var(xci))
                covlist.append(covi)
            self.cov_ = covlist
        elif (self.center_estimation == 'random'):
            random_idx = random.sample(range(N), self.p * 2)
            #random_idx = np.random.randint(0, N, size=(self.p * 2, 1))
            covlist = []
            centers = []
            for i in range(self.p):
                # Get the two random points from list
                x1 = X[random_idx[i], :]
                x2 = X[random_idx[i+1], :]
                center = (x1 + x2) / 2
                radius = np.linalg.norm(x1 - x2) / 2
                points_within_radius = np.linalg.norm(
                    X - center, axis=1) <= radius

                xci = X[points_within_radius, :]
                covi = np.cov(xci, rowvar=False) if n > 1 else np.asarray(
                    np.var(xci))

                covlist.append(covi)
                centers.append(center)
            self.cov_ = covlist
            self.centers_ = np.asarray(centers)

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


# define the sinc function generator
n_train = 100
n_test = 50

X_train = np.random.uniform(-15, 15, size=(n_train, 1))
y_train = np.sin(X_train) / X_train + \
    np.random.normal(loc=0, scale=0.05, size=(n_train, 1))

X_test = np.random.uniform(-15, 15, size=(n_test, 1))
y_test = np.sin(X_test) / X_test + np.random.normal(loc=0,
                                                    scale=0.05, size=(n_test, 1))


def plot_regression_results(X_train, y_train, X_test, y_test, p=5):
    fig, ax = plt.subplots(figsize=(6, 4))
    # horizontal stack vectors to create x1,x2 input for the model
    grid = np.arange(np.min(X_train)-0.1, np.max(X_train) +
                     0.1, 0.01).reshape(-1, 1)

    # train the model
    model = RBF(p=p, center_estimation='random').fit(X_train, y_train)

    # make predictions for the grid
    yhat_grid = model.predict(grid)
    # make predictions for the datasets
    yhat_test = model.predict(X_test)
    yhat_train = model.predict(X_train)

    mse_test = mean_squared_error(y_test, yhat_test.ravel())
    mse_train = mean_squared_error(y_train, yhat_train.ravel())

    ax.scatter(X_train, y_train, color='red', label='Treinamento')
    ax.scatter(X_train, y_train, color='blue', label='Teste')
    ax.plot(grid, yhat_grid, color='black', label='RBF')

    ax.set_xlabel('x')
    ax.set_ylabel('y')

    ax.legend()
    fig.suptitle(
        f'Neurônios:{p}\n MSE treino: {mse_train} \n MSE teste: {mse_test}')
    fig.tight_layout()
    fig.show()


plot_regression_results(X_train, y_train, X_test, y_test, p=50)
