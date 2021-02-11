import numpy as np
import random
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.preprocessing import StandardScaler

class ELMHebbianClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, p=5):
        self.base = ELMHebbianRegressor(p)

    def predict(self, X):
        return np.sign(self.base.predict(X))

    def fit(self, X, y):
        self.base.fit(X, y)
        return self

# http://persoal.citius.usc.es/manuel.fernandez.delgado/programs/dpp_two_class.c
class ELMHebbianRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, p=5):
        self.p = p

    def predict(self, X):
        # input validation
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, ['coef_', 'Z_'])

        N, _ = X.shape
        X_p = self.scaler_.transform(X)
        x_aug = np.hstack((-np.ones((N, 1)), X_p))
        H = np.tanh(x_aug @ self.Z_)
        return H @ self.coef_

    def pre_process(self, X):
        self.scaler_ = StandardScaler().fit(X)
        return self.scaler_.transform(X)

    def fit(self, X, y):
        # check X, y consistency
        X, y = check_X_y(X, y, accept_sparse=True)
        # standardize X
        #X_p = self.pre_process(X)
        # augment X
        # N, _ = X.shape
        # x_aug = np.hstack((-np.ones((N, 1)), X_p))

        # # desired outputs
        # n = self.p / 2
        # m = (n - 1) / 2 
        # w0 = np.zeros((self.p, ))
        # for k in range(N):
        #     dik = y[k] ** n
        #     zeta = random.sample(range(n), m)
        #     dik[zeta] = -dik[zeta]
        #     w0[k]

        # ELMP
        # random_idx = np.arange(N)
        # np.random.shuffle(random_idx 
        # w = np.zeros((self.p,))
        # for idx in random_idx:



        # m = int(N * 0.7)
        # idx_m = random.sample(range(N), m)
        # x_m = x_aug[idx_m, :]
        # y_m = y[idx_m]
        
        # Z = np.random.uniform(-0.5, 0.5, (n+1, self.p))
        # H_m = np.tanh(x_m @ Z)
        # w = y_m @ H_m
        # w = w / np.linalg.norm(w)
        
        # beta = np.max(np.linalg.norm(H_m, axis=1) ** 2)
        # teta = np.max(np.abs(H_m @ w))

        # k = 0
        # while (m < (N-1)) & (k < N * 10):
        #     ridx = random.sample(range(N), 1)[0]
        #     x_mi = x_aug[ridx, :]
        #     H_mi = np.tanh(x_mi @ Z)
            
        #     alfa = abs(w.T @ H_mi)
        #     beta = max(np.max(np.linalg.norm(H_mi) ** 2), beta)
        #     teta = max(np.max(np.abs(H_mi @ w)), teta)
        #     t = (beta + 2*teta) / alfa ** 2

        #     if(t > m):
        #         print(f'm: {m}')
        #         y_mi = y[ridx]
        #         wk = y_mi * H_mi + w
        #         w = wk / np.linalg.norm(wk)
        #         m = m + 1

        #     k += 1

        # # create initial Z matrix
        # Z = np.random.uniform(-0.5, 0.5, (n+1, self.p))
        # # apply activation function: tanh
        # H = np.tanh(x_aug @ Z)
        # # calculate weights
        # w0 = y @ H
        # w = w0 / np.linalg.norm(w0)
        
        # OJA
        # augument x
        N, n = X.shape
        X_p = self.pre_process(X)
        x_aug = np.hstack((-np.ones((N, 1)), X_p))
        # create initial Z matrix
        Z = np.random.uniform(-0.5, 0.5, (n+1, self.p))
        # apply activation function: tanh
        H = np.tanh(x_aug @ Z)
        # calculate weights
        w = np.random.rand(self.p)
        for _ in range(100):
            y_k = H @ w
            w0 = y_k @ H
            #w0 = w0 / np.linalg.norm(w0)

            w = w + 0.001 * w0
        self.coef_ = w
        self.Z_ = Z

        return self