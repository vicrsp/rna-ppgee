import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import r2_score

class Adaline(BaseEstimator,RegressorMixin):
    def __init__(self, eta=0.01, tol=1e-6, max_epochs=100):
        self.eta = eta
        self.tol = tol
        self.max_epochs = max_epochs

    def predict(self, X):
        # input validation
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, ['coef_'])

        return X @ self.coef_

    def fit(self, X, y):
        # check X, y consistency
        X, y = check_X_y(X, y, accept_sparse=True)
        
        N, m = X.shape
        epochs = 0
        w = np.random.rand(m)
        error_epoch = []
        ediff = np.Inf

        while((epochs < self.max_epochs) & (ediff > self.tol)):
            xseq = np.arange(N)
            np.random.shuffle(xseq)
            ei2 = 0
            for iseq in xseq:
                erro = (y[iseq] - X[iseq, :] @ w)
                w = w + self.eta * erro * X[iseq,:]
                ei2 = ei2 + erro ** 2

            ei2 = ei2 / N
            error_epoch.append(ei2)
            epochs = epochs + 1

            if(epochs > 1):
                ediff = np.abs(ediff - ei2)
            else:
                ediff = ei2

        self.coef_ = w
        return self

    def score(self, X, y):
        return r2_score(y, self.predict(X))
