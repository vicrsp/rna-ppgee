import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class PerceptronClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, eta=0.01, max_epochs=100):
        self.eta = eta
        self.max_epochs = max_epochs

    def predict(self, X):
        # input validation
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, 'w_')

        N, _ = X.shape
        X_aug = np.hstack((-np.ones((N, 1)), X))
        u = X_aug @ self.w_
        return np.sign(u)

    def fit(self, X, y):
        # check X, y consistency
        X, y = check_X_y(X, y, accept_sparse=True)

        # initialize the weight matrix
        N, n = X.shape

        # add the bias term
        x_aug = np.hstack((-np.ones((N, 1)), X))

        wt = np.random.rand(n+1) - 0.5
        n_epochs = 0

        while(n_epochs < self.max_epochs):
            # generate random indexes order
            xseq = np.arange(N)
            np.random.shuffle(xseq)

            for i_rand in xseq:
                yhati = np.sign(x_aug[i_rand, :] @ wt)
                ei = y[i_rand] - yhati
                # calculate step size
                dw = self.eta * ei * x_aug[i_rand, :]
                # update weight vector
                wt = wt + dw
            # increment number of epochs
            n_epochs = n_epochs + 1

        self.w_ = wt
        return self        
