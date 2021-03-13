from sklearn.linear_model import Ridge, Perceptron
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import roc_auc_score, r2_score, mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Perceptron
from kerneloptimizer.optimizer import KernelOptimizer

class KernelProjection:
    def __init__(self, kernel='gaussian'):
      self.kernel = kernel
      
    def fit(self, X, y):
        # check X, y consistency
        X, y = check_X_y(X, y, accept_sparse=True)
        if(self.kernel == 'mlp'):
            opt = KernelOptimizer(
                kernel='mlp',
                input_dim=X.shape[1],
                hidden_dim=20,
                output_dim=50
            )
        elif (self.kernel == 'gaussian'):
            opt = KernelOptimizer(kernel='gaussian')
        else:
            raise ValueError()

        # calculate the optimal projection
        opt.fit(X, y)  
        self.kernel_opt_ = opt
      
        # calculate the projection
        self.H_ = self.transform(X)
     
        return self

    def transform(self, X):
        return self.kernel_opt_.get_likelihood_space(X).to_numpy()
