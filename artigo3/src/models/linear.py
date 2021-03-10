from sklearn.linear_model import Ridge, Perceptron
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import roc_auc_score, r2_score, mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Perceptron
from kerneloptimizer.optimizer import KernelOptimizer

class PerceptronClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, kernel, reg_factor=0):
        self.kernel = kernel
        self.reg_factor = reg_factor
        self.base = Perceptron(penalty='l2', alpha=reg_factor)

    def predict(self, X):
        
        return self.base.predict(X)

    def fit(self, X, y):
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
        opt.fit(X,y)
        X_proj = opt.get_likelihood_space(X,y)
        
        # and train in the projected space
        self.base.fit(X_proj, y)
        self.classes_ = unique_labels(y)
        self.opt_ = opt
        return self
    
    def decision_function(self, X):
        return self.base.decision_function(X)

    def score(self, X, y):
        return roc_auc_score(y, self.predict(X))