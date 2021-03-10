from sklearn.linear_model import Ridge, Perceptron
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import roc_auc_score, r2_score, mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Perceptron
from kernel_optimizer.optimizer import KernelOptimizer

class KernelClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, model, kernel='gaussian'):
      self.kernel = kernel
      self.model = model
      
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
        opt.fit(X, y)
        kernel_matrix = self.transform(X,y)
        
        # fit the model on the projected space
        self.model.fit(kernel_matrix, y)        

        self.kernel_opt = opt
        self.kernel_matrix = kernel_matrix

        return self

    def transform(self, X, y):
        return self.kernel_opt.get_likelihood_space(X,y)

    def predict(self, X):
        return self.model.predict(self.kernel_opt.to_likelihood_space(X))

    def decision_function(self, X):
        return self.model.decision_function(X)

    def score(self, X, y):
        return roc_auc_score(y, self.predict(X))