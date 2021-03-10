from sklearn.linear_model import Ridge, Perceptron
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import roc_auc_score, r2_score, mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Perceptron


class AdalineRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, reg_factor=0):
        self.reg_factor = reg_factor
        self.base = MLPRegressor(hidden_layer_sizes=(
            1,), activation='identity', solver='sgd', alpha=reg_factor)

    def predict(self, X):
        return self.base.predict(X)

    def fit(self, X, y):
        self.base.fit(X, y)
        return self
    
    def score(self, X, y):
        return mean_squared_error(y, self.predict(X))

class PerceptronClassifier(BaseEstimator, RegressorMixin):
    def __init__(self, reg_factor=0):
        self.reg_factor = reg_factor
        self.base = Perceptron(penalty='l2', alpha=reg_factor)

    def predict(self, X):
        return self.base.predict(X)

    def fit(self, X, y):
        self.base.fit(X, y)
        self.classes_ = unique_labels(y)
        return self
    
    def decision_function(self, X):
        return self.base.decision_function(X)

    def score(self, X, y):
        return roc_auc_score(y, self.predict(X))