import numpy as np
from .elm import ELMClassifier, ELMRegressor
from .rbf import RBFClassifier, RBFRegressor
from .elm_hebbian import ELMHebbianClassifier, ELMHebbianRegressor
from .linear import AdalineRegressor, PerceptronClassifier

def get_classification_models():
    models = []
    models.append(('Perceptron', PerceptronClassifier(), np.linspace(0,1,50)))
    models.append(('ELM', ELMClassifier(p=20), np.linspace(0,1,50)))
    models.append(('RBF', RBFClassifier(p=20), np.linspace(0,1,50)))
    models.append(('ELMHebbian', ELMHebbianClassifier(p=10), None)) 
    return models

def get_regression_models():
    models = []
    models.append(('Adaline', AdalineRegressor(), np.linspace(0,1,50)))
    models.append(('ELM', ELMRegressor(p=20), np.linspace(0,1,50)))
    models.append(('RBF', RBFRegressor(p=20), np.linspace(0,1,50)))
    return models
