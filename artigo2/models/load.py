import numpy as np
from .elm import ELMClassifier, ELMRegressor
from .perceptron import PerceptronClassifier
from .rbf import RBFClassifier, RBFRegressor
from .adaline import Adaline
from .elm_hebbian import ELMHebbianClassifier, ELMHebbianRegressor

def get_classification_models():
    models = []
    models.append(('Perceptron', PerceptronClassifier(max_epochs=100), None))
    models.append(('ELM', ELMClassifier(p=20), np.linspace(0,1,50)))
    models.append(('RBF', RBFClassifier(p=20), np.linspace(0,1,50)))
    models.append(('ELMHebbian', ELMHebbianClassifier(p=10), None)) 
    return models

def get_regression_models():
    models = []
    models.append(('Adaline', Adaline(max_epochs=100), None))
    models.append(('ELM', ELMRegressor(p=20), np.linspace(0,1,50)))
    models.append(('RBF', RBFRegressor(p=20), np.linspace(0,1,50)))
    return models
