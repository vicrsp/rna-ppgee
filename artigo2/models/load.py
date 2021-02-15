import numpy as np
from elm import ELMClassifier, ELMRegressor
from perceptron import PerceptronClassifier
from rbf import RBFClassifier, RBFRegressor
from adaline import Adaline
from elm_hebbian import ELMHebbianClassifier, ELMHebbianRegressor

import os
os.path.join(os.path.dirname(__file__), '../..')

def get_classification_models():
    models = []
    models.append(('Perceptron', PerceptronClassifier(max_epochs=200), None))
    models.append(('ELM', ELMClassifier(p=15), np.linspace(0,0.001,100)))
    # models.append(('RBF', RBFClassifier(p=15), np.linspace(0,0.001,100)))
    models.append(('ELMHebbian', ELMHebbianClassifier(p=15), None)) 
    return models


def get_regression_models():
    models = []
    models.append(('Adaline', Adaline(max_epochs=200), None))
    models.append(('ELM', ELMRegressor(p=5), None))
    # models.append(('RBF', RBFRegressor(p=10)))
    models.append(('ELMHebbian', ELMHebbianRegressor(p=15), None)) 
    return models
