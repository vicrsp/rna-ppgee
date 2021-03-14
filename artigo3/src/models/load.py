import numpy as np
from .non_linear import ELM, KernelSVM, KernelRBF
from .linear import KernelPerceptron, KernelHebbian

def get_classification_models():
    models = []
    regularization_factors = [2 ** i for i in range(-5,12)]
    models.append(('ELM', ELM(p=100), regularization_factors))
    models.append(('GK-H', KernelHebbian('gaussian'), None)) 
    models.append(('GK-P', KernelPerceptron('gaussian'), None))
    models.append(('MLPK-H', KernelHebbian('mlp'), None)) 
    models.append(('MLPK-P', KernelPerceptron('mlp'), None))
    models.append(('GK-SVM', KernelSVM(), regularization_factors))
    models.append(('GK-RBF', KernelRBF(p=100), regularization_factors))

    return models