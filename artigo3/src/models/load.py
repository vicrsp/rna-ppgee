import numpy as np
from .non_linear import ELM, KernelSVM
from .linear import KernelPerceptron, KernelHebbian

def get_classification_models():
    models = []
    regularization_factors = [2 ** i for i in range(-5,8)]
    models.append(('ELM', ELM(p=30), regularization_factors))
    models.append(('KernelHebbian', KernelHebbian(), None)) 
    models.append(('KernelPerceptron', KernelPerceptron(), None))
    models.append(('KernelSVM', KernelSVM(), regularization_factors))

    return models