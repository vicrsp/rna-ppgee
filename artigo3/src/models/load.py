import numpy as np
from .non_linear import ELM, KernelSVM
from .linear import KernelPerceptron, KernelHebbian

def get_classification_models():
    models = []
    regularization_factors = [2 ** i for i in range(-5,8)]
    models.append(('ELM', ELM(p=30), regularization_factors))
    models.append(('GaussianKernelHebbian', KernelHebbian('gaussian'), None)) 
    models.append(('GaussianKernelPerceptron', KernelPerceptron('gaussian'), None))
    models.append(('MLPKernelHebbian', KernelHebbian('mlp'), None)) 
    models.append(('MLPKernelPerceptron', KernelPerceptron('mlp'), None))
    models.append(('KernelSVM', KernelSVM(), regularization_factors))

    return models