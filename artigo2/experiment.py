import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from models.elm import ELMClassifier, ELMRegressor
from models.perceptron import PerceptronClassifier
from models.rbf import RBFClassifier, RBFRegressor
from models.adaline import Adaline
from models.elm_hebbian import ELMHebbianClassifier, ELMHebbianRegressor
from experiments.evaluation import ModelEvaluationExperiment
from sklearn.datasets import load_digits, load_breast_cancer, load_wine, load_boston, load_diabetes, load_iris
from sklearn.preprocessing import MinMaxScaler

#%% Load the data
# breast_cancer: classificaçao
X_bc, y_bc = load_breast_cancer(return_X_y = True)
y_bc = pd.Series(y_bc).map({0:-1,1:1}).to_numpy()
# digits: classificação
X_dg, y_dg = load_digits(return_X_y = True)
y_dg = pd.Series(y_dg).map(lambda x: -1 if x == 8 else 1).to_numpy()
# iris: classificação
X_ir, y_ir = load_iris(return_X_y = True)
y_ir = pd.Series(y_ir).map({0:-1,1:1,2:1}).to_numpy()
# wine: regressão
X_wn, y_wn = load_wine(return_X_y = True)
y_wn = pd.Series(y_wn).map({0:-1,1:1,2:1}).to_numpy()
# boston: regressão
X_bo, y_bo = load_boston(return_X_y = True)
# diabetes: regressão
X_di, y_di = load_diabetes(return_X_y = True)

#%% Pre-process the data
# Normalization
X_bc = MinMaxScaler().fit_transform(X_bc)
X_dg = MinMaxScaler().fit_transform(X_dg)
X_ir = MinMaxScaler().fit_transform(X_ir)
X_wn = MinMaxScaler().fit_transform(X_wn)
X_bo = MinMaxScaler().fit_transform(X_bo)
X_di = MinMaxScaler().fit_transform(X_di)

#%% Experiment definition
# Classification
models_classification = []
models_classification.append(('Perceptron', PerceptronClassifier(max_epochs=200), None))
models_classification.append(('ELM', ELMClassifier(p=15), np.linspace(0,0.001,100)))
models_classification.append(('RBF', RBFClassifier(p=15), np.linspace(0,0.001,100)))
models_classification.append(('ELMHebbian', ELMHebbianClassifier(p=15), None)) 

datasets_classification = []
datasets_classification.append(('Breast Cancer', (X_bc, y_bc)))
datasets_classification.append(('Digitis (8)', (X_dg, y_dg)))
datasets_classification.append(('Iris', (X_ir, y_ir)))

# Regression
models_regression = []
models_regression.append(('Adaline', Adaline(max_epochs=200)))
models_regression.append(('ELM', ELMRegressor(p=5)))
models_regression.append(('RBF', RBFRegressor(p=10)))

datasets_regression = []
datasets_regression.append(('Wine', (X_wn, y_wn)))
datasets_regression.append(('Boston Housing', (X_bo, y_bo)))
datasets_regression.append(('Diabetes', (X_di, y_di)))


#%% Experimento run
experiment_classification = ModelEvaluationExperiment(datasets_classification)
experiment_classification.start(models_classification)

#%% Experiment analysis
experiment_classification.plot_final_scores()