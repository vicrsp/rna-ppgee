import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from models.load import get_classification_models, get_regression_models
from experiments.evaluation import ModelEvaluationExperiment
from datasets.load import load_regression_datasets, load_classification_datasets

#%% Load experiment definitions
# Classification
models_classification = get_classification_models()
datasets_classification = load_classification_datasets()

# Regression
models_regression = get_regression_models()
datasets_regression = load_regression_datasets()

#%% Experiment run
experiment_classification = ModelEvaluationExperiment(datasets_classification, 'classification')
experiment_classification.start(models_classification)

#%% Experiment analysis
experiment_classification.plot_final_scores()

#%%
experiment_regression = ModelEvaluationExperiment(datasets_classification, 'regression')
experiment_regression.start(models_regression)

#%% Experiment analysis
experiment_regression.plot_final_scores()