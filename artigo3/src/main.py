# %% Load packages
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from models.load import get_classification_models
from experiments.evaluation import ModelEvaluationExperiment
from datasets.load import load_classification_datasets

def run_classification_experiment():
    models_classification = get_classification_models()
    datasets_classification = load_classification_datasets()

    experiment_classification = ModelEvaluationExperiment(
        datasets_classification, 'classification', n_jobs=4)
    experiment_classification.start(models_classification, cv_type='random')

    experiment_classification.plot_final_scores('/home/victor/git/rna-ppgee/artigo3/report/raw')
    experiment_classification.save_final_scores_table('/home/victor/git/rna-ppgee/artigo3/report/raw')#%% Run experiment


# %%
run_classification_experiment()
