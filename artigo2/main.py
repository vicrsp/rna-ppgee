# %% Load packages
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import multiprocessing

from models.load import get_classification_models, get_regression_models
from experiments.evaluation import ModelEvaluationExperiment
from datasets.load import load_regression_datasets, load_classification_datasets


def run_classification_experiment():
    models_classification = get_classification_models()
    datasets_classification = load_classification_datasets()

    experiment_classification = ModelEvaluationExperiment(
        datasets_classification, 'classification', n_jobs=4)
    experiment_classification.start(models_classification)

    experiment_classification.plot_final_scores()
    experiment_classification.save_final_scores_table()


def run_regression_experiment():
    models_regression = get_regression_models()
    datasets_regression = load_regression_datasets()

    experiment_regression = ModelEvaluationExperiment(
        datasets_regression, 'regression', n_jobs=4)
    experiment_regression.start(models_regression)

    experiment_regression.plot_final_scores()
    experiment_regression.save_final_scores_table()


# jobs = [multiprocessing.Process(target=run_regression_experiment), multiprocessing.Process(
#     target=run_classification_experiment)]
# [job.start() for job in jobs]
# [job.join() for job in jobs]

run_regression_experiment()
#run_classification_experiment()