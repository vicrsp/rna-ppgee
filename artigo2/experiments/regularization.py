import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, StratifiedKFold
from tqdm import tqdm

def regularization_experiment():
    pass


def k_fold_cross_validation(datasets, models, scoring, n_splits=10):
    dataset_results = {}
    # For each dataset
    for name, data in tqdm(datasets):
        X, y = data
        results = {}
        # Create k stratified folds
        kfold = StratifiedKFold(n_splits=n_splits)
        for model_name, model in tqdm(models):
            scores = cross_validate(model, X, y, cv=kfold, scoring=scoring)
            results[model_name] = {key: scores[f'test_{key}'] for key in scoring}
        dataset_results[name] = results

    return dataset_results