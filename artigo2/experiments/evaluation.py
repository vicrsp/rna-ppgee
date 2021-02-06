import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, StratifiedKFold
from tqdm import tqdm


def k_fold_cross_validation(datasets, models, scoring, n_splits=10):
    dataset_results = {}
    for name, data in tqdm(datasets):
        X, y = data
        results = {}
        kfold = StratifiedKFold(n_splits=n_splits)
        for model_name, model in tqdm(models):
            scores = cross_validate(model, X, y, cv=kfold, scoring=scoring)
            results[model_name] = {key: scores[f'test_{key}'] for key in scoring}
        dataset_results[name] = results

    return dataset_results


def plot_scores(results):
    n_axes = len(results.keys())
    fig, ax = plt.subplots(1, n_axes, figsize=(5 + n_axes, n_axes + 3))
    for index, score in enumerate(results.keys()):
        for model, values in results[score].items():
            if n_axes > 1:
                sns.distplot(values, ax=ax[index], label=model)
            else:
                sns.distplot(values, ax=ax, label=model)

        if n_axes > 1:
            ax[index].legend()
            ax[index].set_title(score)
        else:
            ax.legend()
            ax.set_title(score)

    return fig, ax

def plot_scores_per_dataset(dataset_results):
    for name in dataset_results.keys():
        fig, ax = plot_scores(dataset_results[name])
        ax.set_title(name)
        
        fig.tight_layout()
        fig.show()

        
