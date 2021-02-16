import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV

class ModelTunerCV:
    def __init__(self, dataset, model, reg_factor):
        self.X, self.y = dataset
        self.model = model
        self.param_grid = {'reg_factor': reg_factor}

    def tune(self, folds=5):
        self.tuner = GridSearchCV(self.model, self.param_grid, cv=folds, n_jobs=2, verbose=3)
        self.tuner.fit(self.X, self.y)
        
        self.opt_param = self.tuner.best_params_['reg_factor']
        return self

    def plot(self, title):
        fig, ax = plt.subplots(figsize=(8,6))
        
        mean_data = self.tuner.cv_results_['mean_test_score']
        std_data = self.tuner.cv_results_['std_test_score']
        reg_factors = self.param_grid['reg_factor']

        ax.plot(reg_factors, mean_data, 'ko-')
        ax.fill_between(reg_factors, mean_data, mean_data + std_data, color='red', alpha=0.5)
        ax.fill_between(reg_factors, mean_data - std_data, mean_data, color='red', alpha=0.5)

        ax.set_title(f'{title} - {self.opt_param}')
        fig.show()



