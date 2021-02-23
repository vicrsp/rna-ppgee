import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats

from .regularization import ModelTunerCV
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, StratifiedKFold, KFold, GridSearchCV
from tqdm import tqdm


class CVFoldResult:
    def __init__(self, model_name, fold, opt_factor, score):
        self.model_name = model_name
        self.fold = fold
        self.opt_factor = opt_factor
        self.score = score

    def get_score(self):
        return (self.model_name, self.fold, self.score)


class ModelEvaluationExperiment:
    def __init__(self, datasets, model_type, n_splits=10, tuning_size=0.3, n_jobs=2):
        self.n_splits = n_splits
        self.tuning_size = tuning_size
        self.datasets = datasets
        self.model_type = model_type
        self.n_jobs = n_jobs

    def start(self, models):
        self.dataset_results = {}
        for name, data in tqdm(self.datasets):
            cv_results = self.k_fold_cross_validation(
                data, models, self.n_splits, self.tuning_size)
            self.dataset_results[name] = cv_results
        return self

    def k_fold_cross_validation(self, data, models, n_splits, tuning_size):
        X, y = data
        results = []
        kfold = StratifiedKFold(
            n_splits=n_splits) if self.model_type == 'classification' else KFold(n_splits=n_splits)
        self.cv_scores = []
        fold = 1
        for train_index, test_index in kfold.split(X, y):
            # create the T and k dataset
            X_t, X_k = X[train_index, :], X[test_index, :]
            y_t, y_k = y[train_index], y[test_index]
            for model_name, model, reg_factors in tqdm(models):
                opt_factor = np.nan
                if(reg_factors is not None):
                    opt_factor = self.tune_model(X_t, y_t, model, reg_factors)
                    model.set_params(reg_factor=opt_factor)
                # Train the tuned model on T
                model.fit(X_t, y_t)
                # Evalute the score on k and store the results
                results.append(CVFoldResult(model_name, fold,
                                            opt_factor, model.score(X_k, y_k)))
            fold += 1

        return results

    def tune_model(self, X_t, y_t, model, regularization_factors):
        cv = ModelTunerCV((X_t, y_t), model, regularization_factors, self.n_jobs).tune()
        opt_factor = cv.opt_param
        return opt_factor

    def plot_final_scores(self, path='/home/victor/git/rna-ppgee/artigo2/report/figures'):
        table = pd.DataFrame()
        
        for name in self.dataset_results.keys():
            results = self.dataset_results[name]
            score_data = []
            [score_data.append(row.get_score()) for row in results]

            score_data = pd.DataFrame(
                score_data, columns=['model_name', 'fold', 'score'])

            fig, ax = plt.subplots(figsize=(8, 6))
            sns.boxplot(data=score_data, x='model_name', y='score', ax=ax)
            ax.set_title(name)
            fig.savefig(f'{path}/{name}_scores')

            score_data['dataset'] = name
            table = pd.concat([score_data, table], ignore_index=True)

        table.to_csv(f'{path}/{self.model_type}_experiment_chart_data.csv',sep=",")

    def save_final_scores_table(self, path='/home/victor/git/rna-ppgee/artigo2/report/tables'):
        
        table = pd.DataFrame()

        for name in self.dataset_results.keys():
            results = self.dataset_results[name]
            score_data = []
            [score_data.append(row.get_score()) for row in results]

            score_data = pd.DataFrame(
                score_data, columns=['model_name', 'fold', 'score'])

            # calculate statistics
            def get_ci(values):
                ci_bs = bs.bootstrap(
                    values.to_numpy(), stat_func=bs_stats.mean, num_iterations=1000, return_distribution=False)
                return f'{ci_bs.lower_bound},{ci_bs.upper_bound},{ci_bs.value}'

            agg_data = score_data.groupby('model_name')['score'].agg(
                [np.mean, np.std, get_ci])

            agg_data[['lower_ci', 'upper_ci', 'average']] = agg_data.apply(lambda x: x['get_ci'].split(','), axis=1, result_type="expand")
            agg_data.drop(columns='get_ci', inplace=True)
            agg_data['dataset'] = name
            agg_data.reset_index(inplace=True)

            table = pd.concat([agg_data, table], ignore_index=True)

        table.to_csv(f'{path}/{self.model_type}_experiment_results.csv',sep=",")
        return table


def plot_scores(filename, ylabel, xlabel, save_path=None):
    data = pd.read_csv(filename, sep=',')

    for ds_name, ds_data in data.groupby('dataset'):
        fig, ax = plt.subplots(figsize=(8, 6))
        q1 = ds_data['score'].quantile(0.25)                 
        q3 = ds_data['score'].quantile(0.75)
        iqr = q3 - q1

        filter = (ds_data['score'] >= q1 - 1.5*iqr) & (ds_data['score'] <= q3 + 1.5*iqr)
        ds_data_filt = ds_data[filter]

        sns.boxplot(data=ds_data_filt, x='model_name', y='score', ax=ax, showfliers=False)
        sns.pointplot(data=ds_data_filt, x='model_name', y='score', ax=ax, join=False,capsize=.2,color="k")
        ax.set_title(ds_name)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        if(save_path is not None):
            fig.savefig(f'{save_path}/{ds_name}_scores')


def plot_ci(filename, ylabel, xlabel, save_path=None):
    data = pd.read_csv(filename, sep=',')

    for ds_name, ds_data in data.groupby('dataset'):
        fig, ax = plt.subplots(figsize=(8, 6))

        q1 = ds_data['score'].quantile(0.25)                 
        q3 = ds_data['score'].quantile(0.75)
        iqr = q3 - q1

        filter = (ds_data['score'] >= q1 - 1.5*iqr) & (ds_data['score'] <= q3 + 1.5*iqr)
        ds_data_filt = ds_data[filter]

        sns.pointplot(data=ds_data_filt, x='model_name', y='score', ax=ax, join=False,capsize=.2,color="k")
        sns.boxplot(data=ds_data, x='model_name', y='score', ax=ax, showfliers=False)
        ax.set_title(ds_name)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        # if(save_path is not None):
        #     fig.savefig(f'{save_path}/{ds_name}_scores')

def plot_ci_all(filename, ylabel, xlabel, save_path=None):
    data = pd.read_csv(filename, sep=',')

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.pointplot(data=data, x='model_name', y='score', hue='dataset', ax=ax, join=False,capsize=.2)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

def display_stats(filename, metric, save_path=None):
    # calculate statistics
    def get_ci(values):
        q1 = values.quantile(0.25)                 
        q3 = values.quantile(0.75)
        iqr = q3 - q1

        filter = (values >= q1 - 1.5*iqr) & (values <= q3 + 1.5*iqr)
        values_filt = values[filter]
        ci_bs = bs.bootstrap(
            values_filt.to_numpy(), stat_func=bs_stats.mean, num_iterations=1000, return_distribution=False)
        return f'{ci_bs.value:.3f} ({ci_bs.lower_bound:.3f},{ci_bs.upper_bound:.3f})'

    data = pd.read_csv(filename, sep=',')
    agg_data = data.groupby(['dataset','model_name'])['score'].agg(get_ci)
    agg_data = agg_data.reset_index().pivot('dataset', 'model_name')

    if(save_path is not None):
        agg_data.to_csv(f'{save_path}/{metric}_scores.csv')
    
    return agg_data