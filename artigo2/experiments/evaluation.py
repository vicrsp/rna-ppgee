import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, StratifiedKFold, KFold
from tqdm import tqdm


class CVFoldResult:
    def __init__(self, model_name, fold, opt_factor, score, tuning_data):
        self.model_name = model_name
        self.fold = fold
        self.opt_factor = opt_factor
        self.score = score
        self.tuning_data = tuning_data

    def get_tuning_data(self):
        if self.tuning_data is not None:
            df = pd.DataFrame(self.tuning_data, columns=['factor', 'metric'])
            df['model'] = self.model_name
            df['fold'] = self.fold
            return df.to_numpy()
        else:
            return np.array([np.nan, np.nan, self.model_name, self.fold])

    def get_score(self):
        return (self.model_name, self.fold, self.score)


class ModelEvaluationExperiment:
    def __init__(self, datasets, model_type, n_splits=10, tuning_size=0.3):
        self.n_splits = n_splits
        self.tuning_size = tuning_size
        self.datasets = datasets
        self.model_type = model_type

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
            # split T into T1 and T2
            X_t1, X_t2, y_t1, y_t2 = train_test_split(
                X_t, y_t, test_size=tuning_size, shuffle=True)
            for model_name, model, reg_factors in tqdm(models):
                opt_factor = np.nan
                tuning_data = None
                if(reg_factors is not None):
                    opt_factor, tuning_data = self.tune_model(
                        X_t1, X_t2, y_t1, y_t2, model, reg_factors)
                    model.set_regularization_factor(opt_factor)
                # Train the tuned model on T
                model.fit(X_t, y_t)
                # Evalute the score on k and store the results
                results.append(CVFoldResult(model_name, fold,
                                            opt_factor, model.score(X_k, y_k), tuning_data))
            fold += 1

        return results

    def tune_model(self, X_t1, X_t2, y_t1, y_t2, model, regularization_factors):
        scores = []
        for factor in tqdm(regularization_factors):
            # train regularized model on T1
            model.set_regularization_factor(factor)
            model.fit(X_t1, y_t1)
            # evaluate it on T2
            scores.append((factor, model.score(X_t2, y_t2)))
        # find the optimal factor
        scores_array = np.asarray(scores)
        factors, metric_value = scores_array[:, 0], scores_array[:, 1]
        i_max = np.argmax(metric_value)
        opt_factor = factors[i_max]

        return opt_factor, scores_array

    def plot_tuning_curves(self):
        for name in self.dataset_results.keys():
            results = self.dataset_results[name]
            # pd.DataFrame(columns=['factor', 'metric', 'model_name','fold'])
            tuning_data = []
            for row in results:
                for x in row.get_tuning_data():
                    tuning_data.append(x)

            tuning_data = pd.DataFrame(tuning_data, columns=[
                                       'factor', 'metric', 'model_name', 'fold'])

            plt.figure()
            sns.catplot(data=tuning_data, x='factor',
                        kind='box', y='metric', col='model_name')

    def plot_final_scores(self, path='/home/victor/git/rna-ppgee/artigo2/report/figures'):
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
