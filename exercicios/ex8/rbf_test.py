import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import train_test_split

from tqdm import tqdm
import random

class RBF(BaseEstimator):
    def __init__(self, p, center_estimation='kmeans'):
        self.p = p
        self.center_estimation = center_estimation

    def apply_transformation(self, X):
        check_is_fitted(self, ['cov_', 'centers_'])
        N, n = X.shape
        H = np.zeros((N, self.p))
        for j in range(N):
            for i in range(self.p):
                mi = self.centers_[i, :]
                covi = self.cov_[i] + 0.001 * np.eye(n)
                H[j, i] = self.gaussian_kernel(X[j, :], mi, covi, n)
        return H

    def predict(self, X):
        # check X, y consistency
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, ['cov_', 'centers_', 'H_', 'coef_'])
        N, _ = X.shape

        H = self.apply_transformation(X)
        H_aug = np.hstack((np.ones((N, 1)), H))
        yhat = H_aug @ self.coef_

        return yhat

    def gaussian_kernel(self, X, m, K, n):
        if n == 1:
            r = np.sqrt(float(K))
            px = (1/(np.sqrt(2*np.pi*r*r)))*np.exp(-0.5 * (float(X-m)/r)**2)
            return px
        else:
            center_distance = (X - m).reshape(-1, 1)
            normalization_factor = np.sqrt(((2*np.pi)**n)*sp.linalg.det(K))
            dist = float(
                np.exp(-0.5 * (center_distance.T @ (sp.linalg.inv(K)) @ center_distance)))
            return dist / normalization_factor
            
    def make_centers(self, X):
        N, n = X.shape

        if(self.center_estimation == 'kmeans'):
            kmeans = KMeans(n_clusters=self.p).fit(X)
            self.centers_ = kmeans.cluster_centers_
            # estimate covariance matrix for all centers
            clusters = kmeans.predict(X)
            covlist = []
            for i in range(self.p):
                xci = X[clusters == i, :]
                covi = np.cov(xci, rowvar=False) if n > 1 else np.asarray(
                    np.var(xci))
                covlist.append(covi)
            self.cov_ = covlist
        elif (self.center_estimation == 'random'):
            random_idx = random.sample(range(N), self.p)
            # define the maximum radius possible from the data
            data_center = np.mean(X, axis=0).reshape(1,-1)
            max_radius = np.max(np.linalg.norm(X - data_center, axis=1))
            # Now define random radius values
            random_radius = np.random.uniform(max_radius * 0.5, max_radius * 0.8, self.p)

            covlist = []
            centers = []
            for i in range(self.p):
                # Get the two random points from list
                center = X[random_idx[i], :]
                points_within_radius = np.linalg.norm(X - center, axis=1) <= random_radius[i]
                xci = X[points_within_radius, :]
                covi = np.cov(xci, rowvar=False) if n > 1 else np.asarray(np.var(xci))

                covlist.append(covi)
                centers.append(center)
            self.cov_ = covlist
            self.centers_ = np.asarray(centers)

        else:
            raise ValueError

    def fit(self, X, y):
        # check X, y consistency
        X, y = check_X_y(X, y, accept_sparse=True)
        N, _ = X.shape

        # define centers
        self.make_centers(X)
        # calculate H matrix
        H = self.apply_transformation(X)

        H_aug = np.hstack((np.ones((N, 1)), H))
        self.coef_ = (sp.linalg.inv(H_aug.T @ H_aug) @ H_aug.T) @ y
        self.H_ = H

        return self


# load the statlog(heart) dataset
heart_df = pd.read_csv('/home/victor/git/rna-ppgee/exercicios/ex8/heart.dat', sep=' ', header=None)
X_hd, y_hd = heart_df.iloc[:,:-1].to_numpy(), heart_df.iloc[:,-1].map({2:1, 1:-1}).to_numpy()
# scale the data
scaler_hd = MinMaxScaler()
X_hd = scaler_hd.fit_transform(X_hd)


def rbf_experiment_multiple_neurons(X_train, y_train, X_test, y_test, neurons, reps=2):
    accuracy_results = []
    accuracy_results_train = []
    for p in tqdm(neurons):
        accuracy_test = []
        accuracy_train = []
        for rep in tqdm(range(reps)):
            model = RBF(p=p,center_estimation='random').fit(X_train, y_train)
            yhat = np.sign(model.predict(X_test))
            yhat_train = np.sign(model.predict(X_train))
            # accuracy
            accuracy_test.append(accuracy_score(y_test, yhat))
            accuracy_train.append(accuracy_score(y_train, yhat_train))
            
        accuracy_results.append(np.mean(accuracy_test))
        accuracy_results_train.append(np.mean(accuracy_train))
    return accuracy_results, accuracy_results_train

def rbf_experiment_dataset_split(X, y, N=30, neurons=[5,10,30,50,100,300], plot=True):
    experiment_values_test = []
    experiment_values_train = []
    data_sets = {}
    for i in tqdm(range(N)):
        # split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)
        data_sets[i] = (X_train, y_train, X_test, y_test)
        # run for every neuron
        test_values, train_values = rbf_experiment_multiple_neurons(X_train, y_train, X_test, y_test, neurons, reps=1)

        experiment_values_test.append(test_values)
        experiment_values_train.append(train_values)

    def convert_results(res, set): 
        df = pd.DataFrame(res, columns=neurons).melt(value_vars=neurons,value_name='Acurácia', var_name='Neurônios')
        df['Conjunto'] = set
        return df

    train_values_df = convert_results(experiment_values_train, 'Treino')
    test_values_df = convert_results(experiment_values_test, 'Teste')
    
    experiment_values_df = pd.concat([train_values_df, test_values_df], ignore_index=True)
    # clear_output(wait=True)
    if (plot==True):
        # Plot the accuracy boxplots
        plt.figure(figsize=(10,6))
        sns.boxplot(data=experiment_values_df, x='Conjunto',y='Acurácia', hue='Neurônios', showfliers=False)

        # # Print the results table
        # display('Tabela: Acurácia Média e Desvio Padrão')
        # display(experiment_values_df.groupby(['Neurônios','Conjunto'])['Acurácia'].agg([np.mean, np.std]))

    return experiment_values_df, data_sets 

results_hd, datasets_hd = rbf_experiment_dataset_split(X_hd, y_hd,neurons=[150], N=10)