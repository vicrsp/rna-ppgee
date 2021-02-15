import os
import pandas as pd
import numpy as np
from sklearn.datasets import load_boston, load_diabetes
from sklearn.preprocessing import MinMaxScaler

path = os.path.join(os.path.dirname(__file__))

# Regression datasets
def load_boston_housing():
    X_bo, y_bo = load_boston(return_X_y = True)
    X_bo = MinMaxScaler(feature_range=(-1,1)).fit_transform(X_bo)
    y_bo = MinMaxScaler(feature_range=(-1,1)).fit_transform(y_bo.reshape(-1,1))
    
    return X_bo, y_bo.reshape(-1,)

def load_airfoil_noise():
    data = pd.read_csv(f'{path}/airfoil_self_noise.dat', sep='\t', header=None)
    data.dropna(inplace=True)
    data_np = data.to_numpy()
    data_np = MinMaxScaler(feature_range=(-1,1)).fit_transform(data_np)
    X, y = data_np[:,:-1], data_np[:,-1]
    return X, y

def load_diabetes_dataset():
    X, y = load_diabetes(return_X_y = True)
    X = MinMaxScaler(feature_range=(-1,1)).fit_transform(X)
    y = MinMaxScaler(feature_range=(-1,1)).fit_transform(y)
    
    return X, y

def load_parkinsons():
    data = pd.read_csv(f'{path}/parkinsons.data', sep=',')
    data.dropna(inplace=True)
    data_np = data.to_numpy()[:,1:]
    data_np = MinMaxScaler(feature_range=(-1,1)).fit_transform(data_np)
    X, y = data_np[:,:-1], data_np[:,-1]
    return X, y

# Classification
# https://archive.ics.uci.edu/ml/datasets/liver+disorders
def load_bupa():
    data = pd.read_csv(f'{path}/bupa.data', sep=',', header=None)
    data.dropna(inplace=True)
    data_np = data.to_numpy()
    X, y = data_np[:,:-1], data_np[:,-1]
    X = MinMaxScaler(feature_range=(-1,1)).fit_transform(X)
    y = pd.Series(y).map({1:-1,2:1}).to_numpy()
    return X, y

# https://archive.ics.uci.edu/ml/datasets/climate+model+simulation+crashes
def load_pop_failures():
    data = pd.read_csv(f'{path}/pop_failures.csv', sep=',')
    data.dropna(inplace=True)
    data_np = data.to_numpy()
    X, y = data_np[:,2:-1], data_np[:,-1]
    X = MinMaxScaler(feature_range=(-1,1)).fit_transform(X)
    y = pd.Series(y).map({1:-1,2:1}).to_numpy()
    return X, y

# https://archive.ics.uci.edu/ml/datasets/statlog+(heart)
def load_heart_disease():
    data = pd.read_csv(f'{path}/heart.dat', sep=' ', header=None)
    data.dropna(inplace=True)
    data_np = data.to_numpy()
    X, y = data_np[:,:-1], data_np[:,-1]
    X = MinMaxScaler(feature_range=(-1,1)).fit_transform(X)
    y = pd.Series(y).map({1:-1,2:1}).to_numpy()
    return X,y


def load_regression_datasets():
    datasets = []
    datasets.append(('Boston Housing', load_boston_housing()))
    datasets.append(('Parkinsons', load_parkinsons()))
    datasets.append(('Airfoil Noise', load_airfoil_noise()))

    return datasets

def load_classification_datasets():
    datasets = []
    datasets.append(('Statlog (Heart)', load_heart_disease()))
    datasets.append(('Climate Model Simluation', load_pop_failures()))
    datasets.append(('Liver Disorder', load_bupa()))

    return datasets
