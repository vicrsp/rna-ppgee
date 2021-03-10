import os
import pandas as pd
import numpy as np
from sklearn.datasets import load_boston, load_diabetes, load_breast_cancer
from sklearn.preprocessing import MinMaxScaler

path = os.path.join(os.path.dirname(__file__))

# Regression datasets
def load_boston_housing():
    X_bo, y_bo = load_boston(return_X_y = True)
    X_bo = MinMaxScaler().fit_transform(X_bo)
    y_bo = MinMaxScaler().fit_transform(y_bo.reshape(-1,1))
    
    return X_bo, y_bo.reshape(-1,)

def load_airfoil_noise():
    data = pd.read_csv(f'{path}/airfoil_self_noise.dat', sep='\t', header=None)
    data.dropna(inplace=True)
    data_np = data.to_numpy()
    data_np = MinMaxScaler().fit_transform(data_np)
    X, y = data_np[:,:-1], data_np[:,-1]
    return X, y

def load_diabetes_dataset():
    X, y = load_diabetes(return_X_y = True)
    X = MinMaxScaler().fit_transform(X)
    y = MinMaxScaler().fit_transform(y.reshape(-1,1)).ravel()
    
    return X, y

def load_red_wine_dataset():
    data = pd.read_csv(f'{path}/winequality-red.csv', sep=';')
    data.dropna(inplace=True)
    data_np = data.to_numpy()
    data_np = MinMaxScaler().fit_transform(data_np)
    X, y = data_np[:,:-1], data_np[:,-1]
    return X, y

def load_white_wine_dataset():
    data = pd.read_csv(f'{path}/winequality-white.csv', sep=';')
    data.dropna(inplace=True)
    data_np = data.to_numpy()
    data_np = MinMaxScaler().fit_transform(data_np)
    X, y = data_np[:,:-1], data_np[:,-1]
    return X, y

# Classification
# https://archive.ics.uci.edu/ml/datasets/liver+disorders
def load_bupa():
    data = pd.read_csv(f'{path}/bupa.data', sep=',', header=None)
    data.dropna(inplace=True)
    data_np = data.to_numpy()
    X, y = data_np[:,:-1], data_np[:,-1]
    X = MinMaxScaler().fit_transform(X)
    y = pd.Series(y).map({1:-1,2:1}).to_numpy()
    return X, y

# https://archive.ics.uci.edu/ml/datasets/climate+model+simulation+crashes
def load_pop_failures():
    data = pd.read_csv(f'{path}/pop_failures.csv', sep=',')
    data.dropna(inplace=True)
    data_np = data.to_numpy()
    X, y = data_np[:,2:-1], data_np[:,-1]
    X = MinMaxScaler().fit_transform(X)
    y = pd.Series(y).map({1:-1,2:1}).to_numpy()
    return X, y

# https://archive.ics.uci.edu/ml/datasets/statlog+(heart)
def load_heart_disease():
    data = pd.read_csv(f'{path}/heart.dat', sep=' ', header=None)
    data.dropna(inplace=True)
    data_np = data.to_numpy()
    X, y = data_np[:,:-1], data_np[:,-1]
    X = MinMaxScaler().fit_transform(X)
    y = pd.Series(y).map({1:-1,2:1}).to_numpy()
    return X,y

def load_breast_cancer_dataset():
    X_bc, y_bc = load_breast_cancer(return_X_y = True)
    y_bc = pd.Series(y_bc).map({0:-1,1:1}).to_numpy()
    X_bc = MinMaxScaler().fit_transform(X_bc)
    
    return X_bc, y_bc

def load_regression_datasets():
    datasets = []
    datasets.append(('Boston Housing', load_boston_housing()))
    datasets.append(('Wine Quality (Red)', load_red_wine_dataset()))
    datasets.append(('Diabetes', load_diabetes_dataset()))

    return datasets

def load_classification_datasets():
    datasets = []
    datasets.append(('Statlog (Heart)', load_heart_disease()))
    datasets.append(('Breast Cancer', load_breast_cancer_dataset()))
    datasets.append(('Liver Disorder', load_bupa()))

    return datasets