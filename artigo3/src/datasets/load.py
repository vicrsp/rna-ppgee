import os
import pandas as pd
import numpy as np
from sklearn.datasets import load_boston, load_diabetes, load_breast_cancer
from sklearn.preprocessing import MinMaxScaler

path = os.path.join(os.path.dirname(__file__))


def read_dat_file(filepath):
    with open(filepath, 'r') as file:
        lines = [line.rstrip('\n') for line in file]
        data_start = lines.index('@data') + 1

        X = np.asarray([np.fromstring(line,sep=',',count=(len(line.split(','))-1)) for line in lines[data_start:]])
        y = np.asarray([line.split(',')[-1].strip() for line in lines[data_start:]])
        return X, y

def load_sonar_uci():
    data = pd.read_csv(f'{path}/sonar.all-data', sep=',', header=None)
    data.dropna(inplace=True)
    data_np = data.to_numpy()
    X, y = data_np[:,:-1], data_np[:,-1]
    X = MinMaxScaler().fit_transform(X)
    y = pd.Series(y).map({'R':-1,'M':1}).to_numpy()
    return X, y

def load_sonar():
    X, y = read_dat_file(f'{path}/sonar.dat')
    X = MinMaxScaler().fit_transform(X)
    y = pd.Series(y).map({'R':-1,'M':1}).to_numpy()
    return X, y

def load_appendicitis():
    X, y = read_dat_file(f'{path}/appendicitis.dat')
    X = MinMaxScaler().fit_transform(X)
    y = pd.Series(y).map({'0':-1,'1':1}).to_numpy()
    return X, y

def load_australian():
    X, y = read_dat_file(f'{path}/australian.dat')
    X = MinMaxScaler().fit_transform(X)
    y = pd.Series(y).map({'0':-1,'1':1}).to_numpy()
    return X, y

def load_haberman():
    X, y = read_dat_file(f'{path}/haberman.dat')
    X = MinMaxScaler().fit_transform(X)
    y = pd.Series(y).map({'negative':-1,'positive':1}).to_numpy()
    return X, y

def load_glass():
    X, y = read_dat_file(f'{path}/glass.dat')
    X = MinMaxScaler().fit_transform(X)
    y = pd.Series(y).map(lambda x: 1 if x == '7' else -1).to_numpy()
    return X, y

def load_segmentation():
    X, y = read_dat_file(f'{path}/segment.dat')
    X = MinMaxScaler().fit_transform(X)
    y = pd.Series(y).map(lambda x: 1 if x == '1' else -1).to_numpy()
    return X, y

def load_pima():
    X, y = read_dat_file(f'{path}/pima.dat')
    X = MinMaxScaler().fit_transform(X)
    y = pd.Series(y).map({'tested_negative':-1,'tested_positive':1}).to_numpy()
    return X, y

def load_ionosphere():
    X, y = read_dat_file(f'{path}/ionosphere.dat')
    X = MinMaxScaler().fit_transform(X)
    y = pd.Series(y).map({'g':-1,'b':1}).to_numpy()
    return X, y

# https://archive.ics.uci.edu/ml/datasets/liver+disorders
def load_bupa():
    data = pd.read_csv(f'{path}/bupa.data', sep=',', header=None)
    data.dropna(inplace=True)
    data_np = data.to_numpy()
    X, y = data_np[:,:-1], data_np[:,-1]
    X = MinMaxScaler().fit_transform(X)
    y = pd.Series(y).map({1:-1,2:1}).to_numpy()
    return X, y

def load_fertility():
    data = pd.read_csv(f'{path}/fertility.txt', sep=',', header=None)
    data.dropna(inplace=True)
    data_np = data.to_numpy()
    X, y = data_np[:,:-1], data_np[:,-1]
    X = MinMaxScaler().fit_transform(X)
    y = pd.Series(y).map({'N':-1,'O':1}).to_numpy()
    return X, y

def load_banknote():
    data = pd.read_csv(f'{path}/banknote.txt', sep=',', header=None)
    data.dropna(inplace=True)
    data_np = data.to_numpy()
    X, y = data_np[:,:-1], data_np[:,-1]
    X = MinMaxScaler().fit_transform(X)
    y = pd.Series(y).map({0:-1,1:1}).to_numpy()
    return X, y

# https://archive.ics.uci.edu/ml/datasets/climate+model+simulation+crashes
def load_pop_failures():
    data = pd.read_csv(f'{path}/pop_failures.csv', sep=',')
    data.dropna(inplace=True)
    data_np = data.to_numpy()
    # ignoring experiment metadata columns
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

# https://archive.ics.uci.edu/ml/datasets/climate+model+simulation+crashes
def load_ilpd():
    data = pd.read_csv(f'{path}/ilpd.csv', sep=',')
    data.dropna(inplace=True)
    data.iloc[:,1] = data.iloc[:,1].map(lambda x: 0 if x=='Male' else 1)
    data_np = data.to_numpy()
    X, y = data_np[:,:-1], data_np[:,-1]
    X = MinMaxScaler().fit_transform(X)
    y = pd.Series(y).map({1:-1,2:1}).to_numpy()
    return X, y


def load_classification_datasets():
    datasets = []
    datasets.append(('appendicitis', load_appendicitis()))
    datasets.append(('australian', load_australian()))
    datasets.append(('banknote', load_banknote()))
    datasets.append(('breastcancer', load_breast_cancer_dataset()))
    datasets.append(('bupa', load_bupa()))
    datasets.append(('climate', load_pop_failures()))
    datasets.append(('fertility', load_fertility()))
    datasets.append(('glass', load_glass()))
    datasets.append(('haberman', load_haberman()))
    datasets.append(('heart', load_heart_disease()))
    datasets.append(('ILPD', load_ilpd()))
    datasets.append(('ionosphere', load_ionosphere()))
    datasets.append(('sonar', load_sonar()))
    datasets.append(('segmentation', load_segmentation()))
    datasets.append(('pima', load_pima()))

    return datasets


def print_stats(datasets):
    stats = []
    for name, ds in datasets:
        X, y = ds
        yp = pd.Categorical(y)
        stats.append((name, X.shape[0], X.shape[1],  f'{yp.describe()["freqs"][0]:.2f}/{yp.describe()["freqs"][1]:.2f}'))

    print(pd.DataFrame(stats, columns=['Dataset','Instâncias','Atributos','Proporção']))

print_stats(load_classification_datasets())