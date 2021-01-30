import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from models.elm import ELMClassifier
from models.perceptron import PerceptronClassifier
from models.rbf import RBFClassifier
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix, make_scorer, f1_score
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, StratifiedKFold
from tqdm import tqdm

# load the statlog(heart) dataset
heart_df = pd.read_csv('datasets/heart.dat', sep=' ', header=None)
X_hd, y_hd = heart_df.iloc[:,:-1].to_numpy(), heart_df.iloc[:,-1].map({2:1, 1:-1}).to_numpy()

# scale the data
scaler_hd = MinMaxScaler()
X_hd = scaler_hd.fit_transform(X_hd)

# load the breast cancer data
X_bc, y_bc = load_breast_cancer(return_X_y = True)
# convert the classes to -1 or 1
y_bc = pd.Series(y_bc).map({0:-1,1:1}).to_numpy()
# scale the data
scaler_bc = MinMaxScaler()
X_bc = scaler_bc.fit_transform(X_bc)

# split the data
X_train, X_test, y_train, y_test = train_test_split(X_hd, y_hd, test_size=0.2)

models = []
models.append(('Perceptron', PerceptronClassifier(max_epochs=200)))
models.append(('ELM', ELMClassifier(p=5)))
models.append(('SVM', SVC(C=1.5, degree=5)))
models.append(('RBF', RBFClassifier(p=100)))

def k_fold_cross_validation(X,y,models,n_splits=30,scoring='accuracy'):
    results = {}
    for scorer in tqdm(scoring):
        kfold = StratifiedKFold(n_splits=n_splits, random_state=1234)
        scores = {}
        for name, model in tqdm(models):
            scores[name] = cross_val_score(model, X, y, cv=kfold, scoring=scorer)

        results[scorer] = scores
    return results

def plot_distributions(results):
    fig, ax = plt.subplots(len(results.keys()), 1, figsize=(10,8))
    for index, score in enumerate(results.keys()):
        for model, values in results[score].items():
            sns.distplot(values, ax=ax[index], label=model)
        
        ax[index].legend()
        ax[index].set_title(score)

    fig.tight_layout()
    fig.show()

scoring = ['f1', 'accuracy', 'precision']
results_bc = k_fold_cross_validation(X_bc, y_bc, models, 10, scoring)
plot_distributions(results_bc)

results_hd = k_fold_cross_validation(X_hd, y_hd, models, 10, scoring)
plot_distributions(results_hd)


