#%%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from models.elm import ELMClassifier, ELMRegressor
from models.perceptron import PerceptronClassifier
from models.rbf import RBFClassifier, RBFRegressor
from models.elm_hebbian import ELMHebbianClassifier, ELMHebbianRegressor
from models.adaline import Adaline
from sklearn.datasets import load_digits, load_breast_cancer, load_wine, load_boston, load_diabetes, load_iris, make_blobs
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, mean_squared_error

#%% 
# blobs
X, y = make_blobs(n_features=2, centers=[[2,2], [4,4]], random_state=1234)
y = pd.Series(y).map({0:-1,1:1}).to_numpy()

# train
heb = ELMHebbianClassifier().fit(X, y)
elm = ELMClassifier().fit(X, y)

#%%
# and plot
fig, ax = plt.subplots()

t_class0 = y == -1
t_class1 = y == 1
ax.scatter(X[t_class0, 0], X[t_class0, 1], color='red')
ax.scatter(X[t_class1, 0], X[t_class1, 1], color='blue')
ax.set_xlabel('x1')
ax.set_ylabel('x2')

x1 = np.arange(-2, 10, step=0.1)
x2 = np.arange(-2, 10, step=0.1)

xx, yy = np.meshgrid(x1, x2)
# flatten each grid to a vector
r1, r2 = xx.flatten(), yy.flatten()
r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))
    
# horizontal stack vectors to create x1,x2 input for the model
grid = np.hstack((r1,r2))
y_heb = heb.predict(grid)
y_elm = elm.predict(grid)

zz_heb = y_heb.reshape(xx.shape)
zz_elm = y_elm.reshape(xx.shape)

ax.contour(xx, yy, zz_heb, colors=['black'])
ax.contour(xx, yy, zz_elm, colors=['green'])

fig.show()

#%% regression
# define the sinc function generator
n_train = 100
n_test = 50

X_train = np.random.uniform(-15,15,size=(n_train,1))
y_train = np.sin(X_train) / X_train + np.random.normal(loc=0,scale=0.05, size=(n_train,1))

X_test = np.random.uniform(-15,15, size=(n_test,1))
y_test = np.sin(X_test) / X_test + np.random.normal(loc=0,scale=0.05, size=(n_test,1))


def plot_regression_results(X_train, y_train, X_test, y_test, reg=0.0):
    fig, ax = plt.subplots(figsize=(6,4))
     # horizontal stack vectors to create x1,x2 input for the model
    grid = np.arange(np.min(X_train)-0.1, np.max(X_train)+0.1, 0.01).reshape(-1, 1)
    
    # train the model
    model = Adaline(reg_factor=reg).fit(X_train, y_train)
    
    # make predictions for the grid
    yhat_grid = model.predict(grid)
    # make predictions for the datasets
    yhat_test = model.predict(X_test)
    yhat_train = model.predict(X_train)

    mse_test = mean_squared_error(y_test, yhat_test.ravel())
    mse_train = mean_squared_error(y_train, yhat_train.ravel())

    ax.scatter(X_train, y_train, color='red', label='Treinamento')
    ax.scatter(X_train, y_train, color='blue', label='Teste')
    ax.plot(grid, yhat_grid, color='black', label='RBF')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    ax.legend()
    fig.suptitle(f'Neurônios:{reg}\n MSE treino: {mse_train} \n MSE teste: {mse_test}')
    fig.tight_layout()
    fig.show()


plot_regression_results(X_train, y_train, X_test, y_test, reg=10)
# %%
