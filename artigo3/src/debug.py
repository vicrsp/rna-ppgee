#%%
import numpy as np
import pandas as pd
from sklearn.datasets import make_moons, make_blobs
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from models.kernel import KernelProjection
from models.load import get_classification_models
from models.non_linear import ELM, KernelSVM
from datasets.load import load_sonar, load_sonar_uci
from experiments.evaluation import ModelEvaluationExperiment, plot_scores, display_stats, plot_ci, plot_ci_all


import seaborn as sns
import matplotlib.pyplot as plt
from pylab import cm

plt.style.use('ggplot')


X, y = make_moons(200, random_state=1234,noise=0.1)
y = pd.Series(y).map({0:-1,1:1}).to_numpy()
#%% 

# X, y = make_blobs(200, n_features=2, centers=[[-1,-1], [1,1]], random_state=1234)
# y = pd.Series(y).map({0:-1,1:1}).to_numpy()

models = get_classification_models()
cmap = cm.get_cmap('Dark2', len(models))
#%%
# and plot
fig, ax = plt.subplots(figsize=(8,6))

t_class0 = y == -1
t_class1 = y == 1
ax.scatter(X[t_class0, 0], X[t_class0, 1], color='red')
ax.scatter(X[t_class1, 0], X[t_class1, 1], color='blue')
ax.set_xlabel('x1')
ax.set_ylabel('x2')

x1 = np.arange(-2, 2, step=0.1)
x2 = np.arange(-2, 2, step=0.1)

xx, yy = np.meshgrid(x1, x2)
# flatten each grid to a vector
r1, r2 = xx.flatten(), yy.flatten()
r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))
    
# horizontal stack vectors to create x1,x2 input for the model
grid = np.hstack((r1,r2))
i = 0
for name, model, _ in models:
    model.fit(X, y) 
    y_hat = model.predict(grid)
    zz = y_hat.reshape(xx.shape)

    cs = ax.contour(xx, yy, zz, colors=[cmap(i)])
    cs.levels = [name for level in cs.levels]
    cs.clabel(cs.levels, inline=True)

    i = i + 1

fig.show()


# %%
def run_classification_experiment():
    models_classification = get_classification_models()
    datasets_classification = [('sonar',load_sonar_uci())]

    experiment_classification = ModelEvaluationExperiment(
        datasets_classification, 'classification', n_jobs=4)
    experiment_classification.start(models_classification)

    experiment_classification.plot_final_scores('/home/victor/git/rna-ppgee/artigo3')
    experiment_classification.save_final_scores_table('/home/victor/git/rna-ppgee/artigo3')


run_classification_experiment()
# %%
plot_ci('/home/victor/git/rna-ppgee/artigo3/classification_experiment_chart_data.csv', 
        'Acurácia',
         'Modelo')
display_stats('/home/victor/git/rna-ppgee/artigo3/classification_experiment_chart_data.csv',
         'Acurácia',
          'std')
# %%
X,y = load_sonar()
X_train, X_test, y_train, y_test = train_test_split(X,y)

m = KernelSVM(reg_factor=100).fit(X_train, y_train)
accuracy_score(y_test, m.predict(X_test))
# %%
