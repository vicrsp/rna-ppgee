#%%
import numpy as np
import pandas as pd
from sklearn.datasets import make_moons, make_blobs
from sklearn.linear_model import Perceptron
from models.kernel import KernelProjection
from models.load import get_classification_models
import seaborn as sns
import matplotlib.pyplot as plt
from pylab import cm

plt.style.use('ggplot')


#%% 

# X, y = make_moons(200, random_state=1234,noise=0.1)
# y = pd.Series(y).map({0:-1,1:1}).to_numpy()

X, y = make_blobs(200, n_features=2, centers=[[-1,-1], [1,1]], random_state=1234)
y = pd.Series(y).map({0:-1,1:1}).to_numpy()

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

# %%
