# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from sklearn.datasets import load_breast_cancer, make_moons, make_gaussian_quantiles
from scipy.optimize import minimize_scalar
from sklearn.neighbors import KernelDensity
from matplotlib import animation
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.svm import SVC

plt.style.use('ggplot')
#%%

def calculate_likelihood(X, y, sigma):
    N, _ = X.shape
    B = np.zeros((N, 2))  # binary problem

    for i in range(N):
        xi = X[i, :]
        for j in range(2):
            Xj = X[np.where(y == np.sign(2*j - 1))[0],:]  # assumes y in [-1,1]
            Nj = Xj.shape[0]
            B[i, j] = np.sum([np.exp(-np.linalg.norm(xi - xk)/(2*(sigma**2))) for xk in Xj]) / Nj
    return B


def calculate_similarities(X, y, sigma):
    V = np.zeros((2, 2))  # binary problem
    for i in range(2):
        # Define Ci and Cj
        Xi = X[(y == np.sign(2*i - 1)), :]
        Ni = Xi.shape[0]
        for j in range(2):
            Xj = X[(y == np.sign(2*j - 1)), :]
            Nj = Xj.shape[0]
            # Calculate the similarities
            Sm = np.exp(-euclidean_distances(Xi, Xj)/(2*(sigma**2)))/(Ni*Nj)
            V[i,j] = np.sum(Sm)

    return V


def kde_estimation(X,y,sigma=0.22):
    N, _ = X.shape
    probs = np.zeros((N, 2))
    
    for i in range(2):
        Xi = X[(y == np.sign(2*i - 1)), :]
        pX_Ci = KernelDensity(kernel='gaussian', bandwidth=sigma).fit(Xi).score_samples(X)
        probs[:,i] = np.exp(pX_Ci)
    return probs


def plot_decision_boundary(X, y, gamma, C):
    fig, ax = plt.subplots(figsize=(6,4))
    
    x1 = np.arange(-2, 6, step=0.01)
    x2 = np.arange(-2, 6, step=0.01)

    xx, yy = np.meshgrid(x1, x2)
    # flatten each grid to a vector
    r1, r2 = xx.flatten(), yy.flatten()
    r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))
    
    # horizontal stack vectors to create x1,x2 input for the model
    grid = np.hstack((r1,r2))
    
    # train the model
    model = SVC(C=C, gamma=gamma).fit(X, y)
    
    # make predictions for the grid
    yhat = model.predict(grid)
    # reshape the predictions back into a grid
    zz = yhat.reshape(xx.shape)
    ax.contour(xx, yy, zz, colors=['black'])
   
    t_class0 = y == -1
    t_class1 = y == 1
    ax.scatter(X[t_class0, 0], X[t_class0, 1], color='red')
    ax.scatter(X[t_class1, 0], X[t_class1, 1], color='blue')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    fig.suptitle(f'Sigma:{gamma}\nAccurácia: {100 * np.sum(y == np.sign(model.predict(X)))/len(y)} %')

    fig.tight_layout()
    fig.show()

# %%
#X_bc, y_bc = load_breast_cancer(return_X_y=True)
#y_bc = pd.Series(y_bc).map({0: -1, 1: 1}).to_numpy()

# data = pd.read_csv('spirals_original.csv')
# X, y = data[['X1','X2']].to_numpy(), data['Class'].to_numpy()
# y = pd.Series(y).map({1:-1,2:1}).to_numpy()

# Construct gaussian quantiles dataset
X1, y1 = make_gaussian_quantiles(cov=2.,
                                 n_samples=100, n_features=2,
                                 n_classes=2, random_state=1)
X2, y2 = make_gaussian_quantiles(mean=(3, 3), cov=1.5,
                                 n_samples=100, n_features=2,
                                 n_classes=2, random_state=1)
X = np.concatenate((X1, X2))
y = np.concatenate((y1, -y2 + 1))
y = pd.Series(y).map({0:-1,1:1}).to_numpy()

# X, y = make_moons(200,random_state=1234,noise=0.1)
# y = pd.Series(y).map({0:-1,1:1}).to_numpy()

plt.scatter(X[y==-1,0], X[y==-1,1], color='r')
plt.scatter(X[y==1,0], X[y==1,1], color='b')

p_kde = kde_estimation(X,y)
px_C1 = p_kde[p_kde[:,1] >= p_kde[:,0],:]
px_C2 = p_kde[p_kde[:,1] < p_kde[:,0],:]

plt.figure()
plt.scatter(px_C1[:, 0], px_C1[:, 1], color='r')
plt.scatter(px_C2[:, 0], px_C2[:, 1], color='b')
plt.plot(np.arange(0,0.3,0.1),np.arange(0,0.3,0.1),'k')
#%%
sigma_values = np.linspace(0.001, 10, 100)
Dvalues = []
for sigma in tqdm(sigma_values):
    Vs = calculate_similarities(X, y, sigma)
    V1, V2 = Vs[0,:], Vs[1,:]
    D = np.dot(V1,V2) * np.linalg.norm(V1 - V2)/(np.linalg.norm(V1)*np.linalg.norm(V1))
    Dvalues.append(D)

plt.plot(sigma_values, Dvalues, 'g')

#%%
def f(x):
    Vs = calculate_similarities(X, y, x)
    V1, V2 = Vs[0,:], Vs[1,:]
    D = np.dot(V1,V2) * np.linalg.norm(V1 - V2)/(np.linalg.norm(V1)*np.linalg.norm(V1))
    return -D

res = minimize_scalar(f, bounds=(0.00001, 2), method='bounded')
print(f'Optimal s = {res.x}')
# %%
p_kde = kde_estimation(X,y,res.x)
px_C1 = p_kde[p_kde[:,1] >= p_kde[:,0],:]
px_C2 = p_kde[p_kde[:,1] < p_kde[:,0],:]

plt.figure()
plt.title(f'$\sigma = {res.x:.3f}$')
plt.scatter(px_C1[:, 0], px_C1[:, 1], color='r')
plt.scatter(px_C2[:, 0], px_C2[:, 1], color='b')
plt.plot(np.arange(0,0.6,0.1),np.arange(0,0.6,0.1),'k')

#%%
plot_decision_boundary(X,y,res.x,1)

# %%
sigma_k = np.linspace(0.0001,2,480,endpoint=True)
sigma_y = np.array([-f(x) for x in tqdm(sigma_k)])

# First set up the figure, the axis, and the plot element we want to animate
fig, ax = plt.subplots(1,2,figsize=(8,6))
line1, = ax[0].plot([], [], 'ro')
line2, = ax[0].plot([], [], 'bo')
point, = ax[1].plot([], [], 'go')

# initialization function: plot the background of each frame
def init():
    # draw the separation surface
    ax[0].plot(np.arange(0,1,0.01),np.arange(0,1,0.01),'k')
    ax[0].set_ylabel('$P(x|C_2)$')
    ax[0].set_xlabel('$P(x|C_1)$')

    ax[1].plot(sigma_k, sigma_y, 'k')
    ax[1].set_ylabel('Distance')
    ax[1].set_xlabel('$\sigma$')

    line1.set_data([], [])
    line2.set_data([], [])
    point.set_data([], [])

    return line1, line2, point

# animation function.  This is called sequentially
def animate(i):
    print(i)
    p_kde = kde_estimation(X,y,sigma_k[i])
    
    px_C1 = p_kde[p_kde[:,1] >= p_kde[:,0],:]
    px_C2 = p_kde[p_kde[:,1] < p_kde[:,0],:]
    ax[0].set_xlim(0, np.max(p_kde[:,0])*1.2)
    ax[0].set_ylim(0, np.max(p_kde[:,1])*1.2)

    ax[0].set_title(f'$\sigma = {sigma_k[i]:.3f}$')

    line1.set_data(px_C1[:, 0], px_C1[:, 1])
    line2.set_data(px_C2[:, 0], px_C2[:, 1])
    point.set_data([sigma_k[i]], [sigma_y[i]])

    return line1,line2,point
# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=len(sigma_k), interval=20, blit=True)
anim.save('projection_animation.gif', fps=24)
fig.show()

# %%
