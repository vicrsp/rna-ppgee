import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def func_simple():
    xc1 = 0.3 * np.random.randn(60, 2) + 2
    xc2 = 0.3 * np.random.randn(60, 2) + 4

    df = pd.DataFrame()
    df['x'] = np.hstack((xc1[:, 0], xc2[:, 0]))
    df['y'] = np.hstack((xc1[:, 1], xc2[:, 1]))
    df['class'] = np.hstack((np.zeros(60), np.ones(60)))

    return df


def func_circle(radius=0.6):
    x = np.arange(-1, 1 + 0.1, step=0.1)
    y = np.arange(-1, 1 + 0.1, step=0.1)

    xy = np.array([(xi, yi) for xi in x for yi in y])

    z = np.sum(xy ** 2, axis=1)
    z_class = 1 * (z > radius)

    df = pd.DataFrame()
    df['x'] = xy[:, 0]
    df['y'] = xy[:, 1]
    df['class'] = z_class.flatten()

    return df


def plot_func(data):
    t_class0 = data['class'] == 0
    t_class1 = data['class'] == 1

    plt.scatter(data.loc[t_class0, 'x'],
                data.loc[t_class0, 'y'], color='black')
    plt.scatter(data.loc[t_class1, 'x'], data.loc[t_class1, 'y'], color='red')

    plt.show()


#data = func_simple()
data = func_circle()
plot_func(data)


class LinearPerceptron:
    def __init__(self, eta=0.1, tol=0.0001, max_epochs=100):
        self.eta = eta
        self.tol = tol
        self.max_epochs = max_epochs

    def predict(self, x, w):
        N, _ = x.shape
        x_aug = np.hstack((-np.ones((N, 1)), x))
        u = x_aug @ w
        return 1.0 * (u >= 0)

    def train(self, x_train, y_train):
        # initialize the weight matrix
        N, n = x_train.shape
        x_aug = np.hstack((-np.ones((N, 1)), x_train))

        wt = np.random.rand(n+1) - 0.5
        n_epochs = 0
        e_vec = []

        while(n_epochs < self.max_epochs):
            # generate random indexes order
            xseq = np.arange(N)
            np.random.shuffle(xseq)
            error_array = []

            for i_rand in xseq:
                yhati = 1.0 * ((x_aug[i_rand, :] @ wt) >= 0)
                ei = y_train[i_rand] - yhati
                # calculate step size
                dw = self.eta * ei * x_aug[i_rand, :]
                # update weight vector
                wt = wt + dw

                error_array.append(ei ** 2)

            # increment number of epochs
            n_epochs = n_epochs + 1
            e_vec.append(np.sum(error_array) / N)

        return wt, e_vec


# %%
lperceptron = LinearPerceptron()
x_train = data[['x', 'y']].to_numpy()
y_train = data['class'].to_numpy()

radius = 0.6
def rbf(x): return np.expm1((x ** 2) / (2*(radius**2)))

x_train = rbf(x_train)
# data_rbf = pd.DataFrame()
# data_rbf['x'] = x_train[:,0]
# data_rbf['y'] = x_train[:,1]
# data_rbf['class'] = x_train[:,1]


w, e = lperceptron.train(x_train, y_train)
yp = lperceptron.predict(x_train, w)

# %%
