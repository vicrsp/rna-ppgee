#%%
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor

class MLP:
    def __init__(self, p, m, max_epochs=100, eta=0.01):
      self.p = p
      self.m = m
      self.eta = eta
      self.max_epochs = max_epochs

    def sech2(self, u):
        return ((2/(np.exp(u)+np.exp(-u)))*(2/(np.exp(u)+np.exp(-u))))
    
    # def tanh(self, x):
    #     t=(np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
    #     dt=1-t**2
    #     return t,dt

    def fit(self, X, y):
        # augment X
        N, n = X.shape
        x_aug = np.hstack((np.ones((N, 1)), X))

        # initialize the weight and hidden layer matrixes
        w = np.random.uniform(-0.5, 0.5, (self.p+1, self.m))
        Z = np.random.uniform(-0.5, 0.5, (n+1, self.p))

        # initialize the main loop
        epochs = 0

        while((epochs < self.max_epochs)):
            xseq = np.arange(N)
            np.random.shuffle(xseq)
            ei2 = 0
            for iseq in xseq:  
                # current input/output pair
                xa = x_aug[iseq,:]
                ya = y[iseq]
                
                # hidden layer pass
                U = xa.T @ Z
                H = np.tanh(U)
                H_aug = np.hstack((1, H)) #np.hstack((np.ones((N, 1)), H))
                
                # output layer pass
                O = H_aug @ w
                y_hat = O # np.tanh(O) apply linear activatio function
                
                # propagate output error
                e = ya - y_hat
                dO = e * self.sech2(O) 
                # propagate hidden layer error
                # the bias should not be considered
                ehidden = dO @ w[:-1,:].T
                dU = ehidden * self.sech2(U)

                # update w and Z
                w = w + self.eta * (H_aug.reshape(1,-1).T @ dO.reshape(1,-1))
                Z = Z + self.eta * (xa.reshape(-1,1) @ dU.reshape(1,-1))

                ei2 = ei2 + (e @ e.T)
            print(f'Epoch: {epochs}; MSE: {ei2}')
            epochs = epochs + 1
        self.coef_ = w
        self.Z_ = Z

        return self

    def predict(self, X):
        N, _ = X.shape
        x_aug = np.hstack((np.ones((N, 1)), X))
        # forward pass through hidden layer
        H = np.tanh(x_aug @ self.Z_)
        # add bias and forward pass on output layer
        H_aug = np.hstack((np.ones((N, 1)), H))
        yhat = H_aug @ self.coef_
        return yhat

#%%
Ntrain = 45
X = np.linspace(0, 2 * np.pi, Ntrain).reshape(-1,1)
y = np.sin(X) + np.random.uniform(-0.1,0.1, (len(X),1))

X_test = np.arange(0, 2 * np.pi, step=0.01).reshape(-1,1)
y_test = np.sin(X_test)

mlp_sk = MLPRegressor((3,), activation='tanh',solver='adam',learning_rate_init=0.01).fit(X,y)
mlp = MLP(3,1,200,0.01).fit(X,y)

y_hat = mlp.predict(X_test)
y_hat_sk = mlp_sk.predict(X_test)

#%%
print(mean_squared_error(y_test, y_hat))
plt.plot(X, y, 'ro')
plt.plot(X_test, y_test, 'b-')
plt.plot(X_test, y_hat, 'g--')
plt.plot(X_test, y_hat_sk, 'k--')

# %%
