import numpy as np
import pandas as pd
import torch

from kerneloptimizer.functions import loss_functions
from kerneloptimizer.neural_nets.mlp.neural_svm import NeuralKernel
from scipy.optimize import minimize_scalar, fmin, minimize, Bounds
from scipy.spatial import distance_matrix

class KernelOptimizer:

    def __init__(self, kernel='gaussian', **kwargs):

        params = kwargs.keys()
        self.kernel = kernel

        if kernel == 'mlp':
            input_dim = kwargs.get('input_dim')
            hidden_dim = kwargs.get('hidden_dim', 20)
            output_dim = kwargs.get('output_dim', 50)

            self.model = NeuralKernel(input_dim=input_dim,
                                      hidden_dim=hidden_dim,
                                      output_dim=output_dim)

        if kernel == 'gaussian':
            self.width = None


    def fit(self, X, y, **kwargs):

        if self.kernel == 'mlp':
            n_epochs = kwargs.get('n_epochs', 1000)
            #X_tensor = torch.tensor(X, dtype=torch.float32)
            self.model.fit(X,y,n_epochs=n_epochs)

        if self.kernel == 'gaussian':
            fun = lambda sigma: loss_functions.supervised_rbf_loss(
                X, y, sigma)
            result = minimize_scalar(fun, bounds=(0.,np.inf))
            self.width = result['x']


    def to_likelihood_space(self, X):
        if self.kernel == 'mlp':
            Xout = self.model.transform(X)
            gram_matrix = np.dot(Xout, Xout.transpose())

        if self.kernel == 'gaussian':
            dist_matrix = distance_matrix(X, X)
            gram_matrix = np.exp(-.5 * ((dist_matrix**2)/(self.width**2)))

        N = gram_matrix.shape[0]
        gram_matrix = gram_matrix * (1 - np.eye(N))

        return gram_matrix

    def get_likelihood_space(self, X, y):

        if self.kernel == 'mlp':
            Xout = self.model.transform(X)
            gram_matrix = np.dot(Xout, Xout.transpose())

        if self.kernel == 'gaussian':
            dist_matrix = distance_matrix(X, X)
            gram_matrix = np.exp(-.5 * ((dist_matrix**2)/(self.width**2)))

        N = gram_matrix.shape[0]
        gram_matrix = gram_matrix * (1 - np.eye(N))
        classes = sorted(np.unique(y))
        sims = [
            np.sum(gram_matrix[
                :, np.where(y == c)[0]
            ], axis=1)/(N-1) for c in classes
        ]

        sims = np.array(sims).transpose()

        sim_df = pd.DataFrame(
            sims,
            columns=[
                'sim_to_c' + str(c) for c in classes
            ]
        )

        return sim_df
