import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim

from kerneloptimizer.functions import loss_functions
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


class DeepMLP(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DeepMLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.layer1 = torch.nn.Linear(self.input_dim, self.hidden_dim)
        self.layer1.weight.data.normal_(0.0, 10.0)
        self.layer2 = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
        self.layer2.weight.data.normal_(0.0, 10.0)
        self.layer3 = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
        self.layer3.weight.data.normal_(0.0, 10.0)
        self.layer4 = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
        self.layer4.weight.data.normal_(0.0, 10.0)
        self.layer5 = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
        self.layer5.weight.data.normal_(0.0, 10.0)
        self.layer6 = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
        self.layer6.weight.data.normal_(0.0, 10.0)
        self.layer7 = torch.nn.Linear(self.hidden_dim, self.output_dim)
        self.layer7.weight.data.normal_(0.0, 10.0)
        self.tanh = torch.nn.Tanh()
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.layer1(x)
        x = self.sigmoid(x)
        x = self.layer2(x)
        x = self.sigmoid(x)
        x = self.layer3(x)
        x = self.sigmoid(x)
        x = self.layer4(x)
        x = self.sigmoid(x)
        x = self.layer5(x)
        x = self.sigmoid(x)
        x = self.layer6(x)
        x = self.sigmoid(x)
        x = self.layer7(x)
        output = self.sigmoid(x) / np.sqrt(self.output_dim)
        output = F.normalize(output, dim=1)

        return output

class MLP(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.layer1 = torch.nn.Linear(self.input_dim, self.hidden_dim)
        self.layer1.weight.data.uniform_(-1.0, 1.0)
        self.tanh = torch.nn.Tanh()
        self.layer2 = torch.nn.Linear(self.hidden_dim, self.output_dim)
        self.layer2.weight.data.uniform_(-1.0, 1.0)
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        x = self.layer1(x)
        x = self.tanh(x)
        x = self.layer2(x)
        output = self.sigmoid(x) #/ np.sqrt(self.output_dim)
        output = F.normalize(output, dim=1)

        return output


class NeuralKernel:

    def __init__(self, input_dim, hidden_dim=20, output_dim=50):
        self.model = MLP(input_dim, hidden_dim, output_dim)

    def fit(self, X, y, n_epochs=1000):

        if type(X) != np.ndarray:
            X = np.array(X)
        if type(y) != np.ndarray:
            y = np.array(y)


        #X, X_dev, y, y_dev = train_test_split(
        #    X, y, test_size=0.2, stratify=y, random_state=42
        #)

        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        #X_dev = torch.tensor(X_dev, dtype=torch.float)

        #criterion = None
        #if method == 'supervised':
        criterion = loss_functions.supervisedDotProductLoss_torch
        #elif method == 'unsupervised':
        #    criterion = loss_functions.dotProductLoss_torch
        #elif method == 'mmd':
        #    criterion = loss_functions.MMDLoss_torch

        supervised = True
        #if method == 'unsupervised':
        #    supervised = False


        optimizer = torch.optim.RMSprop(self.model.parameters(), weight_decay=1e-6)

        # Optimizing Neural Network

        self.model.train()
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            X_out = self.model(X)
            #X_dev_out = self.model(X_dev)
            if supervised:
                loss = criterion(X_out, y, normalize=False)
                #loss_dev = criterion(X_dev_out, y_dev)
            else:
                loss = criterion(X_out, normalize=False)
                #loss_dev = criterion(X_dev_out)

            loss.backward()
            optimizer.step()

        self.model.eval()



    def transform(self, X):

        assert type(X) == np.ndarray

        X = torch.tensor(X, dtype=torch.float)

        self.model.eval()
        X_out = self.model(X)
        X_np = X_out.detach().numpy()

        return np.array(X_np, dtype=np.float64)
