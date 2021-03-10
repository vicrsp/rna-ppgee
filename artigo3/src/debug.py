import numpy as np
import pandas as pd
from sklearn.datasets import make_moons
from sklearn.linear_model import Perceptron
from models.kernel import KernelClassifier

X, y = make_moons(200,random_state=1234,noise=0.1)
y = pd.Series(y).map({0:-1,1:1}).to_numpy()

pc = Perceptron()
kc = KernelClassifier(pc, kernel='gaussian')







