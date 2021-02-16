import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import train_test_split, StratifiedKFold

class ModelTuner:
    def __init__(self, dataset, model, reg_factor=(0,100,20), tuning_size=0.3):
        self.X, self.y = dataset
        self.model = model
        self.tuning_size = tuning_size
        reg_min, reg_max, reg_num = reg_factor
        self.reg_factor = np.linspace(reg_min, reg_max, reg_num)

    def tune(self, repetitions=30):
        tuning_data = []
        for i in range(repetitions):
            X_t1, X_t2, y_t1, y_t2 = train_test_split(self.X, self.y, test_size=self.tuning_size, shuffle=True)
            for factor in tqdm(self.reg_factor):
                scores = []
                for _ in range(int(repetitions/3)):
                    # train regularized model on T1
                    self.model.set_regularization_factor(factor)
                    self.model.fit(X_t1, y_t1)
                    # evaluate it on T2
                    score = self.model.score(X_t2, y_t2)
                    scores.append(score)

                tuning_data.append((i, factor, np.mean(score)))
          
        self.results = pd.DataFrame(tuning_data, columns=['i','factor','score'])
        return self

    def plot(self, title):
        fig, ax = plt.subplots(figsize=(8,6))
        sns.lineplot(data=self.results, x='factor',y='score', ax=ax)
        ax.set_title(title)
        fig.show()



