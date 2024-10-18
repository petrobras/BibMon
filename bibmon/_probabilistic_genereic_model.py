import numpy as np
import pandas as pd
from scipy.stats import norm
from abc import ABC, abstractmethod

class ProbabilisticGenericModel(ABC):
    @abstractmethod
    def train_core(self):
        pass

    @abstractmethod
    def predict_proba(self, X):
        pass

    def set_hyperparameters(self, params_dict):
        for key, value in params_dict.items():
            setattr(self, key, value)

    def load_model(self, *args, **kwargs):
        pass

    def pre_train(self, X_train, Y_train=None, *args, **kwargs):
        # Garantir que os dados são numéricos
        self.X_train = pd.DataFrame(X_train).astype(float)
        self.Y_train = pd.DataFrame(Y_train)

    def train(self, *args, **kwargs):
        start_time = time.time()
        self.train_core()
        end_time = time.time()
        self.train_time = end_time - start_time

    def pre_test(self, X_test, Y_test=None, *args, **kwargs):
        # Garantir que os dados são numéricos
        self.X_test = pd.DataFrame(X_test).astype(float)
        self.Y_test = pd.DataFrame(Y_test)

    def test(self, *args, **kwargs):
        start_time = time.time()
        probabilities = self.predict_proba(self.X_test.values)
        end_time = time.time()
        self.test_time = end_time - start_time
        self.probabilities = pd.DataFrame(probabilities, index=self.X_test.index)

    def fit(self, X_train, Y_train=None, *args, **kwargs):
        self.pre_train(X_train, Y_train, *args, **kwargs)
        self.train(*args, **kwargs)

    def predict(self, X_test, Y_test=None, *args, **kwargs):
        self.pre_test(X_test, Y_test, *args, **kwargs)
        self.test(*args, **kwargs)