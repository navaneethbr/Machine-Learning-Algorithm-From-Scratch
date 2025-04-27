import pandas as pd 
import numpy as np
import math
from random import randint

class PerceptronLogitReg:
    def __init__(self, epochs: int=100, learning_rate: float=0.01) -> None:
        self.beta = None
        self.epochs = epochs
        self.learning_rate = learning_rate
    
    def _step(pred_val: float) -> int:
        return 1 if pred_val > 0 else 0
        
    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> float:
        X_train = X.values
        y_train = y.values
        
        X_train = np.insert(X_train, 0, 1, axis=1)
        self.beta = np.ones((X_train.shape[1], 1))
        
        for _ in range(self.epochs):
            random_index = randint(0, X_train.shape[0]-1)
            y_pred = self._step(X_train[random_index]@self.beta)
            self.beta = self.beta + self.learning_rate*((y_train[random_index] - y_pred)*X_train[random_index])
            
        return self.beta
    
# if __name__ == "__main__":
    