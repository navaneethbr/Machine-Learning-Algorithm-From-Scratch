import pandas as pd
import numpy as np
from scipy.special import softmax
from sklearn.preprocessing import OneHotEncoder
onehot_encoder = OneHotEncoder(sparse=False)
from sklearn.datasets import load_iris

class SoftmaxRegression:
    def __init__(self, epochs: int=1000, learning_rate: float=0.1) -> None:
        """Initializing epochs, learning_rate

        Args:
            epochs (int, optional): _description_. Defaults to 10.
            learning_rate (float, optional): _description_. Defaults to 0.01.
        """
        self.lr = learning_rate
        self.epochs = epochs
        
    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> np.ndarray:
        """Fits the training data to Gradient GD Linear Regression model

        Args:
            X_train (pd.DataFrame): Independent Variable
            y_train (pd.DataFrame): Dependent Variable

        Returns:
            pd.Series: returns intercept and coefficient
        """
        Y_onehot = onehot_encoder.fit_transform(y.reshape(-1,1))
        n = X.shape[0]
        # X = np.insert(X, 0, 1, axis=1)
        self.W = np.zeros((X.shape[1], Y_onehot.shape[1]))
        
        for _ in range(self.epochs):
            y_pred = softmax(X@self.W, axis=1)
            dw = (1/n)*(X.T@(y_pred - Y_onehot))
            step_w = self.lr*dw 
            self.W -= step_w 
            
        return -self.W
    
    def loss(self, X: pd.DataFrame, y: pd.DataFrame) -> float:
        """Calculats loss for the test dataset

        Args:
            X (pd.DataFrame): test data
            y (pd.DataFrame): test data

        Returns:
            float: _description_
        """
        N = y.shape[0]  # Batch size
        Y_onehot = onehot_encoder.fit_transform(y.reshape(-1,1))
        self.W = np.zeros((X.shape[1], Y_onehot.shape[1]))
        y_pred = softmax(X@self.W, axis=1)
        # y_pred = np.exp(y_pred - np.max(y_pred, axis=1, keepdims=True))  
        # y_pred = y_pred / np.sum(y_pred, axis=1, keepdims=True)
        loss = -np.sum(y * np.log(y_pred)) / N  # Average loss across the batch
        return loss
        
            
if __name__ == "__main__":
    X = load_iris().data
    y = load_iris().target
    model = SoftmaxRegression()
    w = model.fit(X, y)
    print(w)