from typing import Tuple, Union, List
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

class GDLogisticRegression():
    def __init__(self) -> None:
        self.coeff = None
        self.loss_track = []
        
    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        Activation function

        Args:
            z (np.ndarray): predicited values

        Returns:
            np.ndarray: transformed output between range [0,1]
        """
        return 1/(1+np.exp(-z))
        
    def _cost_function(self, X: np.ndarray, y: np.ndarray, y_pred: np.ndarray) -> np.float64 :
        """
        Loss function

        Args:
            X (np.ndarray): Independent Variable
            y (np.ndarray): Dependent Variable
            y_pred (np.ndarray): _description_

        Returns:
            np.float64: returns loss for given coefficients
        """
        cols = X.shape[0]
        return np.float64((-1/cols)*((y.T@np.log(y_pred))+((1-y).T@np.log(1-y_pred))))
    
    def fit(self, X: np.ndarray, y: np.ndarray, learning_rate: float=0.01, iteration: int=2000) -> Tuple[np.ndarray, List[float]]:
        """
        fits the data to logistic model 

        Args:
            X (np.ndarray): Independent Variable
            y (np.ndarray): Dependent Variable
            learning_rate (float, optional): _description_. Defaults to 0.01.
            iteration (int, optional): _description_. Defaults to 2000.

        Returns:
            Tuple[np.ndarray, List[float]]: returns coefficients and loss for every iteration 
        """
        X = np.insert(X, 0, 1, axis=1)
        rows, cols = X.shape
        self.coeff = np.ones((cols, 1))
        for _ in range(iteration):
            y_pred = X@self.coeff
            y_pred = self._sigmoid(y_pred)
            dl = (X.T@(y-y_pred))/rows
            self.coeff =  self.coeff + (learning_rate*dl)
            self.loss_track.append(self._cost_function(X, y, y_pred))
        return self.coeff, self.loss_track
    
    def predict(self, X: np.ndarray, threshold: float=0.5) -> np.ndarray:
        """
        predicts output for the given test data 

        Args:
            X (np.ndarray): Independent Variable
            threshold (float, optional): _description_. Defaults to 0.5.

        Returns:
            np.ndarray: Probabilit of event occuring 
                        False : 0
                        True  : 1
        """
        pred = self._sigmoid(X@self.coeff)
        return (threshold <= pred).astype(int)
        
    def predict_prob(self, X: np.ndarray) -> np.ndarray:
        """
        predicts output for the given test data ranging between (0,1)

        Args:
            X (np.ndarray): Independent Variable

        Returns:
            np.ndarray: Probabilit of event occuring between (0,1)
        """
        return self._sigmoid(X@self.coeff)
    
    def accuracy(y_hat: np.ndarray, y: np.ndarray) -> int:
        """
        give the accuracy for the given model

        Args:
            y_hat (np.ndarray): Predicted Output
            y (np.ndarray): Actual Output

        Returns:
            int: accuracy of model between (0,1)
                0 beging the least score 
                1 beging the highest 
        """
        return np.sum(y_hat==y)/len(y)
    
if __name__ == "__main__":
    # Load the Iris dataset (binary classification)
    iris = load_iris()
    X = iris.data
    y = (iris.target == 0).astype(int)# Setosa vs. Rest

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the logistic regression model
    sklearn_model = LogisticRegression(random_state=42, solver='sag', max_iter=1000)
    sklearn_model.fit(X_train, y_train.ravel())
    sklearn_predict = sklearn_model.predict(X_test)

    our_model = GDLogisticRegression()
    our_model.fit(X_train, y_train)
    our_model.predict(X_test)
    
    
    