import pandas as pd
import numpy as np
import random
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import math
from sklearn.metrics import mean_squared_error

class MBGDRegression:
    def __init__(self, batch_size: int=3, epochs: int=100, learning_rate: float=0.01) -> None:
        """Initializing epochs, learning_rate, batch_size

        Args:
            epochs (int, optional): number of iteration. Defaults to 100.
            learning_rate (float, optional): set intial learning rate. Defaults to 0.01.
            batch_size (int, optional): how many batchs should the data be divided into. Default to 3.
        """
        self.epochs = epochs
        self.lr = learning_rate
        self.batch_size = batch_size
        self.t0, self.t1 = 5, 50 
        
    def _learning_schedule(self, t) -> np.float64:
        """Used to alter learning rate dynamically after each epochs so that it can find the local minima
        """
        return self.t0/(t + self.t1)
        
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.DataFrame) -> pd.Series:
        """Fits the training data to Stochastic GD Linear Regression model

        Args:
            X_train (pd.DataFrame): Independent Variable
            y_train (pd.DataFrame): Dependent Variable

        Returns:
            pd.Series: returns intercept and coefficient
        """
        self.beta0 = 0
        self.beta  = np.ones((X_train.shape[1],1))
        
        for i in range(self.epochs):
            for j in range(X_train.shape[0]//self.batch_size):
                # self.lr = self._learning_schedule(i * X_train.shape[0] + j) ## dynamic learning rate
                
                rand = random.sample(range(X_train.shape[0]), self.batch_size)  
                
                dfb0 = np.mean(-2*(y_train[rand] - self.beta0 - (X_train[rand]@self.beta)))
                dfb  = -2*(X_train[rand].T@(y_train[rand] - self.beta0 - (X_train[rand]@self.beta)))
                
                step_size_b0 = dfb0*self.lr
                step_size_b = dfb*self.lr
                
                self.beta0 = self.beta0 - step_size_b0
                self.beta  = self.beta - step_size_b
        
        return self.beta0, self.beta
        
    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        """Predicts on test data using the trained model

        Args:
            X_test (pd.DataFrame): Independent test Variable

        Returns:
            pd.Series: 
        """
        return self.beta0 + (X_test@self.beta)
    
    def rmse(self, y_pred: pd.Series, y_actual: pd.Series) -> np.float64:
        """Evaluation metrics Room Mean Squared Error(RMSE).

        Args:
            y_acutal (array): Actual Output 
            y_pred (array): Predicted Output

        Returns:
        RMSE error: integer
                retuns the error for give model
        """
        return math.sqrt(((y_pred - y_actual).T@(y_pred - y_actual))/y_actual.shape[0])
    
if __name__ == "__main__":
    X,y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    modelsk = LinearRegression().fit(X_train, y_train)
    model = MBGDRegression()
    beta, beta0 = model.fit(X_train, y_train.reshape(-1,1))
    y_pred = model.predict(X_test)
    print(math.sqrt(mean_squared_error(y_pred, y_test)))