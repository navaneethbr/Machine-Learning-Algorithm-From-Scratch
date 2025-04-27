import numpy as np 
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import math
from sklearn.metrics import mean_squared_error

class GDLassoRegression:
    def __init__(self, alpha: float=0.01, learning_rate: float=0.01, epochs: int=50) -> None:
        """Initializing epochs, learning_rate, alpha

        Args:
            alpha (float, optional): _description_. Defaults to 0.01.
            learning_rate (float, optional): _description_. Defaults to 0.01.
            epochs (int, optional): _description_. Defaults to 10.
        """
        self.alpha = alpha
        self.lr = learning_rate
        self.epochs = epochs
        self.beta = None
        self.intercept = 1
        
    def fit(self, X_train: pd.DataFrame, y_train: pd.DataFrame) -> pd.Series:
        """Fits the training data to Gradient GD Lasso Regression model

        Args:
            X_train (pd.DataFrame): Independent Variable
            y_train (pd.DataFrame): Dependent Variable

        Returns:
            pd.Series: returns intercept and coefficient
        """
        X_train = np.insert(X_train, 0, self.intercept, axis=1)
        
        self.beta = np.ones((X_train.shape[1], 1))
        
        for _ in range(self.epochs):
            db = X_train.T@((X_train@self.beta)-y_train) + (self.alpha*np.sign(self.beta))
            self.beta -= (self.lr*db)
            
        return self.beta
    
    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        """Predicts on test data using the trained model

        Args:
            X_test (pd.DataFrame): Independent test Variable

        Returns:
            pd.Series: 
        """
        X_test = np.insert(X_test, 0, self.intercept, axis=1)
        return X_test@self.beta
    
    def rmse(self, y_pred: pd.Series, y_actual: pd.Series) -> float:
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
    model = GDLassoRegression()
    beta = model.fit(X_train, y_train.reshape(-1,1))
    y_pred = model.predict(X_test)
    print(math.sqrt(mean_squared_error(y_pred, y_test)))
    