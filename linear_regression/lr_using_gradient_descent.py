import pandas as pd 
import numpy as np 
from typing_extensions import Annotated
from typing import Tuple
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import math
from sklearn.metrics import mean_squared_error

class GDRgression:
    def __init__(self, epochs: int=25, learning_rate: float=0.01) -> None:
        """Initializing epochs, learning_rate

        Args:
            epochs (int, optional): _description_. Defaults to 10.
            learning_rate (float, optional): _description_. Defaults to 0.01.
        """
        self.epocs = epochs
        self.learning_rate = learning_rate
        self.beta = None
        self.beta0 = None
        
    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> pd.Series:
        """Fits the training data to Gradient GD Linear Regression model

        Args:
            X_train (pd.DataFrame): Independent Variable
            y_train (pd.DataFrame): Dependent Variable

        Returns:
            pd.Series: returns intercept and coefficient
        """
        self.beta = np.ones((X.shape[1],1))
        self.beta0 = 0
        
        X_train = X.values
        y_train = y.values
        
        for _ in range(self.epocs):
            db0 = np.mean(-2*(y_train - self.beta0 - (X_train@self.beta)) )
            db = 2*(X_train.T@(y_train - self.beta0 - (X_train@self.beta)))
            
            step_b0 = self.learning_rate*db0
            step_b = self.learning_rate*db
            
            self.beta0 = self.beta0 - step_b0
            self.beta = self.beta - step_b
        
        return self.beta, self.beta0
    
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Predicts on test data using the trained model

        Args:
            X_test (pd.DataFrame): Independent test Variable

        Returns:
            pd.Series: 
        """
        X_test = X.values
        return (self.beta0 + X_test@self.beta)
    
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
    df = pd.read_csv("/Users/navaneeth/Documents/Work/MACHINE_LEARNING_FROM_SCRATCH/dataset/Student_Performance.csv")
    independent_var = ['Hours Studied','Previous Scores','Extracurricular Activities','Sleep Hours','Sample Question Papers Practiced']
    X = df[independent_var]
    y = df.iloc[:,-1:]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    modelsk = LinearRegression().fit(X_train.values, y_train.values)
    model = GDRgression()
    beta, beta0 = model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(math.sqrt(mean_squared_error(y_pred, y_test.values)))
    