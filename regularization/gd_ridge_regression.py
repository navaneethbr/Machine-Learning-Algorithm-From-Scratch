import pandas as pd 
import numpy as np 
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import math
from sklearn.metrics import mean_squared_error

class GDRidgeRgression:
    def __init__(self, alpha: float=0.01, epochs: int=100, learning_rate: float=0.01) -> None:
        """Initializing epochs, learning_rate, alpha

        Args:
            epochs (int, optional): _description_. Defaults to 10.
            learning_rate (float, optional): _description_. Defaults to 0.01.
            alpha (float, optional): _description_. Defaults to 0.01.
        """
        self.epocs = epochs
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.beta = None
        
    def fit(self, X_train: pd.DataFrame, y_train: pd.DataFrame) -> pd.Series:
        """Fits the training data to Gradient GD Linear Regression model

        Args:
            X_train (pd.DataFrame): Independent Variable
            y_train (pd.DataFrame): Dependent Variable

        Returns:
            pd.Series: returns intercept and coefficient
        """
        # X_train = X.values
        # y_train = y.values
        
        intercept = 1
        self.beta = np.ones((X_train.shape[1]+1,1))
        X_train = np.insert(X_train, 0, intercept, axis=1) 
        
        for _ in range(self.epocs):
            
            db = ((X_train.T@X_train)@self.beta) + (self.alpha*self.beta) - (X_train.T@y_train)
            
            step_b = self.learning_rate*db
            
            self.beta = self.beta - step_b
        
        return self.beta
    
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Predicts on test data using the trained model

        Args:
            X_test (pd.DataFrame): Independent test Variable

        Returns:
            pd.Series: 
        """
        intercept = 1
        X_test = np.insert(X, 0, intercept, axis=1)
        
        return (X_test@self.beta)
    
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
    # df = pd.read_csv("/Users/navaneeth/Documents/Work/MACHINE_LEARNING_FROM_SCRATCH/dataset/Student_Performance.csv")
    # independent_var = ['Hours Studied','Previous Scores','Extracurricular Activities','Sleep Hours','Sample Question Papers Practiced']
    # X = df[independent_var]
    # y = df.iloc[:,-1:]
    X,y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    modelsk = LinearRegression().fit(X_train, y_train)
    model = GDRidgeRgression()
    beta = model.fit(X_train, y_train.reshape(-1,1))
    y_pred = model.predict(X_test)
    print(math.sqrt(mean_squared_error(y_pred, y_test)))
    