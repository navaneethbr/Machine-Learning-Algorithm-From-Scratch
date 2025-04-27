from typing import Union, List
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

class MultipleRidgeRegression:
    """This class implements Multiple Ridge Regression, with one or more independent varibale and one dependent variable
    """
    def __init__(self, alpha: float=0.01) -> None:
        self.beta = None
        self.alpha = alpha
    
    def fit(self, X: List[List[Union[int, float]]], y: List[List[Union[int, float]]]) -> List[List[float]]:
        """Funciton to find the coefficient and the intercept.

        Args:
            X (array): Training Data (Independent Variable)
            y (array): Training Data (Dependent Variable)

        Returns:
        beta value: array
            returns the intercept and coefficient values
        """
        # inserting 1 for intercept in the train matrix
        intercept = 1
        X = np.insert(X, 0, intercept, axis=1)
        
        I = np.identity(X.shape[1])
        
        # calculating the intercept and coefficient
        self.beta = np.linalg.inv((X.T@X)+(self.alpha*I))@(X.T@y)
        return self.beta
    
    def predict(self, X_train: List[List[Union[int, float]]]) -> List[List[float]]:
        """Function to predict values for given test data.

        Args:
            X_train (array): Testing data (Independent Variable)

        Returns:
        y_predicted : array of integers
                returns the predicted values for give test data
        """
        intercept = 1
        X_train = np.insert(X_train, 0, intercept, axis=1)
        y_pred = X_train@self.beta
        return y_pred
    
    def rmse(self, y_acutal: List[Union[int, float]] ,y_pred: List[Union[int, float]]) -> float:
        """Evaluation metrics Room Mean Squared Error(RMSE).

        Args:
            y_acutal (array): Actual Output 
            y_pred (array): Predicted Output

        Returns:
        RMSE error: integer
                retuns the error for give model
        """
        n = y_acutal.shape
        diff = (y_pred-y_acutal)
        return float(np.sqrt(((diff.T)@(diff))/n))
    
    
if __name__ == "__main__":
    
    # prediction from the model we built 
    our_model = MultipleRidgeRegression()
    
    df = pd.read_csv("/Users/navaneeth/Documents/Work/MACHINE_LEARNING_FROM_SCRATCH/dataset/Student_Performance.csv")
    
    independent_var = ['Hours Studied','Previous Scores','Extracurricular Activities','Sleep Hours','Sample Question Papers Practiced']
    X_train, y_train, X_test, y_test = np.array(np.array(df.loc[:900, independent_var])), np.array(df.loc[:900, "Performance Index"]), np.array(df.loc[901:,independent_var]), np.array(df.loc[901:, "Performance Index"])
    
    beta = our_model.fit(X_train, y_train)
    y_pred = our_model.predict(X_test)
    our_model_error = our_model.rmse(y_test, y_pred)
    
    # prediction form sklearn Linear Regression
    sklearn_model = linear_model.LinearRegression()
    
    X_train_copy = X_train.copy()
    X_train_copy = X_train_copy
    
    sklearn_model.fit(X_train_copy, y_train)
    y_pred_copy = sklearn_model.predict(X_test)
    sklearn_error = np.sqrt(mean_squared_error(y_test, y_pred_copy)) 
    
    # comparing the coefficient and intercept from our model and sklearn model 
    print("coefficient(our model) : ", beta[1:])
    print("coefficient(sklearn)   : ", sklearn_model.coef_)
    print()
    print("intercept(our model)   : ", beta[0])
    print("intercept(sklearn)     : ", sklearn_model.intercept_)
    print()
    # compartint the error from our model and sklearn model
    print("Error(our model)       : ", our_model_error)
    print("Error(sklearn)         : ", sklearn_error)
    