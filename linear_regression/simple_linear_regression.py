from typing import Union, List
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

class SimpleLinearRegression:
    """This class implements simple Linear Regression, with one independent and one dependent variable 
    """
    def __init__(self) -> None:
        self.a = 1
        self.b = 0
    
    def fit(self, X: List[Union[int, float]], y: List[Union[int, float]]) -> float :
        """Funciton to find the coefficient and the intercept.

        Args:
        X (array): Training Data (Independent Variable)
        y (array): Training Data (Dependent Variable)

        Returns:
        a, b: int or tuple(of size 2)
            returns the coefficient and the intercept
        """
        X_mean, y_mean = np.mean(X), np.mean(y)
        sum_denominator, sum_numerator = 0, 0
        
        for i in range(len(X)):
            sum_numerator += (y[i]-y_mean)*(X[i]-X_mean)
            sum_denominator += (X[i]-X_mean)**2
        self.a = sum_numerator/sum_denominator
        self.b = y_mean - (self.a*X_mean)
        
        return self.a, self.b

    def predict(self, X_test: List[Union[int, float]]) -> list:
        """Fuction to predict values for given test data.

        Args:
        X_test (array): Testing data (Independent Variable)

        Returns:
        y_predicted : array of integers
                    returns the predicted values for give test data
        """
        return (self.a*X_test)+self.b
        
    def rmse(self, y_acutal: List[Union[int, float]], y_pred: List[Union[int, float]]) -> float:
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
    our_model = SimpleLinearRegression()
    
    df = pd.read_csv("/Users/navaneeth/Documents/Work/Machine Learning from Scratch/Dataset/Student_Performance.csv")
    
    X_train, y_train, X_test, y_test = np.array(df.loc[:901, "Hours Studied"]), np.array(df.loc[:901, "Performance Index"]), np.array(df.loc[901:,'Hours Studied']), np.array(df.loc[901:, "Performance Index"])
    
    a, b = our_model.fit(X_train, y_train)
    y_pred = our_model.predict(X_test)
    our_model_error = our_model.rmse(y_test, y_pred)
    
    # prediction form sklearn Linear Regression
    sklearn_model = linear_model.LinearRegression()
    
    X_train_copy = X_train.copy()
    X_train_copy = X_train_copy.reshape(-1, 1)
    
    sklearn_model.fit(X_train_copy, y_train)
    y_pred_copy = sklearn_model.predict(X_test.reshape(-1,1))
    sklearn_error = np.sqrt(mean_squared_error(y_test, y_pred_copy)) 
    
    # comparing the coefficient and intercept from our model and sklearn model 
    print("coefficient(our model) : ", a)
    print("coefficient(sklearn)   : ",float(sklearn_model.coef_))
    print()
    print("intercept(our model)   : ", b)
    print("intercept(sklearn)     : ", sklearn_model.intercept_)
    print()
    # compartint the error from our model and sklearn model
    print("Error(our model)       : ", our_model_error)
    print("Error(sklearn)         : ", sklearn_error)
    
    
    