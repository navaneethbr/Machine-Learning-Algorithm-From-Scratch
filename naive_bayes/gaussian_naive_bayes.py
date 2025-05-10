from typing import List
import numpy as np
from numpy.typing import ArrayLike
from sklearn import datasets
from sklearn.model_selection import train_test_split
import sys


class GaussianNaiveBayes:
    def __init__(self):
        pass
    
    def _gaussian_distribution(self, std: float, mean: float, value: List[float]):
        """returns the gaussian distribution for give std & mean

        Args:
            std (float): _description_
            mean (float): _description_
            value (List[float]): _description_

        Returns:
            _type_: _description_
        """
        exp_cal = -np.square(value-mean)/(2*np.square(std))
        distribution = np.exp(exp_cal)/(np.sqrt(2*np.pi*np.square(std)))
        return distribution
    
    def fit(self, X: ArrayLike, y: ArrayLike) -> None:
        """This take the dependent and independent features and calculated the mean, standard deviation and prior for all unique class. 

        Args:
            X (ArrayLike): Training Data (Independent Variable)
            y (ArrayLike): Training Data (Dependent Variable)
        """
        _, n_features = X.shape
        self.unique_value = np.unique(y)
        uniq_val_len = len(self.unique_value)
        self.features_mean = np.empty((uniq_val_len, n_features))
        self.features_std = np.empty((uniq_val_len, n_features))
        self.features_prior = np.empty((uniq_val_len))
        
        for idx, value in enumerate(self.unique_value):
            X_ft = X[y==value, :]
            self.features_mean[idx] = np.mean(X_ft, axis=0)
            self.features_std[idx] = np.std(X_ft, axis=0)
            self.features_prior[idx] = len(X_ft)/len(y)
        
    def predict(self, X: ArrayLike) -> ArrayLike:
        """predicts the output for the given data using the data calculated in fit method.

        Args:
            X (ArrayLike): Testing data (Independent Variable)

        Returns:
            ArrayLike: returns the predicted values for give data
        """
        y_pred = np.empty(len(X))
        
        for o_idx, sample in enumerate(X):
            max_poster = -sys.float_info.max
            max_idx = 0
            for idx, value in enumerate(self.unique_value):
                prob = self._gaussian_distribution(self.features_std[idx], self.features_mean[idx], sample)
                posterior = np.sum(np.log(prob))
                prior = np.log(self.features_prior[idx])
                posterior = posterior + prior
                if posterior > max_poster:
                    max_poster = posterior
                    max_idx = idx
            y_pred[o_idx] = max_idx       
            
        return y_pred
                
                
        
if __name__ == "__main__":
    
    X, y = datasets.make_classification(
        n_samples=1000, n_features=10, n_classes=2, random_state=123
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123
    )

    nb = GaussianNaiveBayes()
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_test)
    
    accuracy = np.sum(y_test == y_pred) / len(y_test)
    
    print("Naive Bayes classification accuracy", accuracy)