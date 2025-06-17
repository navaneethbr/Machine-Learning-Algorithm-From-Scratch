import numpy as np
import pandas as pd
from collections import Counter

from sklearn.model_selection import train_test_split

class Node:
    def __init__(self, *, threshold=None, left=None, right=None, feature=None, value=None):
        """
        Represents a node in the regression tree.

        Attributes:
            threshold (float): The threshold value used for splitting.
            left (Node): Left child node.
            right (Node): Right child node.
            feature (int): Feature index used for the split.
            value (float): Predicted value if it's a leaf node.
        """
        self.left = left
        self.right = right
        self.threshold = threshold 
        self.feature = feature 
        self.value = value
        
    def is_leaf_node(self):
        """
        Check if the current node is a leaf node.

        Returns:
            bool: True if it's a leaf node, else False.
        """
        return self.value is not None
    
class RegressionTree:
    def __init__(self, *, max_depth=10, min_samples_split=2, n_features=None):
        """
        A simple implementation of a decision tree for regression.

        Parameters:
            max_depth (int): Maximum depth of the tree.
            min_samples_split (int): Minimum number of samples required to split a node.
            n_features (int or None): Number of features to consider when looking for the best split.

        Attributes:
            root (Node): The root node of the regression tree.
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.root = None
        
    def fit(self, X, y):
        """
        Fits the regression tree to the training data.

        Args:
            X (np.ndarray): Training features.
            y (np.ndarray): Target values.
        """
        self.n_features = X.shape[1] if self.n_features is None else min(X.shape[1], self.n_features)
        self.root = self._grow_tree(X, y)
        
    def _grow_tree(self, X, y, depth=0):
        """
        Recursively grows the regression tree.

        Args:
            X (np.ndarray): Subset of training features.
            y (np.ndarray): Subset of target values.
            depth (int): Current depth of the tree.

        Returns:
            Node: The current node in the tree.
        """
        n_sample, n_feats = X.shape
        if depth>=self.max_depth or n_sample<self.min_samples_split:
            value = self._mean_leaf(y)
            return Node(value=value)
        
        features = np.random.choice(n_feats, self.n_features, replace=False)
        
        feature, threshold = self._best_split(X, y, features)
        left_indx, right_indx = self._split(X, feature, threshold)
        left = self._grow_tree(X[left_indx,:], y[left_indx], depth+1)
        right = self._grow_tree(X[right_indx,:], y[right_indx], depth+1)
        
        return Node(left=left, right=right, threshold=threshold, feature=feature)
        
    def _best_split(self, X, y, features):
        """
        Finds the best feature and threshold for splitting the data.

        Args:
            X (np.ndarray): Features.
            y (np.ndarray): Target values.
            features (np.ndarray): Subset of features to evaluate.

        Returns:
            tuple: (best_feature, best_threshold)
        """
        error = -1
        min_error_feature_split = None
        split_threshold = None
        
        for feature in features:
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                weighted_avg_error = self._split_error(X, y, feature, threshold)
                if weighted_avg_error > error:
                    error = weighted_avg_error
                    min_error_feature_split = feature
                    split_threshold = threshold
                
        return min_error_feature_split, split_threshold
            
    def _split_error(self, X, y, feature, threshold):
        """
        Calculates the reduction in mean squared error from a potential split.

        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target values.
            feature (int): Feature index to split on.
            threshold (float): Threshold value for the split.

        Returns:
            float: Reduction in error due to the split.
        """
        left_indx, right_indx = self._split(X, feature, threshold)
        
        if len(left_indx)==0 or len(right_indx)==0:
            return 0
        
        weight_left, weight_right = len(y[left_indx])/len(y), len(y[right_indx])/len(y)
        parent = self._mean_square_error(y)
        left_error, right_error = self._mean_square_error(y[left_indx]), self._mean_square_error(y[right_indx])
        
        weight_avg = parent - ((weight_left*left_error)+(weight_right*right_error))
        return weight_avg
    
    def _split(self, X, feature, threshold):
        """
        Splits the data based on a feature and threshold.

        Args:
            X (np.ndarray): Feature matrix.
            feature (int): Feature index to split on.
            threshold (float): Threshold value for splitting.

        Returns:
            tuple: (left_indices, right_indices)
        """
        left_split = np.argwhere(X[:, feature]<=threshold).flatten()
        right_split = np.argwhere(X[:, feature]>threshold).flatten()
        return left_split, right_split
            
    def _mean_square_error(self, y):
        """
        Calculates the mean squared error of the target values.

        Args:
            y (np.ndarray): Target values.

        Returns:
            float: Mean squared error.
        """
        mean = np.mean(y)
        return np.sum(np.square(y-mean))
            
    def _mean_leaf(self, y):
        """
        Calculates the value for a leaf node (mean of the target values).

        Args:
            y (np.ndarray): Target values.

        Returns:
            float: Mean of the target values.
        """
        return np.mean(y)
    
    def predict(self, X):
        """
        Predicts target values for given feature inputs.

        Args:
            X (np.ndarray): Feature matrix.

        Returns:
            np.ndarray: Predicted values.
        """
        return np.array([self._traverse_node(x, self.root) for x in X])
    
    def _traverse_node(self, X, node):
        """
        Traverses the tree recursively to make a prediction for a single sample.

        Args:
            X (np.ndarray): Single feature vector.
            node (Node): Current node in the tree.

        Returns:
            float: Predicted value.
        """
        if node.is_leaf_node():
            return node.value
        
        if X[node.feature] <= node.threshold:
            return self._traverse_node(X, node.left)
        return self._traverse_node(X, node.right)
    
if __name__ == "__main__":
    data = pd.read_csv('/Users/navaneeth/Documents/Work/Machine Learning from Scratch/Dataset/airfoil_noise_data.csv')
    
    X = data.iloc[:, :-1].values
    Y = data.iloc[:, -1].values.reshape(-1,1)
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=41)
    
    model = RegressionTree()
    model.fit(X_train, Y_train)
    
    predictions = model.predict(X_test)
    print(predictions)
    print(Y_test)
    