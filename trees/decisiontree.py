import numpy as np 
from collections import Counter
from sklearn import datasets
from sklearn.model_selection import train_test_split
from numpy.typing import ArrayLike
from typing import Literal

class Node:
    """
    A class representing a node in the decision tree.

    Args:
        feature (int): Index of the feature to split on.
        threshold (float): Threshold value for the feature split.
        left (Node): Left child node.
        right (Node): Right child node.
        value (int): Predicted class label for leaf nodes.
    """
    def __init__(self, threshold=None, feature=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        
    def is_leaf_node(self):
        """
        Check whether the node is a leaf node.
        """
        return self.value is not None
        
class DecisionTree:
    """
    A decision tree classifier.

    Args:
        min_sample_split (int): The minimum number of samples required to split an internal node.
        max_depth (int): The maximum depth of the tree.
        n_features (int, optional): The number of features to consider when looking for the best split.
        criterion (str): The function to measure the quality of a split ('gini' or 'entropy').
    """
    def __init__(self, *, min_sample_split: int=2, max_depth: int=100, n_features: int=None, criterion: Literal['gini', 'entropy']='gini'):
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.criterion = criterion
        self.root = None
        
    def fit(self, X: ArrayLike, y: ArrayLike):
        """
        Build the decision tree classifier from the training set (X, y).

        Args:
            X (ArrayLike): Feature matrix.
            y (ArrayLike): Target labels.
        """
        self.n_features = X.shape[1] if self.n_features is None else min(X.shape[1], self.n_features)
        self.root = self._grow_tree(X, y)
        
    def _grow_tree(self, X: ArrayLike, y:ArrayLike, depth: int=0):
        """
        Recursively build the decision tree.

        Args:
            X (ArrayLike): Feature matrix.
            y (ArrayLike): Target labels.
            depth (int): Current depth of the tree.

        Returns:
            Node: The root node of the constructed subtree.
        """
        n_sample, n_feat = X.shape
        n_label = len(np.unique(y))
        
        if depth>=self.max_depth or n_label==1 or n_sample<self.min_sample_split:
            leaf_value = self._common_label(y)
            return Node(value=leaf_value)
        
        feats = np.random.choice(n_feat, self.n_features, replace=False)
        
        feature, threshold = self._split_tree(X, y, feats)
        left_index, right_index = self._split(X, threshold, feature)
        left = self._grow_tree(X[left_index,:], y[left_index], depth+1)
        right = self._grow_tree(X[right_index,:], y[right_index], depth+1)
        
        return Node(threshold, feature, left, right)
        
    def _split_tree(self, X: ArrayLike, y: ArrayLike, features: ArrayLike):
        """
        Find the best feature and threshold to split on.

        Args:
            X (ArrayLike): Feature matrix.
            y (ArrayLike): Target labels.
            features (ArrayLike): Subset of features to evaluate.

        Returns:
            tuple: Best feature and threshold for the split.
        """
        gain = -1
        split_threshold = None
        split_feature = None 
        
        if self.criterion == 'gini':
            info_gain_cal = self._information_gain_gini
        elif self.criterion == 'entropy':
            info_gain_cal = self._information_gain_entropy
        else:
            raise ValueError('enter the correct criterion gini or entropy')
            
        for feature in features:
            thresholds = np.unique(X[:,feature])
            for threshold in thresholds:
                
                info_gain = info_gain_cal(X, y, threshold, feature)
                
                if info_gain > gain:
                    gain = info_gain
                    split_feature = feature
                    split_threshold = threshold
                    
        return split_feature, split_threshold
                              
    def _information_gain_entropy(self, X: ArrayLike, y: ArrayLike, threshold: ArrayLike, feature: ArrayLike):
        """
        Compute the information gain using entropy.

        Args:
            X (ArrayLike): Feature matrix.
            y (ArrayLike): Target labels.
            threshold (ArrayLike): calculated threshold to split.
            feature (ArrayLike): feature that needs to be split.

        Returns:
            float: Information gain.
        """
        # entropy of parent
        parent_entropy = self._entropy(y)
        
        
        left_index, right_index = self._split(X, threshold, feature)
        if len(left_index)==0 or len(right_index)==0:
            return 0
        
        # child entropy
        left_entropy = self._entropy(y[left_index])
        right_entropy = self._entropy(y[right_index])
        child_entropy = (len(left_index)/len(y))*left_entropy+(len(right_index)/len(y))*right_entropy
        
        #infomation gain
        info_gain =  parent_entropy - child_entropy
        return info_gain
    
    def _information_gain_gini(self, X: ArrayLike, y: ArrayLike, threshold: ArrayLike, feature: ArrayLike):
        """
        Compute the information gain using gini.

        Args:
            X (ArrayLike): Feature matrix.
            y (ArrayLike): Target labels.
            threshold (ArrayLike): calculated threshold to split.
            feature (ArrayLike): feature that needs to be split.

        Returns:
            float: Information gain.
        """
        # gini of parent
        parent_gini = self._gini(y)
        
        left_index, right_index = self._split(X, threshold, feature)
        if len(left_index)==0 or len(right_index)==0:
            return 0
        
        # gini of child
        left_gini = self._gini(y[left_index])
        right_gini = self._gini(y[right_index])
        child_gini = (len(left_index)/len(y))*left_gini+(len(right_index)/len(y))*right_gini
        
        #infomation gain
        info_gain = parent_gini - child_gini
        return info_gain
        
    def _split(self, X: ArrayLike, threshold: ArrayLike, feature: ArrayLike):
        """
        Split the dataset based on the given feature and threshold.

        Returns:
            tuple: Indices for left and right splits.
        """
        left_index = np.argwhere(X[:, feature]<=threshold).flatten()
        right_index = np.argwhere(X[:, feature]>threshold).flatten()
        return left_index, right_index
        
    def _entropy(self, y: ArrayLike):
        """
        Compute entropy of the label distribution.

        Returns:
            float: Entropy value.
        """
        label_count = np.bincount(y)
        prob = label_count/len(y)
        return -np.sum([(p * np.log(p)) for p in prob if p>0])
    
    def _gini(self, y: ArrayLike):
        """
        Compute Gini impurity of the label distribution.

        Returns:
            float: Gini impurity value.
        """
        label_count = np.bincount(y)
        prob = label_count/len(y)
        return (1 - np.sum(np.square(prob)))
        
    def _common_label(self, y: ArrayLike):
        """
        Find the most common label in the dataset.

        Returns:
            int: Most frequent label.
        """
        count = Counter(y)
        return count.most_common(1)[0][0]
    
    def predict(self, X: ArrayLike):
        """
        Predict class labels for samples in X.

        Args:
            X (ArrayLike): Feature matrix.

        Returns:
            np.ndarray: Predicted class labels.
        """
        return np.array([self._traverse_node(x, self.root) for x in X])
    
    def _traverse_node(self, X: ArrayLike, node: Node):
        """
        Traverse the tree to make a prediction for a single sample.

        Args:
            X (ArrayLike): Feature vector.
            node (Node): Current node in the tree.

        Returns:
            int: Predicted label.
        """
        if node.is_leaf_node():
            return node.value
        
        if X[node.feature] <= node.threshold:
            return self._traverse_node(X, node.left)
        return self._traverse_node(X, node.right)
    
if __name__=='__main__':
    data = datasets.load_breast_cancer()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )
    
    clf = DecisionTree(criterion='gini')
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    def accuracy(y_test, y_pred):
        return np.sum(y_test == y_pred) / len(y_test)

    acc = accuracy(y_test, predictions)
    print(acc)