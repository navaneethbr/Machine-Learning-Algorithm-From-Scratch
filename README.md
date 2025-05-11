# Machine-Learning-Algorithm-From-Scratch

This repository contains implementations of various machine learning algorithms built from scratch, without using high-level machine learning libraries like Scikit-learn. The aim is to provide a deeper understanding of how these algorithms work internally by building them using only basic Python libraries such as NumPy.

## Implemented Algorithms

### Logistic Regression
- **`perceptron_logistic_regression.py`**  
  Implements logistic regression using the perceptron approach, which is a type of linear classifier that maps inputs to binary outputs using a linear decision boundary.

- **`gd_logistic_regression.py`**  
  Implements logistic regression using gradient descent, an optimization algorithm that minimizes the cost function by iteratively moving towards the steepest descent.

- **`softmax_regression.py`**  
  Implements multiclass logistic regression using the softmax function, which generalizes logistic regression to multiple classes by calculating the probability distribution over multiple classes.

### Trees
- **`decisiontree.py`**  
  Implements a decision tree classifier using the ID3 algorithm, which selects the attribute with the highest information gain to split the data at each node.

- **`regressiontree.py`**  
  Implements a decision tree for regression tasks, where the tree predicts continuous values by splitting the data based on feature values.

### Regularization Techniques
- **`gd_lasso_regression.py`**  
  Implements Lasso regression using gradient descent, which adds an L1 penalty to the cost function to encourage sparsity in the model coefficients.

- **`gd_ridge_regression.py`**  
  Implements Ridge regression using gradient descent, which adds an L2 penalty to the cost function to prevent overfitting by shrinking the coefficients.

- **`multiple_ridge_regression.py`**  
  Implements Ridge regression with multiple features, extending the regularization technique to datasets with more than one feature.

- **`simple_ridge_regression.py`**  
  Implements Ridge regression with a single feature, applying L2 regularization to a simple linear regression model.

### Clustering
- **`k_means.py`**  
  Implements the K-Means clustering algorithm, which partitions data into K clusters by minimizing the variance within each cluster.

### Linear Regression
- **`multiple_linear_regression.py`**  
  Implements multiple linear regression, modeling the relationship between two or more features and a continuous target variable.

- **`lr_stochastic_gd.py`**  
  Implements linear regression using stochastic gradient descent, an optimization technique that updates the model parameters using one data point at a time.

- **`lr_using_gradient_descent.py`**  
  Implements linear regression using batch gradient descent, an optimization technique that updates the model parameters using the entire dataset.

- **`lr_mini_batch_gd.py`**  
  Implements linear regression using mini-batch gradient descent, an optimization technique that updates the model parameters using a small subset of the dataset.

- **`simple_linear_regression.py`**  
  Implements simple linear regression, modeling the relationship between a single feature and a continuous target variable.

### Naive Bayes
- **`gaussian_naive_bayes.py`**  
  Implements the Gaussian Naive Bayes classifier, which assumes that the features follow a normal distribution and uses Bayes' theorem for classification.

---

Feel free to explore, modify, and contribute to the code to deepen your understanding of how machine learning algorithms work at a fundamental level.
