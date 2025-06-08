import numpy as np 
from numpy.typing import ArrayLike
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets


class LinearSVC:
    """
    A simple implementation of Linear Support Vector Classifier using hinge loss and gradient descent.

    Args:
    learning_rate : float -> Defaults to 0.01.
        The step size used for updating weights during gradient descent.
    c : float -> Defaults to 1.0.
        Regularization parameter. Controls the trade-off between achieving a low error on the training data and minimizing margin width (street).
    epochs : int -> Defaults to 100.
        Number of passes over the training dataset.
    """
    def __init__(self, learning_rate: float=0.01, c: float=1.0, epochs: int=100):
        self.lr = learning_rate 
        self.c = c
        self.C = c
        self.epochs = epochs
        self.w = None
        self.b = None
    
    def fit(self, X: ArrayLike, y: ArrayLike):
        """
        Train the Linear SVM model using the training data.

        Args:
        X : array-like
            Training data.
        y : array-like
            Target labels.
        """
        self.w = np.random.rand(X.shape[1])
        self.b = np.random.rand(1)[0]
        
        for _ in range(self.epochs):
            dw, db = self._calculate_gradient(X=X, y=y)
            self.w = self.w - self.lr*(dw)
            self.b = self.b - self.lr*(db)
        
    def _calculate_gradient(self, X, y):
        """
        Compute the gradients of the loss function with respect to weights and bias.

        Args:
        X : array-like
            Input features.
        y : array-like
            Target labels.

        Returns:
        dw : np.ndarray
            Gradient of the loss with respect to weights.
        db : float
            Gradient of the loss with respect to bias.
        """
        dw = np.zeros(X.shape[1])
        db = 0
        for i, x_i in enumerate(X):
            pred = y[i]*(x_i@self.w + self.b)
            hing_loss = pred < 1 # hing loss 
            if hing_loss:
                dw = dw + (y[i]*x_i)
                db = db + (y[i])
        dw = self.w - self.c*dw
        db = -self.c*db
        return dw, db         
      
    def predict(self, X: ArrayLike):
        """
        Make predictions using the trained Linear SVM model.

        Args:
        X : array-like
            Input data.

        Returns:
        np.ndarray
            Predicted class labels (-1 or 1).
        """
        pred = []
        if self.w is None:
            raise ValueError("No values found for weigth. Please train the model")
        
        results = X@self.w + self.b
        pred.append(np.sign(results))
            
        return np.array(pred)
            
    def plot_decision_boundary(self, X: ArrayLike, y: ArrayLike):
        """
        Plot the decision boundary and margin lines for 2D data.

        Args:
        X : array-like
            Input data.
        y : array-like
            Target labels.
        """
        X = np.array(X)
        y = np.array(y)

        if X.shape[1] != 2:
            raise ValueError("Plotting only works with 2D data")

        plt.figure(figsize=(8, 6))

        # Plot data points
        for label in np.unique(y):
            plt.scatter(X[y == label][:, 0], X[y == label][:, 1], label=f"Class {label}")

        # Create a grid for plotting
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        x_vals = np.linspace(x_min, x_max, 100)

        # Compute corresponding y values for decision boundary and margins
        w, b = self.w, self.b
        if w[1] != 0:
            # Decision boundary
            y_vals = -(w[0] * x_vals + b) / w[1]
            # Margin lines
            margin_pos = -(w[0] * x_vals + b - 1) / w[1]
            margin_neg = -(w[0] * x_vals + b + 1) / w[1]

            plt.plot(x_vals, y_vals, 'k--', label="Decision Boundary (w·x + b = 0)")
            plt.plot(x_vals, margin_pos, 'r--', label="Margin +1 (w·x + b = 1)")
            plt.plot(x_vals, margin_neg, 'b--', label="Margin -1 (w·x + b = -1)")
        else:
            # Vertical line case (avoid division by zero)
            x_line = -b / w[0]
            plt.axvline(x=x_line, color='k', linestyle='--', label="Decision Boundary")

        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.title("SVM Decision Boundary and Margins")
        plt.legend()
        plt.grid(True)
        plt.show()

        
if __name__ == "__main__":
    X, y = datasets.make_blobs(
        n_samples=50,
        n_features=2,
        centers=2,
        cluster_std=1.05,
        random_state=40
    )
    y = np.where(y == 0, -1, 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123
    )
    
    model = LinearSVC()
    
    model.fit(X=X_train, y=y_train)
    predictions = model.predict(X=X_test)
    print(np.sum(predictions==y_test)/len(y_test))
    model.plot_decision_boundary(X=X_train, y=y_train)
    model.plot_decision_boundary(X=X_test, y=y_test)
    
        
        