import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from numpy.typing import ArrayLike

class KMeans:
    def __init__(self, *, clusters: int, n_iter: int):
        """
        Initializes the KMeans clustering model.

        Args:
            clusters (int): Number of clusters.
            n_iter (int): Number of iterations to run the algorithm.
        """
        self.clusters = clusters
        self.n_iter = n_iter
        self.centroids = None
        
    def fit(self, X: ArrayLike):
        """
        Fits the KMeans model to the data.

        Args:
            X (ArrayLike): Containing the training data.

        Returns:
            np.ndarray: Final centroid positions after fitting.
        """
        # random.seed(42)
        n_smaple, _ = X.shape
        centroid_idx = random.sample(range(0, n_smaple), self.clusters)
        self.centroids = X[centroid_idx]
        
        for _ in range(self.n_iter):
            min_distance_idx = self._calculate_clusters_distance(self.centroids, X)
            self.centroids = self._calculate_new_centroids(min_distance_idx, self.centroids, X)                
            
        return self.centroids
    
    def predict(self, X: ArrayLike):
        """
        Predicts the closest cluster each sample in X belongs to.

        Args:
            X (ArrayLike): Testing data.

        Returns:
            np.ndarray: An array of cluster indices for each sample.
        """
        cluster_array = self._calculate_clusters_distance(self.centroids, X)
        prediction = np.empty(X.shape[0])
        for row in cluster_array:
            prediction[row[1]] = row[0]
        return prediction
                    
            
    def _calculate_new_centroids(self, min_distance_idx, centroids, X):
        """
        Recalculates centroids as the mean of all points assigned to each cluster.

        Args:
            min_distance_idx (np.ndarray): Array of shape (n_samples, 2) with [cluster_index, sample_index].
            centroids (np.ndarray): Current centroid positions.
            X (np.ndarray): Input data.

        Returns:
            np.ndarray: New centroid positions.
        """
        coordinates = {i:np.empty((0, X.shape[1])) for i in range(centroids.shape[0])}
        new_centroids = np.empty((0, centroids.shape[1]))
        for dist in min_distance_idx:
            coordinates[dist[0]] = np.append(coordinates[dist[0]], [X[dist[1]]], axis=0)
            
        for i, centroid_array in coordinates.items():
            new_coordinates = np.mean(centroid_array, axis=0)
            new_centroids = np.append(new_centroids, [new_coordinates], axis=0)
            
        return new_centroids
            
            
    def _calculate_clusters_distance(self, centroids, X):
        """
        Calculates the closest centroid for each point in X.

        Args:
            centroids (np.ndarray): Current centroid positions.
            X (np.ndarray): Input data.

        Returns:
            np.ndarray: Array of shape (n_samples, 2) where each row contains [closest_cluster_index, sample_index].
        """
        distance = []
        for centroid in centroids:
            distance.append(self._euclidean_distance(centroid, X))
        distance = np.array(distance)

        row, col = distance.shape      
        cluster_array = []
                
        for j in range(col):
            min_val = distance[0, j]
            min_idx = None
            for i in range(row):
                distance_val = distance[i, j]
                if distance_val <= min_val:
                    min_val = distance_val
                    min_idx = [i, j]
    
            cluster_array.append(min_idx)
        
        return np.array(cluster_array)
        
    def _euclidean_distance(self, centroid, data_points):
        """
        Computes Euclidean distance from a centroid to each point in the dataset.

        Args:
            centroid (np.ndarray): A single centroid.
            data_points (np.ndarray): Data points to measure against.

        Returns:
            np.ndarray: Array of distances.
        """
        return np.sqrt(np.sum(np.square(data_points-centroid), axis=1))

    def plot(self, X, n_components=2):
        """
        Plots the clustered data in 2D using PCA for dimensionality reduction.

        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features).
            n_components (int): Number of dimensions to reduce to for plotting (default is 2).

        Raises:
            ValueError: If input X is not a 2D array or is empty.
        """
        # Check that X is a 2D array (samples, features)
        if len(X.shape) != 2 or X.shape[0] == 0:
            raise ValueError("X must be a 2D array with samples as rows and features as columns.")
        
        # Reduce the dimensions to 2 using PCA or any other dimensionality reduction method
        if X.shape[1] > 2:
            pca = PCA(n_components=n_components)
            X_reduced = pca.fit_transform(X)  # Reduce to 2D
        else:
            X_reduced = X  # Already 2D, no reduction necessary
        
        # Get the cluster assignments (predictions)
        predictions = self.predict(X)
        
        # Create a plot
        fig, ax = plt.subplots(figsize=(12, 8))

        # Plot each data point in the corresponding cluster
        for i in range(self.clusters):
            # Get points assigned to cluster i
            cluster_points = X_reduced[predictions == i]
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {i + 1}")
        
        # Plot the centroids (reduced to 2D if necessary)
        centroids_reduced = pca.transform(self.centroids) if X.shape[1] > 2 else self.centroids
        ax.scatter(centroids_reduced[:, 0], centroids_reduced[:, 1], marker="x", color="black", linewidth=2, label="Centroids")

        # Add legends and labels
        ax.legend()
        plt.title("K-Means Clustering")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.show()


if __name__ == '__main__':
    np.random.seed(42)
    from sklearn.datasets import make_blobs

    X, y = make_blobs(
        centers=3, n_samples=500, n_features=3, shuffle=True, random_state=40
    )
    
    clusters = len(np.unique(y))
    model = KMeans(clusters=clusters, n_iter=10)
    model.fit(X)
    y_pred = model.predict(X)
    model.plot(X)