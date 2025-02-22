import numpy as np
import pandas as pd
import sys
from sklearn.preprocessing import OrdinalEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_samples, silhouette_score, calinski_harabasz_score, rand_score
from preprocessor import preprocess_data, load_data

# try it with `python kmeans.py csv/iris.csv 3`

class Kmeans:
    def __init__(self, n_clusters=2, tol=1e-4, max_iter=300, random_state=42):
        self.n_clusters = n_clusters
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state
        self.labels = None
        self.centroids = None
        
        if self.random_state:
            np.random.seed(self.random_state) # for reproducibility

    def initialize_centroids(self, X):
        '''
        Initialize centroids using KMeans++
        '''
        n_samples, n_features = X.shape
        centroids = np.zeros((self.n_clusters, n_features))
        centroids[0] = X[np.random.randint(n_samples)]
        
        for i in range(1, self.n_clusters):
            distances = np.min([np.linalg.norm(X - c, axis=1)**2 for c in centroids[:i]], axis=0)
            probs = distances / np.sum(distances)
            centroids[i] = X[np.random.choice(n_samples, p=probs)]
        return centroids

    def assign_clusters(self, X):
        '''
        Assign data points to the cluster with the closest centroid (Euclidean distance)
        '''
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    def recompute_centroids(self, X):
        '''
        Compute new centroids as the mean of the assigned points in the cluster
        '''
        return np.array([X[self.labels == i].mean(axis=0) if np.any(self.labels == i) else self.centroids[i]
            for i in range(self.n_clusters)
        ])

    def compute_sse(self, X, centroids):
        '''
        Compute Sum of Squared Errors (SSE) for each point from the centroids
        '''
        return np.sum((X - centroids[self.labels])**2)
    
    def fit(self, X):
        '''
        Fit KMeans model to the data with SSE-based stopping criterion
        '''
        self.centroids = self.initialize_centroids(X)
        prev_sse = 0
        for i in range(self.max_iter):
            self.labels = self.assign_clusters(X)
            new_centroids = self.recompute_centroids(X)
            
            sse = self.compute_sse(X, new_centroids)
            #print(prev_sse)
            if prev_sse != 0 and abs(prev_sse - sse) / prev_sse < self.tol:
                #print(f"Converged at iteration {i}")
                break

            self.centroids = new_centroids
            prev_sse = sse

    def predict(self, X):
        '''
        Predict the cluster assignment for new data points
        '''
        return self.assign_clusters(X)


    def compute_purity(self, X, y):
        '''
        Compute the cluster purity for the given dataset X and ground truth y
        '''
        purity = 0
        for i in range(self.n_clusters):
            cluster_labels = y[self.labels == i]
            unique_labels, counts = np.unique(cluster_labels, return_counts=True)
            most_common_label = unique_labels[np.argmax(counts)]
            
            purity += np.sum(cluster_labels == most_common_label)
        
        return purity / len(X)

    def plot_clusters(self, X):
        '''
        Plot the clusters and centroids in 2D (using PCA if needed)
        '''
        # If the data is more than 2D, apply PCA to reduce it to 2D
        if X.shape[1] > 2:
            pca = PCA(n_components=2)
            X_reduced = pca.fit_transform(X)
            centroids_reduced = pca.transform(self.centroids)  # Apply same PCA transformation to centroids
        else:
            X_reduced = X
            centroids_reduced = self.centroids

        # Plot the data points, colored by their assigned cluster label
        plt.figure(figsize=(8, 6))
        plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=self.labels, cmap='viridis', marker='o', s=50, alpha=0.6)

        # Plot the centroids with a contrasting color (red 'X')
        plt.scatter(centroids_reduced[:, 0], centroids_reduced[:, 1], c='red', s=200, marker='X', label='Centroids')

        # Add labels and title
        plt.title('KMeans Clusters (PCA-reduced)')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.legend(loc='best')

        plt.show()

    def cluster_stats(self, X):
        '''
        Compute evaluation metrics for each cluster along with intercluster distances and ratio of radiuses to intercluster distances
        '''
        silhouette_vals = silhouette_samples(X, self.labels) if self.n_clusters > 1 else np.zeros(len(X))
        # Initialize with inf to ignore self-distances
        intercluster_dists = np.full((self.n_clusters, self.n_clusters), np.inf)
        radius_distance_ratios = []

        cluster_stats = []
        for i in range(self.n_clusters):
            cluster_points = X[self.labels == i]
            distances = np.linalg.norm(cluster_points - self.centroids[i], axis=1)
            cluster_silhouette = silhouette_vals[self.labels == i] if len(cluster_points) > 1 else [0]
            cluster_radius = np.max(distances) if len(distances) > 0 else 0

            for j in range(i + 1, self.n_clusters):  
                dist = np.linalg.norm(self.centroids[i] - self.centroids[j])
                intercluster_dists[i, j] = dist
                intercluster_dists[j, i] = dist  # Symmetric matrix
                if dist > 0:  # Avoid division by zero
                    other_radius = np.max(np.linalg.norm(X[self.labels == j] - self.centroids[j], axis=1))
                    ratio = (cluster_radius + other_radius) / dist
                    radius_distance_ratios.append(ratio)

            cluster_stats.append({
                'label': i,
                'size': len(cluster_points),
                'centroid': self.centroids[i],
                'radius': cluster_radius,
                'silhouette': np.mean(cluster_silhouette),
                'sse': np.sum(distances ** 2),
                'points': cluster_points
            })
        return cluster_stats, intercluster_dists, radius_distance_ratios

    def score(self, X, y=None):
        silhouette = silhouette_score(X, self.labels) if self.n_clusters > 1 else 0
        ch_index = calinski_harabasz_score(X, self.labels) if self.n_clusters > 1 else 0
        rand_index = rand_score(y, self.labels) if y is not None else None
        purity = self.compute_purity(X, y) if y is not None else None

        stats, intercluster_dists, radius_distance_ratios = self.cluster_stats(X)

        avg_radius_distance_ratio = np.mean(radius_distance_ratios) if radius_distance_ratios else np.nan

        return {
            "radius_distance_ratio": avg_radius_distance_ratio,
            "silhouette_score": silhouette,
            "ch_index": ch_index,
            "rand_index": rand_index,
            "purity": purity,
            "stats": stats
        }

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 kmeans.py <Filename> <k>")
        sys.exit(1)
    csv = sys.argv[1]
    k = int(sys.argv[2])
    
    X, y = load_data(csv, target=False)
    X, y = preprocess_data(X, y, "normal")

    model = Kmeans(n_clusters=k)
    model.fit(X)
    score = model.score(X, y)
    for cluster in score['stats']:
        print(f"Cluster {cluster['label']}:\n"
              f"  Center: {', '.join(f'{val:.2f}' for val in cluster['centroid'])}\n"
              f"  Radius: {cluster['radius']:.4f}\n"
              f"  Silhouette Score: {cluster['silhouette']:.4f}\n"
              f"  SSE: {cluster['sse']:.4f}\n"
              f"  {cluster['size']} Points")
        #for point in cluster['points']:
         #   print("  ", ', '.join(f'{val:.2f}' for val in point))
 
    print("\n=== Clustering Metrics ===")
    print(f"Avg Radius-to-Intercluster Distance Ratio: {score['radius_distance_ratio']:.4f}")
    print(f"Silhouette Score: {score['silhouette_score']:.4f}")
    print(f"Calinski-Harabasz Index: {score['ch_index']:.4f}")
    if score['rand_index'] is not None:
        print(f"Rand Index: {score['rand_index']:.4f}")
        print(f"Total Cluster Purity: {score['purity']:.4f}")

    model.plot_clusters(X)
