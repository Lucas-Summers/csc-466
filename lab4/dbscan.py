import numpy as np
import pandas as pd
import sys
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_samples, silhouette_score, calinski_harabasz_score, rand_score
from preprocessor import preprocess_data
from preprocessor import load_data

# try it with `python dbscan csv/iris.csv 0.4 10`

class DBScan:
    def __init__(self, epsilon=0.5, minpts=4):
        self.epsilon = epsilon
        self.minpts = minpts
        self.clusters = None
        self.noise = None
    
    def find_neighbors(self, X):
        '''
        Computes the pairwise distance matrix and finds neighboring points
        '''
        n = len(X)
        distance_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                distance_matrix[i, j] = np.linalg.norm(X[i] - X[j])

        return {i: np.where(distance_matrix[i] <= self.epsilon)[0].tolist() for i in range(len(distance_matrix))}

    def expand_cluster(self, point, cluster_id, cluster_labels, neighbors, core_points):
        '''
        Recursively expands the cluster.
        '''
        stack = [point]
        while stack:
            p = stack.pop()
            if cluster_labels[p] == -1:  # If unclassified, mark as border
                cluster_labels[p] = cluster_id
            elif cluster_labels[p] == 0:  # If core, continue expansion
                cluster_labels[p] = cluster_id
                if p in core_points:
                    stack.extend(neighbors[p])
        return cluster_labels


    def fit(self, X):
        '''
        Implements the DBSCAN model to the provided data
        '''
        n = len(X)
        neighbors = self.find_neighbors(X)
        
        cluster_labels = [0] * n  # 0 indicates unclassified
        core_points = {i for i in range(n) if len(neighbors[i]) >= self.minpts}
        
        current_cluster = 0
        
        for point in core_points:
            if cluster_labels[point] == 0:
                current_cluster += 1
                cluster_labels = self.expand_cluster(point, current_cluster, cluster_labels, neighbors, core_points)
        
        self.noise = [i for i, label in enumerate(cluster_labels) if label == 0]
        self.clusters = {i: [] for i in range(1, current_cluster + 1)}
        
        for i, label in enumerate(cluster_labels):
            if label > 0:
                self.clusters[label].append(i)

    def predict(self, X):
        '''
        Predict the cluster assignment for new data points
        '''
        pass

    def plot_clusters(self, X):
        '''
        Plot the clusters and centroids in 2D (using PCA if needed)
        '''
        # If the data is more than 2D, apply PCA to reduce it to 2D
        if X.shape[1] > 2:
            pca = PCA(n_components=2)
            X_reduced = pca.fit_transform(X)
        else:
            X_reduced = X

        plt.figure(figsize=(8, 6))
        
        # Plot the noise points
        noise_points_reduced = X_reduced[self.noise]
        plt.scatter(noise_points_reduced[:, 0], noise_points_reduced[:, 1], c='gray', marker='x', label='Noise', alpha=0.5)

        # Create a color map for different clusters
        for cluster_id, cluster_points in self.clusters.items():
            cluster_points_reduced = X_reduced[cluster_points]
            plt.scatter(cluster_points_reduced[:, 0], cluster_points_reduced[:, 1], label=f'Cluster {cluster_id}', alpha=0.6)

        # Add labels and title
        plt.title('DBSCAN Clusters (PCA-reduced)')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.legend(loc='best')

        plt.show()

    def compute_purity(self, X, y):
        '''
        Compute the cluster purity for the given dataset X and ground truth y
        '''
        purity = 0
        n_points = len(X)

        for cluster_id, cluster_points in self.clusters.items():
            if cluster_id == -1:
                continue
            cluster_labels = y[cluster_points]
            unique_labels, counts = np.unique(cluster_labels, return_counts=True)
            most_common_label = unique_labels[np.argmax(counts)]

            purity += np.sum(cluster_labels == most_common_label)
        
        return purity / (n_points)

    def cluster_stats(self, X):
        '''
        Compute evaluation metrics for each cluster along with intercluster distances and ratio of radiuses to intercluster distances
        '''
        n_clusters = len(self.clusters)
        
        if n_clusters == 0:
            return [], np.array([]), []

        cluster_labels = np.zeros(len(X)) - 1  # -1 for noise
        centroids = {}

        for cluster_id, indices in self.clusters.items():
            cluster_labels[indices] = cluster_id
            centroids[cluster_id] = np.mean(X[indices], axis=0)

        silhouette_vals = silhouette_samples(X, cluster_labels) if n_clusters > 1 else np.zeros(len(X))
        intercluster_dists = np.full((n_clusters, n_clusters), np.inf)
        radius_distance_ratios = []

        cluster_stats = []
        cluster_id_list = list(self.clusters.keys())

        for i, cluster_id in enumerate(cluster_id_list):
            cluster_points = X[self.clusters[cluster_id]]
            distances = np.linalg.norm(cluster_points - centroids[cluster_id], axis=1)
            cluster_radius = np.max(distances) if len(distances) > 0 else 0
            cluster_silhouette = silhouette_vals[self.clusters[cluster_id]] if len(cluster_points) > 1 else [0]

            for j in range(i + 1, n_clusters):  
                other_cluster_id = cluster_id_list[j]
                dist = np.linalg.norm(centroids[cluster_id] - centroids[other_cluster_id])
                intercluster_dists[i, j] = dist
                intercluster_dists[j, i] = dist  # Symmetric matrix
                if dist > 0:
                    other_radius = np.max(np.linalg.norm(X[self.clusters[other_cluster_id]] - centroids[other_cluster_id], axis=1))
                    ratio = (cluster_radius + other_radius) / dist
                    radius_distance_ratios.append(ratio)

            cluster_stats.append({
                'label': cluster_id,
                'size': len(cluster_points),
                'centroid': centroids[cluster_id],
                'radius': cluster_radius,
                'silhouette': np.mean(cluster_silhouette),
                'sse': np.sum(distances ** 2),
                'points': cluster_points
            })

        return cluster_stats, intercluster_dists, radius_distance_ratios

    def score(self, X, y=None):
        '''
        Compute all total cluster metrics as well as metrics for each cluster
        '''
        n_clusters = len(self.clusters)
        if n_clusters == 0:
            return {
                "radius_distance_ratio": np.nan,
                "silhouette_score": np.nan,
                "ch_index": np.nan,
                "rand_index": np.nan if y is not None else None,
                "purity": np.nan if y is not None else None,
                "stats": [],
            }

        cluster_labels = np.zeros(len(X)) - 1  # Default to noise (-1)
        for cluster_id, indices in self.clusters.items():
            cluster_labels[indices] = cluster_id

        silhouette = silhouette_score(X, cluster_labels) if n_clusters > 1 else -1
        ch_index = calinski_harabasz_score(X, cluster_labels) if n_clusters > 1 else -1
        rand_index = rand_score(y, cluster_labels) if y is not None else None
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
    if len(sys.argv) < 4:
        print("Usage: python3 dbscan.py <Filename> <epsilon> <NumPoints>")
        sys.exit(1)
    csv = sys.argv[1]
    epsilon = float(sys.argv[2])
    numPoints = int(sys.argv[3])

    X, y = load_data(csv, target=False)
    X, y = preprocess_data(X, y, "normal")
    
    model = DBScan(epsilon=epsilon, minpts=numPoints)
    model.fit(X)
    score = model.score(X, y)

    for cluster in score['stats']:
        print(f"Cluster {cluster['label']}:\n"
              f"  Center: {', '.join(f'{val:.2f}' for val in cluster['centroid'])}\n"
              f"  Radius: {cluster['radius']:.4f}\n"
              f"  Silhouette Score: {cluster['silhouette']:.4f}\n"
              f"  SSE: {cluster['sse']:.4f}\n"
              f"  {cluster['size']} Points:")
        #for point in cluster['points']:
        #    print("  ", ', '.join(f'{val:.2f}' for val in point))
 
    print("\n=== Clustering Metrics ===")
    print(f"Avg Radius-to-Intercluster Distance Ratio: {score['radius_distance_ratio']:.4f}")
    print(f"Silhouette Score: {score['silhouette_score']:.4f}")
    print(f"Calinski-Harabasz Index: {score['ch_index']:.4f}")
    if score['rand_index'] is not None:
        print(f"Rand Index: {score['rand_index']:.4f}")
        print(f"Total Cluster Purity: {score['purity']:.4f}")

    model.plot_clusters(X)


