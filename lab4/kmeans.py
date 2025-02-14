import numpy as np
import pandas as pd
import sys
from sklearn.preprocessing import OrdinalEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# try it with `python kmeans.py csv/iris.csv 3`

class KMeans:
    def __init__(self, n_clusters=2, tol=1e-4, max_iter=300, random_state=42):
        self.n_clusters = n_clusters
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state
        self.labels = None
        self.centroids = None
        
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

    def move_centroids(self, X):
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
            new_centroids = self.move_centroids(X)
            
            sse = self.compute_sse(X, new_centroids)
            print(prev_sse)
            if prev_sse != 0 and abs(prev_sse - sse) / prev_sse < self.tol:
                print(f"Converged at iteration {i}")
                break

            self.centroids = new_centroids
            prev_sse = sse

    def predict(self, X):
        '''
        Predict the cluster assignment for new data points
        '''
        return self.assign_clusters(X)

    def cluster_stats(self, X):
        '''
        Compute evaluation metrics for each cluster
        '''
        cluster_stats = []
        for i in range(len(self.centroids)):
            cluster_points = X[self.labels == i]
            distances = np.linalg.norm(cluster_points - self.centroids[i], axis=1)
            cluster_stats.append({
                'Cluster': i,
                'Size': len(cluster_points),
                'Centroid': self.centroids[i],
                'Max Dist': np.max(distances),
                'Min Dist': np.min(distances),
                'Avg Dist': np.mean(distances),
                'SSE': np.sum(distances ** 2),
                'Points': cluster_points
            })
        return cluster_stats

    def compute_purity(self, X, y):
        '''
        Compute the cluster purity for the given dataset X and ground truth y
        '''
        purity = 0
        for i in range(self.n_clusters):
            # Get the true labels of the points in this cluster
            cluster_labels = y[self.labels == i]
            
            # Find the most frequent true label in this cluster using np.unique
            unique_labels, counts = np.unique(cluster_labels, return_counts=True)
            
            # Get the label with the maximum count (mode)
            most_common_label = unique_labels[np.argmax(counts)]
            
            # Count how many points in this cluster have the most common label
            purity += np.sum(cluster_labels == most_common_label)
        
        # Compute overall purity as the proportion of correctly classified points
        return purity / len(X)

    def compute_intercluster_distances(self):
        '''
        Compute the pairwise distances between centroids of clusters
        '''
        n_clusters = len(self.centroids)
        distances = np.zeros((n_clusters, n_clusters))
        for i in range(n_clusters):
            for j in range(i + 1, n_clusters):
                dist = np.linalg.norm(self.centroids[i] - self.centroids[j])
                distances[i, j] = dist
                distances[j, i] = dist  # Since distance is symmetric
        
        # Convert to DataFrame for better readability
        distance_df = pd.DataFrame(distances, columns=[f"Cluster {i}" for i in range(n_clusters)],
                                   index=[f"Cluster {i}" for i in range(n_clusters)])
        return distance_df
    
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

def remove_outliers_zscore(X, threshold=3):
    # Calculate Z-scores
    mean = np.mean(X, axis=0)
    std_dev = np.std(X, axis=0)
    z_scores = np.where(std_dev != 0, (X - mean) / std_dev, 0)
    # Filter out rows where any feature has a Z-score greater than the threshold
    X_clean = X[np.all(np.abs(z_scores) < threshold, axis=1)]
    return X_clean

def remove_outliers_iqr(X):
    # Calculate Q1 (25th percentile) and Q3 (75th percentile)
    Q1 = np.percentile(X, 25, axis=0)
    Q3 = np.percentile(X, 75, axis=0)
    IQR = Q3 - Q1
    
    # Define outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Filter out rows that are outside of the bounds
    X_clean = X[np.all((X >= lower_bound) & (X <= upper_bound), axis=1)]
    return X_clean

def preprocess_data(X):
    # Example with Normalization
    #scaler = MinMaxScaler()
    #X_scaled = scaler.fit_transform(X)
    #return remove_outliers_iqr(X_scaled)
    
    # Example with Standardization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return remove_outliers_zscore(X_scaled)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 kmeans.py <Filename> <k>")
        sys.exit(1)
    csv = sys.argv[1]
    k = int(sys.argv[2])

    df = pd.read_csv(csv)
    categorical_columns = df.select_dtypes(include=["object"]).columns.tolist()
    if categorical_columns:
        encoder = OrdinalEncoder()
        df[categorical_columns] = encoder.fit_transform(df[categorical_columns])

    X = preprocess_data(df.to_numpy())

    #y = df.iloc[:, -1].to_numpy() # for iris

    model = KMeans(n_clusters=k)
    model.fit(X)
    stats = model.cluster_stats(X)
    for cluster in stats:
        print(f"Cluster {cluster['Cluster']}:\n"
              f"  Center: {', '.join(f'{val:.2f}' for val in cluster['Centroid'])}\n"
              f"  Max Dist: {cluster['Max Dist']}\n"
              f"  Min Dist: {cluster['Min Dist']}\n"
              f"  Avg Dist: {cluster['Avg Dist']}\n"
              f"  SSE: {cluster['SSE']}\n"
              f"  {cluster['Size']} Points:")
        for point in cluster['Points']:
            print("  ", ', '.join(map(str, point)))

    intercluster_distances = model.compute_intercluster_distances()
    print("\nInter-cluster distances:")
    print(intercluster_distances.to_string())

    # for iris
    #purity = model.compute_purity(X, y)
    #print(f"\nTotal Cluster Purity: {purity * 100:.2f}%")

    model.plot_clusters(X)
