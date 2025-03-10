import numpy as np
import argparse
from sklearn.cluster import KMeans

if __name__ == "__main__":
    # try it with `python KMeansAuthorship.py matrix.npy kmeans.out 50`
    parser = argparse.ArgumentParser(description="K-Means Clustering for Authorship Attribution")
    parser.add_argument('input_file', type=str, help="Path to the file with vectorized document representations")
    parser.add_argument('output_file', type=str, help="Path to save the clustering output (cluster labels)")
    parser.add_argument('k', type=int, help="Number of clusters (k for K-Means)")
    parser.add_argument('--n_init', type=int, default=10, help="Number of K-Means initializations")
    parser.add_argument('--max_iter', type=int, default=300, help="Maximum number of K-Means iterations")

    args = parser.parse_args()
    
    # Load the vectorized documents
    documents = np.load(args.input_file)
    
    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=args.k, n_init=args.n_init, max_iter=args.max_iter, random_state=42)
    cluster_labels = kmeans.fit_predict(documents)

    # Save the results
    np.savetxt(args.output_file, cluster_labels, fmt='%d')
    
    print(f"Clustering complete. Results saved to {args.output_file}")
