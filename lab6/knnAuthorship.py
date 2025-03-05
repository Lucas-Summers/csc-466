import numpy as np
import argparse
from sklearn.neighbors import KNeighborsClassifier
import csv
from tqdm import tqdm

if __name__ == "__main__":
    # try it with `python knnAuthorship.py matrix.npy knn5.out 5`
    parser = argparse.ArgumentParser(description="K-Means Clustering for Authorship Attribution")
    parser.add_argument('input_file', type=str, help="Path to the file with vectorized document representations")
    parser.add_argument('output_file', type=str, help="Path to save the clustering output (cluster labels)")
    parser.add_argument('n', type=int, help="Number of neighbors")

    args = parser.parse_args()
    
    # Load the vectorized documents
    documents = np.load(args.input_file)
    gt = csv.reader(open('ground_truth.csv'))
    gt_no_header = [row for row in list(gt)[1:]]
    y = [row[1] for row in gt_no_header]

    # Perform all-but-one K-Means clustering
    cluster_labels = []
    kmeans = KNeighborsClassifier(n_neighbors=5)
    num_docs = len(documents)
    pbar = tqdm(range(num_docs))
    correct = 0
    for idx in pbar:
        all_but_one = np.delete(documents, idx, axis=0)
        all_labels_but_one = np.delete(y, idx)
        kmeans.fit(all_but_one, all_labels_but_one)
        cluster_labels.append(kmeans.predict([documents[idx]])[0])

        if cluster_labels[-1] == y[idx]:
            correct += 1
        pbar.set_description(f"Accuracy: {correct / (idx + 1):.2f}")

    cluster_labels = np.array(cluster_labels)
        
    # Check accuracy
    accuracy = sum(cluster_labels == y) / len(y)
    print(f"Accuracy: {accuracy}")

    # Save the results
    np.savetxt(args.output_file, cluster_labels, fmt='%s')
    
    print(f"Classification complete. Results saved to {args.output_file}")
