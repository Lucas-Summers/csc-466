import numpy as np
import argparse
from sklearn.neighbors import KNeighborsClassifier
import csv
from tqdm import tqdm

def okapi_function(all_docs):
    '''
    Takes all the documents in the dataset 
    and returns the okapi similarity function that can 
    be used to calculate the similarity between two documents
    '''
    # Calculate the average document length
    avg_doc_length = np.mean([len(doc) for doc in all_docs])

    def okapi(doc1, doc2):
        '''
        Takes two documents and returns the okapi BM25 similarity between them
        '''
        # Calculate the term frequency of each word in the documents
        doc1_tf = {word: doc1.count(word) for word in doc1}
        doc2_tf = {word: doc2.count(word) for word in doc2}

        # Calculate the document frequency of each word in the documents
        doc1_df = {word: sum([1 for doc in all_docs if word in doc]) for word in doc1} 
        doc2_df = {word: sum([1 for doc in all_docs if word in doc]) for word in doc2}

        # Calculate the similarity between the two documents
        similarity = 0
        for term in set(doc1 + doc2):
            k_1, b, k_2 = 1.5, 0.75, 100
            idf = np.log((len(all_docs) - doc1_df[term] + 0.5) / (doc1_df[term] + 0.5))
            doc1_score = idf * ((k_1 + 1) * doc1_tf[term]) / (k_1 * (1 - b + b * len(doc1) / avg_doc_length) + doc1_tf[term])
            doc1_score *= ((k_2 + 1) * doc2_tf[term]) / (k_2 + doc2_tf[term])

            similarity += doc1_score
        return similarity

    return okapi
            
if __name__ == "__main__":
    # try it with `python knnAuthorship.py matrix.npy knn5.out 5`
    parser = argparse.ArgumentParser(description="K-Means Clustering for Authorship Attribution")
    parser.add_argument('input_file', type=str, help="Path to the file with vectorized document representations")
    parser.add_argument('output_file', type=str, help="Path to save the clustering output (cluster labels)")
    parser.add_argument('n', type=int, help="Number of neighbors")
    parser.add_argument('similarity', type=str, help="Similarity metric")

    args = parser.parse_args()
    
    # Load the vectorized documents
    documents = np.load(args.input_file)
    gt = csv.reader(open('ground_truth.csv'))
    gt_no_header = [row for row in list(gt)[1:]]
    y = [row[1] for row in gt_no_header]

    # Create the similarity function
    similarities = {
        'cosine': 'cosine',
        'okapi': okapi_function(documents)
    }

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
