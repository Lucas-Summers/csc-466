import numpy as np
import argparse
from sklearn.metrics import rand_score
from collections import Counter
import csv
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd

def load_ground_truth(ground_truth_file):
    '''
    Load the ground truth data from a csv file
    '''
    ground_truth = {}
    with open(ground_truth_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for filename, author in reader:
            ground_truth[filename] = author
    return ground_truth

def evaluate_clusters(cluster_labels, ground_truth, k):
    '''
    Evaluate each cluster for size, plurality, purity, and distribution info
    '''
    cluster_info = {}
    for i in range(k):
        cluster_indices = np.where(cluster_labels == i)[0]
        cluster_documents = [list(ground_truth.keys())[idx] for idx in cluster_indices]
        authors_in_cluster = [ground_truth[doc] for doc in cluster_documents]
        author_counts = Counter(authors_in_cluster)
        plurality_label, plurality_count = author_counts.most_common(1)[0]
        purity = plurality_count / len(cluster_documents)
        author_distribution = dict(author_counts)
        
        cluster_info[i] = {
            'documents': cluster_documents,
            'size': len(cluster_documents),
            'purity': purity,
            'author_distribution': author_distribution,
            'plurality_label': plurality_label
        }
    
    return cluster_info

def evaluate_authors(cluster_info, cluster_labels, ground_truth, k):
    '''
    Evaluate each author to get total clusters they are in, total clusters they are plurality,
    precision score, recall score, and F1-score.
    '''
    author_clusters = defaultdict(set)
    # Populate the author_clusters dictionary where key = author, value = set of clusters this author appears in
    for cluster_id, cluster_data in cluster_info.items():
        for doc in cluster_data['documents']:
            author = ground_truth.get(doc, None)
            if author:
                author_clusters[author].add(cluster_id)

    author_info = {}
    for author, clusters in author_clusters.items():
        total_clusters = len(clusters)  # The number of clusters the author has appeared in
        plurality_clusters = sum(1 for cluster_id in clusters if cluster_info[cluster_id]['plurality_label'] == author)

        # Compute total articles written by the author
        total_articles = sum(cluster_info[c]['author_distribution'].get(author, 0) for c in cluster_info)

        # Count articles written by the author in clusters where they are the plurality
        num_articles_in_plurality_clusters = sum(
            cluster_info[c]['author_distribution'].get(author, 0)
            for c in cluster_info if cluster_info[c]['plurality_label'] == author
        )

        # Total number of articles in clusters where this author is the plurality
        total_articles_in_plurality_clusters = sum(
            cluster_info[c]['size'] for c in cluster_info if cluster_info[c]['plurality_label'] == author
        )

        recall = num_articles_in_plurality_clusters / total_articles if total_articles > 0 else 0
        precision = num_articles_in_plurality_clusters / total_articles_in_plurality_clusters if total_articles_in_plurality_clusters > 0 else 0
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        author_info[author] = {
            'total_clusters': total_clusters,
            'plurality_clusters': plurality_clusters,
            'recall': recall,
            'precision': precision,
            'f1_score': f1_score
        }

    return author_info

def plot_author_metrics(author_info):
    '''
    Plot precision, recall, and f1 score for each author as a barchart
    '''
    authors = list(author_info.keys())
    # Convert to percentages
    recalls = [info['recall'] * 100 for info in author_info.values()]
    precisions = [info['precision'] * 100 for info in author_info.values()]
    f1_scores = [info['f1_score'] * 100 for info in author_info.values()]
    
    x = range(len(authors))
    plt.figure(figsize=(12, 6))
    
    plt.bar(x, recalls, width=0.25, label="Recall (%)", align='center', alpha=0.8)
    plt.bar(x, precisions, width=0.25, label="Precision (%)", align='edge', alpha=0.8)
    plt.bar(x, f1_scores, width=0.25, label="F1-score (%)", align='edge', alpha=0.8, color='green')

    plt.xticks(x, authors, rotation=90)
    plt.ylabel("Percentage")
    plt.title("Author Classification Metrics (Precision, Recall, F1-score)")
    plt.legend()
    plt.tight_layout()
    plt.show()

def rank_authors_by_difficulty(author_info):
    '''
    Ranks authors based on their F1-score and prints the top 5 easiest and hardest authors to identify
    '''
    sorted_authors = sorted(author_info.items(), key=lambda x: x[1]['f1_score'], reverse=True)

    print("\nTop 5 Easiest Authors To Identify (Highest F1-Score):")
    for author, info in sorted_authors[:5]:
        print(f"  {author}: {info['f1_score'] * 100:.2f}%")

    print("\nTop 5 Hardest Authors To Identify (Lowest F1-Score):")
    for author, info in sorted_authors[-5:]:
        print(f"  {author}: {info['f1_score'] * 100:.2f}%")

def author_metrics_to_csv(author_info):
    '''
    Create and save csv of all author metrics
    '''
    df = pd.DataFrame(author_info).T  # Transpose dictionary to DataFrame
    df.columns = ["Total Clusters", "Plurality Clusters", "Recall", "Precision", "F1 Score"]
    
    # Convert Recall and Precision to percentages
    df["Recall"] *= 100  
    df["Precision"] *= 100  
    df["F1 Score"] *= 100

    df.to_csv("output.csv", index=True)


if __name__ == "__main__":
    # try it with `python clusteringEvaluation.py kmeans.out gt.csv`
    parser = argparse.ArgumentParser(description="Evaluate K-Means Clustering for Authorship Attribution")
    parser.add_argument('cluster_labels_file', type=str, help="Path to the K-Means clustering output file")
    parser.add_argument('ground_truth_file', type=str, help="Path to the ground truth file")
    
    args = parser.parse_args()

    # Load the ground truth and cluster labels
    ground_truth = load_ground_truth(args.ground_truth_file)
    cluster_labels = np.loadtxt(args.cluster_labels_file, dtype=int)

    # Evaluate clusters
    k = len(set(cluster_labels))  # Number of clusters
    cluster_info = evaluate_clusters(cluster_labels, ground_truth, k)
    
    # Report cluster information
    print("== Cluster Metrics ==")
    for cluster_id, info in cluster_info.items():
        print(f"Cluster {cluster_id}:")
        print(f"  Size: {info['size']}")
        print(f"  Plurality Label: {info['plurality_label']}")
        print(f"  Purity: {info['purity'] * 100:.2f}%")
        print(f"  Author Distribution: {info['author_distribution']}")
    
    print("\n== Summary ==")
    # Evaluate average cluster purity
    avg_purity = np.mean([info['purity'] for info in cluster_info.values()])
    print(f"Average Cluster Purity: {avg_purity:.2f}")
    
    # Evaluate the Rand Score
    document_filenames = list(ground_truth.keys())
    ground_truth_labels = [ground_truth[filename] for filename in document_filenames]
    rand = rand_score(ground_truth_labels, cluster_labels)
    print(f"Rand Score: {rand:.2f}")
    
    # Evaluate individual authors
    author_info = evaluate_authors(cluster_info, cluster_labels, ground_truth, k)
    
    # Report author information
    print("\n== Author Metrics ==")
    for author, info in author_info.items():
        print(f"Author {author}:")
        print(f"  Clusters with author: {info['total_clusters']}")
        print(f"  Clusters with author as plurality: {info['plurality_clusters']}")
        print(f"  Recall: {info['recall'] * 100:.2f}%")
        print(f"  Precision: {info['precision'] * 100:.2f}%")

    #plot_author_metrics(author_info)
    #rank_authors_by_difficulty(author_info)
    #author_metrics_to_csv(author_info)
