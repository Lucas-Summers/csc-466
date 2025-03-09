import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

if __name__ == "__main__":
    # try it with `python classifierEvaluation.py knn5.out ground_truth.csv`
    parser = argparse.ArgumentParser(description="Evaluate KNN for Authorship Attribution")
    parser.add_argument('knn_labels_file', type=str, help="Path to the K-Means clustering output file")
    parser.add_argument('ground_truth_file', type=str, help="Path to the ground truth file")
    parser.add_argument('--plt', action='store_true', help="Plot confusion matrix")
    
    args = parser.parse_args()

    # Load the ground truth and cluster labels
    gt = csv.reader(open('ground_truth.csv'))
    gt_no_header = [row for row in list(gt)[1:]]
    gt_labels = [row[1] for row in gt_no_header]

    knn_labels = np.loadtxt(args.knn_labels_file, dtype=str)

    # Evaluate clusters
    authors = set([row[1] for row in gt_no_header])
    hits, strikes, misses = defaultdict(int), defaultdict(int), defaultdict(int)
    for pred, actual in zip(knn_labels, gt_labels):
        if pred == actual:
            hits[actual] += 1
        else:
            strikes[pred] += 1
            misses[actual] += 1
    
    for author in authors:
        precision = hits[author] / (hits[author] + strikes[author])
        recall = hits[author] / (hits[author] + misses[author])
        f1 = 2 * precision * recall / (precision + recall)
        print(f"Author: {author}")
        print(f"\tHits, Strikes, Misses: {hits[author]}, {strikes[author]}, {misses[author]}")
        print(f"\tPrecision: {precision}")
        print(f"\tRecall: {recall}")
        print(f"\tF1 Score: {f1}")
        print()

    print("top 5 authors by f1 score")
    sorted_authors = sorted(authors, key=lambda x: hits[x] / (hits[x] + strikes[x] + misses[x]), reverse=True)
    for author in sorted_authors[:5]:
        print(f"Author: {author}")
        print(f"precision, recall, f1: {hits[author] / (hits[author] + strikes[author]):.2f}, {hits[author] / (hits[author] + misses[author]):.2f}, {2 * hits[author] / (hits[author] + strikes[author]) * hits[author] / (hits[author] + misses[author]) / (hits[author] / (hits[author] + strikes[author]) + hits[author] / (hits[author] + misses[author])):.2f}") 
    print("bottom 5 authors by f1 score")
    for author in sorted_authors[-5:]:
        print(f"Author: {author}")
        print(f"precision, recall, f1: {hits[author] / (hits[author] + strikes[author]):.2f}, {hits[author] / (hits[author] + misses[author]):.2f}, {2 * hits[author] / (hits[author] + strikes[author]) * hits[author] / (hits[author] + misses[author]) / (hits[author] / (hits[author] + strikes[author]) + hits[author] / (hits[author] + misses[author])):.2f}")


    print("== Overall Metrics ==")
    correct = sum(hits.values())
    total = len(gt_labels)
    accuracy = correct / total
    print(f"Accuracy: {accuracy}")
    print(f"Total Hits: {correct}")
    print(f"Total Misses: {total - correct}")

    # save confusion matrix as csv
    confusion_matrix = np.zeros((len(authors), len(authors)))
    for i, author in enumerate(authors):
        for j, author2 in enumerate(authors):
            confusion_matrix[i, j] = sum(1 for pred, actual in zip(knn_labels, gt_labels) if pred == author and actual == author2)
    np.savetxt('confusion_matrix.csv', confusion_matrix, delimiter=',', fmt='%d')

    # plt confusion matrix
    if args.plt:
        plt.figure(figsize=(12, 10))  # Make the figure larger
        plt.imshow(confusion_matrix, cmap='viridis')
        plt.colorbar()
        plt.xticks(range(len(authors)), authors, rotation=90, fontsize=8)  # Smaller font size for labels
        plt.yticks(range(len(authors)), authors, fontsize=8)  # Smaller font size for labels
        plt.xlabel("Predicted", fontsize=10)
        plt.ylabel("Actual", fontsize=10)
        plt.title("Confusion Matrix", fontsize=12)

        plt.savefig(f'confusion_matrix_{args.knn_labels_file.split(".")[0]}.png')

        print(f"Confusion matrix saved to confusion_matrix.png")
    
    print(f"Confusion matrix saved to confusion_matrix.csv")

        
