import numpy as np
import pandas as pd
import sys
from sklearn.preprocessing import OrdinalEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from preprocessor import preprocess_data
from hclustering import AgglomerativeClustering as AgglomerativeClustering466
from dbscan import DBScan
from kmeans import Kmeans
from itertools import product
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from tqdm import tqdm
import os

def kmeans_elbow(hyps, X, y=None, filename='', method='466'):
    scores = {'ch_index': [], 'silhouette_score': []}
    k_values = hyps['k']

    for k in k_values:
        if method == 'sklearn':
            model = KMeans(n_clusters=k)
        elif method == '466':
            model = Kmeans(n_clusters=k)

        model.fit(X)

        if method == "466":
            scores['ch_index'].append(model.score(X, y)['ch_index'])
            scores['silhouette_score'].append(model.score(X, y)['silhouette_score'])
        elif method == "sklearn":
            labels = model.fit_predict(X)
            scores['ch_index'].append(calinski_harabasz_score(X, labels))
            scores['silhouette_score'].append(silhouette_score(X, labels))

    # --- Plotting ---
    plt.figure(figsize=(12, 6))

    # Plot CH Index
    plt.subplot(1, 2, 1)
    plt.plot(k_values, scores['ch_index'], marker='o', color='b', label='CH Index')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('CH Index')
    plt.title(f'KMeans - CH Index ({method}, {filename})')
    plt.grid(True)

    # Plot Silhouette Score
    plt.subplot(1, 2, 2)
    plt.plot(k_values, scores['silhouette_score'], marker='o', color='g', label='Silhouette Score')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title(f'KMeans - Silhouette Score ({method}, {filename})')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'elbow/kmeans-{method}-{filename}.png', dpi=300)
    plt.show()

def dbscan_elbow(hyps, X, y=None, filename='', method='466', metric='ch_index'):
    scores = {}
    eps_values = hyps['epsilon']
    minpts_values = hyps['minpts']

    for minpts in minpts_values:
        scores[minpts] = {'ch_index': [], 'silhouette_score': []}

        for eps in eps_values:
            if method == 'sklearn':
                model = DBSCAN(eps=eps, min_samples=minpts)
            else:
                model = DBScan(epsilon=eps, minpts=minpts)
            
            model.fit(X)

            if method == "466":
                scores[minpts]['ch_index'].append(model.score(X, y)['ch_index'])
                scores[minpts]['silhouette_score'].append(model.score(X, y)['silhouette_score'])
            elif method == "sklearn":
                labels = model.fit_predict(X)
                # Avoid invalid scores if DBSCAN produces a single cluster or all noise
                if len(set(labels)) <= 1:
                    scores[minpts]['ch_index'].append(-1)
                    scores[minpts]['silhouette_score'].append(-1)
                else:
                    scores[minpts]['ch_index'].append(calinski_harabasz_score(X, labels))
                    scores[minpts]['silhouette_score'].append(silhouette_score(X, labels))
    # --- Plotting ---
    plt.figure(figsize=(12, 6))

    # Plot CH Index
    plt.subplot(1, 2, 1)
    for minpts, values in scores.items():
        plt.plot(eps_values, values['ch_index'], marker='o', label=f'MinPts={minpts}')
    plt.xlabel('Epsilon')
    plt.ylabel('CH Index')
    plt.title(f'DBSCAN - CH Index ({method}, {filename})')
    plt.legend()
    plt.grid(True)

    # Plot Silhouette Score
    plt.subplot(1, 2, 2)
    for minpts, values in scores.items():
        plt.plot(eps_values, values['silhouette_score'], marker='o', label=f'MinPts={minpts}')
    plt.xlabel('Epsilon')
    plt.ylabel('Silhouette Score')
    plt.title(f'DBSCAN - Silhouette Score ({method}, {filename})')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'elbow/dbscan-{method}-{filename}.png', dpi=300)
    plt.show()

def hcluster_elbow(hyps, X, filename='', method='466'):
    scores = {'ch_index': [], 'silhouette_score': []}
    threshs = hyps['threshold']

    for thresh in tqdm(threshs):
        if method == 'sklearn':
            model = AgglomerativeClustering(linkage='single', n_clusters=thresh)   
        elif method == '466':
            model = AgglomerativeClustering466()

        model.fit(X)

        if method == "466":
            X, y = model.get_clusters(thresh, X)
            scores['ch_index'].append(model.score(X, y)['ch_index'])
            scores['silhouette_score'].append(model.score(X, y)['silhouette_score'])
        elif method == "sklearn":
            labels = model.fit_predict(X)
            scores['ch_index'].append(calinski_harabasz_score(X, labels))
            scores['silhouette_score'].append(silhouette_score(X, labels))

    # --- Plotting ---
    plt.figure(figsize=(12, 6))

    # Plot CH Index
    plt.subplot(1, 2, 1)
    plt.plot(threshs, scores['ch_index'], marker='o', color='b', label='CH Index')
    plt.xlabel('Height to Cut')
    plt.ylabel('CH Index')
    plt.title(f'AggCluster - CH Index ({method}, {filename})')
    plt.grid(True)

    # Plot Silhouette Score
    plt.subplot(1, 2, 2)
    plt.plot(threshs, scores['silhouette_score'], marker='o', color='g', label='Silhouette Score')
    plt.xlabel('Height to Cut')
    plt.ylabel('Silhouette Score')
    plt.title(f'AggCluster - Silhouette Score ({method}, {filename})')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'elbow/hcluster-{method}-{filename}.png', dpi=300)
    plt.show()
    
if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python3 elbow.py <Filename>")
        sys.exit(1)

    csv = sys.argv[1]

    
    df = pd.read_csv(csv)
    categorical_columns = df.select_dtypes(include=["object"]).columns.tolist()
    if categorical_columns:
        encoder = OrdinalEncoder()
        df[categorical_columns] = encoder.fit_transform(df[categorical_columns])
    X = preprocess_data(df.to_numpy(), "normal")
    #X = df.to_numpy()
    y = None

    # for iris or data with ground truth
    #y = X[:, -1]
    #X = X[:, :-1]
    
    # Generate hyperparam ranges
    kmeans_hyps = {'k': range(2, 10)}
    dbscan_hyps = {'epsilon': np.linspace(0.1, 1, 20), 'minpts': range(3, 15)}
    fname = os.path.splitext(os.path.basename(csv))[0]
    kmeans_elbow(kmeans_hyps, X, y, filename=fname, method='466')
    kmeans_elbow(kmeans_hyps, X, y, filename=fname, method='sklearn')
    dbscan_elbow(dbscan_hyps, X, y, filename=fname, method='466')
    dbscan_elbow(dbscan_hyps, X, y, filename=fname,method='sklearn')

    df = pd.read_csv(csv)
    X = df.select_dtypes(include=[np.number])
    y = df.select_dtypes(include=[object])
    X = X.to_numpy()
    hcluster_hyps = {'threshold': range(2, len(X)-1)}
    hcluster_elbow(hcluster_hyps, X, filename=fname, method='466')
    hcluster_elbow(hcluster_hyps, X, filename=fname, method='sklearn')


