import numpy as np
import pandas as pd
import os
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

def perform_pca(data, features, variance_threshold=0.8, max_components=30):
    """
    Perform PCA to reduce dimensionality
    """
    # Make sure features are in the dataset and handle missing values
    valid_features = [f for f in features if f in data.columns]
    feature_df = data[valid_features].fillna(0)
    
    # First run PCA to determine explained variance
    pca_full = PCA()
    pca_full.fit(feature_df)
    
    # Find number of components needed for threshold
    explained_variance = np.cumsum(pca_full.explained_variance_ratio_)
    n_components_for_threshold = np.argmax(explained_variance >= variance_threshold) + 1
    
    # Use the smaller of threshold-based or max components
    n_components = min(n_components_for_threshold, max_components, len(feature_df) - 1, len(valid_features))
    print(f"Using {n_components} PCA components (threshold would need {n_components_for_threshold})")
    
    # Run PCA with the determined number of components
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(feature_df)
    
    explained_variance_used = np.sum(pca.explained_variance_ratio_)
    print(f"Variance explained with {n_components} components: {explained_variance_used:.4f}")
    
    return pca_result, pca, n_components

def agglomerative_grid_search(data, features, variance_threshold=0.8, max_components=30):
    """
    Perform Agglomerative clustering with grid search (PCA applied first)
    """
    pca_result, pca, n_components = perform_pca(data, features, variance_threshold=variance_threshold, max_components=max_components)
    
    # Parameters to try
    n_clusters_range = [5, 6, 7, 8, 10, 11, 12, 13, 14, 15]
    linkage_options = ['ward', 'complete', 'average', 'single']
    metric_options = ['euclidean', 'manhattan', 'cosine']
    
    print("\nGrid search for Agglomerative Clustering parameters:")
    
    results = []
    
    for n_clusters in n_clusters_range:
        for linkage in linkage_options:
            for metric in metric_options:
                # Skip invalid combinations
                if linkage == 'ward' and n_components < n_clusters:
                    continue
                    
                # Apply clustering
                model = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    linkage=linkage,
                    metric=metric
                )
                
                try:
                    labels = model.fit_predict(pca_result)
                    
                    # Calculate evaluation metrics
                    silhouette_avg = silhouette_score(pca_result, labels)
                    ch_score = calinski_harabasz_score(pca_result, labels)
                    db_score = davies_bouldin_score(pca_result, labels)
                    
                    result = {
                        'n_clusters': n_clusters,
                        'linkage': linkage,
                        'metric': metric,
                        'silhouette': silhouette_avg,
                        'ch_score': ch_score,
                        'db_score': db_score,
                        'labels': labels
                    }
                    
                    results.append(result)
                    
                    print(f"n_clusters={n_clusters}, linkage={linkage}, metric={metric}: silhouette={silhouette_avg:.4f}, CH={ch_score:.1f}, DB={db_score:.4f}")
                except:
                    print(f"n_clusters={n_clusters}, linkage={linkage}, metric={metric}: Failed to calculate metrics")
    
    if results:
        # Create a composite score: high silhouette, high CH, low DB
        for r in results:
            #r['composite_score'] = (
            #    r['silhouette'] / max([res['silhouette'] for res in results]) +
            #    r['ch_score'] / max([res['ch_score'] for res in results]) -
            #    r['db_score'] / max([res['db_score'] for res in results])
            #)
            
            # Just going with CH score by itself because it provides the best results
            r['composite_score'] = (
                r['ch_score']
            )
        
        # Sort by composite score
        results = sorted(results, key=lambda x: x['composite_score'], reverse=True)
        best_result = results[0]
        best_n_clusters = best_result['n_clusters']
        best_linkage = best_result['linkage']
        best_metric = best_result['metric']
        best_labels = best_result['labels']

        
        print(f"\nBest Agglomerative parameters: n_clusters={best_n_clusters}, linkage={best_linkage}")
        print(f"Silhouette: {best_result['silhouette']:.4f}, CH: {best_result['ch_score']:.1f}, DB: {best_result['db_score']:.4f}")
        
        scores = {
            'silhouette': best_result['silhouette'],
            'ch_score': best_result['ch_score'],
            'db_score': best_result['db_score'],
            'n_clusters': best_n_clusters
        }
    else:
        # Fallback to n_clusters=3, linkage='ward'
        best_n_clusters = 3
        best_linkage = 'ward'
        best_metric = 'euclidean'
        
        model = AgglomerativeClustering(
            n_clusters=best_n_clusters,
            linkage=best_linkage,
            metric=best_metric
        )
        best_labels = model.fit_predict(pca_result)
        
        print(f"\nNo valid Agglomerative results. Using fallback n_clusters={best_n_clusters}, linkage={best_linkage}, metric={best_metric}")
        
        scores = {
            'silhouette': -1,
            'ch_score': -1,
            'db_score': -1,
            'n_clusters': best_n_clusters
        }
    
    return (best_n_clusters, best_linkage, best_metric), pca_result, best_labels, scores, pca, n_components
