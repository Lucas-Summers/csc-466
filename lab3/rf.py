from c45 import c45 
import numpy as np
import json
import concurrent.futures
from joblib import Parallel, delayed

class RandomForest:
    def __init__(self, numAttrs, numPoints, numTrees, metric="info_gain", threshold=0.1):
        self.numAttrs = numAttrs
        self.numPoints = numPoints
        self.numTrees = numTrees
        self.metric = metric
        self.threshold = threshold
        self.forest = []

    def fit_regular(self, X, y, labels):
        '''
        Train the Random Forest with bootstrapped datasets
        '''
        np.random.seed(42)  # For reproducibility
        n_samples, n_features = X.shape
        for _ in range(self.numTrees):
            # Bootstrap sampling
            sample_size = int(self.numPoints * n_samples) if self.numPoints < 1 else min(self.numPoints, n_samples)
            indices = np.random.choice(n_samples, sample_size, replace=True)
            X_sample, y_sample = X[indices], y[indices]
            
            # Randomly select subset of attributes
            attribute_indices = np.random.choice(n_features, size=min(self.numAttrs, n_features), replace=False)

            # This will reorder the columns of X_sample, but that's fine since label_sample is also reordered
            X_sample = X_sample[:, attribute_indices]
            #label_sample = [labels[i] for i in attribute_indices]
            label_sample = np.array(labels)[attribute_indices]
            
            # Train a decision tree (C45 instance)
            tree = c45(metric=self.metric, threshold=self.threshold)
            tree.fit(X_sample, y_sample, label_sample, "random_forest_tree.json")
            self.forest.append(tree)

    def fit(self, X, y, labels):
        '''
        Train the Random Forest with bootstrapped datasets in parallel
        '''
        np.random.seed(42)  # For reproducibility
        n_samples, n_features = X.shape

        # Parallel execution of tree training using joblib
        trees = Parallel(n_jobs=-1)(delayed(self._train_tree)(X, y, labels, n_samples, n_features) for _ in range(self.numTrees))
        self.forest.extend(trees)

    def _train_tree(self, X, y, labels, n_samples, n_features):
        '''
        Train a single decision tree and return it (used by parallel fit)
        '''
        sample_size = int(self.numPoints * n_samples) if self.numPoints < 1 else min(self.numPoints, n_samples)
        indices = np.random.choice(n_samples, sample_size, replace=True)
        X_sample, y_sample = X[indices], y[indices]

        # Randomly select subset of attributes
        attribute_indices = np.random.choice(n_features, size=min(self.numAttrs, n_features), replace=False)

        # Reorder the columns of X_sample (based on selected attributes)
        X_sample = X_sample[:, attribute_indices]
        label_sample = np.array(labels)[attribute_indices]

        # Train the tree
        tree = c45(metric=self.metric, threshold=self.threshold)
        tree.fit(X_sample, y_sample, label_sample, "random_forest_tree.json")
        return tree

    def predict_regular(self, X, labels, verbose=False):
        '''
        Predict the class labels for given data using majority voting
        '''
        predictions = np.vstack([tree.predict(X, labels, verbose=verbose) for tree in self.forest])
        return np.apply_along_axis(self.majority_vote, axis=0, arr=predictions)

    def predict(self, X, labels, verbose=False):
        '''
        Predict the class labels for given data in parallel using majority voting
        '''
        predictions = np.vstack(Parallel(n_jobs=-1)(delayed(tree.predict)(X, labels, verbose=verbose) for tree in self.forest))
        return np.apply_along_axis(self.majority_vote, axis=0, arr=predictions)

    def majority_vote(self, preds):
        '''
        Resolve plurality ties by choosing the smallest class lexicographically
        '''
        uniq, counts = np.unique(preds, return_counts=True)
        max_count = counts.max()
        return uniq[counts == max_count].min()

    def export(self, filename):
        '''
        Export each tree in the Random Forest to a single JSON file
        '''
        with open(filename, "w") as f:
            f.write("[")
            for tree in self.forest:
                f.write(json.dumps(tree.tree))
                f.write(",\n")
            f.write("]")

    def load_forest(self, filename):
        '''
        Load each tree in a Random Forest from a JSON file
        '''
        with open(filename, 'r') as f:
            data = json.load(f)
            forest = []
            for js in data:
                tree = c45()
                tree.tree = js
                forest.append(tree)
            self.forest = forest
