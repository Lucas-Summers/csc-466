from c45 import c45 
import numpy as np
from collections import Counter

class RandomForest:
    def __init__(self, numAttrs, numPoints, numTrees, metric="info_gain", threshold=0.4):
        self.numAttrs = numAttrs
        self.numPoints = numPoints
        self.numTrees = numTrees
        self.metric = metric
        self.threshold = threshold
        self.forest = []

    def fit(self, X, y):
        """Train the Random Forest with bootstrapped datasets."""
        n_samples, n_features = X.shape
        for _ in range(self.numTrees):
            # Bootstrap sampling
            if self.numPoints < 1:
                sample_size = int(self.numPoints * n_samples)
            else:
                sample_size = min(self.numPoints, n_samples)
            indices = np.random.choice(n_samples, sample_size, replace=True)
            X_sample, y_sample = X.iloc[indices], y.iloc[indices]
            
            # Randomly select subset of attributes
            attributes = np.random.choice(X.columns, size=min(self.numAttrs, n_features), replace=False)
            X_sample = X_sample[attributes]  # Select only chosen attributes
            
            # Train a decision tree (C45 instance)
            tree = c45(metric=self.metric, threshold=self.threshold)
            tree.fit(X_sample, y_sample, "random_forest_tree.json")
            self.forest.append(tree)

    def predict(self, X):
        """Predict the class labels for given data using majority voting."""
        predictions = np.array([tree.predict(X) for tree in self.forest])  # Get predictions from all trees
        final_predictions = [self.majority_vote(preds) for preds in predictions.T]  # Majority vote per sample
        return final_predictions

    def majority_vote(self, preds):
            """Resolve ties by choosing the smallest lexicographically."""
            counter = Counter(preds)
            max_count = max(counter.values())
            top_classes = [cls for cls, count in counter.items() if count == max_count]
            return min(top_classes)  # Break ties by choosing the smallest class

