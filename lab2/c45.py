import pandas as pd
import numpy as np
import json

class c45:
    def __init__(self, metric="info_gain", threshold=0.01):
        assert metric in ["info_gain", "gain_ratio"]

        self.metric = metric
        self.threshold = threshold
        self.tree = None

    def fit(self, train_x, train_y):
        pass

    def predict(self, X_test):
        if self.tree is None:
            print("Tree is not trained, call fit() or read_tree() first")
            return None
        pass

    def save_tree(self, filename):
        pass

    def read_tree(self, filename):
        try:
            self.tree = json.load(open(filename, "r"))
            print("Tree loaded from", filename)
        except:
            print("Error reading tree from", filename)

