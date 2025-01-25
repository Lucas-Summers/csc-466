import pandas as pd
import numpy as np
import json

class c45:
    def __init__(self, metric="info_gain", threshold=0.01):
        assert metric in ["info_gain", "gain_ratio"]

        self.metric = metric
        self.threshold = threshold
        self.tree = None
        self.labels = None

    def fit(self, train_x, train_y):
        pass

    def predict(self, X_test_df):
        if self.tree is None:
            print("Tree is not trained, call fit() or read_tree() first")

        self.labels = X_test_df.keys()

        return [self.predict_one(self.tree["node"], x) for x in X_test_df.data]
    
    def predict_one(self, node, x):
        x_value = x[self.labels.get_loc(node["var"])]

        for edge_dict in node["edges"]:
            edge = edge_dict["edge"]
            if edge["value"] == x_value:
                if "leaf" in edge:
                    return edge["leaf"]
                else:
                    return self.predict_one(edge["node"], x)
                
        Exception("No edge found for value", x_value, "in node", node)
        
    def save_tree(self, filename):
        pass

    def read_tree(self, filename):
        try:
            self.tree = json.load(open(filename, "r"))
            print("Tree loaded from", filename)
        except Exception as e:
            print("Error reading tree from", filename, ":", e)

