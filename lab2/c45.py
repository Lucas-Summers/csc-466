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
    
    def predict(self, X_test_df):
        '''
        Predicts the class for each row in the given dataframe. Expects
        a dataframe with labeled columns.
        '''
        if self.tree is None:
            print("Tree is not trained, call fit() or read_tree() first")

        # iterate and build a list of predictions
        result = []
        for index, row in X_test_df.iterrows():
            pred = self.predict_row(row, self.tree["node"])
            result.append(pred)
        return result

    def predict_row(self, row, node):
        '''
        Get the predicted class for a row by traversing the given node
        returns a dict {'decision': 'not_recom', 'p': 0.74}
        or None if the row does not reach a leaf
        '''
        # what label the tree is splitting on
        split_var = node["var"]
        # value the row has for that label
        row_value = row[split_var]
        
        for edge_dict in node["edges"]:
            edge = edge_dict["edge"]
            if edge["value"] == row_value:
                if "leaf" in edge:
                    print("Predicted class:", edge["leaf"])
                    return edge["leaf"]
                else:
                    print("Going to node", edge["node"])
                    return self.predict_row(row, edge["node"])
        return None

    def save_tree(self, filename):
        pass

    def read_tree(self, filename):
        '''
        Just reads the json file as a dict and stores it, no modifications
        '''
        try:
            self.tree = json.load(open(filename, "r"))
            print("Tree loaded from", filename)
        except Exception as e:
            print("Error reading tree from", filename, ":", e)

