import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import OrdinalEncoder

class c45:
    def __init__(self, metric="info_gain", threshold=0.6):
        assert metric in ["info_gain", "gain_ratio"]

        self.metric = metric
        self.threshold = threshold
        self.tree = None

        # for unique node ids when making a graphviz dot file
        self.id_ctr = 0 

        self.encoder = OrdinalEncoder()
        self.class_labels = None
    
    def metric_score(self, X, y, attr, threshold=None):
        '''
        Selects the correct metric to use based on metric variable
        '''
        if self.metric == "info_gain":
            return self.info_gain(X, y, attr, threshold)
        elif self.metric == "gain_ratio":
            return self.info_gain_ratio(X, y, attr, threshold)
        else:
            raise ValueError("Invalid metric given.")

    def build_tree(self, X, y):
        '''
        Recursively builds the decision tree
        '''

        X = np.array(X) if not isinstance(X, np.ndarray) else X
        y = np.array(y) if not isinstance(y, np.ndarray) else y

        if len(np.unique(y)) == 1:
            decision_label = self.encoder.inverse_transform(y[0].reshape(-1, 1))[0][0]  # Get the actual class label
            return {"leaf": {"decision": decision_label, "p": 1.0}}
        
        if len(X) == 0 or X.shape[1] == 0:
        #if X.empty or len(X.columns) == 0:
            #decision = y.value_counts.idxmax()
            decision = np.bincount(y).argmax()
            decision_label = self.encoder.inverse_transform([[decision]])[0][0]  # Get the actual class label
            #prob = y.value_counts(normalize=True).iloc[0]
            prob = np.max(np.bincount(y) / len(y))
            return {"leaf": {"decision": decision_label, "p": prob}}

        best_attr, best_score, best_threshold = self.best_split(X, y)
        if best_score < self.threshold:
            #decision = y.value_counts().idxmax()
            decision = np.bincount(y).argmax()
            decision_label = self.encoder.inverse_transform([[decision]])[0][0]  # Get the actual class label
            #prob = y.value_counts(normalize=True).iloc[0]
            prob = np.max(np.bincount(y) / len(y))
            return {"leaf": {"decision": decision_label, "p": prob}}

        tree = {"node": {"var": best_attr, "edges": []}}
        if best_threshold is not None:  # Numeric split
            left_mask = X[:, best_attr] <= best_threshold
            right_mask = ~left_mask
            #left_mask = X[best_attr] <= best_threshold
            #right_mask = X[best_attr] > best_threshold

            left_subtree = self.build_tree(X[left_mask], y[left_mask])
            right_subtree = self.build_tree(X[right_mask], y[right_mask])

            tree["node"]["edges"].extend([
                {"edge": {"op": "<=", "value": best_threshold, **left_subtree}},
                {"edge": {"op": ">", "value": best_threshold, **right_subtree}},
            ])
        else:  # Categorical split
            for val in np.unique(X[:, best_attr]):
            #for val in X[best_attr].unique():
                subset_X = X[X[:, best_attr] == val]
                subset_y = y[X[:, best_attr] == val]
                #subset_X = X[X[best_attr] == val].drop(columns=[best_attr])
                #subset_y = y[X[best_attr] == val]
                subtree = self.build_tree(subset_X, subset_y)
                tree["node"]["edges"].append({"edge": {"value": val, **subtree}})

        return tree

    def best_split(self, X, y):
        '''
        Determines the best attribute to split on based on the chosen metric
        '''
        best_attr = None
        best_score = -np.inf
        best_threshold = None
        
        for col_idx in range(X.shape[1]):
            col_values = X[:, col_idx]
        #for attr in X.columns:
            #if X[attr].dtype in [np.float64, np.int64]:  # Numeric attribute
            if np.issubdtype(col_values.dtype, np.number):  # Numeric attribute
                sorted_indices = np.argsort(col_values)  # Sort indices for fast threshold selection
                sorted_values = col_values[sorted_indices]

                unique_values = np.unique(sorted_values)
                thresholds = (unique_values[:-1] + unique_values[1:]) / 2  # Midpoints

                for threshold in thresholds:
                #thresholds = X[attr].sort_values().unique()
                #for i in range(1, len(thresholds)):
                 #   threshold = (thresholds[i - 1] + thresholds[i]) / 2
                    score = self.metric_score(X, y, col_idx, threshold)
                    if score > best_score:
                        best_score = score
                        best_attr = col_idx
                        best_threshold = threshold
            else: # Categorical attribute
                score = self.metric_score(X, y, col_idx)
                if score > best_score:
                    best_score = score
                    best_attr = col_idx
                    best_threshold = None
        return best_attr, best_score, best_threshold
    
    def entropy(self, y):
        '''
        Compute the entropy of a label distribution
        '''
        #probs = y.value_counts(normalize=True)
        #return -sum(probs * np.log2(probs))
        if len(y) == 0:
            return 0
        _, counts = np.unique(y, return_counts=True)
        probs = counts / counts.sum()
        return -np.sum(probs * np.log2(probs))

    def info_gain(self, X, y, attr, threshold=None):
        '''
        Calculate information gain for a given attribute
        '''
        total_entropy = self.entropy(y)

        if threshold is not None:  # Numeric attribute
            left_mask = X[:, attr] <= threshold
            right_mask = ~left_mask

            left_y, right_y = y[left_mask], y[right_mask]
            weighted_entropy = (len(left_y) / len(y)) * self.entropy(left_y) + \
                               (len(right_y) / len(y)) * self.entropy(right_y)
        else: # Categorical attribute
            #values = X[attr].unique()
            #weighted_entropy = sum((len(y[X[attr] == val]) / len(y)) * self.entropy(y[X[attr] == val]) for val in values)
   #         values, counts = np.unique(X[attr], return_counts=True)
  #          weighted_entropy = np.sum(
 #               (counts / len(y)) * np.array([self.entropy(y[X[:, attr] == val]) for val in values])
#            )
            values, counts = np.unique(X[attr], return_counts=True)
            weighted_entropy = np.sum(
                (counts / len(y)) * np.array([self.entropy(y[np.isin(X[:, attr], val)]) for val in values])
            )

        return total_entropy - weighted_entropy

    def info_gain_ratio(self, X, y, attr, threshold=None):
        '''
        Calculate information gain ratio for a given attribute
        '''
        info_gain = self.info_gain(X, y, attr, threshold)

        if threshold is not None:  # Numeric attribute
            left_mask = X[:, attr] <= threshold
            right_mask = ~left_mask

            #left_ratio = len(y[left_mask]) / len(y)
            #right_ratio = len(y[right_mask]) / len(y)
            left_ratio = np.sum(left_mask) / len(y)
            right_ratio = np.sum(right_mask) / len(y)
            #split_info = -sum(r * np.log2(r) for r in [left_ratio, right_ratio] if r > 0)
            ratios = np.array([left_ratio, right_ratio])
            split_info = -np.sum(ratios * np.log2(ratios, where=ratios > 0))
        else:  # Categorical attribute
            unique_vals, counts = np.unique(col_values, return_counts=True)
            #split_info = -sum((len(y[X[attr] == val]) / len(y)) * np.log2(len(y[X[attr] == val]) / len(y)) for val in X[attr].unique())
            probs = counts / len(y)
            split_info = -np.sum(probs * np.log2(probs, where=probs > 0))


        return info_gain / split_info if split_info > 0 else 0

    def fit(self, X_train, y_train, filename):
        '''
        Train the C45 decision tree on the given dataset
        '''
        # Encode the features
        X_encoded = self.encoder.fit_transform(X_train)
        
        # Encode the target variable (y)
        #y_encoded = encoder.fit_transform(y.reshape(-1, 1)).flatten()
        y_encoded = self.encoder.fit_transform(y_train.to_numpy().reshape(-1, 1)).flatten()
        y_encoded = y_encoded.astype(int)
        #self.class_labels = self.encoder.fit(y_train.reshape(-1, 1)).categories_[0]
        #print(self.class_labels)
        self.tree = self.build_tree(X_encoded, y_encoded)
        self.tree = {"dataset": filename, **self.tree}
    
    def predict(self, X_test_df, prob=False):
        '''
        Predicts the class for each row in the given dataframe. Expects
        a dataframe with labeled columns. 
        Returns a list of class predictions if prob=False, or a list of
        tuples with the class and the probability if prob=True
        '''
        if self.tree is None:
            print("Tree is not trained, call fit() or read_tree() first")

        # iterate and build a list of predictions
        result = []
        for index, row in X_test_df.iterrows():
            if "node" in self.tree:
                pred = self.predict_row(row, self.tree["node"], prob)
            elif "leaf" in self.tree:
                pred = (self.tree["leaf"]["decision"], self.tree["leaf"]["p"]) if prob else self.tree["leaf"]["decision"]
            else:
                print("Error: Invalid tree format")
                return None
            result.append(pred)
        return result
    
    def predict_row(self, row, node, prob=False):
        '''
        Get the predicted class for a row by traversing the given node
        returns the class if prob=False, a tuple (class, p) if prob=True,
        or None if the row does not reach a leaf
        '''
        # what label the tree is splitting on
        split_var = node["var"]
        # value the row has for that label
        row_value = row[split_var]
        
        for edge_dict in node["edges"]:
            edge = edge_dict["edge"]

            if "op" in edge:  # Numeric split
                if edge["op"] == "<=" and row[split_var] <= edge["value"]:
                    if "leaf" in edge:
                        result = edge["leaf"]
                        return (result["decision"], result["p"]) if prob else result["decision"]
                    else:
                        return self.predict_row(row, edge["node"], prob)
                elif edge["op"] == ">" and row[split_var] > edge["value"]:
                    if "leaf" in edge:
                        result = edge["leaf"]
                        return (result["decision"], result["p"]) if prob else result["decision"]
                    else:
                        return self.predict_row(row, edge["node"], prob)
            else: # Categorical split
                if edge["value"] == row_value:
                    if "leaf" in edge:
                        result = edge["leaf"]
                        if not prob:
                            return result["decision"]
                        else:
                            return (result["decision"], result["p"])
                    else:
                        return self.predict_row(row, edge["node"])
        return None

    #def serialize_tree(self, tree):
    #    ''' Converts numpy.int64 and other non-serializable values to standard Python types '''
    #    if isinstance(tree, dict):
    #        return {key: self.serialize_tree(value) for key, value in tree.items()}
    #    elif isinstance(tree, np.int64):
    #        return int(tree)  # Convert numpy.int64 to int
    #    else:
    #        return tree

    def serialize_tree(self, tree):
        ''' Converts numpy.int64 and other non-serializable values to standard Python types '''
        if isinstance(tree, dict):
            return {key: self.serialize_tree(value) for key, value in tree.items()}
        elif isinstance(tree, list):
            return [self.serialize_tree(item) for item in tree]
        elif isinstance(tree, np.int64):
            return int(tree)  # Convert numpy.int64 to int
        else:
            return tree

    def save_tree(self, filename):
        '''
        Saves the tree dict as a json file that can be loaded with read_tree
        '''
        try:
            serializable_tree = self.serialize_tree(self.tree)
            print(json.dumps(serializable_tree, indent=2))
            json.dump(serializable_tree, open(filename, "w"), indent=2)
            print("Tree written to", filename)
        except Exception as e:
            print("Error writing tree to", filename, ":", e)

    def read_tree(self, filename):
        '''
        Just reads the json file as a dict and stores it, no modifications
        '''
        try:
            self.tree = json.load(open(filename, "r"))
            print("Tree loaded from", filename)
        except Exception as e:
            print("Error reading tree from", filename, ":", e)

    def tree_size(self):
        '''
        Returns the number of nodes in the tree
        '''
        if self.tree is None:
            print("Tree is not trained, call fit() or read_tree() first")
            return None
        else:
            if "leaf" in self.tree:
                return 1
            return self.count_nodes(self.tree["node"])
        
    def count_nodes(self, node):
        '''
        Recursively counts the number of nodes in the tree
        '''
        ct = 1
        for edge in node["edges"]:
            if "leaf" in edge["edge"]:
                ct += 1
            else:
                ct += self.count_nodes(edge["edge"]["node"])
        return ct

    def to_graphviz_dot(self):
        '''
        Returns the tree in dot format, to be used with graphviz.
        Paste into https://dreampuf.github.io/GraphvizOnline/?engine=dot to
        visualize the tree.
        '''
        if self.tree is None:
            print("Tree is not trained, call fit() or read_tree() first")
            return None
        else:
            dot_lines = ['digraph DecisionTree {\n',
                         '    node [fontname = "Monospace", shape="rectangle", style="rounded", width=3];\n',
                         '    edge [fontname = "Monospace", fontsize="10", fontcolor="grey"];\n']
            dot_lines.extend(self.tree_to_dot(self.tree["node"]))
            dot_lines.append('}\n')
            return "".join(dot_lines)
    
    def tree_to_dot(self, node):
        '''
        Recursively creates a list of string with the node in dot format.
        '''
        dot_lines = []
        unique_id = self.id_ctr
        # Create a unique node identifier
        node_name = f"{node['var']}_{unique_id}"
        dot_lines.append(f'    {node_name} [label="{node["var"]}"];\n')
        
        for edge in node['edges']:
            edge_value = edge['edge']['value']
            if "leaf" in edge['edge']:
                # is leaf edge
                decision = edge['edge']['leaf']['decision']
                p = edge['edge']['leaf']['p']
                self.id_ctr += 1
                child_node_name = f"{decision}_{self.id_ctr}"
                dot_lines.append(f'    {child_node_name} [label="{decision} (p={p:.3f})"];\n')
            else:
                # is a node edge
                child_node = edge['edge']['node']
                self.id_ctr += 1
                child_node_name = f"{child_node['var']}_{self.id_ctr}"
                child_dot_lines = self.tree_to_dot(child_node)   
                dot_lines.extend(child_dot_lines)
                
            dot_lines.append(f'    {node_name} -> {child_node_name} [label="{edge_value}"];\n')
    
        return dot_lines

