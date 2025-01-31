import pandas as pd
import numpy as np
import json

class c45:
    def __init__(self, metric="info_gain", threshold=0.4):
        assert metric in ["info_gain", "gain_ratio"]

        self.metric = metric
        self.threshold = threshold
        self.tree = None

        # for unique node ids when making a graphviz dot file
        self.id_ctr = 0 
    
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

        if len(y.unique()) == 1:
            return {"leaf": {"decision": y.iloc[0], "p": 1.0}}
        
        if X.empty or len(X.columns) == 0:
            decision = y.value_counts.idxmax()
            prob = y.value_counts(normalize=True).iloc[0]
            return {"leaf": {"decision": decision, "p": prob}}

        best_attr, best_score, best_threshold = self.best_split(X, y)
        if best_score < self.threshold:
            decision = y.value_counts().idxmax()
            prob = y.value_counts(normalize=True).iloc[0]
            return {"leaf": {"decision": decision, "p": prob}}

        tree = {"node": {"var": best_attr, "edges": []}}
        if best_threshold is not None:  # Numeric split
            left_mask = X[best_attr] <= best_threshold
            right_mask = X[best_attr] > best_threshold

            left_subtree = self.build_tree(X[left_mask], y[left_mask])
            right_subtree = self.build_tree(X[right_mask], y[right_mask])

            tree["node"]["edges"].extend([
                {"edge": {"op": "<=", "value": best_threshold, **left_subtree}},
                {"edge": {"op": ">", "value": best_threshold, **right_subtree}},
            ])
        else:  # Categorical split
            for val in X[best_attr].unique():
                subset_X = X[X[best_attr] == val].drop(columns=[best_attr])
                subset_y = y[X[best_attr] == val]
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

        for attr in X.columns:
            if X[attr].dtype in [np.float64, np.int64]:  # Numeric attribute
                thresholds = X[attr].sort_values().unique()
                for i in range(1, len(thresholds)):
                    threshold = (thresholds[i - 1] + thresholds[i]) / 2
                    score = self.metric_score(X, y, attr, threshold)
                    if score > best_score:
                        best_score = score
                        best_attr = attr
                        best_threshold = threshold
            else: # Categorical attribute
                score = self.metric_score(X, y, attr)
                if score > best_score:
                    best_score = score
                    best_attr = attr
                    best_threshold = None
        return best_attr, best_score, best_threshold
    
    def entropy(self, y):
        '''
        Compute the entropy of a label distribution
        '''
        probs = y.value_counts(normalize=True)
        return -sum(probs * np.log2(probs))

    def info_gain(self, X, y, attr, threshold=None):
        '''
        Calculate information gain for a given attribute
        '''
        total_entropy = self.entropy(y)

        if threshold is not None:  # Numeric attribute
            left_mask = X[attr] <= threshold
            right_mask = X[attr] > threshold

            left_y, right_y = y[left_mask], y[right_mask]
            weighted_entropy = (len(left_y) / len(y)) * self.entropy(left_y) + \
                               (len(right_y) / len(y)) * self.entropy(right_y)
        else: # Categorical attribute
            values = X[attr].unique()
            weighted_entropy = sum((len(y[X[attr] == val]) / len(y)) * self.entropy(y[X[attr] == val]) for val in values)

        return total_entropy - weighted_entropy

    def info_gain_ratio(self, X, y, attr, threshold=None):
        '''
        Calculate information gain ratio for a given attribute
        '''
        info_gain = self.info_gain(X, y, attr, threshold)

        if threshold is not None:  # Numeric attribute
            left_mask = X[attr] <= threshold
            right_mask = X[attr] > threshold

            left_ratio = len(y[left_mask]) / len(y)
            right_ratio = len(y[right_mask]) / len(y)
            split_info = -sum(r * np.log2(r) for r in [left_ratio, right_ratio] if r > 0)
        else:  # Categorical attribute
            split_info = -sum((len(y[X[attr] == val]) / len(y)) * np.log2(len(y[X[attr] == val]) / len(y)) 
                              for val in X[attr].unique())

        return info_gain / split_info if split_info != 0 else 0

    def fit(self, X_train, y_train, filename):
        '''
        Train the C45 decision tree on the given dataset
        '''
        self.tree = self.build_tree(X_train, y_train)
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

    def save_tree(self, filename):
        '''
        Saves the tree dict as a json file that can be loaded with read_tree
        '''
        try:
            json.dump(self.tree, open(filename, "w"), indent=2)
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

