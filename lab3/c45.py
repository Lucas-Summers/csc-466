import pandas as pd
import numpy as np
import json
from collections import defaultdict

class c45:
    """
    C45 class from lab 2
    """
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

    def proportion_of_most_common(self, y):
        '''
        Returns the proportion of the most common class in y
        '''
        uniq, counts = np.unique(y, return_counts=True)
        decision = uniq[counts.argmax()]
        normalized_counts = counts / counts.sum()
        prob = max(normalized_counts)
        return prob, decision
    
    def build_tree(self, X, y, labels):
        '''
        Recursively builds the decision tree
        '''

        if len(np.unique(y)) == 1:
            return {"leaf": {"decision": y[0], "p": 1.0}}
        
        if X.size == 0 or X.shape[1] == 0:
            prob, decision = self.proportion_of_most_common(y)
            return {"leaf": {"decision": decision, "p": prob}}

        best_attr, best_score, best_threshold = self.best_split(X, y)
        if best_score < self.threshold:
            prob, decision = self.proportion_of_most_common(y)
            return {"leaf": {"decision": decision, "p": prob}}

        tree = {"node": {"var": labels[best_attr], "edges": []}}
        if best_threshold is not None:  # Numeric split
            left_mask = X[:, best_attr] <= best_threshold
            right_mask = X[:, best_attr] > best_threshold

            left_subtree = self.build_tree(X[left_mask], y[left_mask], labels)
            right_subtree = self.build_tree(X[right_mask], y[right_mask], labels)

            tree["node"]["edges"].extend([
                {"edge": {"op": "<=", "value": best_threshold, **left_subtree}},
                {"edge": {"op": ">", "value": best_threshold, **right_subtree}},
            ])
        else:  # Categorical split
            for val in np.unique(X[:, best_attr]):
                subset_X = X[X[:, best_attr] == val] # np.delete(X[X[:, best_attr] == val], 1, best_attr)
                subset_y = y[X[:, best_attr] == val]
                subtree = self.build_tree(subset_X, subset_y, labels)
                tree["node"]["edges"].append({"edge": {"value": val, **subtree}})

        return tree
      
    def best_split(self, X, y):
        '''
        Determines the best attribute to split on based on the chosen metric
        '''
        best_attr = None
        best_score = -np.inf
        best_threshold = None

        cols = X.shape[1]
        for attr in range(cols):
            if np.issubdtype(X[:, attr].dtype, np.number):  # Numeric attribute
                thresholds = np.unique(np.sort(X[:, attr]))
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
        unique_values, counts = np.unique(y, return_counts=True)
        probs = counts / counts.sum()
        return -sum(probs * np.log2(probs))

    def info_gain(self, X, y, attr, threshold=None):
        '''
        Calculate information gain for a given attribute
        '''
        total_entropy = self.entropy(y)

        if threshold is not None:  # Numeric attribute
            left_mask = X[:, attr] <= threshold
            right_mask = X[:, attr] > threshold

            left_y, right_y = y[left_mask], y[right_mask]
            weighted_entropy = (len(left_y) / len(y)) * self.entropy(left_y) + \
                               (len(right_y) / len(y)) * self.entropy(right_y)
        else: # Categorical attribute
            values = np.unique(X[:, attr])
            weighted_entropy = sum((len(y[X[:, attr] == val]) / len(y)) * self.entropy(y[X[:, attr] == val]) for val in values)

        return total_entropy - weighted_entropy

    def info_gain_ratio(self, X, y, attr, threshold=None):
        '''
        Calculate information gain ratio for a given attribute
        '''
        info_gain = self.info_gain(X, y, attr, threshold)

        if threshold is not None:  # Numeric attribute
            left_mask = X[:, attr] <= threshold
            right_mask = X[:, attr] > threshold

            left_ratio = len(y[left_mask]) / len(y)
            right_ratio = len(y[right_mask]) / len(y)
            split_info = -sum(r * np.log2(r) for r in [left_ratio, right_ratio] if r > 0)
        else:  # Categorical attribute
            split_info = -sum((len(y[X[:, attr] == val]) / len(y)) * np.log2(len(y[X[:, attr] == val]) / len(y)) 
                              for val in np.unique(X[:, attr]))

        return info_gain / split_info if split_info != 0 else 0

    def fit(self, X_train, y_train, labels, filename):
        '''
        Train the C45 decision tree on the given dataset
        '''
        self.tree = self.build_tree(X_train, y_train, labels)
        self.tree = {"dataset": filename, **self.tree}
    
    def predict(self, X_test, labels, prob=False, verbose=False):
        '''
        Predicts the class for each row in the given np array. Expects
        a dataframe with labeled columns. 
        labels is the labels for the classes in the order they are given in X_test
        Returns a list of class predictions if prob=False, or a list of
        tuples with the class and the probability if prob=True
        '''
        if self.tree is None:
            print("Tree is not trained, call fit() or read_tree() first")

        if verbose:
            print("Predicting", "with tree", self.tree, "labels", labels)
        # iterate and build a list of predictions
        result = []
        for index in range(X_test.shape[0]):
            row = X_test[index]
            if "node" in self.tree:
                pred = self.predict_row(row, self.tree["node"], labels, prob)
            elif "leaf" in self.tree:
                pred = (self.tree["leaf"]["decision"], self.tree["leaf"]["p"]) if prob else self.tree["leaf"]["decision"]
            else:
                print("Error: Invalid tree format")
                return None
            result.append(pred)
        return result
    
    def predict_row(self, row, node, labels, prob=False):
        '''
        Get the predicted class for a row by traversing the given node
        returns the class if prob=False, a tuple (class, p) if prob=True,
        or None if the row does not reach a leaf
        '''
        # what label the tree is splitting on
        split_var = node["var"]
        split_var_idx = labels.index(split_var)
        # value the row has for that label
        row_value = row[split_var_idx]
        
        for edge_dict in node["edges"]:
            edge = edge_dict["edge"]

            if "op" in edge:  # Numeric split
                if edge["op"] == "<=" and row_value <= edge["value"]:
                    if "leaf" in edge:
                        result = edge["leaf"]
                        return (result["decision"], result["p"]) if prob else result["decision"]
                    else:
                        return self.predict_row(row, edge["node"], labels, prob)
                elif edge["op"] == ">" and row_value > edge["value"]:
                    if "leaf" in edge:
                        result = edge["leaf"]
                        return (result["decision"], result["p"]) if prob else result["decision"]
                    else:
                        return self.predict_row(row, edge["node"], labels, prob)
            else: # Categorical split
                if edge["value"] == row_value:
                    if "leaf" in edge:
                        result = edge["leaf"]
                        if not prob:
                            return result["decision"]
                        else:
                            return (result["decision"], result["p"])
                    else:
                        return self.predict_row(row, edge["node"], labels)
        # plurality voting
        count_leaves = self.count_leaves(node)
        max_class = max(count_leaves, key=count_leaves.get)
        return max_class
    
    def count_leaves(self, node):
        '''
        Recursively counts the number of leaf nodes in the tree by decision
        '''
        leaf_classes = defaultdict(int)
        for edge in node["edges"]:
            if "leaf" in edge["edge"]:
                leaf_classes[edge["edge"]["leaf"]["decision"]] += 1
            else:
                lc_dict = self.count_leaves(edge["edge"]["node"])
                for k, v in lc_dict.items():
                    leaf_classes[k] += v
        return leaf_classes

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

