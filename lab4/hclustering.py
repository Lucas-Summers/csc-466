import argparse
import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import OrdinalEncoder

# try with `python hclustering.py csv/mammal_milk.csv --dot mm.dot`

def nums_only(x):
    '''
    Filter out non-numeric values from an array
    '''
    return x[np.vectorize(lambda val: isinstance(val, (int, float)))(x)]
def euclidean_dist(x, y):
    x, y = nums_only(x), nums_only(y)
    return np.sqrt(np.sum((x - y)**2))

class AgglomerativeClustering:
    def __init__(self, metric=euclidean_dist, linkage="single"):
        self.tree = None
        self.metric = metric
        self.linkage = linkage

    def cluster_distance(self, cluster1_idxs, cluster2_idxs, X):
        '''
        Compute the distance between two clusters using the specified linkage method
        - cluster1_idxs: indices of the first cluster in X
        - cluster2_idxs: indices of the second cluster in X
        - X: data points
        '''
        if self.linkage == "single":
            return min(self.metric(X[i], X[j]) for i in cluster1_idxs for j in cluster2_idxs)
        elif self.linkage == "complete":
            return max(self.metric(X[i], X[j]) for i in cluster1_idxs for j in cluster2_idxs)
        elif self.linkage == "average":
            return np.mean([self.metric(X[i], X[j]) for i in cluster1_idxs for j in cluster2_idxs])
        
    def fit(self, X):
        # initialize each point as a cluster (indices of X)
        clusters = [[i] for i in range(X.shape[0])]
        nodes = [{"type": "leaf", "height": 0, "data": X[i].tolist()} for i in range(X.shape[0])]
  
        while len(clusters) > 1:
            min_dist = np.inf
            for i in range(len(clusters)):
                for j in range(i+1, len(clusters)):
                    dist = self.cluster_distance(clusters[i], clusters[j], X)
                    if dist < min_dist:
                        min_dist = dist
                        min_i, min_j = i, j
                        
            # merge the two closest clusters
            curr_height = max(nodes[min_i]["height"], nodes[min_j]["height"]) + 1
            node = {"type": "node", "height": curr_height, "nodes": [nodes[min_i], nodes[min_j]]}
            nodes[min_i] = node
            nodes.pop(min_j)

            clusters[min_i].extend(clusters[min_j])
            clusters.pop(min_j)

        self.tree = nodes[0]
        self.tree["type"] = "root"

    def cut_tree(self, threshold, node=None, print_dot=False):
        '''
        Cut the dendrogram at the specified threshold.
        Return a list of nodes representing the clusters below the threshold.
        '''
        if node is None:
            node = self.tree

        if node['type'] == 'leaf' or node['height'] < threshold:
            return [node]
        ret = []
        for child in node['nodes']:
            ret.extend(self.cut_tree(threshold, node=child, print_dot=False))

        if print_dot:
            dot_str = "digraph G {\n"
            for node in ret:
                dot_str += self.generate_dot_recur(node)
            dot_str += "}"
            print(dot_str)

        return ret
    
    def generate_dot(self, nodes=None):
        '''
        For visualizing the dendrogram using Graphviz. 
        https://dreampuf.github.io/GraphvizOnline/
        '''
        if self.tree is None:
            print("No tree to generate")
            return

        dot_str = "digraph G {\n"
        dot_str = self.generate_dot_recur(self.tree, dot_str)
        dot_str += "}"
        return dot_str
        
    def generate_dot_recur(self, node, dot_str=""):
        if node['type'] == 'leaf':
            # Create a node for the leaf with its data as the label
            label = ', '.join(map(str, node['data']))
            dot_str += f'    "{id(node)}" [label="{label}"];\n'
            return dot_str
        
        # Create a node for the current level
        dot_str += f'    "{id(node)}" [label="Height {node["height"]}"];\n'
        
        # If there are children, process them recursively
        for child in node.get('nodes', []):
            dot_str += f'    "{id(node)}" -> "{id(child)}";\n'
            dot_str = self.generate_dot_recur(child, dot_str)
        
        return dot_str
        



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("filename", help="CSV file name")
    ap.add_argument("threshold", type=float, help="Threshold for cutting the dendrogram", nargs='?', default=None)
    ap.add_argument("--linkage", help="Linkage method (single, complete, average)", 
                    default="single", 
                    choices=["single", "complete", "average"])
    ap.add_argument("--json", help="Output json file name", default=None)
    ap.add_argument("--dot", help="Output DOT file name", default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.filename)

    ## Not needed for now...
    # Encode categorical columns as integers
    # categorical_columns = df.select_dtypes(include=["object"]).columns.tolist()
    # if categorical_columns:
    #     encoder = OrdinalEncoder()
    #     df[categorical_columns] = encoder.fit_transform(df[categorical_columns])
    X = df.to_numpy()

    model = AgglomerativeClustering()
    model.fit(X)

    if args.threshold is not None:
        # TODO: return just the clusters, not the nodes.
        print("Cutting the dendrogram at threshold", args.threshold)
        model.cut_tree(args.threshold, print_dot=True)

    if args.json:
        with open(args.out, "w") as f:
            json.dump(model.tree, f, indent=2)
    if args.dot:
        with open(args.dot, "w") as f:
            f.write(model.generate_dot())


