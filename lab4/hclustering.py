import argparse
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, rand_score
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
        self.cut = None

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
        If print_dot is True, print DOT for visualizing cut tree.
        Returns a list of nodes at the cut level.
        '''
        if node is None:
            node = self.tree

        if node['type'] == 'leaf' or node['height'] < threshold:
            self.cut = [node]
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

        self.cut = ret
        return ret
    
    def get_clusters(self, threshold):
        '''
        Get the clusters at the specified threshold.
        If threshold < 1, it is treated as a proportion of the maximum height.
        '''
        if threshold < 1:
            threshold = self.tree["height"] * threshold
        nodes = self.cut_tree(threshold)
        X, y = [], []
        for i, node in enumerate(nodes):
            leaves = self.get_leaves(node)
            for leaf in leaves:
                X.append(leaf['data'])
                y.append(i)
        return np.array(X), np.array(y)
        
    def get_leaves(self, node):
        if node['type'] == 'leaf':
            return [node]
        leaves = []
        for child in node['nodes']:
            leaves += self.get_leaves(child)
        return leaves
    
    def plot_clusters(self, threshold):
        '''
        Plot the clusters and centroids in 2D (using PCA if needed)
        '''
        X, y = self.get_clusters(threshold)
        # If the data is more than 2D, apply PCA to reduce it to 2D
        if X.shape[1] > 2:
            # encode categorical columns as integers
            encoder = OrdinalEncoder()
            cat_cols = np.vectorize(lambda x: isinstance(x, str))(X[0])
            if np.any(cat_cols):
                X[:, cat_cols] = encoder.fit_transform(X[:, cat_cols])
            pca = PCA(n_components=2)
            X = pca.fit_transform(X)
        
        # Plot the clusters
        for i in np.unique(y):
            plt.scatter(X[y == i, 0], X[y == i, 1], label=f'Cluster {i}')

        # Add labels and title
        plt.title('Agg Clusters (PCA-reduced)')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.legend(loc='best')

        plt.show()

    def score(self, X, y):
        silhouette = silhouette_score(X, y) if len(self.cut) > 1 else 0
        ch_index = calinski_harabasz_score(X, y) if len(self.cut) > 1 else 0

        return {
            "silhouette_score": silhouette,
            "ch_index": ch_index
        }

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
    ap.add_argument("threshold", type=float, help="Threshold for cutting the dendrogram, if < 1, proportion of maximum height", nargs='?', default=None)
    ap.add_argument("--linkage", help="Linkage method (single, complete, average)", 
                    default="single", 
                    choices=["single", "complete", "average"])
    ap.add_argument("--json", help="Output json file name", default=None)
    ap.add_argument("--dot", help="Output DOT file name", default=None)
    ap.add_argument("--plot", help="Plot the clusters", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.filename)

    # filter by column name
    X = df.select_dtypes(include=[np.number])
    y = df.select_dtypes(include=[object])

    ## Not needed for now...
    # Encode categorical columns as integers
    # categorical_columns = df.select_dtypes(include=["object"]).columns.tolist()
    # if categorical_columns:
    #     encoder = OrdinalEncoder()
    #     df[categorical_columns] = encoder.fit_transform(df[categorical_columns])
    X = X.to_numpy()

    model = AgglomerativeClustering()
    print("Fitting the model...")
    model.fit(X)

    if args.threshold is not None:
        print("Cutting the dendrogram at threshold", args.threshold)
        X, y = model.get_clusters(args.threshold)
        print(model.tree["height"])
        if args.plot:
            model.plot_clusters(args.threshold)
        print(model.score(X, y))
    if args.json:
        with open(args.out, "w") as f:
            json.dump(model.tree, f, indent=2)
    if args.dot:
        with open(args.dot, "w") as f:
            f.write(model.generate_dot())


