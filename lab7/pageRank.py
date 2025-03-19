import sys
import csv
import time
import numpy as np
from collections import defaultdict, Counter

class PageRank:
    
    def __init__(self, damping_factor=0.85, epsilon=1e-6, max_iterations=200):
        '''
        Parameters:
        - damping_factor: Probability of following a link (default 0.85)
        - epsilon: Convergence threshold (default 1e-6)
        - max_iterations: Maximum number of iterations to perform (default 200)
        '''
        self.damping_factor = damping_factor
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        
        # Graph data structures
        self.nodes = []
        self.node_to_idx = {}
        self.adjacency_list = defaultdict(list)
        self.out_degrees = Counter()
        
        # Performance metrics
        self.read_time = 0
        self.process_time = 0
        self.iterations = 0
        
        # Results
        self.page_ranks = None
    
    def read_csv_data(self, filename):
        '''
        Read the CSV data file and build a graph structure.
        The function handles both directed and undirected graphs.

        Returns:
        - Number of nodes in the network
        - Number of edges in the network
        - read_time: The time it took to read in the graph from the csv file
        '''
        start_time = time.time()
        
        # Read file and extract node names and edges
        edges = []
        nodes_set = set()
        
        with open(filename, 'r') as f:
            reader = csv.reader(f)
            
            for row in reader:
                # Remove any whitespace and quotes from node names
                node1 = row[0].strip().strip('"')
                node2 = row[2].strip().strip('"')
                
                # Clean score values (handle both integer and string formats)
                if len(row) >= 4:
                    try:
                        score1 = int(''.join(filter(str.isdigit, row[1])))
                        score2 = int(''.join(filter(str.isdigit, row[3])))
                    except ValueError:
                        score1 = 0
                        score2 = 0
                else:
                    score1 = 0
                    score2 = 0
                
                nodes_set.add(node1)
                nodes_set.add(node2)
                
                # Determine edge direction based on dataset
                if "football" in filename.lower():
                    # NCAA Football dataset: edge goes from loser to winner
                    if score1 > score2:
                        edges.append((node2, node1))
                    else:
                        edges.append((node1, node2))
                else:
                    # For other datasets, use the default direction given in the file
                    edges.append((node1, node2))
                    
                    # If it's a file that represents undirected graphs (no 'Dir' in filename),
                    # add the reverse edge as well
                    if "Dir" not in filename:
                        edges.append((node2, node1))
        
        # Convert nodes set to list for indexing
        self.nodes = sorted(list(nodes_set))
        self.node_to_idx = {node: idx for idx, node in enumerate(self.nodes)}
        
        # Create adjacency list representation of the graph
        for source, target in edges:
            self.adjacency_list[self.node_to_idx[source]].append(self.node_to_idx[target])
        
        # Calculate out-degree for each node
        for source, targets in self.adjacency_list.items():
            self.out_degrees[source] = len(targets)
        
        # Handle nodes with no outgoing edges (dangling nodes)
        for node_idx in range(len(self.nodes)):
            if node_idx not in self.adjacency_list:
                self.adjacency_list[node_idx] = []
        
        self.read_time = time.time() - start_time
        
        return len(self.nodes), len(edges), self.read_time
    
    def compute_pagerank(self):
        '''
        Implements the PageRank algorithm using NumPy matrix operations for efficiency.
        
        Returns:
        - page_ranks: List of PageRank scores for each node
        - iterations: Number of iterations performed
        - process_time: Time taken for computation
        '''
        start_time = time.time()
        
        n = len(self.nodes)
        
        # Initialize PageRank with uniform distribution
        self.page_ranks = np.ones(n) / n
        
        # Convert adjacency list to a sparse transition matrix
        # M[i,j] = probability of moving from node j to node i
        M = np.zeros((n, n))
        
        for source, targets in self.adjacency_list.items():
            if targets:  # If the node has outgoing links
                for target in targets:
                    M[target, source] = 1.0 / self.out_degrees[source]
        
        # Identify dangling nodes and create teleportation vector
        dangling_nodes = np.array([i for i in range(n) if not self.adjacency_list[i]])
        teleport = np.ones(n) / n
        
        self.iterations = 0
        diff = float('inf')
        
        while diff > self.epsilon and self.iterations < self.max_iterations:
            self.iterations += 1
            prev_page_ranks = self.page_ranks.copy()
            
            # Handle dangling nodes: add their contribution to all nodes
            dangling_contrib = np.sum(self.page_ranks[dangling_nodes]) * teleport if len(dangling_nodes) > 0 else 0
            
            # Update PageRank: 
            self.page_ranks = (1 - self.damping_factor) * teleport + \
                            self.damping_factor * (M @ self.page_ranks + dangling_contrib)
            self.page_ranks = self.page_ranks / np.sum(self.page_ranks)

            # Check for convergence
            diff = np.sum(np.abs(self.page_ranks - prev_page_ranks))
        
        self.process_time = time.time() - start_time
        return self.page_ranks, self.iterations, self.process_time
    
    def get_ranked_nodes(self):
        '''
        Returns nodes sorted by their PageRank scores in descending order.
        '''
        if self.page_ranks is None:
            return []
        
        # Sort nodes by PageRank in descending order
        ranked_nodes = [(self.nodes[i], self.page_ranks[i]) for i in range(len(self.nodes))]
        ranked_nodes.sort(key=lambda x: x[1], reverse=True)
        
        return ranked_nodes
    
    def print_stats(self):
        '''
        Print statistics about the graph and the PageRank computation.
        '''
        print(f"Read time: {self.read_time:.4f} seconds")
        print(f"Number of nodes: {len(self.nodes)}")
        print(f"Number of edges: {sum(len(targets) for targets in self.adjacency_list.values())}")
        print(f"Processing time: {self.process_time:.4f} seconds")
        print(f"Number of iterations: {self.iterations}")
    
    def print_results(self):
        '''
        Print the PageRank results in descending order.
        '''
        ranked_nodes = self.get_ranked_nodes()
        
        print("\nPageRank Results:")
        for i, (node, rank) in enumerate(ranked_nodes, 1):
            print(f"{i} {node} with pagerank: {rank}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pageRank.py <input_file>")
        sys.exit(1)
    input_file = sys.argv[1]

    print(f"Processing file: {input_file}")
    pagerank = PageRank()
    pagerank.read_csv_data(input_file)
    pagerank.compute_pagerank()
    pagerank.print_stats()
    pagerank.print_results()
