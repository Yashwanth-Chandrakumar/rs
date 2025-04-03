# exp 10 google page rank

import numpy as np

def pagerank(graph, damping=0.85, epsilon=1e-8, max_iter=100):
    # Convert adjacency dict to matrix
    nodes = list(graph.keys())
    n = len(nodes)
    node_idx = {nodes[i]: i for i in range(n)}
    
    # Build transition matrix
    M = np.zeros((n, n))
    for node, links in graph.items():
        if links:  # Avoid division by zero
            for target in links:
                M[node_idx[target], node_idx[node]] = 1.0 / len(links)
    
    # Initialize ranks
    ranks = np.ones(n) / n
    
    # Power iteration
    for _ in range(max_iter):
        prev_ranks = ranks.copy()
        ranks = (1 - damping) / n + damping * (M @ ranks)
        if np.sum(np.abs(ranks - prev_ranks)) < epsilon:
            break
            
    return {nodes[i]: ranks[i] for i in range(n)}

graph = {'A': ['B', 'C'], 'B': ['C'], 'C': ['A']}
result = pagerank(graph)
