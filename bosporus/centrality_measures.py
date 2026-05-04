import numpy as np
try:
    from graph_tool.all import Graph
    HAS_GRAPH_TOOL = True
except ImportError:
    HAS_GRAPH_TOOL = False
    
from graph_tool.centrality import betweenness, pagerank, closeness
from graph_tool.clustering import local_clustering


def compute_centrality_measures(edge_list, N, measures=["degree", "closeness", "betweenness", "harmonic", "clustering", "pagerank"]):
    g = Graph(directed=False)
    g.add_edge_list(edge_list)
    
    # Pre-allocate results with .a.copy() to ensure data is returned 
    # as standard numpy arrays before the process terminates.
    
    results = dict()
    for m in measures: 
        try:
            if m == "degree":
                results["degree"] = g.get_total_degrees(range(g.num_vertices())).copy()
            elif m == "pagerank":
                results["pagerank"] = pagerank(g).a.copy()
            elif m == "betweenness":
                results["betweenness"] = betweenness(g)[0].a.copy()
            elif m == "closeness":
                results["closeness"] = closeness(g).a.copy()
            elif m == "harmonic":
                results["harmonic"] = closeness(g, harmonic=True).a.copy()
            elif m == "clustering":
                results["clustering"] = local_clustering(g).a.copy()
            else:
                raise ValueError(f"Unknown centrality measure: {m}")
        except Exception as e:
            print(f"Error computing {m}: {e}")

    
    for key in results:
        arr = results[key]
        if len(arr) < N:
            padded = np.zeros(N)
            padded[:len(arr)] = arr
            results[key] = padded
    
    return {k: list(v) for k, v in results.items()}
