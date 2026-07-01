import warnings
import numpy as np
import pandas as pd

try:
    from graph_tool.all import Graph
    from graph_tool.centrality import betweenness, pagerank, closeness
    from graph_tool.clustering import local_clustering
    HAS_GRAPH_TOOL = True
except ImportError:
    HAS_GRAPH_TOOL = False

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

__all__ = ['compute_centrality_measures']

_VALID_MEASURES = {"degree", "pagerank", "betweenness", "closeness", "harmonic", "clustering"}


def _compute_with_graph_tool(edge_list, N, measures):
    g = Graph(directed=False)
    g.add_edge_list(edge_list)

    results = {}
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
        except Exception as e:
            warnings.warn(f"Failed to compute {m}: {e}", RuntimeWarning)

    for key in results:
        arr = results[key]
        if len(arr) < N:
            padded = np.zeros(N)
            padded[:len(arr)] = arr
            results[key] = padded

    return pd.DataFrame({k: list(v) for k, v in results.items()})


def _compute_with_networkx(edge_list, N, measures):
    G = nx.Graph()
    G.add_nodes_from(range(N))
    G.add_edges_from(edge_list)

    results = {}
    for m in measures:
        try:
            if m == "degree":
                d = dict(G.degree())
                results["degree"] = np.array([d.get(i, 0) for i in range(N)], dtype=float)
            elif m == "pagerank":
                pr = nx.pagerank(G)
                results["pagerank"] = np.array([pr.get(i, 0.0) for i in range(N)])
            elif m == "betweenness":
                bc = nx.betweenness_centrality(G)
                results["betweenness"] = np.array([bc.get(i, 0.0) for i in range(N)])
            elif m == "closeness":
                cc = nx.closeness_centrality(G)
                results["closeness"] = np.array([cc.get(i, 0.0) for i in range(N)])
            elif m == "harmonic":
                hc = nx.harmonic_centrality(G)
                # networkx returns raw sums; divide by (N-1) to match graph-tool's normalized convention
                norm = N - 1 if N > 1 else 1
                results["harmonic"] = np.array([hc.get(i, 0.0) / norm for i in range(N)])
            elif m == "clustering":
                cl = nx.clustering(G)
                results["clustering"] = np.array([cl.get(i, 0.0) for i in range(N)])
        except Exception as e:
            warnings.warn(f"Failed to compute {m}: {e}", RuntimeWarning)

    return pd.DataFrame(results)


def compute_centrality_measures(edge_list, N, measures, backend=None):
    """Compute graph centrality measures.

    Parameters
    ----------
    edge_list : iterable of (u, v) pairs or frozenset({u, v})
        Edges. The graph is always treated as undirected.
    N : int
        Total number of nodes. Isolated nodes not in edge_list are zero-padded.
    measures : list of str
        Measures to compute. Supported: "degree", "pagerank", "betweenness",
        "closeness", "harmonic", "clustering".
    backend : str or None
        "graph_tool", "networkx", or None (auto: graph-tool if installed, else networkx).

    Returns
    -------
    pd.DataFrame with one column per measure and N rows.
    """
    for m in measures:
        if m not in _VALID_MEASURES:
            raise ValueError(f"Unknown centrality measure: '{m}'. Supported: {sorted(_VALID_MEASURES)}")

    if backend is None:
        use_graph_tool = HAS_GRAPH_TOOL
    elif backend == "graph_tool":
        use_graph_tool = True
    elif backend == "networkx":
        use_graph_tool = False
    else:
        raise ValueError(f"Unknown backend: '{backend}'. Use 'graph_tool', 'networkx', or None.")

    if use_graph_tool:
        if not HAS_GRAPH_TOOL:
            raise ImportError(
                "graph-tool is not installed. Install via conda: "
                "conda install -c conda-forge graph-tool"
            )
        return _compute_with_graph_tool(edge_list, N, measures)
    else:
        if not HAS_NETWORKX:
            raise ImportError(
                "Neither graph-tool nor networkx is installed. "
                "Install networkx via pip: pip install networkx"
            )
        return _compute_with_networkx(edge_list, N, measures)
