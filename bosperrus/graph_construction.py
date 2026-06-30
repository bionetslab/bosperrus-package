from itertools import combinations
from scipy.spatial import Delaunay
from sklearn.neighbors import NearestNeighbors


def construct_graph(coordinates, graph_type, k=None, r=None):
    if graph_type == "delaunay":
        edge_list = delaunay_edges(coordinates)
        
    elif graph_type == "knn":
        if k is None:
            raise ValueError("For knn graph construction, 'k' must be provided.")

        edge_list = knn_edges(coordinates, k=k)
    elif graph_type == "rnn":
        if r is None:
            raise ValueError("For rnn graph construction, 'r' must be provided.")
        edge_list = rnn_edges(coordinates, r=r)
    else:
        raise ValueError(f"Unknown graph type: {graph_type}")
    return edge_list
    
    
def knn_edges(coords, k):
    """Directed, asymmetric kNN on full set."""
    nbrs = NearestNeighbors(n_neighbors=int(k) + 1).fit(coords)
    _, indices = nbrs.kneighbors(coords)

    edges = set()
    for u, neighbors in enumerate(indices):
        for v in neighbors[1:]:
            edges.add((u, v))
    return edges


def rnn_edges(coords, r):
    """Undirected rNN on full set."""
    nbrs = NearestNeighbors(radius=r).fit(coords)
    _, indices = nbrs.radius_neighbors(coords, radius=r)

    edges = set()
    for u, neighbors in enumerate(indices):
        for v in neighbors:
            if u != v:
                edges.add(frozenset((u, v)))
    return edges


def delaunay_edges(coords):
    """Undirected Delaunay on full set."""
    tri = Delaunay(coords)
    edges = set()

    for simplex in tri.simplices:
        for u, v in combinations(simplex, 2):
            edges.add(frozenset((u, v)))

    return edges
