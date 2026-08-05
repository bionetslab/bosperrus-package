from itertools import combinations
import numpy as np
from scipy.spatial import Delaunay
from sklearn.neighbors import NearestNeighbors

__all__ = ['construct_graph', 'knn_edges', 'rnn_edges', 'delaunay_edges', 'grid_edges']

_GRID_NEIGHBOR_OFFSETS = {
    "hex": [(0, -2), (0, 2), (-1, -1), (-1, 1), (1, -1), (1, 1)],
    "rect": [(-1, 0), (1, 0), (0, -1), (0, 1)],
}


def construct_graph(coordinates, graph_type, k=None, r=None, row=None, col=None, grid_type="hex"):
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
    elif graph_type == "grid":
        if row is None or col is None:
            raise ValueError("For grid graph construction, 'row' and 'col' must be provided.")
        edge_list = grid_edges(row, col, grid_type=grid_type)
    else:
        raise ValueError(f"Unknown graph type: {graph_type}")
    return edge_list
    
    
def knn_edges(coordinates, k):
    """Directed, asymmetric kNN on full set."""
    nbrs = NearestNeighbors(n_neighbors=int(k) + 1).fit(coordinates)
    _, indices = nbrs.kneighbors(coordinates)

    edges = set()
    for u, neighbors in enumerate(indices):
        for v in neighbors[1:]:
            edges.add((u, v))
    return edges


def rnn_edges(coordinates, r):
    """Undirected rNN on full set."""
    nbrs = NearestNeighbors(radius=r).fit(coordinates)
    _, indices = nbrs.radius_neighbors(coordinates, radius=r)

    edges = set()
    for u, neighbors in enumerate(indices):
        for v in neighbors:
            if u != v:
                edges.add(frozenset((u, v)))
    return edges


def delaunay_edges(coordinates):
    """Undirected Delaunay on full set."""
    tri = Delaunay(coordinates)
    edges = set()

    for simplex in tri.simplices:
        for u, v in combinations(simplex, 2):
            edges.add(frozenset((u, v)))

    return edges


def grid_edges(row, col, grid_type="hex"):
    """Undirected grid adjacency from discrete grid indices.

    Unlike every other function in this module, `row`/`col` are discrete grid
    indices (e.g. Visium's `array_row`/`array_col`), not continuous spatial or
    pixel coordinates -- adjacency is exact (a lookup), not nearest-neighbor.

    Parameters
    ----------
    row, col : array-like of int
        Grid indices, one pair per node, aligned by position.
    grid_type : {"hex", "rect"}
        "hex": Visium's offset hex grid; 6 neighbors at (row, col +/- 2) and
        (row -/+ 1, col +/- 1).
        "rect": square grid (e.g. Visium HD); 4 neighbors at (row +/- 1, col) and
        (row, col +/- 1).

    Returns
    -------
    set of frozenset({u, v})
        Undirected edges, indexed positionally into `row`/`col` (matching
        rnn_edges/delaunay_edges; not knn_edges' directed tuples, since grid
        adjacency is inherently symmetric).
    """
    if grid_type not in _GRID_NEIGHBOR_OFFSETS:
        raise ValueError(f"Unknown grid_type: {grid_type!r}. Expected one of {list(_GRID_NEIGHBOR_OFFSETS)}.")
    row = np.asarray(row)
    col = np.asarray(col)
    if len(row) != len(col):
        raise ValueError("row and col must have the same length.")

    offsets = _GRID_NEIGHBOR_OFFSETS[grid_type]
    lookup = {(int(r), int(c)): i for i, (r, c) in enumerate(zip(row, col))}

    edges = set()
    for i, (r, c) in enumerate(zip(row, col)):
        for dr, dc in offsets:
            j = lookup.get((int(r) + dr, int(c) + dc))
            if j is not None:
                edges.add(frozenset((i, j)))
    return edges
