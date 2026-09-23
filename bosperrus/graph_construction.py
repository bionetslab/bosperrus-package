from itertools import combinations
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import Delaunay
from sklearn.neighbors import NearestNeighbors

__all__ = ['construct_graph', 'knn_edges', 'rnn_edges', 'delaunay_edges', 'grid_edges',
           'split_into_connected_components', 'find_grid_border']

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


def _validate_grid_inputs(row, col, n_counts):
    row = np.asarray(row)
    col = np.asarray(col)
    if len(row) != len(col):
        raise ValueError("row and col must have the same length.")

    if n_counts is None:
        keep = np.ones(len(row), dtype=bool)
    else:
        n_counts = np.asarray(n_counts)
        if len(n_counts) != len(row):
            raise ValueError("n_counts must have the same length as row/col.")
        keep = n_counts > 0
    return row, col, keep


def split_into_connected_components(row, col, n_counts=None, grid_type="rect", min_size=0):
    """Label each node by its spatially-connected grid component (adjacency
    via grid_edges), largest component first.

    If `n_counts` is given, nodes with `n_counts <= 0` are dropped from the
    adjacency graph *before* splitting -- an empty grid position never
    counts as a real neighbor, so it can't silently bridge two components
    that aren't actually spatially connected once only real signal counts
    (e.g. a TMA's individual cores, kept separate rather than merged through
    a strip of zero-count background bins that technically sit between them).

    Parameters
    ----------
    row, col : array-like of int
        Grid indices, one pair per node, aligned by position.
    n_counts : array-like, optional
        Per-node count/signal. Nodes with `n_counts <= 0` are excluded and
        always get label -1, regardless of `min_size`.
    grid_type : {"hex", "rect"}, default "rect"
        See `grid_edges`.
    min_size : int, default 0
        Components with `<= min_size` surviving members are dropped (label -1).

    Returns
    -------
    np.ndarray of int, shape (len(row),)
        Component label per node: 0 = largest surviving component, 1 = next,
        etc. -1 for excluded nodes (n_counts <= 0, or in a dropped component).
    """
    row, col, keep = _validate_grid_inputs(row, col, n_counts)
    labels = np.full(len(row), -1, dtype=int)
    kept_idx = np.flatnonzero(keep)
    if len(kept_idx) == 0:
        return labels

    edges = grid_edges(row[keep], col[keep], grid_type=grid_type)
    n_kept = len(kept_idx)
    rows_e, cols_e = zip(*edges) if edges else ((), ())
    adjacency = csr_matrix(
        (np.ones(2 * len(rows_e)), (rows_e + cols_e, cols_e + rows_e)), shape=(n_kept, n_kept)
    )
    _, kept_labels = connected_components(adjacency, directed=False)
    sizes = np.bincount(kept_labels)

    rank = 0
    for raw_label in np.argsort(-sizes):
        if sizes[raw_label] <= min_size:
            continue
        labels[kept_idx[kept_labels == raw_label]] = rank
        rank += 1

    return labels


def find_grid_border(row, col, n_counts=None, grid_type="rect"):
    """Flag border nodes: grid positions with fewer than the full complement
    of grid neighbors actually present (4 for "rect", 6 for "hex" -- see
    grid_edges). A purely local, per-node property -- unlike distance-to-
    border, it does not need components split apart first, since a node's
    true degree already reflects only neighbors that actually exist,
    regardless of which disconnected fragment it belongs to.

    If `n_counts` is given, nodes with `n_counts <= 0` are dropped from the
    adjacency graph before computing degree (an empty grid position never
    counts as a real neighbor) and are always flagged False -- they aren't
    part of the tissue at all, so "border" doesn't apply to them.

    Parameters
    ----------
    row, col : array-like of int
        Grid indices, one pair per node, aligned by position.
    n_counts : array-like, optional
        Per-node count/signal, see above.
    grid_type : {"hex", "rect"}, default "rect"
        See `grid_edges`.

    Returns
    -------
    np.ndarray of bool, shape (len(row),)
    """
    if grid_type not in _GRID_NEIGHBOR_OFFSETS:
        raise ValueError(f"Unknown grid_type: {grid_type!r}. Expected one of {list(_GRID_NEIGHBOR_OFFSETS)}.")
    row, col, keep = _validate_grid_inputs(row, col, n_counts)
    is_border = np.zeros(len(row), dtype=bool)
    kept_idx = np.flatnonzero(keep)
    if len(kept_idx) == 0:
        return is_border

    edges = grid_edges(row[keep], col[keep], grid_type=grid_type)
    degree = np.zeros(len(kept_idx), dtype=int)
    for u, v in edges:
        degree[u] += 1
        degree[v] += 1

    max_degree = len(_GRID_NEIGHBOR_OFFSETS[grid_type])
    is_border[kept_idx] = degree < max_degree
    return is_border
