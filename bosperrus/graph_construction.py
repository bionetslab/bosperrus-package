from itertools import combinations
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import Delaunay
from sklearn.neighbors import NearestNeighbors

__all__ = ['construct_graph', 'knn_edges', 'rnn_edges', 'delaunay_edges', 'grid_edges',
           'grid_to_physical_coords', 'grid_neighbor_graph',
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


def grid_to_physical_coords(row, col, grid_type="rect", bin_size_um=1.0):
    """Convert discrete grid indices to physical (x, y) coordinates (same
    unit as `bin_size_um`), such that Euclidean distance in the result
    equals true physical distance -- unlike treating (row, col) as Cartesian
    and scaling by a single scalar, which is exact for "rect" (isotropic)
    but wrong for "hex": its two step directions, (0, +-2) and (+-1, +-1)
    (see `grid_edges`), are NOT an isotropic scaling of one physical pitch
    -- a naive scalar multiply reports a (0,2) hex step as 2x farther than a
    (1,1) step, when both are exactly one hex-neighbor-pitch apart.

    "rect": `x = col * bin_size_um`, `y = row * bin_size_um`.
    "hex": 10x Genomics' offset hex convention -- `x = col * bin_size_um/2`,
    `y = row * bin_size_um*sqrt(3)/2`, which maps every one of `grid_edges`'
    6 hex offsets to exactly `bin_size_um` apart (regular hexagonal packing).

    Parameters
    ----------
    row, col : array-like of int
        Grid indices, one pair per node, aligned by position.
    grid_type : {"hex", "rect"}, default "rect"
        See `grid_edges`.
    bin_size_um : float, default 1.0
        Physical size (um) of one grid step/spot pitch.

    Returns
    -------
    np.ndarray, shape (len(row), 2)
        (x, y) physical coordinates, one row per node.
    """
    if grid_type not in _GRID_NEIGHBOR_OFFSETS:
        raise ValueError(f"Unknown grid_type: {grid_type!r}. Expected one of {list(_GRID_NEIGHBOR_OFFSETS)}.")
    row = np.asarray(row, dtype=float)
    col = np.asarray(col, dtype=float)
    if grid_type == "hex":
        x, y = col * (bin_size_um / 2), row * (bin_size_um * np.sqrt(3) / 2)
    else:
        x, y = col * bin_size_um, row * bin_size_um
    return np.column_stack([x, y])


def grid_neighbor_graph(row, col, n_counts=None, grid_type="rect"):
    """Build a grid's adjacency graph once, so callers needing more than one
    of `split_into_connected_components`/`find_grid_border` (e.g.
    `distances.distance_to_grid_border`, which needs both) on the same
    (row, col, n_counts, grid_type) don't each rebuild `grid_edges` from
    scratch -- matters for large (>1M spot) grids.

    Parameters
    ----------
    row, col : array-like of int
        Grid indices, one pair per node, aligned by position.
    n_counts : array-like, optional
        Per-node count/signal; nodes with `n_counts <= 0` are dropped before
        building adjacency (see `split_into_connected_components`).
    grid_type : {"hex", "rect"}, default "rect"
        See `grid_edges`.

    Returns
    -------
    kept_idx : np.ndarray of int
        Indices into the original `row`/`col` of nodes that survived the
        `n_counts` filter (all of them, if `n_counts` is None).
    adjacency : scipy.sparse.csr_matrix or None
        Symmetric `(len(kept_idx), len(kept_idx))` adjacency matrix,
        positions relative to `kept_idx` (not the original row/col indices).
        None if no nodes survive the `n_counts` filter.
    """
    row, col, keep = _validate_grid_inputs(row, col, n_counts)
    kept_idx = np.flatnonzero(keep)
    if len(kept_idx) == 0:
        return kept_idx, None

    edges = grid_edges(row[keep], col[keep], grid_type=grid_type)
    n_kept = len(kept_idx)
    rows_e, cols_e = zip(*edges) if edges else ((), ())
    adjacency = csr_matrix(
        (np.ones(2 * len(rows_e)), (rows_e + cols_e, cols_e + rows_e)), shape=(n_kept, n_kept)
    )
    return kept_idx, adjacency


def split_into_connected_components(row, col, n_counts=None, grid_type="rect", min_size=0, neighbor_graph=None):
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
        always get label -1, regardless of `min_size`. Ignored if
        `neighbor_graph` is given (the graph already reflects any filtering).
    grid_type : {"hex", "rect"}, default "rect"
        See `grid_edges`.
    min_size : int, default 0
        Components with `<= min_size` surviving members are dropped (label -1).
    neighbor_graph : (kept_idx, adjacency), optional
        A precomputed result of `grid_neighbor_graph`, built with matching
        `row`/`col`/`n_counts`/`grid_type` -- reuse this to avoid rebuilding
        the adjacency graph when also calling `find_grid_border` on the same
        data (see `grid_neighbor_graph`). Computed internally if not given.

    Returns
    -------
    np.ndarray of int, shape (len(row),)
        Component label per node: 0 = largest surviving component, 1 = next,
        etc. -1 for excluded nodes (n_counts <= 0, or in a dropped component).
    """
    row = np.asarray(row)
    col = np.asarray(col)
    if len(row) != len(col):
        raise ValueError("row and col must have the same length.")

    labels = np.full(len(row), -1, dtype=int)
    kept_idx, adjacency = (
        neighbor_graph if neighbor_graph is not None else grid_neighbor_graph(row, col, n_counts, grid_type)
    )
    if len(kept_idx) == 0:
        return labels

    _, kept_labels = connected_components(adjacency, directed=False)
    sizes = np.bincount(kept_labels)

    rank = 0
    for raw_label in np.argsort(-sizes):
        if sizes[raw_label] <= min_size:
            continue
        labels[kept_idx[kept_labels == raw_label]] = rank
        rank += 1

    return labels


def find_grid_border(row, col, n_counts=None, grid_type="rect", neighbor_graph=None):
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
        Per-node count/signal, see above. Ignored if `neighbor_graph` is
        given (the graph already reflects any filtering).
    grid_type : {"hex", "rect"}, default "rect"
        See `grid_edges`.
    neighbor_graph : (kept_idx, adjacency), optional
        A precomputed result of `grid_neighbor_graph`, built with matching
        `row`/`col`/`n_counts`/`grid_type` -- reuse this to avoid rebuilding
        the adjacency graph when also calling `split_into_connected_components`
        on the same data. Computed internally if not given.

    Returns
    -------
    np.ndarray of bool, shape (len(row),)
    """
    if grid_type not in _GRID_NEIGHBOR_OFFSETS:
        raise ValueError(f"Unknown grid_type: {grid_type!r}. Expected one of {list(_GRID_NEIGHBOR_OFFSETS)}.")
    row = np.asarray(row)
    col = np.asarray(col)
    if len(row) != len(col):
        raise ValueError("row and col must have the same length.")

    is_border = np.zeros(len(row), dtype=bool)
    kept_idx, adjacency = (
        neighbor_graph if neighbor_graph is not None else grid_neighbor_graph(row, col, n_counts, grid_type)
    )
    if len(kept_idx) == 0:
        return is_border

    degree = np.asarray(adjacency.sum(axis=1)).ravel().astype(int)
    max_degree = len(_GRID_NEIGHBOR_OFFSETS[grid_type])
    is_border[kept_idx] = degree < max_degree
    return is_border
