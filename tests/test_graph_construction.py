import numpy as np
import pytest
from bosperrus.graph_construction import (
    knn_edges,
    rnn_edges,
    delaunay_edges,
    construct_graph,
    grid_edges,
    split_into_connected_components,
    find_grid_border,
    grid_to_physical_coords,
    grid_neighbor_graph,
)


@pytest.fixture
def grid_coords():
    """5x2 grid of points, deterministic."""
    rng = np.random.default_rng(42)
    # 10 points arranged in a 5x2 grid with small jitter to avoid degeneracy
    xs, ys = np.meshgrid(np.linspace(0, 1, 5), np.linspace(0, 1, 2))
    coords = np.column_stack([xs.ravel(), ys.ravel()])
    coords += rng.uniform(-0.05, 0.05, coords.shape)
    return coords


# ---------------------------------------------------------------------------
# knn_edges tests
# ---------------------------------------------------------------------------

def test_knn_edges_returns_tuples(grid_coords):
    edges = knn_edges(grid_coords, k=2)
    assert isinstance(edges, set)
    assert len(edges) > 0
    for edge in edges:
        assert isinstance(edge, tuple), f"Expected tuple, got {type(edge)}"
        assert len(edge) == 2


def test_knn_edges_count(grid_coords):
    k = 2
    n = len(grid_coords)
    edges = knn_edges(grid_coords, k=k)
    # Each of the N nodes emits exactly k directed edges
    assert len(edges) <= n * k


def test_knn_edges_each_node_has_k_outgoing(grid_coords):
    k = 2
    edges = knn_edges(grid_coords, k=k)
    n = len(grid_coords)
    # Count outgoing edges per source node
    out_degree = {u: 0 for u in range(n)}
    for u, v in edges:
        out_degree[u] += 1
    for u in range(n):
        assert out_degree[u] == k, (
            f"Node {u} has {out_degree[u]} outgoing edges, expected {k}"
        )


# ---------------------------------------------------------------------------
# rnn_edges tests
# ---------------------------------------------------------------------------

def test_rnn_edges_returns_frozensets(grid_coords):
    edges = rnn_edges(grid_coords, r=0.5)
    assert isinstance(edges, set)
    for edge in edges:
        assert isinstance(edge, frozenset), f"Expected frozenset, got {type(edge)}"


def test_rnn_edges_symmetry(grid_coords):
    """frozenset({u,v}) == frozenset({v,u}) — check that no duplicate pairs exist."""
    edges = rnn_edges(grid_coords, r=0.5)
    # Verify idempotency: re-adding the reversed pair produces the same set
    edges_check = set()
    for fs in edges:
        u, v = tuple(fs)
        edges_check.add(frozenset((u, v)))
        edges_check.add(frozenset((v, u)))  # identical frozenset
    assert edges_check == edges


def test_rnn_edges_zero_radius(grid_coords):
    """With r=0, no pair of distinct points can be within radius 0 of each other."""
    edges = rnn_edges(grid_coords, r=0)
    # All resulting frozensets must not contain two distinct nodes
    for fs in edges:
        nodes = tuple(fs)
        assert len(nodes) == 1 or nodes[0] == nodes[1], (
            "r=0 should produce no edges between distinct nodes"
        )
    # More directly: the set should be empty (no self-edges since u != v is enforced)
    assert len(edges) == 0


# ---------------------------------------------------------------------------
# delaunay_edges tests
# ---------------------------------------------------------------------------

def test_delaunay_edges_returns_frozensets(grid_coords):
    edges = delaunay_edges(grid_coords)
    assert isinstance(edges, set)
    assert len(edges) > 0
    for edge in edges:
        assert isinstance(edge, frozenset), f"Expected frozenset, got {type(edge)}"


def test_delaunay_edges_covers_all_nodes(grid_coords):
    edges = delaunay_edges(grid_coords)
    n = len(grid_coords)
    nodes_in_edges = set()
    for fs in edges:
        nodes_in_edges.update(fs)
    for i in range(n):
        assert i in nodes_in_edges, f"Node {i} not covered by any Delaunay edge"


# ---------------------------------------------------------------------------
# grid_edges tests
# ---------------------------------------------------------------------------

@pytest.fixture
def hex_grid_rowcol():
    """3 rows x offset hex layout, deterministic. Node (1,3) (index 6) has a full
    complement of 6 neighbors; corner node (0,0) (index 0) has only 2."""
    row = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2])
    col = np.array([0, 2, 4, 6, 8, 1, 3, 5, 7, 0, 2, 4, 6, 8])
    return row, col


@pytest.fixture
def rect_grid_rowcol():
    """3x3 square grid, deterministic. Center node (1,1) (index 4) has 4 neighbors;
    corner node (0,0) (index 0) has only 2."""
    rows, cols = np.meshgrid(np.arange(3), np.arange(3), indexing="ij")
    return rows.ravel(), cols.ravel()


def test_grid_edges_returns_frozensets(hex_grid_rowcol):
    row, col = hex_grid_rowcol
    edges = grid_edges(row, col, grid_type="hex")
    assert isinstance(edges, set)
    assert len(edges) > 0
    for edge in edges:
        assert isinstance(edge, frozenset), f"Expected frozenset, got {type(edge)}"


def test_grid_edges_hex_interior_node_has_six_neighbors(hex_grid_rowcol):
    row, col = hex_grid_rowcol
    edges = grid_edges(row, col, grid_type="hex")
    degree = {i: 0 for i in range(len(row))}
    for u, v in (tuple(e) for e in edges):
        degree[u] += 1
        degree[v] += 1
    assert degree[6] == 6, f"Interior node (1,3) expected 6 neighbors, got {degree[6]}"


def test_grid_edges_hex_corner_node_has_fewer_neighbors(hex_grid_rowcol):
    row, col = hex_grid_rowcol
    edges = grid_edges(row, col, grid_type="hex")
    degree = {i: 0 for i in range(len(row))}
    for u, v in (tuple(e) for e in edges):
        degree[u] += 1
        degree[v] += 1
    assert degree[0] == 2, f"Corner node (0,0) expected 2 neighbors, got {degree[0]}"


def test_grid_edges_rect_interior_node_has_four_neighbors(rect_grid_rowcol):
    row, col = rect_grid_rowcol
    edges = grid_edges(row, col, grid_type="rect")
    degree = {i: 0 for i in range(len(row))}
    for u, v in (tuple(e) for e in edges):
        degree[u] += 1
        degree[v] += 1
    assert degree[4] == 4, f"Center node (1,1) expected 4 neighbors, got {degree[4]}"


def test_grid_edges_rect_corner_node_has_fewer_neighbors(rect_grid_rowcol):
    row, col = rect_grid_rowcol
    edges = grid_edges(row, col, grid_type="rect")
    degree = {i: 0 for i in range(len(row))}
    for u, v in (tuple(e) for e in edges):
        degree[u] += 1
        degree[v] += 1
    assert degree[0] == 2, f"Corner node (0,0) expected 2 neighbors, got {degree[0]}"


def test_grid_edges_invalid_grid_type(rect_grid_rowcol):
    row, col = rect_grid_rowcol
    with pytest.raises(ValueError, match="Unknown grid_type"):
        grid_edges(row, col, grid_type="invalid")


def test_grid_edges_mismatched_length_raises():
    with pytest.raises(ValueError, match="same length"):
        grid_edges(np.array([0, 1, 2]), np.array([0, 1]), grid_type="rect")


# ---------------------------------------------------------------------------
# construct_graph dispatch tests
# ---------------------------------------------------------------------------

def test_construct_graph_dispatches_knn(grid_coords):
    result = construct_graph(grid_coords, "knn", k=3)
    expected = knn_edges(grid_coords, 3)
    assert result == expected


def test_construct_graph_dispatches_rnn(grid_coords):
    result = construct_graph(grid_coords, "rnn", r=0.5)
    expected = rnn_edges(grid_coords, 0.5)
    assert result == expected


def test_construct_graph_dispatches_delaunay(grid_coords):
    result = construct_graph(grid_coords, "delaunay")
    expected = delaunay_edges(grid_coords)
    assert result == expected


def test_construct_graph_invalid_type(grid_coords):
    with pytest.raises(ValueError, match="Unknown graph type"):
        construct_graph(grid_coords, "invalid")


def test_construct_graph_dispatches_grid(rect_grid_rowcol):
    row, col = rect_grid_rowcol
    # "grid" construction never touches `coordinates` (row/col are the actual input),
    # so passing None here is deliberate, not an oversight.
    result = construct_graph(None, "grid", row=row, col=col, grid_type="rect")
    expected = grid_edges(row, col, grid_type="rect")
    assert result == expected


def test_construct_graph_grid_requires_row_and_col(grid_coords):
    with pytest.raises(ValueError, match="'row' and 'col' must be provided"):
        construct_graph(grid_coords, "grid")


# ---------------------------------------------------------------------------
# split_into_connected_components
# ---------------------------------------------------------------------------

@pytest.fixture
def line_grid_with_gap():
    """5 nodes in a row (row=0, col=0..4); n_counts=0 at col=2 splits it
    into two components of 2 once excluded."""
    row = np.zeros(5, dtype=int)
    col = np.arange(5)
    n_counts = np.array([5, 5, 0, 5, 5])
    return row, col, n_counts


def test_split_into_connected_components_single_block_is_one_component(rect_grid_rowcol):
    row, col = rect_grid_rowcol
    labels = split_into_connected_components(row, col, grid_type="rect")
    assert (labels == 0).all()


def test_split_into_connected_components_without_n_counts_is_one_component(line_grid_with_gap):
    row, col, _ = line_grid_with_gap
    labels = split_into_connected_components(row, col, grid_type="rect")
    assert (labels == 0).all()


def test_split_into_connected_components_n_counts_filters_zero_and_splits(line_grid_with_gap):
    row, col, n_counts = line_grid_with_gap
    labels = split_into_connected_components(row, col, n_counts=n_counts, grid_type="rect")
    assert labels[2] == -1  # zero-count node always excluded
    assert labels[0] == labels[1]
    assert labels[3] == labels[4]
    assert labels[0] != labels[3]
    assert set(labels.tolist()) == {-1, 0, 1}


def test_split_into_connected_components_min_size_drops_small(line_grid_with_gap):
    row, col, n_counts = line_grid_with_gap
    labels = split_into_connected_components(row, col, n_counts=n_counts, grid_type="rect", min_size=2)
    # both surviving pieces have exactly 2 members, <= min_size=2 -> all dropped
    assert (labels == -1).all()


def test_split_into_connected_components_mismatched_length_raises():
    with pytest.raises(ValueError, match="same length"):
        split_into_connected_components(np.array([0, 1, 2]), np.array([0, 1]))


def test_split_into_connected_components_n_counts_mismatched_length_raises(rect_grid_rowcol):
    row, col = rect_grid_rowcol
    with pytest.raises(ValueError, match="n_counts must have the same length"):
        split_into_connected_components(row, col, n_counts=np.ones(3))


# ---------------------------------------------------------------------------
# find_grid_border
# ---------------------------------------------------------------------------

def test_find_grid_border_rect_corner_is_border_center_is_not(rect_grid_rowcol):
    row, col = rect_grid_rowcol
    is_border = find_grid_border(row, col, grid_type="rect")
    assert is_border[0]        # corner, degree 2 < 4
    assert not is_border[4]    # center, degree 4


def test_find_grid_border_hex_corner_is_border_interior_is_not(hex_grid_rowcol):
    row, col = hex_grid_rowcol
    is_border = find_grid_border(row, col, grid_type="hex")
    assert is_border[0]      # corner, degree 2 < 6
    assert not is_border[6]  # interior, degree 6


def test_find_grid_border_n_counts_excludes_zero_count_and_updates_neighbor_degree(rect_grid_rowcol):
    """Zeroing one of the center's 4 neighbors should exclude that neighbor
    (always False) and drop the center's real degree to 3, making it a
    border node even though its raw grid degree is still 4."""
    row, col = rect_grid_rowcol
    n_counts = np.ones(9)
    n_counts[1] = 0  # node (0,1), one of the center (1,1)'s neighbors

    is_border = find_grid_border(row, col, n_counts=n_counts, grid_type="rect")
    assert not is_border[1]   # excluded node is always False
    assert is_border[4]       # center now has only 3 real neighbors


def test_find_grid_border_invalid_grid_type(rect_grid_rowcol):
    row, col = rect_grid_rowcol
    with pytest.raises(ValueError, match="Unknown grid_type"):
        find_grid_border(row, col, grid_type="invalid")


def test_find_grid_border_mismatched_length_raises():
    with pytest.raises(ValueError, match="same length"):
        find_grid_border(np.array([0, 1, 2]), np.array([0, 1]))


# ---------------------------------------------------------------------------
# grid_to_physical_coords
# ---------------------------------------------------------------------------

def test_grid_to_physical_coords_rect_is_isotropic_scale():
    row = np.array([0, 1, 2])
    col = np.array([0, 1, 2])
    coords = grid_to_physical_coords(row, col, grid_type="rect", bin_size_um=2.5)
    np.testing.assert_allclose(coords, np.column_stack([col * 2.5, row * 2.5]))


def test_grid_to_physical_coords_hex_all_offsets_equidistant():
    """Every one of grid_edges' hex neighbor offsets must map to exactly
    bin_size_um apart -- the whole point of not treating (row, col) as
    isotropic Cartesian for hex (unlike "rect", (0,2) and (1,1) steps are
    NOT equidistant in raw index space, sqrt(2) vs 2, even though they are
    physically equidistant on the real hex lattice)."""
    bin_size_um = 3.0
    origin = grid_to_physical_coords(np.array([5]), np.array([5]), grid_type="hex", bin_size_um=bin_size_um)[0]
    for dr, dc in [(0, -2), (0, 2), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
        neighbor = grid_to_physical_coords(
            np.array([5 + dr]), np.array([5 + dc]), grid_type="hex", bin_size_um=bin_size_um
        )[0]
        assert np.linalg.norm(neighbor - origin) == pytest.approx(bin_size_um)


def test_grid_to_physical_coords_invalid_grid_type():
    with pytest.raises(ValueError, match="Unknown grid_type"):
        grid_to_physical_coords(np.array([0]), np.array([0]), grid_type="invalid")


# ---------------------------------------------------------------------------
# grid_neighbor_graph
# ---------------------------------------------------------------------------

def test_grid_neighbor_graph_empty_when_all_excluded():
    row = np.array([0, 1, 2])
    col = np.array([0, 1, 2])
    kept_idx, adjacency = grid_neighbor_graph(row, col, n_counts=np.zeros(3), grid_type="rect")
    assert len(kept_idx) == 0
    assert adjacency is None


def test_grid_neighbor_graph_reused_matches_default_split(rect_grid_rowcol):
    row, col = rect_grid_rowcol
    neighbor_graph = grid_neighbor_graph(row, col, grid_type="rect")
    labels_reused = split_into_connected_components(row, col, grid_type="rect", neighbor_graph=neighbor_graph)
    labels_default = split_into_connected_components(row, col, grid_type="rect")
    np.testing.assert_array_equal(labels_reused, labels_default)


def test_grid_neighbor_graph_reused_matches_default_border(rect_grid_rowcol):
    row, col = rect_grid_rowcol
    neighbor_graph = grid_neighbor_graph(row, col, grid_type="rect")
    border_reused = find_grid_border(row, col, grid_type="rect", neighbor_graph=neighbor_graph)
    border_default = find_grid_border(row, col, grid_type="rect")
    np.testing.assert_array_equal(border_reused, border_default)


def test_grid_neighbor_graph_reused_matches_default_hex(hex_grid_rowcol):
    row, col = hex_grid_rowcol
    neighbor_graph = grid_neighbor_graph(row, col, grid_type="hex")
    labels_reused = split_into_connected_components(row, col, grid_type="hex", neighbor_graph=neighbor_graph)
    labels_default = split_into_connected_components(row, col, grid_type="hex")
    np.testing.assert_array_equal(labels_reused, labels_default)
    border_reused = find_grid_border(row, col, grid_type="hex", neighbor_graph=neighbor_graph)
    border_default = find_grid_border(row, col, grid_type="hex")
    np.testing.assert_array_equal(border_reused, border_default)
