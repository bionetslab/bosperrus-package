import numpy as np
import pytest
from bosperrus.graph_construction import (
    knn_edges,
    rnn_edges,
    delaunay_edges,
    construct_graph,
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
