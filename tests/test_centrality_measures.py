import numpy as np
import pytest

from bosperrus.centrality_measures import HAS_GRAPH_TOOL, compute_centrality_measures
from bosperrus.graph_construction import knn_edges

BACKENDS = ["networkx"] + (["graph_tool"] if HAS_GRAPH_TOOL else [])
MEASURES = ["degree", "clustering", "closeness"]


@pytest.mark.parametrize("backend", BACKENDS)
def test_mutual_directed_edges_counted_once(backend):
    """A mutual kNN pair, (0, 1) and (1, 0), is one undirected edge."""
    directed = {(0, 1), (1, 0), (1, 2), (2, 0)}
    undirected = {frozenset(e) for e in [(0, 1), (1, 2), (2, 0)]}
    a = compute_centrality_measures(directed, 3, MEASURES, backend=backend)
    b = compute_centrality_measures(undirected, 3, MEASURES, backend=backend)
    np.testing.assert_allclose(a.values, b.values)
    np.testing.assert_allclose(a["degree"], [2, 2, 2])
    np.testing.assert_allclose(a["clustering"], [1, 1, 1])


@pytest.mark.skipif(not HAS_GRAPH_TOOL, reason="graph-tool not installed")
def test_backends_agree_on_knn_graph():
    """On a connected kNN graph, graph-tool and networkx give the same degree,
    clustering and closeness."""
    coords = np.random.default_rng(42).uniform(size=(200, 2))
    edges = knn_edges(coords, 6)
    gt = compute_centrality_measures(edges, 200, MEASURES, backend="graph_tool")
    nx = compute_centrality_measures(edges, 200, MEASURES, backend="networkx")
    assert (gt["clustering"] <= 1).all()
    np.testing.assert_allclose(gt["degree"], nx["degree"])
    np.testing.assert_allclose(gt["clustering"], nx["clustering"])
    np.testing.assert_allclose(gt["closeness"], nx["closeness"])
