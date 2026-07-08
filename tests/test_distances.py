import numpy as np
import pandas as pd
import pytest
from bosperrus.distances import (
    distance_to_rectangular_border,
    distance_to_convex_hull,
    distance_to_pointset,
    distance_to_mask,
    distance_to_alpha_shape,
)


# ---------------------------------------------------------------------------
# distance_to_rectangular_border
# ---------------------------------------------------------------------------

def test_distance_to_rectangular_border_center_is_maximum():
    """Center of a unit square is farther from the border than the corners."""
    # Unit square corners + center
    coords = np.array([
        [0.0, 0.0],  # corner
        [1.0, 0.0],  # corner
        [0.0, 1.0],  # corner
        [1.0, 1.0],  # corner
        [0.5, 0.5],  # center
    ])
    d = distance_to_rectangular_border(coords)
    center_dist = d.iloc[4]
    corner_dists = d.iloc[:4]
    assert (corner_dists == 0.0).all(), "Corners should have distance 0"
    assert center_dist > 0.0, "Center should have positive distance"
    assert (center_dist >= corner_dists).all()


def test_distance_to_rectangular_border_requires_2d():
    coords_3d = np.ones((5, 3))
    with pytest.raises(ValueError, match="Nx2"):
        distance_to_rectangular_border(coords_3d)


def test_distance_to_rectangular_border_returns_series():
    rng = np.random.default_rng(42)
    coords = rng.uniform(0, 1, (10, 2))
    result = distance_to_rectangular_border(coords)
    assert isinstance(result, pd.Series)
    assert len(result) == 10


# ---------------------------------------------------------------------------
# distance_to_convex_hull (2D)
# ---------------------------------------------------------------------------

def test_distance_to_convex_hull_2d_interior_vs_border():
    """Interior points of a square have larger hull distance than boundary points."""
    # Boundary points of a [0,2]x[0,2] square
    boundary = np.array([
        [0.0, 0.0], [1.0, 0.0], [2.0, 0.0],
        [0.0, 2.0], [1.0, 2.0], [2.0, 2.0],
        [0.0, 1.0], [2.0, 1.0],
    ])
    # Interior points
    interior = np.array([
        [1.0, 1.0],  # centre
        [0.5, 0.5],
        [1.5, 1.5],
    ])
    all_coords = np.vstack([boundary, interior])
    d = distance_to_convex_hull(all_coords)

    d_boundary = d.iloc[: len(boundary)]
    d_interior = d.iloc[len(boundary):]

    assert d_interior.min() > d_boundary.min(), (
        "Interior points should have larger distance to convex hull than boundary points"
    )


def test_distance_to_convex_hull_2d_returns_series():
    rng = np.random.default_rng(42)
    coords = rng.uniform(0, 1, (15, 2))
    result = distance_to_convex_hull(coords)
    assert isinstance(result, pd.Series)
    assert len(result) == 15


def test_distance_to_convex_hull_3d():
    """distance_to_convex_hull works for 3D coords."""
    rng = np.random.default_rng(42)
    # Points on/near the surface of a unit sphere
    angles = rng.uniform(0, np.pi, (20, 2))
    coords = np.column_stack([
        np.sin(angles[:, 0]) * np.cos(angles[:, 1]),
        np.sin(angles[:, 0]) * np.sin(angles[:, 1]),
        np.cos(angles[:, 0]),
    ])
    result = distance_to_convex_hull(coords)
    assert isinstance(result, pd.Series)
    assert len(result) == len(coords)
    assert (result >= 0).all()


# ---------------------------------------------------------------------------
# distance_to_pointset
# ---------------------------------------------------------------------------

def test_distance_to_pointset_basic():
    """A query point that coincides with a pointset member has distance 0."""
    pointset = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    query = np.array([[1.0, 0.0]])  # exactly on a pointset member
    result = distance_to_pointset(query, pointset)
    assert result.iloc[0] == pytest.approx(0.0)


def test_distance_to_pointset_returns_series():
    rng = np.random.default_rng(42)
    coords = rng.uniform(0, 1, (8, 2))
    pointset = rng.uniform(0, 1, (5, 2))
    result = distance_to_pointset(coords, pointset)
    assert isinstance(result, pd.Series)
    assert len(result) == 8


# ---------------------------------------------------------------------------
# distance_to_mask
# ---------------------------------------------------------------------------

def test_distance_to_mask_2d():
    """Coords far from the True (foreground) region have larger distance than nearby ones.

    distance_to_mask uses distance_transform_edt on the *inverted* mask, which
    assigns each pixel its distance to the nearest True-foreground pixel.
    - Pixels inside the True block -> distance 0.
    - Pixels just outside the block -> small positive distance.
    - Pixels far from the block -> large positive distance.
    """
    # 20x20 boolean mask: True (foreground) in a 10x10 centre block [5:15, 5:15]
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:15, 5:15] = True

    # Inside the block -> distance 0
    inside_coords = np.array([
        [7, 7], [10, 10], [12, 12],
    ], dtype=float)

    # Just outside the block (1 pixel away from border)
    near_outside_coords = np.array([
        [4, 10], [15, 10], [10, 4], [10, 15],
    ], dtype=float)

    # Far from the block (corner of the 20x20 grid)
    far_outside_coords = np.array([
        [0, 0], [0, 19], [19, 0], [19, 19],
    ], dtype=float)

    all_coords = np.vstack([inside_coords, near_outside_coords, far_outside_coords])
    result = distance_to_mask(all_coords, mask)

    assert isinstance(result, pd.Series)
    assert len(result) == len(all_coords)

    n_inside = len(inside_coords)
    n_near   = len(near_outside_coords)

    d_inside      = result.iloc[:n_inside]
    d_near_outside = result.iloc[n_inside: n_inside + n_near]
    d_far_outside  = result.iloc[n_inside + n_near:]

    # Inside foreground pixels have distance 0
    assert (d_inside == 0.0).all(), "Pixels inside the True block should have distance 0"

    # Farther background pixels have larger distances than near background pixels
    assert d_far_outside.min() > d_near_outside.max(), (
        "Pixels far from the foreground block should have larger distance than nearby ones"
    )


# ---------------------------------------------------------------------------
# distance_to_alpha_shape
# ---------------------------------------------------------------------------

alphashape = pytest.importorskip("alphashape", reason="alphashape not installed")


def test_distance_to_alpha_shape_returns_series():
    rng = np.random.default_rng(42)
    coords = rng.uniform(0, 100, (60, 2))
    result = distance_to_alpha_shape(coords, alpha=0)
    assert isinstance(result, pd.Series)
    assert len(result) == 60
    assert result.name == "distance_to_alpha_shape"


def test_distance_to_alpha_shape_nonnegative():
    rng = np.random.default_rng(42)
    coords = rng.uniform(0, 100, (60, 2))
    result = distance_to_alpha_shape(coords, alpha=0)
    assert (result >= 0).all()


def test_distance_to_alpha_shape_interior_farther_than_boundary():
    """Dense ring of points: the centre has larger alpha-shape distance than the ring."""
    angles = np.linspace(0, 2 * np.pi, 80, endpoint=False)
    ring = np.column_stack([10 * np.cos(angles), 10 * np.sin(angles)])
    centre = np.array([[0.0, 0.0]])
    coords = np.vstack([ring, centre])
    d = distance_to_alpha_shape(coords, alpha=0)
    assert d.iloc[-1] > d.iloc[:-1].max() * 0.5, (
        "Centre point should have clearly larger distance than the ring boundary"
    )


def test_distance_to_alpha_shape_concave_boundary():
    """Alpha shape with sufficient alpha follows concavities; distances near the
    concave notch should be small, not inflated as the convex hull would give."""
    # U-shape: left column, right column, connecting base — opens upward
    left   = np.column_stack([np.zeros(20),        np.linspace(0, 10, 20)])
    right  = np.column_stack([np.full(20, 10.0),   np.linspace(0, 10, 20)])
    bottom = np.column_stack([np.linspace(0, 10, 15), np.zeros(15)])
    # Point inside the U gap (near the open top, between the two arms)
    gap_pt = np.array([[5.0, 8.0]])
    coords = np.vstack([left, right, bottom, gap_pt])

    # With alpha=0 (convex hull), the gap point is interior → large distance
    d_convex = distance_to_alpha_shape(coords, alpha=0)
    # With a concave alpha, the boundary wraps around the gap → smaller distance
    d_concave = distance_to_alpha_shape(coords, alpha=0.3)

    gap_idx = len(coords) - 1
    assert d_concave.iloc[gap_idx] < d_convex.iloc[gap_idx], (
        "Alpha shape should give smaller distance for a point in a concave gap "
        "than the convex hull (alpha=0) does"
    )


def test_distance_to_alpha_shape_requires_2d():
    coords = np.ones((10, 3))
    with pytest.raises(ValueError, match="Nx2"):
        distance_to_alpha_shape(coords, alpha=0)


def test_distance_to_alpha_shape_requires_at_least_3_points():
    coords = np.array([[0.0, 0.0], [1.0, 0.0]])
    with pytest.raises(ValueError, match="3 points"):
        distance_to_alpha_shape(coords, alpha=0)


def test_distance_to_alpha_shape_multipolygon_warns():
    """Alpha that splits the shape into fragments should raise a UserWarning."""
    # Two well-separated clusters: a large enough alpha will produce one hull,
    # but an intermediate alpha that keeps only within-cluster edges produces
    # two disconnected polygons.
    cluster_a = np.random.default_rng(0).uniform(0, 1, (30, 2))
    cluster_b = np.random.default_rng(1).uniform(10, 11, (30, 2))
    coords = np.vstack([cluster_a, cluster_b])
    # alpha small enough to keep intra-cluster triangles but too small to
    # bridge the gap between clusters → MultiPolygon
    with pytest.warns(UserWarning, match="MultiPolygon"):
        distance_to_alpha_shape(coords, alpha=2.0)


def test_distance_to_alpha_shape_empty_raises():
    """Alpha large enough to exclude all Delaunay triangles → empty shape."""
    # Five points spread over a 1000x1000 grid produce large circumradii;
    # alpha=0.01 (threshold = 100 units) excludes all triangles → empty.
    rng = np.random.default_rng(42)
    coords = rng.uniform(0, 1000, (5, 2))
    with pytest.raises(ValueError, match="empty"):
        distance_to_alpha_shape(coords, alpha=0.01)
