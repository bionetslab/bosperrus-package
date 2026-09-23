import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree, ConvexHull, distance

from .graph_construction import split_into_connected_components, find_grid_border

__all__ = ['distance_to_rectangular_border', 'distance_to_pointset', 'distance_to_mask',
           'distance_to_convex_hull', 'distance_to_alpha_shape', 'distance_to_grid_border']


def distance_to_rectangular_border(coordinates):
    if coordinates.shape[1] != 2:
        raise ValueError("Spatial coordinates must be Nx2.")
    x = coordinates[:, 0]
    y = coordinates[:, 1]

    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()

    # distances to each of the four borders
    d_left   = x - xmin
    d_right  = xmax - x
    d_bottom = y - ymin
    d_top    = ymax - y

    # distance to the rectangle boundary = smallest distance to any border
    d_border = np.vstack([d_left, d_right, d_bottom, d_top]).min(axis=0)
    return pd.Series(d_border, name="distance_to_rectangular_border")


def distance_to_pointset(coordinates, pointset):
    coordinates = np.asarray(coordinates, dtype=float)
    pointset = np.asarray(pointset, dtype=float)

    if coordinates.shape[1] != pointset.shape[1]:
        raise ValueError("Coordinates and pointset must have the same dimensionality.")
    if len(pointset) == 0:
        raise ValueError("Pointset must contain at least one point.")

    tree = cKDTree(pointset)
    d_min, _ = tree.query(coordinates, k=1)
    return pd.Series(d_min, name="distance_to_pointset")


def distance_to_grid_border(row, col, bin_size_um, n_counts=None, grid_type="rect", component_labels=None):
    """Per-node physical distance (um) to the nearest border node (see
    `find_grid_border`), computed within each spatially-connected grid
    component separately (see `split_into_connected_components`) so a node
    is never "nearest" to a border point belonging to a different,
    physically disconnected fragment that just happens to sit close by in
    raw grid coordinates.

    Parameters
    ----------
    row, col : array-like of int
        Grid indices, one pair per node, aligned by position.
    bin_size_um : float
        Physical size (um) of one grid step. Assumes an isotropic grid
        (row-step and col-step both span `bin_size_um`) -- exact for
        `grid_type="rect"`, but only an approximation for `"hex"`, whose two
        axes are not an isotropic scaling of a single physical pitch.
    n_counts : array-like, optional
        Per-node count/signal. Nodes with `n_counts <= 0` are excluded (see
        `split_into_connected_components`/`find_grid_border`) and get NaN.
    grid_type : {"hex", "rect"}, default "rect"
        See `grid_edges`.
    component_labels : array-like of int, optional
        Precomputed labels (e.g. reused from a prior `split_into_connected_
        components` call, such as when the same labels are also needed
        elsewhere, like cross-sample component matching). Computed
        internally from `row`/`col`/`n_counts`/`grid_type` if not given.

    Returns
    -------
    pd.Series, name "distance_to_grid_border"
        NaN for excluded nodes (n_counts <= 0, or in a dropped component).
    """
    row = np.asarray(row)
    col = np.asarray(col)
    if len(row) != len(col):
        raise ValueError("row and col must have the same length.")

    if component_labels is None:
        component_labels = split_into_connected_components(row, col, n_counts=n_counts, grid_type=grid_type)
    else:
        component_labels = np.asarray(component_labels)
    is_border = find_grid_border(row, col, n_counts=n_counts, grid_type=grid_type)

    coords = np.column_stack([row, col])
    distance = np.full(len(row), np.nan)
    for label in np.unique(component_labels):
        if label < 0:
            continue
        member_mask = component_labels == label
        border_mask = member_mask & is_border
        if not border_mask.any():
            continue
        distance[member_mask] = distance_to_pointset(coords[member_mask], coords[border_mask]).to_numpy()

    return pd.Series(distance * bin_size_um, name="distance_to_grid_border")


def distance_to_mask(coordinates, mask, pixel_size_um=1.0):
    """`coordinates` must already be in the mask's own pixel-index space
    (i.e. caller-side responsibility to convert e.g. a spot's native pixel
    coordinate into the mask's coordinate system first). `pixel_size_um`
    converts the resulting pixel-space distance into physical units in one
    step -- physical size (in your chosen unit, e.g. um) of one pixel in
    that same coordinate system. Default 1.0 preserves the raw-pixel-distance
    behavior from before this parameter existed."""
    coordinates = np.asarray(coordinates, dtype=float)

    mask_arr = np.asarray(mask)
    if mask_arr.ndim != coordinates.shape[1]:
        raise ValueError("Mask dimensionality must match coordinate dimensionality.")

    inverted = ~mask_arr.astype(bool)
    dmap = distance_transform_edt(inverted)

    rounded = np.round(coordinates).astype(int)
    for dim in range(coordinates.shape[1]):
        rounded[:, dim] = np.clip(rounded[:, dim], 0, mask_arr.shape[dim] - 1)

    # multi-dimensional indexing
    d_vals = dmap[tuple(rounded[:, i] for i in range(rounded.shape[1]))]
    return pd.Series(d_vals * pixel_size_um, name="distance_to_mask")


def _point_to_segment_distance(points, a, b):
    ab = b - a
    ab_len2 = np.dot(ab, ab)
    p_vec = points - a
    if ab_len2 == 0:
        return np.linalg.norm(p_vec, axis=1)
    t = np.dot(p_vec, ab) / ab_len2
    t = np.clip(t, 0.0, 1.0)
    proj = a + np.outer(t, ab)
    return np.linalg.norm(points - proj, axis=1)


def _point_to_triangle_distance(points, a, b, c):
    ab = b - a
    ac = c - a
    ap = points - a

    d00 = np.dot(ab, ab)
    d01 = np.dot(ab, ac)
    d11 = np.dot(ac, ac)

    d20 = np.dot(ap, ab)
    d21 = np.dot(ap, ac)

    denom = d00 * d11 - d01 * d01
    if denom == 0:
        d_ab = _point_to_segment_distance(points, a, b)
        d_ac = _point_to_segment_distance(points, a, c)
        d_bc = _point_to_segment_distance(points, b, c)
        return np.minimum(np.minimum(d_ab, d_ac), d_bc)

    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom

    inside = (v >= 0) & (w >= 0) & (v + w <= 1)

    n = np.cross(ab, ac)
    n_len = np.linalg.norm(n)
    if n_len == 0:
        d_ab = _point_to_segment_distance(points, a, b)
        d_ac = _point_to_segment_distance(points, a, c)
        d_bc = _point_to_segment_distance(points, b, c)
        return np.minimum(np.minimum(d_ab, d_ac), d_bc)
    n_unit = n / n_len

    dist = np.full(points.shape[0], np.inf, dtype=float)
    dist[inside] = np.abs(np.dot(ap[inside], n_unit))

    outside = ~inside
    if np.any(outside):
        d_ab = _point_to_segment_distance(points[outside], a, b)
        d_bc = _point_to_segment_distance(points[outside], b, c)
        d_ca = _point_to_segment_distance(points[outside], c, a)
        dist[outside] = np.minimum(np.minimum(d_ab, d_bc), d_ca)

    return dist


def distance_to_convex_hull(coordinates):
    coordinates = np.asarray(coordinates, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] not in (2, 3):
        raise ValueError("Spatial coordinates must be Nx2 or Nx3.")

    n_points = coordinates.shape[0]
    if n_points == 0:
        return pd.Series([], dtype=float, name="distance_to_convex_hull")

    if n_points <= coordinates.shape[1]:
        # Not enough points to define a full hull; close-distance to points
        if n_points == 1:
            d_vals = np.zeros(1)
        else:
            d_vals = distance.cdist(coordinates, coordinates, metric="euclidean").min(axis=1)
        return pd.Series(d_vals, name="distance_to_convex_hull")

    hull = ConvexHull(coordinates)
    d_min = np.full(n_points, np.inf, dtype=float)

    for simplex in hull.simplices:
        if coordinates.shape[1] == 2 and simplex.shape[0] == 2:
            a, b = coordinates[simplex[0]], coordinates[simplex[1]]
            d_seg = _point_to_segment_distance(coordinates, a, b)
            d_min = np.minimum(d_min, d_seg)
        elif coordinates.shape[1] == 3 and simplex.shape[0] == 3:
            a, b, c = coordinates[simplex[0]], coordinates[simplex[1]], coordinates[simplex[2]]
            d_tri = _point_to_triangle_distance(coordinates, a, b, c)
            d_min = np.minimum(d_min, d_tri)
        else:
            raise ValueError("Unexpected hull simplex shape.")

    return pd.Series(d_min, name="distance_to_convex_hull")


def distance_to_alpha_shape(coordinates, alpha):
    """Compute distance from each point to the boundary of the alpha shape
    (concave hull) of the point cloud.

    The alpha shape generalises the convex hull: larger ``alpha`` values
    produce more concave (tighter) boundaries that follow nooks and crannies
    in the data; ``alpha=0`` recovers the convex hull. Too large an ``alpha``
    may split the shape into disconnected pieces or produce an empty geometry.

    .. warning::
        ``alpha`` must be chosen carefully and validated visually before use.
        The right value is data- and scale-dependent. Recommended workflow::

            import alphashape, geopandas as gpd, matplotlib.pyplot as plt
            shape = alphashape.alphashape(coordinates, alpha=YOUR_ALPHA)
            gpd.GeoSeries([shape]).boundary.plot()
            plt.scatter(coordinates[:, 0], coordinates[:, 1], s=1)
            plt.show()

        Increase ``alpha`` until the boundary traces the ROI nooks without
        punching holes through the interior or splitting into fragments.
        A :class:`UserWarning` is raised automatically if the shape becomes
        disconnected (``MultiPolygon``), which is a reliable sign that
        ``alpha`` is too large.

    Only 2-D coordinates are supported.

    Parameters
    ----------
    coordinates : array-like, shape (N, 2)
        2-D node coordinates.
    alpha : float
        Concaveness parameter. ``alpha=0`` gives the convex hull limit.
        Larger values follow concavities more tightly; too large produces
        holes, disconnected fragments, or an empty shape.

    Returns
    -------
    pd.Series
        Named ``"distance_to_alpha_shape"``. Distance of each point to the
        nearest point on the alpha shape boundary, in the same units as
        ``coordinates``.

    Raises
    ------
    ImportError
        If ``alphashape`` is not installed.
        Install with ``pip install alphashape`` or ``pip install bosperrus[alphashape]``.
    ValueError
        If ``coordinates`` is not Nx2, has fewer than 3 rows, or the resulting
        alpha shape is empty (try a smaller ``alpha``).

    Warns
    -----
    UserWarning
        If the alpha shape is a ``MultiPolygon`` (disconnected fragments),
        indicating ``alpha`` is likely too large.
    """
    try:
        import alphashape
    except ImportError:
        raise ImportError(
            "alphashape is required for distance_to_alpha_shape. "
            "Install it with: pip install alphashape or pip install bosperrus[alphashape]"
        )
    import warnings
    import shapely
    from shapely.geometry import MultiPolygon

    coordinates = np.asarray(coordinates, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError(
            "Spatial coordinates must be Nx2 for distance_to_alpha_shape."
        )
    if len(coordinates) < 3:
        raise ValueError(
            "At least 3 points are required to compute an alpha shape."
        )

    shape = alphashape.alphashape(coordinates, alpha)

    if shape is None or shape.is_empty:
        raise ValueError(
            f"Alpha shape is empty for alpha={alpha}. Try a smaller alpha value."
        )

    if isinstance(shape, MultiPolygon):
        warnings.warn(
            f"Alpha shape with alpha={alpha} produced a MultiPolygon "
            f"({len(list(shape.geoms))} disconnected fragments). "
            "This usually means alpha is too large: the boundary has been split "
            "into pieces, and distances will be measured to the nearest fragment "
            "rather than a single enclosing boundary. Consider reducing alpha "
            "and inspecting the result visually.",
            UserWarning,
            stacklevel=2,
        )

    boundary = shape.boundary
    point_geoms = shapely.points(coordinates)
    d_vals = shapely.distance(point_geoms, boundary)

    return pd.Series(d_vals, name="distance_to_alpha_shape")
