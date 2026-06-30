import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree, ConvexHull, distance


def distance_to_rectangular_border(coords):
    if coords.shape[1] != 2:
        raise ValueError("Spatial coordinates must be Nx2.")
    x = coords[:, 0]
    y = coords[:, 1]

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


def distance_to_pointset(coords, pointset):
    coords = np.asarray(coords, dtype=float)
    pointset = np.asarray(pointset, dtype=float)
    
    if coords.shape[1] != pointset.shape[1]:
        raise ValueError("Coords and pointset must have the same dimensionality.")
    if len(pointset) == 0:
        raise ValueError("Pointset must contain at least one point.")

    tree = cKDTree(pointset)
    d_min, _ = tree.query(coords, k=1)
    return pd.Series(d_min, name="distance_to_pointset")


def distance_to_mask(coords, mask):
    coords = np.asarray(coords, dtype=float)
    
    mask_arr = np.asarray(mask)
    if mask_arr.ndim != coords.shape[1]:
        raise ValueError("Mask dimensionality must match coordinate dimensionality.")

    inverted = ~mask_arr.astype(bool)
    dmap = distance_transform_edt(inverted)

    rounded = np.round(coords).astype(int)
    for dim in range(coords.shape[1]):
        rounded[:, dim] = np.clip(rounded[:, dim], 0, mask_arr.shape[dim] - 1)

    # multi-dimensional indexing
    d_vals = dmap[tuple(rounded[:, i] for i in range(rounded.shape[1]))]
    return pd.Series(d_vals, name="distance_to_mask")


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


def distance_to_convex_hull(coords):
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] not in (2, 3):
        raise ValueError("Spatial coordinates must be Nx2 or Nx3.")

    n_points = coords.shape[0]
    if n_points == 0:
        return pd.Series([], dtype=float, name="distance_to_convex_hull")

    if n_points <= coords.shape[1]:
        # Not enough points to define a full hull; close-distance to points
        if n_points == 1:
            d_vals = np.zeros(1)
        else:
            d_vals = distance.cdist(coords, coords, metric="euclidean").min(axis=1)
        return pd.Series(d_vals, name="distance_to_convex_hull")

    hull = ConvexHull(coords)
    d_min = np.full(n_points, np.inf, dtype=float)

    for simplex in hull.simplices:
        if coords.shape[1] == 2 and simplex.shape[0] == 2:
            a, b = coords[simplex[0]], coords[simplex[1]]
            d_seg = _point_to_segment_distance(coords, a, b)
            d_min = np.minimum(d_min, d_seg)
        elif coords.shape[1] == 3 and simplex.shape[0] == 3:
            a, b, c = coords[simplex[0]], coords[simplex[1]], coords[simplex[2]]
            d_tri = _point_to_triangle_distance(coords, a, b, c)
            d_min = np.minimum(d_min, d_tri)
        else:
            raise ValueError("Unexpected hull simplex shape.")

    return pd.Series(d_min, name="distance_to_convex_hull")
