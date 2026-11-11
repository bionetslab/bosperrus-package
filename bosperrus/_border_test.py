"""Interior-window test of a fitted border effect (used by `anndata_api.interior_window_test`).

A component's border profile (the rising piecewise-linear elbow of `fit_border_profile`) is compared with the
profiles of square windows placed well inside the same component, where there is no physical border: any elbow
there comes from tissue structure.

Test statistic: the steepness of the dip, -effect strength / max(half-life, `min_half_life_um`) (score units per um;
the effect strength is negative for a dip at the border, the half-life is the distance from the border to the
midpoint of the rise). It is large only for a dip that is both deep and narrow; a broad tissue gradient (deep but
wide) and a small wiggle (narrow but shallow) both score low. Profiles without a rising elbow have steepness 0.
The p-value is the profile's rank among the windows:
    p = (1 + #windows at least as steep) / (1 + #windows),
so the smallest reachable p is 1 / (1 + #windows).

Windows are drawn so that no two overlap by more than a given fraction of their area (`spread_corners`), which keeps
them from being near-copies of one another.
"""
import numpy as np


def window_corners(eligible, size):
    """(row, col) offsets of every size x size square lying entirely on True cells of the 2-D array `eligible`."""
    if size > min(eligible.shape):
        return np.empty((0, 2), dtype=np.int64)
    s = np.zeros((eligible.shape[0] + 1, eligible.shape[1] + 1), dtype=np.int64)
    s[1:, 1:] = eligible.astype(np.int64).cumsum(0).cumsum(1)
    full = s[size:, size:] - s[:-size, size:] - s[size:, :-size] + s[:-size, :-size]
    return np.argwhere(full == size * size)


def overlap_fraction(a, b, size):
    """Shared area fraction of size x size squares with corners a (2,) and b (n, 2)."""
    d = np.abs(np.asarray(b) - np.asarray(a))
    return np.prod(np.clip(1 - d / size, 0, None), axis=-1)


def spread_corners(corners, size, n, max_overlap, rng, max_candidates=20000):
    """Up to n corners in random order, each overlapping every earlier one by at most `max_overlap` of the area.

    More than `max_candidates` candidates are first thinned to a lattice of step ~size/10 (random offset) to bound
    the work on large components; that step is far below the spacing the overlap cap enforces for any
    max_overlap < 0.8."""
    corners = np.asarray(corners)
    if len(corners) == 0:
        return corners
    step = max(1, size // 10) if len(corners) > max_candidates else 1
    offset = rng.integers(step, size=2)
    corners = corners[((corners - offset) % step == 0).all(axis=1)] if step > 1 else corners
    chosen = np.empty((0, 2), dtype=corners.dtype)
    for c in corners[rng.permutation(len(corners))]:
        if len(chosen) == 0 or overlap_fraction(c, chosen, size).max() <= max_overlap:
            chosen = np.vstack([chosen, c])
            if len(chosen) == n:
                break
    return chosen


def steepness(info, min_half_life_um=0.0):
    """-effect strength / max(half-life, min_half_life_um) (um^-1 times score units) of a fit record; 0 without a
    rising elbow. The floor keeps knots at the first bin (half-life = half a bin) from dominating: below the
    precision of the border itself a narrower dip is not a steeper one."""
    if not info["border_effect"]:
        return 0.0
    eff, half = float(info["observed_effect_strength"]), float(info["half_life_um"])
    if not np.isfinite(eff) or not np.isfinite(half):
        return 0.0
    half = max(half, float(min_half_life_um))
    return -eff / half if half > 0 else (np.inf if eff < 0 else 0.0)


def window_p_value(observed, null):
    """Rank p-value of an observed steepness against the windows' steepness (NaN without windows)."""
    null = np.asarray(null, dtype=float)
    if len(null) == 0:
        return np.nan
    return (1 + int((null >= observed).sum())) / (1 + len(null))
