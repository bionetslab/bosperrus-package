"""Generic plotting helpers for score-vs-distance fits.

Unlike anndata_api.py/image_masks.py, this module has no optional-dependency
gate -- matplotlib is a core bosperrus dependency (see pyproject.toml), not
an extra, since it's common/light enough that gating it behind an extra
would just be friction for the common case.
"""
import numpy as np
import pandas as pd

from .fit import ConstantFit, PiecewiseLinearFit, ExponentialSaturationFit, ExponentialDecayFit, MichaelisMentenFit

__all__ = ["plot_fit", "FIT_PALETTE"]

# Name -> color, one entry per built-in Fit subclass, for code that only has
# the serialized best_fit_type string (e.g. anndata_api.py's
# plot_border_effect/plot_diffusion, reconstructing a predictor from
# adata.uns-stored params rather than a live Fit instance -- see
# _predict_from_params). Each subclass's `color` class attribute (e.g.
# `ExponentialSaturationFit.color`) is the actual source of truth; this dict
# is just a name-keyed view over those, built here so it can never drift out
# of sync with them. Code holding a live Fit instance should just read
# `fit.color` directly instead of going through this dict.
FIT_PALETTE = {
    name: cls.color
    for cls, name in [
        (ConstantFit, "Constant Fit"),
        (PiecewiseLinearFit, "Piecewise Linear Fit"),
        (ExponentialSaturationFit, "Exponential Saturation Fit"),
        (ExponentialDecayFit, "Exponential Decay Fit"),
        (MichaelisMentenFit, "Michaelis-Menten Fit"),
    ]
}


def plot_fit(ax, distance, score, predict_fn, d_grid=None, n_grid=200, bins=100,
             quantiles=((0.05, 0.95), (0.25, 0.75)), band_colors=("0.88", "0.7"),
             mean_kwargs=None, line_kwargs=None):
    """`score` vs. `distance` summarised per distance bin -- quantile bands
    (a "fan") and the mean -- with a fitted model curve overlaid on top.

    The summary line is the mean, not the median: every bosperrus fit is a
    least-squares fit, i.e. it models the mean score at each distance, so
    with skewed scores (e.g. counts) a fit would look off against the median.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw into.
    distance : array-like
        Distance values (x-axis).
    score : array-like
        Observed score values (y-axis), aligned with `distance`.
    predict_fn : callable
        `d -> predicted score`, e.g. a fitted `Fit` instance's bound
        `.predict` method (`fit.predict`) -- any callable works, so this
        isn't tied to a `Fit` instance specifically (e.g. `anndata_api`'s
        `plot_border_effect`/`plot_diffusion` reconstruct a predictor from
        stored params instead of keeping a live `Fit` object around, since
        `Fit` instances aren't AnnData/h5ad-serializable).
    d_grid : array-like, optional
        Distance values to evaluate/draw the fitted curve at. Defaults to
        `n_grid` points spanning `[min(distance), max(distance)]`.
    n_grid : int, default 200
        Number of points in the default `d_grid`, if not given explicitly.
    bins : int or array-like, default 100
        Distance bins: a number of equal-width bins over the data range, or
        the bin edges. Points outside the edges are not summarised; empty
        bins are skipped.
    quantiles : sequence of (low, high), default ((0.05, 0.95), (0.25, 0.75))
        Quantile bands per distance bin, drawn in this order (outer first).
    band_colors : sequence of colors, default ("0.88", "0.7")
        One fill color per band in `quantiles`.
    mean_kwargs : dict, optional
        Extra keyword arguments forwarded to `ax.plot` for the per-bin mean
        (defaults: black, linewidth 1, label "mean").
    line_kwargs : dict, optional
        Extra keyword arguments forwarded to `ax.plot` for the fitted curve
        (e.g. `color`, `linewidth`). Drawn last, on top.

    Returns
    -------
    matplotlib.axes.Axes
        The same `ax`, for chaining.
    """
    distance = np.asarray(distance, dtype=float)
    score = np.asarray(score, dtype=float)
    if len(distance) != len(score):
        raise ValueError(f"distance and score must have the same length, got {len(distance)} and {len(score)}.")
    valid = np.isfinite(distance) & np.isfinite(score)
    distance, score = distance[valid], score[valid]
    if len(distance) == 0:
        raise ValueError("No finite (distance, score) pairs to plot.")
    if len(band_colors) != len(quantiles):
        raise ValueError(f"need one band color per quantile band, got {len(band_colors)} for {len(quantiles)}.")

    if isinstance(bins, (int, np.integer)):
        edges = np.linspace(distance.min(), distance.max(), int(bins) + 1)
        if edges[0] == edges[-1]:  # all distances equal: one bin around them
            edges = np.array([edges[0] - 0.5, edges[0] + 0.5])
    else:
        try:
            edges = np.asarray(bins, dtype=float)
        except ValueError:  # ragged, e.g. old 2-D hist2d bins [x_edges, y_edges]
            edges = None
        if edges is None or edges.ndim != 1 or len(edges) < 2:
            raise ValueError("bins must be a number of bins or a 1-D array of at least 2 bin edges.")
    inside = (distance >= edges[0]) & (distance <= edges[-1])
    idx = np.clip(np.searchsorted(edges, distance[inside], side="right") - 1, 0, len(edges) - 2)
    groups = pd.Series(score[inside]).groupby(idx)
    centres = ((edges[:-1] + edges[1:]) / 2)
    if len(idx):
        x = centres[np.asarray(sorted(set(idx)))]
        q = groups.quantile(sorted({v for band in quantiles for v in band})).unstack()
        for (lo, hi), color in zip(quantiles, band_colors):
            ax.fill_between(x, q[lo].to_numpy(), q[hi].to_numpy(), color=color, linewidth=0,
                            label=f"{lo:.0%}-{hi:.0%}")
        mean_kwargs = dict(mean_kwargs or {})
        mean_kwargs.setdefault("color", "black")
        mean_kwargs.setdefault("linewidth", 1)
        mean_kwargs.setdefault("label", "mean")
        ax.plot(x, groups.mean().to_numpy(), **mean_kwargs)

    if d_grid is None:
        d_grid = np.linspace(distance.min(), distance.max(), n_grid)
    else:
        d_grid = np.asarray(d_grid, dtype=float)
    predicted = predict_fn(d_grid)

    line_kwargs = dict(line_kwargs or {})
    line_kwargs.setdefault("color", "C1")
    line_kwargs.setdefault("linewidth", 2)
    ax.plot(d_grid, predicted, **line_kwargs)

    ax.set_xlabel("distance")
    ax.set_ylabel("score")
    return ax
