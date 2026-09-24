"""Generic plotting helpers for score-vs-distance fits.

Unlike anndata_api.py/image_masks.py, this module has no optional-dependency
gate -- matplotlib is a core bosperrus dependency (see pyproject.toml), not
an extra, since it's common/light enough that gating it behind an extra
would just be friction for the common case.
"""
import numpy as np

__all__ = ["plot_fit"]


def plot_fit(ax, distance, score, predict_fn, d_grid=None, n_grid=200, bins=40,
             hist_kwargs=None, line_kwargs=None):
    """2D histogram of `score` vs. `distance`, with a fitted model curve
    overlaid on top.

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
    bins : int or (int, int), default 40
        Passed to `ax.hist2d`.
    hist_kwargs : dict, optional
        Extra keyword arguments forwarded to `ax.hist2d` (e.g. `cmap`).
    line_kwargs : dict, optional
        Extra keyword arguments forwarded to `ax.plot` for the fitted curve
        (e.g. `color`, `linewidth`).

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

    hist_kwargs = dict(hist_kwargs or {})
    hist_kwargs.setdefault("cmap", "Greys")
    ax.hist2d(distance, score, bins=bins, **hist_kwargs)

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
