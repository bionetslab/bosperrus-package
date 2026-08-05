"""AnnData-native convenience wrappers around Flow for spatial-transcriptomics use.

Requires the optional `anndata` dependency (`pip install bosperrus[anndata]`). This
module never imports `anndata` at the top level -- only inside the two public
functions below, via `_require_anndata()` -- so the rest of `bosperrus` (and this
module's own presence in `bosperrus.__all__`) stays importable without it installed,
mirroring `distances.distance_to_alpha_shape`'s optional-dependency pattern.
"""
import numpy as np
import pandas as pd

from .distances import distance_to_pointset
from .fit import ConstantFit, PiecewiseLinearFit, ExponentialSaturationFit
from .graph_construction import grid_edges
from .pipeline import Flow

__all__ = ["identify_analysis_buffer", "correct_layer"]

_GRID_DEGREE = {"hex": 6, "rect": 4}


def _require_anndata():
    try:
        import anndata  # noqa: F401
    except ImportError:
        raise ImportError(
            "anndata is required for this function. "
            "Install it with: pip install anndata or pip install bosperrus[anndata]"
        )


def _resolve_distance_to_border(adata, row_key, col_key, grid_type, distance_key):
    if distance_key is not None:
        if distance_key not in adata.obs:
            raise KeyError(f"distance_key={distance_key!r} not found in adata.obs.")
        return adata.obs[distance_key].to_numpy()

    if grid_type not in _GRID_DEGREE:
        raise ValueError(f"Unknown grid_type: {grid_type!r}. Expected one of {list(_GRID_DEGREE)}.")

    row = adata.obs[row_key].to_numpy()
    col = adata.obs[col_key].to_numpy()
    edges = grid_edges(row, col, grid_type=grid_type)

    degree = np.zeros(len(row), dtype=int)
    for u, v in edges:
        degree[u] += 1
        degree[v] += 1

    is_border = degree < _GRID_DEGREE[grid_type]
    if not is_border.any():
        raise ValueError(
            "No border points detected (every node has a full complement of grid "
            "neighbors) -- check row_key/col_key/grid_type, or pass distance_key "
            "directly if you already have a distance-to-border column."
        )

    coords = np.column_stack([row, col])
    return distance_to_pointset(coords, coords[is_border]).to_numpy()


def identify_analysis_buffer(
    adata,
    score,
    row_key="array_row",
    col_key="array_col",
    grid_type="hex",
    distance_key=None,
    key_added="analysis_buffer",
    copy=False,
):
    """Flag spots within the piecewise-linear-fit elbow of the tissue border.

    Fits `PiecewiseLinearFit` (vs. the `ConstantFit` null) of `score` against
    distance to the nearest border point -- spots with fewer than the full
    complement of grid neighbors. Writes a boolean column to
    `adata.obs[key_added]`: True for spots closer to the border than the fitted
    breakpoint. If `ConstantFit` wins (no detected effect), every spot is flagged
    False -- there is nothing to call a buffer.

    Parameters
    ----------
    adata : AnnData
    score : str or array-like
        Either the name of an `.obs` column, or an array of values aligned with
        `adata.obs_names` to fit against distance-to-border. Single measure only --
        per-column elbows are typically too heterogeneous to aggregate into one
        buffer without an arbitrary threshold choice; fit each measure you care
        about separately.
    row_key, col_key : str, default "array_row", "array_col"
        `.obs` columns holding discrete grid indices (e.g. Visium's
        `array_row`/`array_col`). Not used if `distance_key` is given.
    grid_type : {"hex", "rect"}, default "hex"
        "hex" (6 neighbors, e.g. Visium) or "rect" (4 neighbors, e.g. Visium HD).
    distance_key : str, optional
        If given, must already exist in `adata.obs` -- reused directly, skipping
        grid construction and border-point detection entirely.
    key_added : str, default "analysis_buffer"
        `.obs` column name for the output boolean buffer flag. Fit diagnostics are
        stored in `adata.uns[f"{key_added}_fit"]`.
    copy : bool, default False
        If True, return a modified copy of `adata` instead of mutating in place.

    Returns
    -------
    AnnData or None
        The modified AnnData if `copy=True`, else None (`adata` is mutated in place).
    """
    _require_anndata()
    adata = adata.copy() if copy else adata

    distance = _resolve_distance_to_border(adata, row_key, col_key, grid_type, distance_key)

    if isinstance(score, str):
        score_values = adata.obs[score].to_numpy()
        score_name = score
    else:
        score_values = np.asarray(score)
        score_name = "score"

    flow = Flow.from_distances_and_scores(
        distances=pd.Series(distance, name="distance_to_border"),
        scores=pd.DataFrame({score_name: score_values}),
    )
    flow.flow(fits=[ConstantFit, PiecewiseLinearFit])

    best_fit = flow.best_fits[score_name]
    if isinstance(best_fit, PiecewiseLinearFit):
        buffer = distance < best_fit.params["piecewise_linear_b"]
    else:
        buffer = np.zeros(len(distance), dtype=bool)

    adata.obs[key_added] = buffer
    adata.uns[f"{key_added}_fit"] = {
        "best_fit_type": best_fit.name,
        "params": dict(best_fit.params),
        "observed_effect_strength": best_fit.observed_effect_strength,
        "observed_half_life": best_fit.observed_half_life,
        "affected_fraction": best_fit.fraction_not_converged,
        "grid_type": grid_type,
    }

    return adata if copy else None


def correct_layer(
    adata,
    layer=None,
    row_key="array_row",
    col_key="array_col",
    grid_type="hex",
    distance_key=None,
    key_added="bosperrus_corrected",
    copy=False,
):
    """Correct each feature toward its exponential-saturation asymptote.

    Fits `ExponentialSaturationFit` (vs. the `ConstantFit` null) of every feature
    (column) in `layer` against distance to the nearest border point,
    independently per feature. Writes the corrected matrix to
    `adata.layers[key_added]`: features where `ConstantFit` wins (no detected
    effect) pass through unchanged.

    Note: independently fitting many features (e.g. thousands of genes) means
    thousands of independent curve fits -- expect runtime to scale roughly
    linearly with feature count.

    Parameters
    ----------
    adata : AnnData
    layer : str, optional
        Name of the `.layers` entry to correct. If None, uses `adata.X`.
    row_key, col_key, grid_type, distance_key : see `identify_analysis_buffer`.
    key_added : str, default "bosperrus_corrected"
        `.layers` key for the corrected matrix. Per-feature fit diagnostics are
        stored as new `.var` columns prefixed with `key_added`.
    copy : bool, default False
        If True, return a modified copy of `adata` instead of mutating in place.

    Returns
    -------
    AnnData or None
        The modified AnnData if `copy=True`, else None (`adata` is mutated in place).
    """
    _require_anndata()
    from scipy import sparse

    adata = adata.copy() if copy else adata

    distance = _resolve_distance_to_border(adata, row_key, col_key, grid_type, distance_key)

    matrix = adata.X if layer is None else adata.layers[layer]
    if sparse.issparse(matrix):
        matrix = matrix.toarray()
    matrix = np.asarray(matrix)

    scores = pd.DataFrame(matrix, columns=adata.var_names)
    flow = Flow.from_distances_and_scores(
        distances=pd.Series(distance, name="distance_to_border"),
        scores=scores,
    )
    flow.flow(fits=[ConstantFit, ExponentialSaturationFit])

    corrected_cols = [f"BOSPERRUS corrected {g}" for g in adata.var_names]
    adata.layers[key_added] = flow.observations[corrected_cols].to_numpy()

    fit_quality = flow.fit_quality.T
    adata.var[f"{key_added}_best_fit_type"] = fit_quality["best_fit_type"].to_numpy()
    adata.var[f"{key_added}_effect_strength"] = fit_quality["observed_effect_strength"].to_numpy()
    adata.var[f"{key_added}_half_life"] = fit_quality["observed_half_life"].to_numpy()

    return adata if copy else None
