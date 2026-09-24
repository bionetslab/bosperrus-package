"""AnnData-native convenience wrappers around Flow for spatial-transcriptomics use.

Requires the optional `anndata` dependency (`pip install bosperrus[anndata]`). This
module never imports `anndata` at the top level -- only inside the two public
functions below, via `_require_anndata()` -- so the rest of `bosperrus` (and this
module's own presence in `bosperrus.__all__`) stays importable without it installed,
mirroring `distances.distance_to_alpha_shape`'s optional-dependency pattern.
"""
import numpy as np
import pandas as pd

from .distances import distance_to_grid_border
from .fit import ConstantFit, PiecewiseLinearFit, ExponentialSaturationFit
from .graph_construction import split_into_connected_components, find_grid_border
from .pipeline import Flow

__all__ = ["identify_analysis_buffer", "correct_layer"]


def _require_anndata():
    try:
        import anndata  # noqa: F401
    except ImportError:
        raise ImportError(
            "anndata is required for this function. "
            "Install it with: pip install anndata or pip install bosperrus[anndata]"
        )


def _components_and_distance(adata, row_key, col_key, grid_type, n_counts_key,
                              bin_size_um, min_component_size, distance_key):
    """Shared core of identify_analysis_buffer/correct_layer: split into
    spatially-connected grid components (see split_into_connected_components),
    flag border nodes (see find_grid_border), and get each node's distance to
    its own component's border (see distance_to_grid_border) -- both callers
    then fit independently per component, never pooled, since components are
    e.g. a TMA's individual cores, physically disconnected pieces of tissue.

    is_border is masked to False wherever components < 0 (n_counts <= 0, or
    a component dropped by min_component_size), matching the same
    exclusion semantics as the buffer/correction outputs: nothing meaningful
    is reported for spots outside the actual analysis.
    """
    row = adata.obs[row_key].to_numpy()
    col = adata.obs[col_key].to_numpy()
    n_counts = adata.obs[n_counts_key].to_numpy() if n_counts_key is not None else None

    components = split_into_connected_components(
        row, col, n_counts=n_counts, grid_type=grid_type, min_size=min_component_size,
    )
    is_border = find_grid_border(row, col, n_counts=n_counts, grid_type=grid_type) & (components >= 0)

    if distance_key is not None:
        if distance_key not in adata.obs:
            raise KeyError(f"distance_key={distance_key!r} not found in adata.obs.")
        distance = adata.obs[distance_key].to_numpy()
    else:
        distance = distance_to_grid_border(
            row, col, bin_size_um=bin_size_um, n_counts=n_counts,
            grid_type=grid_type, component_labels=components,
        ).to_numpy()

    if not (components >= 0).any():
        raise ValueError(
            "No connected components survived n_counts/min_component_size filtering -- "
            "nothing to fit. Check row_key/col_key/n_counts_key/grid_type/min_component_size."
        )
    return components, distance, is_border


def identify_analysis_buffer(
    adata,
    score,
    row_key="array_row",
    col_key="array_col",
    grid_type="rect",
    n_counts_key=None,
    bin_size_um=1.0,
    min_component_size=0,
    distance_key=None,
    key_added="analysis_buffer",
    border_key="border",
    copy=False,
):
    """Flag spots within the piecewise-linear-fit elbow of the tissue border,
    fit independently per spatially-connected grid component (see
    `split_into_connected_components`) -- e.g. a TMA's individual cores are
    never pooled into one global elbow.

    Fits `PiecewiseLinearFit` (vs. the `ConstantFit` null) of `score` against
    distance to the nearest border point, separately within each component.
    Writes a boolean column to `adata.obs[key_added]`: True for spots closer
    to their own component's border than that component's fitted breakpoint.
    A component where `ConstantFit` wins (no detected effect) gets an
    all-False buffer for its own spots.

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
        `array_row`/`array_col`).
    grid_type : {"hex", "rect"}, default "rect"
        "rect" (4 neighbors, e.g. Visium HD/STOmics) or "hex" (6 neighbors,
        classic Visium's offset hex grid).
    n_counts_key : str, optional
        `.obs` column of per-spot counts/signal. If given, spots with
        `n_counts <= 0` are excluded before splitting into components and
        computing distances (see `split_into_connected_components`) -- an
        empty grid position never counts as real tissue.
    bin_size_um : float, default 1.0
        Physical size (um) of one grid step, passed to
        `distance_to_grid_border`. The default of 1.0 fits/reports distances
        in raw grid-step units; pass the real spot pitch to get `params`
        (e.g. the elbow breakpoint) in um instead.
    min_component_size : int, default 0
        Components with `<= min_component_size` surviving spots are dropped
        entirely (never fit).
    distance_key : str, optional
        If given, must already exist in `adata.obs` -- reused directly as
        each spot's distance-to-border, instead of computing it via
        `distance_to_grid_border`. Components are still (re)computed from
        `row_key`/`col_key`/`n_counts_key`/`grid_type` regardless, since
        fitting is always per-component.
    key_added : str, default "analysis_buffer"
        `.obs` column name for the output boolean buffer flag. Per-component
        fit diagnostics are stored in
        `adata.uns[f"{key_added}_fit"]["per_component"]`, keyed by component
        label (int).
    border_key : str, default "border"
        `.obs` column name for a boolean flag: True if the spot is itself a
        border node (see `find_grid_border`), False otherwise -- including
        for spots excluded by `n_counts_key`/`min_component_size`.
    copy : bool, default False
        If True, return a modified copy of `adata` instead of mutating in place.

    Returns
    -------
    AnnData or None
        The modified AnnData if `copy=True`, else None (`adata` is mutated in place).
    """
    _require_anndata()
    adata = adata.copy() if copy else adata

    components, distance, is_border = _components_and_distance(
        adata, row_key, col_key, grid_type, n_counts_key, bin_size_um, min_component_size, distance_key,
    )
    adata.obs[border_key] = is_border

    if isinstance(score, str):
        score_values = adata.obs[score].to_numpy()
        score_name = score
    else:
        score_values = np.asarray(score)
        score_name = "score"

    buffer = np.zeros(len(components), dtype=bool)
    per_component = {}
    for label in np.unique(components):
        if label < 0:
            continue
        mask = (components == label) & np.isfinite(distance)
        if not mask.any():
            continue

        flow = Flow.from_distances_and_scores(
            distances=pd.Series(distance[mask], name="distance_to_border"),
            scores=pd.DataFrame({score_name: score_values[mask]}),
        )
        flow.flow(fits=[ConstantFit, PiecewiseLinearFit])
        best_fit = flow.best_fits[score_name]
        if isinstance(best_fit, PiecewiseLinearFit):
            buffer[mask] = distance[mask] < best_fit.params["piecewise_linear_b"]

        per_component[int(label)] = {
            "best_fit_type": best_fit.name,
            "params": dict(best_fit.params),
            "observed_effect_strength": best_fit.observed_effect_strength,
            "observed_half_life": best_fit.observed_half_life,
            "affected_fraction": best_fit.fraction_not_converged,
            "n_spots": int(mask.sum()),
        }

    adata.obs[key_added] = buffer
    adata.uns[f"{key_added}_fit"] = {
        "grid_type": grid_type, "bin_size_um": bin_size_um, "per_component": per_component,
    }

    return adata if copy else None


def correct_layer(
    adata,
    layer=None,
    row_key="array_row",
    col_key="array_col",
    grid_type="rect",
    n_counts_key=None,
    bin_size_um=1.0,
    min_component_size=0,
    distance_key=None,
    key_added="bosperrus_corrected",
    border_key="border",
    copy=False,
):
    """Correct each feature toward its exponential-saturation asymptote,
    fit independently per spatially-connected grid component (see
    `identify_analysis_buffer`) -- e.g. a TMA's individual cores each get
    their own correction curve per gene, never pooled into one global fit.

    Fits `ExponentialSaturationFit` (vs. the `ConstantFit` null) of every
    feature (column) in `layer` against distance to the nearest border
    point, independently per feature *and* per component. Writes the
    corrected matrix to `adata.layers[key_added]`: a feature where
    `ConstantFit` wins within a component (no detected effect) passes
    through unchanged for that component's spots.

    Note: independently fitting many features across many components means
    `n_features * n_components` curve fits -- expect runtime to scale
    accordingly.

    Parameters
    ----------
    adata : AnnData
    layer : str, optional
        Name of the `.layers` entry to correct. If None, uses `adata.X`.
    row_key, col_key, grid_type, n_counts_key, bin_size_um,
    min_component_size, distance_key : see `identify_analysis_buffer`.
    key_added : str, default "bosperrus_corrected"
        `.layers` key for the corrected matrix. Per-component fit-quality
        DataFrames (mirroring `Flow.fit_quality`: columns = features, rows =
        `Fit.params_summary()` keys) are stored in
        `adata.uns[f"{key_added}_fit_quality"]`, keyed by component label (int)
        -- not `.var` columns, since a feature's winning model can differ
        between components, which a single flat per-gene column can't represent.
    border_key : str, default "border"
        `.obs` column name for a boolean flag: True if the spot is itself a
        border node (see `find_grid_border`), False otherwise -- including
        for spots excluded by `n_counts_key`/`min_component_size`.
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

    components, distance, is_border = _components_and_distance(
        adata, row_key, col_key, grid_type, n_counts_key, bin_size_um, min_component_size, distance_key,
    )
    adata.obs[border_key] = is_border

    matrix = adata.X if layer is None else adata.layers[layer]
    if sparse.issparse(matrix):
        matrix = matrix.toarray()
    matrix = np.asarray(matrix, dtype=float)

    corrected = matrix.copy()
    fit_quality_by_component = {}
    for label in np.unique(components):
        if label < 0:
            continue
        mask = (components == label) & np.isfinite(distance)
        if not mask.any():
            continue

        scores = pd.DataFrame(matrix[mask], columns=adata.var_names)
        flow = Flow.from_distances_and_scores(
            distances=pd.Series(distance[mask], name="distance_to_border"),
            scores=scores,
        )
        flow.flow(fits=[ConstantFit, ExponentialSaturationFit])
        corrected_cols = [f"BOSPERRUS corrected {g}" for g in adata.var_names]
        corrected[mask] = flow.observations[corrected_cols].to_numpy()
        fit_quality_by_component[int(label)] = flow.fit_quality

    adata.layers[key_added] = corrected
    adata.uns[f"{key_added}_fit_quality"] = fit_quality_by_component

    return adata if copy else None
