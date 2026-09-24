"""AnnData-native convenience wrappers around Flow for spatial-transcriptomics use.

Requires the optional `anndata` dependency (`pip install bosperrus[anndata]`);
`quantify_diffusion` additionally requires `image-masks`
(`pip install bosperrus[image-masks]`, for `image_masks.get_tissue_mask`). This
module never imports `anndata`/`scikit-image` at the top level -- only inside
the functions that need them, via `_require_anndata()` (and, transitively,
`image_masks`'s own lazy imports) -- so the rest of `bosperrus` (and this
module's own presence in `bosperrus.__all__`) stays importable without them
installed, mirroring `distances.distance_to_alpha_shape`'s optional-dependency
pattern.
"""
import warnings

import numpy as np
import pandas as pd

from .distances import distance_to_grid_border, distance_to_mask
from .fit import ConstantFit, PiecewiseLinearFit, ExponentialSaturationFit, MichaelisMentenFit
from .graph_construction import split_into_connected_components, find_grid_border, grid_edges
from .image_masks import get_tissue_mask
from .pipeline import Flow
from .plotting import plot_fit, FIT_PALETTE

__all__ = [
    "identify_analysis_buffer", "correct_layer", "quantify_diffusion",
    "plot_border_effect", "plot_diffusion",
]

# Below this average component size (kept spots / number of components),
# warn that components look unreasonably numerous -- usually a sign of
# grid_type mismatch, a bad components_key override, or otherwise
# over-fragmented data that per-component fits won't handle reliably.
_MIN_REASONABLE_AVG_COMPONENT_SIZE = 20


def _require_anndata():
    try:
        import anndata  # noqa: F401
    except ImportError:
        raise ImportError(
            "anndata is required for this function. "
            "Install it with: pip install anndata or pip install bosperrus[anndata]"
        )


def _validate_numeric_nonneg_column(values, descriptor):
    """descriptor is a pre-formatted string identifying the offending
    parameter/column for the error message, e.g. "distance_key='my_col'"."""
    if not np.issubdtype(values.dtype, np.number):
        raise ValueError(f"{descriptor} column must be numeric, got dtype {values.dtype}.")
    finite = np.isfinite(values)
    if finite.any() and (values[finite] < 0).any():
        raise ValueError(f"{descriptor} column contains negative values -- distances must be >= 0.")


def _get_components(adata, row_key, col_key, grid_type, n_counts_key, min_component_size,
                     components_key, stacklevel):
    """Shared component-labeling core of identify_analysis_buffer/
    correct_layer/quantify_diffusion: split into spatially-connected grid
    components (see split_into_connected_components) -- every caller then
    fits independently per component, never pooled, since components are
    e.g. a TMA's individual cores, physically disconnected pieces of tissue.

    components_key is dual-purpose: if that column already exists in
    adata.obs, it's reused as-is (letting a caller supply their own
    components, or reuse labels from an earlier call) -- sanity-checked
    (must cast to int) so a bad override fails clearly here, not confusingly
    inside the fitting loop. Otherwise it's computed and written there. Pass
    None to skip persisting a value that wasn't already present.

    Raises if nothing survived filtering, and warns (at the given
    stacklevel, so it points at each caller's own caller) if the resulting
    components look unreasonably numerous relative to how much data
    survived -- usually a grid_type mismatch, a components_key column that
    isn't really component labels, or genuinely over-fragmented data.
    """
    row = adata.obs[row_key].to_numpy()
    col = adata.obs[col_key].to_numpy()
    n_counts = adata.obs[n_counts_key].to_numpy() if n_counts_key is not None else None

    if components_key is not None and components_key in adata.obs:
        try:
            components = adata.obs[components_key].to_numpy().astype(int)
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"components_key={components_key!r} column isn't usable as integer "
                f"component labels: {e}"
            ) from e
    else:
        components = split_into_connected_components(
            row, col, n_counts=n_counts, grid_type=grid_type, min_size=min_component_size,
        )

    if components_key is not None:
        # Always (re)write the int-cast array back, even when reused from an
        # existing column -- otherwise a column stored in some other dtype
        # (e.g. str, from adata.obs["components"] = components.astype(str))
        # keeps that dtype on disk while every fit ran against the in-memory
        # int-cast copy, so a later caller re-reading adata.obs[components_key]
        # directly (e.g. plot_border_effect/plot_diffusion) gets values that
        # never match the int component labels stored in *_fit["per_component"].
        adata.obs[components_key] = components

    n_kept = int((components >= 0).sum())
    if n_kept == 0:
        raise ValueError(
            "No connected components available for fitting -- either none survived "
            "n_counts/min_component_size filtering, or components_key is all-excluded. "
            "Check row_key/col_key/n_counts_key/grid_type/min_component_size/components_key."
        )

    n_components = len(np.unique(components[components >= 0]))
    if n_kept / n_components < _MIN_REASONABLE_AVG_COMPONENT_SIZE:
        warnings.warn(
            f"{n_components} components for {n_kept} kept spots (average size "
            f"{n_kept / n_components:.1f}) -- unusually fragmented. This can mean a "
            f"grid_type mismatch, a components_key column that isn't really component "
            f"labels, or genuinely many small/noisy fragments; per-component fits on "
            f"very small components are unreliable. Check grid_type/components_key, or "
            f"raise min_component_size to drop the smallest fragments.",
            UserWarning,
            stacklevel=stacklevel,
        )

    return row, col, n_counts, components


def _components_and_distance(adata, row_key, col_key, grid_type, n_counts_key, bin_size_um,
                              min_component_size, components_key, distance_key):
    """Shared core of identify_analysis_buffer/correct_layer: components (see
    _get_components), border flags (see find_grid_border), and each node's
    distance to its own component's border (see distance_to_grid_border).

    distance_key is dual-purpose exactly like components_key (see
    _get_components) -- reused if already present (sanity-checked: numeric,
    non-negative), computed and written otherwise.

    is_border is masked to False wherever components < 0 (n_counts <= 0, or
    a component dropped by min_component_size), matching the same
    exclusion semantics as the buffer/correction outputs: nothing meaningful
    is reported for spots outside the actual analysis.
    """
    row, col, n_counts, components = _get_components(
        adata, row_key, col_key, grid_type, n_counts_key, min_component_size, components_key, stacklevel=4,
    )
    is_border = find_grid_border(row, col, n_counts=n_counts, grid_type=grid_type) & (components >= 0)

    if distance_key is not None and distance_key in adata.obs:
        distance = adata.obs[distance_key].to_numpy()
        _validate_numeric_nonneg_column(distance, f"distance_key={distance_key!r}")
    else:
        distance = distance_to_grid_border(
            row, col, bin_size_um=bin_size_um, n_counts=n_counts,
            grid_type=grid_type, component_labels=components,
        ).to_numpy()
        if distance_key is not None:
            adata.obs[distance_key] = distance

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
    components_key="components",
    distance_key="distance_to_border",
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
    components_key : str, default "components"
        `.obs` column for component labels (see
        `split_into_connected_components`). If this column already exists,
        it's reused as-is instead of being recomputed -- e.g. to supply your
        own component boundaries, or reuse labels from an earlier call.
        Otherwise it's computed and written here. Pass None to skip writing
        it (still computed internally either way, since fitting is always
        per-component). Note: an existing column is *always* reused, even if
        you've changed `row_key`/`n_counts_key`/`grid_type`/
        `min_component_size` since it was written -- rename or delete the
        column first if you want it recomputed.
    distance_key : str, default "distance_to_border"
        `.obs` column for each spot's distance to its own component's border
        (see `distance_to_grid_border`). Same reuse-if-present,
        compute-and-write-otherwise behavior as `components_key`.
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
        adata, row_key, col_key, grid_type, n_counts_key, bin_size_um,
        min_component_size, components_key, distance_key,
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
    components_key="components",
    distance_key="distance_to_border",
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
    min_component_size, components_key, distance_key : see `identify_analysis_buffer`.
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
        adata, row_key, col_key, grid_type, n_counts_key, bin_size_um,
        min_component_size, components_key, distance_key,
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


def _native_pixel_size_um(row, col, spatial, bin_size_um, grid_type, max_edges=2000):
    """um per native adata.obsm["spatial"] pixel, measured empirically from
    the known physical grid pitch (bin_size_um) vs. the pixel distance
    between grid-adjacent spots -- native pixel size is NOT a fixed
    constant, it varies per scan, so this can't be a hardcoded conversion
    factor. Cast to float64 first: some loaders store obsm["spatial"] as an
    unsigned integer dtype, and a plain difference would silently wrap
    around instead of going negative."""
    spatial = np.asarray(spatial, dtype=np.float64)
    edges = list(grid_edges(row, col, grid_type=grid_type))
    if not edges:
        raise ValueError("No grid-adjacent spot pairs found -- can't calibrate native pixel size.")
    if len(edges) > max_edges:
        idx = np.random.default_rng(0).choice(len(edges), max_edges, replace=False)
        edges = [edges[i] for i in idx]
    u, v = zip(*edges)
    pixel_pitch = np.median(np.linalg.norm(spatial[list(u)] - spatial[list(v)], axis=1))
    return bin_size_um / pixel_pitch


def quantify_diffusion(
    adata,
    library_id,
    score,
    row_key="array_row",
    col_key="array_col",
    grid_type="rect",
    n_counts_key=None,
    bin_size_um=1.0,
    min_component_size=0,
    components_key="components",
    image_key="hires",
    mask_distance_key="distance_to_mask",
    mask_image_key="mask",
    key_added="diffusion",
    segment_kwargs=None,
    copy=False,
):
    """Quantify RNA/signal diffusion outside the tissue boundary, fit
    independently per spatially-connected grid component (see
    `identify_analysis_buffer`) -- e.g. a TMA's individual cores each get
    their own diffusion curve, never pooled into one global fit.

    Finds the sample's image-derived tissue mask (see
    `image_masks.get_tissue_mask`), computes each spot's physical distance
    (um) outside that mask (see `distances.distance_to_mask`), and fits
    `ExponentialSaturationFit` (vs. the `ConstantFit` null) of `score`
    against that distance -- for spots outside the mask only -- separately
    within each component.

    Reports two cross-sample-comparable diffusion parameters per component
    (see `per_component` below), mirroring the manuscript's own convention:
    - `alpha` (counts/um^2): `-a / bin_size_um**2`. Sign-flipped because
      counts decay *away* from tissue (the raw fit parameter `a` comes out
      negative), and divided by spot area because raw counts -- and thus
      `a` -- scale with capture area, not comparable across differently
      binned samples otherwise.
    - `beta` (1/um): the fitted decay rate `b`, already in inverse-um since
      distances are fit directly in um (not native pixels or grid steps).
    Both are `None` for a component where `ConstantFit` won (no detected
    diffusion effect) or where fitting didn't converge.

    Requires the `image-masks` extra in addition to `anndata`
    (`pip install bosperrus[image-masks]`) -- see `image_masks.get_tissue_mask`.

    The default mask lookup only works out of the box for platforms whose
    reader embeds an image directly in the AnnData, following the
    scanpy/squidpy `adata.uns["spatial"][library_id]` convention (e.g. 10x
    Visium). Platforms that ship tissue images/masks as separate files
    instead (e.g. STOmics/Stereo-seq) aren't covered -- compute a
    distance-to-mask array yourself (see `distances.distance_to_mask`) and
    pass it via `mask_distance_key` to skip the image lookup entirely.

    Parameters
    ----------
    adata : AnnData
    library_id : str
        Key into `adata.uns["spatial"]` (see `image_masks.get_hires_image`).
    score : str or array-like
        Either the name of an `.obs` column, or an array of values aligned
        with `adata.obs_names` (typically total counts) to fit against
        distance outside the mask.
    row_key, col_key, grid_type, n_counts_key, min_component_size,
    components_key : see `identify_analysis_buffer`.
    bin_size_um : float, default 1.0
        Physical size (um) of one grid step/spot pitch -- used both to
        calibrate native pixel size (see `_native_pixel_size_um`) and to
        scale `alpha` by spot area (`bin_size_um ** 2`). Unlike
        `identify_analysis_buffer`, this isn't just a cosmetic unit choice:
        leaving it at the default 1.0 directly biases `alpha`'s
        cross-sample comparability, so pass the real spot pitch.
    image_key : str, default "hires"
        Which embedded image to segment -- see `image_masks.get_tissue_mask`.
    mask_distance_key : str, default "distance_to_mask"
        `.obs` column for each spot's physical distance (um) outside the
        tissue mask (0 for spots inside it). Same reuse-if-present,
        compute-and-write-otherwise behavior as `identify_analysis_buffer`'s
        `distance_key` -- sanity-checked (numeric, non-negative) if reused.
    mask_image_key : str, default "mask"
        Writes the segmented mask into `adata.uns["spatial"][library_id]
        ["images"][mask_image_key]` (plus the matching `scalefactors
        ["tissue_{mask_image_key}_scalef"]`) -- the same convention
        `image_masks.get_hires_image` reads from, so you can immediately
        plot it with `sc.pl.spatial(adata, library_id=library_id,
        img_key=mask_image_key)`, spots and all, without recomputing it.
        Only written when the mask is actually computed fresh here (i.e.
        not when reusing an existing `mask_distance_key`, since then no
        mask is computed at all). Pass None to skip.
    key_added : str, default "diffusion"
        Per-component fit diagnostics (including `alpha`/`beta`) are stored
        in `adata.uns[f"{key_added}_fit"]["per_component"]`, keyed by
        component label (int). There's no natural per-spot boolean output
        analogous to `identify_analysis_buffer`'s buffer flag here, so
        nothing else is written besides `components_key`/`mask_distance_key`.
    segment_kwargs : dict, optional
        Extra keyword arguments forwarded to
        `image_masks.segment_tissue_from_rgb` (e.g. `sigma`, `close_radius`).
    copy : bool, default False
        If True, return a modified copy of `adata` instead of mutating in place.

    Returns
    -------
    AnnData or None
        The modified AnnData if `copy=True`, else None (`adata` is mutated in place).
    """
    _require_anndata()
    adata = adata.copy() if copy else adata

    row, col, _, components = _get_components(
        adata, row_key, col_key, grid_type, n_counts_key, min_component_size, components_key, stacklevel=3,
    )

    if isinstance(score, str):
        score_values = adata.obs[score].to_numpy()
        score_name = score
    else:
        score_values = np.asarray(score)
        score_name = "score"

    if mask_distance_key is not None and mask_distance_key in adata.obs:
        distance = adata.obs[mask_distance_key].to_numpy()
        _validate_numeric_nonneg_column(distance, f"mask_distance_key={mask_distance_key!r}")
    else:
        spatial = adata.obsm["spatial"]
        pixel_size_um = _native_pixel_size_um(row, col, spatial, bin_size_um, grid_type=grid_type)
        mask, pixel_scale = get_tissue_mask(adata, library_id, image_key=image_key, **(segment_kwargs or {}))
        if mask_image_key is not None:
            adata.uns["spatial"][library_id]["images"][mask_image_key] = mask.astype(np.uint8) * 255
            adata.uns["spatial"][library_id]["scalefactors"][f"tissue_{mask_image_key}_scalef"] = pixel_scale
        # obsm["spatial"] is (x, y) = (pixel_col, pixel_row); distance_to_mask indexes
        # the mask array as [row, col], hence the swap.
        spatial = np.asarray(spatial, dtype=np.float64)
        coords_mask_space = np.stack([spatial[:, 1] * pixel_scale, spatial[:, 0] * pixel_scale], axis=1)
        distance = distance_to_mask(
            coords_mask_space, mask, pixel_size_um=pixel_size_um / pixel_scale,
        ).to_numpy()
        if mask_distance_key is not None:
            adata.obs[mask_distance_key] = distance

    spot_area_um2 = bin_size_um ** 2
    per_component = {}
    for label in np.unique(components):
        if label < 0:
            continue
        mask_outside = (components == label) & (distance > 0) & np.isfinite(distance)
        if not mask_outside.any():
            continue

        flow = Flow.from_distances_and_scores(
            distances=pd.Series(distance[mask_outside], name="distance_outside_mask"),
            scores=pd.DataFrame({score_name: score_values[mask_outside]}),
        )
        flow.flow(fits=[ConstantFit, ExponentialSaturationFit])
        best_fit = flow.best_fits[score_name]

        alpha = beta = None
        if isinstance(best_fit, ExponentialSaturationFit):
            alpha = -best_fit.params["exponential_saturation_a"] / spot_area_um2
            beta = best_fit.params["exponential_saturation_b"]

        per_component[int(label)] = {
            "best_fit_type": best_fit.name,
            "params": dict(best_fit.params),
            "alpha": alpha,
            "beta": beta,
            "observed_effect_strength": best_fit.observed_effect_strength,
            "observed_half_life": best_fit.observed_half_life,
            "n_spots": int(mask_outside.sum()),
        }

    adata.uns[f"{key_added}_fit"] = {
        "grid_type": grid_type, "bin_size_um": bin_size_um, "per_component": per_component,
    }

    return adata if copy else None


_PREDICT_FORMULAS = {
    "Constant Fit": lambda d, p: np.full_like(np.asarray(d, dtype=float), p["constant_c"]),
    "Piecewise Linear Fit": lambda d, p: PiecewiseLinearFit.piecewise_plateau(
        d, p["piecewise_linear_b"], p["piecewise_linear_m"], p["piecewise_linear_c"]),
    "Exponential Saturation Fit": lambda d, p: ExponentialSaturationFit.exp_sat(
        d, p["exponential_saturation_a"], p["exponential_saturation_b"], p["exponential_saturation_c"]),
    "Michaelis-Menten Fit": lambda d, p: MichaelisMentenFit.michaelis_menten(
        d, p["michaelis_menten_a"], p["michaelis_menten_b"], p["michaelis_menten_c"]),
}


def _predict_from_params(best_fit_type, params):
    """Reconstruct a `d -> predicted score` callable from a stored
    best_fit_type/params pair (as saved in adata.uns[...]["per_component"]),
    without needing a live Fit instance -- Fit objects aren't
    AnnData/h5ad-serializable, so only best_fit_type/params (already plain
    dicts/floats) are ever persisted to .uns, and this rebuilds a usable
    predictor from just those, mirroring each Fit subclass's own predict()."""
    try:
        formula = _PREDICT_FORMULAS[best_fit_type]
    except KeyError:
        raise ValueError(f"Unknown best_fit_type {best_fit_type!r} -- can't reconstruct the fitted curve.")
    return lambda d: formula(d, params)


def _plot_per_component_fit(score_values, distance, components, per_component, xlabel, ncols, figsize, **plot_fit_kwargs):
    import matplotlib.pyplot as plt

    labels = sorted(per_component)
    if not labels:
        raise ValueError("No per-component fits to plot.")

    nrows = int(np.ceil(len(labels) / ncols))
    if figsize is None:
        figsize = (4 * ncols, 3 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes = axes.ravel()

    for ax, label in zip(axes, labels):
        info = per_component[label]
        member_mask = components == label
        predict_fn = _predict_from_params(info["best_fit_type"], info["params"])

        # Color the fitted curve by model type (FIT_PALETTE), same convention
        # every other bosperrus plot uses -- unless the caller already asked
        # for a specific color via their own line_kwargs.
        kwargs = dict(plot_fit_kwargs)
        line_kwargs = dict(kwargs.pop("line_kwargs", {}))
        line_kwargs.setdefault("color", FIT_PALETTE.get(info["best_fit_type"], "C1"))
        kwargs["line_kwargs"] = line_kwargs

        plot_fit(ax, distance[member_mask], score_values[member_mask], predict_fn, **kwargs)
        ax.set_xlabel(xlabel)
        ax.set_title(f"component {label} (n={info['n_spots']}, {info['best_fit_type']})", fontsize=9)

    for ax in axes[len(labels):]:
        ax.axis("off")

    fig.tight_layout()
    return fig


def plot_border_effect(adata, score, distance_key="distance_to_border", components_key="components",
                        key_added="analysis_buffer", ncols=4, figsize=None, **plot_fit_kwargs):
    """Plot `score` vs. distance-to-border with each component's fitted
    elbow curve overlaid on top (see `identify_analysis_buffer`, which must
    be run first -- this reads its stored results, it doesn't fit anything
    itself). One subplot per component.

    Parameters
    ----------
    adata : AnnData
    score : str or array-like
        Same score `identify_analysis_buffer` was run with.
    distance_key, components_key, key_added : see `identify_analysis_buffer`
        -- must match the values it was actually called with, since this
        reads `adata.obs[distance_key]`/`adata.obs[components_key]` and
        `adata.uns[f"{key_added}_fit"]["per_component"]`.
    ncols : int, default 4
        Subplot grid width.
    figsize : (float, float), optional
        Defaults to `(4 * ncols, 3 * nrows)`.
    **plot_fit_kwargs
        Forwarded to `plotting.plot_fit` (e.g. `bins`, `hist_kwargs`, `line_kwargs`).

    Returns
    -------
    matplotlib.figure.Figure
    """
    _require_anndata()
    fit_key = f"{key_added}_fit"
    if fit_key not in adata.uns:
        raise KeyError(f"{fit_key!r} not found in adata.uns -- run identify_analysis_buffer first.")

    score_values = adata.obs[score].to_numpy() if isinstance(score, str) else np.asarray(score)
    distance = adata.obs[distance_key].to_numpy()
    components = adata.obs[components_key].to_numpy()
    per_component = adata.uns[fit_key]["per_component"]

    return _plot_per_component_fit(
        score_values, distance, components, per_component,
        xlabel="distance to border", ncols=ncols, figsize=figsize, **plot_fit_kwargs,
    )


def plot_diffusion(adata, score, mask_distance_key="distance_to_mask", components_key="components",
                    key_added="diffusion", ncols=4, figsize=None, **plot_fit_kwargs):
    """Plot `score` vs. distance outside the tissue mask with each
    component's fitted diffusion curve overlaid on top (see
    `quantify_diffusion`, which must be run first -- this reads its stored
    results, it doesn't fit anything itself). Only spots outside the mask
    (distance > 0) are shown, matching exactly what `quantify_diffusion`
    itself fits against. One subplot per component.

    Parameters
    ----------
    adata : AnnData
    score : str or array-like
        Same score `quantify_diffusion` was run with.
    mask_distance_key, components_key, key_added : see `quantify_diffusion`
        -- must match the values it was actually called with, since this
        reads `adata.obs[mask_distance_key]`/`adata.obs[components_key]` and
        `adata.uns[f"{key_added}_fit"]["per_component"]`.
    ncols : int, default 4
        Subplot grid width.
    figsize : (float, float), optional
        Defaults to `(4 * ncols, 3 * nrows)`.
    **plot_fit_kwargs
        Forwarded to `plotting.plot_fit` (e.g. `bins`, `hist_kwargs`, `line_kwargs`).

    Returns
    -------
    matplotlib.figure.Figure
    """
    _require_anndata()
    fit_key = f"{key_added}_fit"
    if fit_key not in adata.uns:
        raise KeyError(f"{fit_key!r} not found in adata.uns -- run quantify_diffusion first.")

    score_values = adata.obs[score].to_numpy() if isinstance(score, str) else np.asarray(score)
    distance = adata.obs[mask_distance_key].to_numpy()
    outside = distance > 0
    components = np.where(outside, adata.obs[components_key].to_numpy(), -1)
    distance = np.where(outside, distance, np.nan)
    per_component = adata.uns[fit_key]["per_component"]

    return _plot_per_component_fit(
        score_values, distance, components, per_component,
        xlabel="distance outside mask", ncols=ncols, figsize=figsize, **plot_fit_kwargs,
    )
