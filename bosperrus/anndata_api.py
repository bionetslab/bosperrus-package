"""AnnData-native convenience wrappers around Flow for spatial-transcriptomics use.

Two entry points read a pipeline's own output folder directly (Space Ranger
`binned_outputs/` for Visium HD, a SAW run folder for STOmics/Stereo-seq):

- `load_filtered` + `identify_analysis_buffer_from_filtered`: the tissue
  border effect ("analysis buffer") on the pipeline's tissue-filtered bins.
- `quantify_diffusion_from_raw`: RNA diffusion outside an image-only tissue
  mask, on the raw (whole capture area) bins.

Requires the optional `anndata` dependency (`pip install bosperrus[anndata]`);
the two path-based readers additionally need the `raw` extra
(`pip install bosperrus[raw]`: h5py, pyarrow, tifffile, scikit-image). This
module never imports those at the top level -- only inside the functions
that need them -- so the rest of `bosperrus` (and this module's own presence
in `bosperrus.__all__`) stays importable without them installed, mirroring
`distances.distance_to_alpha_shape`'s optional-dependency pattern.
"""
import warnings

import numpy as np
import pandas as pd

from . import _border_test, _readers
from .distances import distance_to_grid_border, distance_to_mask
from .fit import ConstantFit, PiecewiseLinearFit, RisingPiecewiseLinearFit, ExponentialSaturationFit, ExponentialDecayFit, MichaelisMentenFit
from .graph_construction import split_into_connected_components, find_grid_border, small_grid_holes
from .image_masks import segment_tissue_from_rgb, segment_tissue_from_ssdna
from .pipeline import Flow
from .plotting import plot_fit, FIT_PALETTE

__all__ = [
    "load_filtered", "identify_analysis_buffer_from_filtered", "interior_window_test", "fit_border_profile",
    "correct_layer",
    "quantify_diffusion_from_raw", "quantify_diffusion_visium_HD", "quantify_diffusion_stereoseq",
    "plot_border_effect",
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


def _reuse_if_matching_params(adata, key, current_params, stacklevel):
    """Guards the "reuse an existing *_key column if present" pattern shared
    by components_key/distance_key/mask_distance_key: an AnnData round-tripped
    through h5ad has no other way to tell "this column is stale" apart from
    comparing the parameters it was actually computed with against the
    current call's -- without this, changing row_key/grid_type/bin_size_um/
    etc. between calls silently reuses a now-mismatched column instead of
    recomputing, which used to be the documented (if surprising) behavior.

    The parameters used to compute `adata.obs[key]` are stored alongside it
    in `adata.uns[f"{key}_bosperrus_params"]`. If that metadata is missing
    entirely, the column is treated as externally/user-supplied (e.g. custom
    component boundaries) rather than stale -- trusted as-is, silently, same
    as before this guard existed. Only warns (and signals recompute) when
    bosperrus's own prior computation is being contradicted by the current
    call.

    Returns True if it's safe to reuse `adata.obs[key]` as-is, False if it
    should be (re)computed.
    """
    if key not in adata.obs:
        return False
    params_key = f"{key}_bosperrus_params"
    if params_key not in adata.uns:
        return True
    stored_params = adata.uns[params_key]
    if stored_params == current_params:
        return True
    warnings.warn(
        f"adata.obs[{key!r}] already exists but was computed with different "
        f"parameters ({stored_params!r}) than this call ({current_params!r}) -- "
        f"recomputing rather than reusing a stale column. Pass a different "
        f"key, or delete adata.obs[{key!r}]/adata.uns[{params_key!r}], to "
        f"silence this once you're done changing parameters.",
        UserWarning,
        stacklevel=stacklevel,
    )
    return False


def _fill_small_holes(row, col, n_counts, grid_type, max_hole_area_um2, bin_size_um):
    """Effective tissue footprint with small enclosed holes filled (see
    `small_grid_holes`). Returns (row_aug, col_aug, footprint_aug, n_real):
    the real nodes first, then one extra "virtual" node per hole position
    that has no node at all; footprint_aug is 1.0 for every node that is part
    of the tissue (n_counts > 0, or inside a filled hole) and 0.0 otherwise --
    passed as n_counts to the grid functions, so virtual nodes take part in
    adjacency (components, border, distance) but are never fit or reported."""
    n_real = len(row)
    footprint = n_counts > 0 if n_counts is not None else np.ones(n_real, dtype=bool)
    if max_hole_area_um2 is None:
        return row, col, footprint.astype(float), n_real
    if grid_type != "rect":
        raise ValueError(
            f"max_hole_area_um2 is only supported for grid_type='rect' (got {grid_type!r}); pass max_hole_area_um2=None."
        )
    max_hole_size = int(np.floor(max_hole_area_um2 / bin_size_um ** 2 + 1e-9))
    hole_row, hole_col = small_grid_holes(row, col, n_counts=footprint.astype(float), max_hole_size=max_hole_size)
    footprint = footprint.copy()
    virtual_row = virtual_col = np.array([], dtype=np.int64)
    if len(hole_row):
        row64, col64 = np.asarray(row, dtype=np.int64), np.asarray(col, dtype=np.int64)
        width = int(max(col64.max(), hole_col.max())) + 1
        existing = pd.Index(row64 * width + col64).get_indexer(hole_row * width + hole_col)
        footprint[existing[existing >= 0]] = True
        virtual_row, virtual_col = hole_row[existing < 0], hole_col[existing < 0]
    row_aug = np.concatenate([np.asarray(row, dtype=np.int64), virtual_row])
    col_aug = np.concatenate([np.asarray(col, dtype=np.int64), virtual_col])
    footprint_aug = np.concatenate([footprint, np.ones(len(virtual_row), dtype=bool)]).astype(float)
    return row_aug, col_aug, footprint_aug, n_real


def _get_components(adata, row_key, col_key, grid_type, n_counts_key, min_component_size,
                     components_key, stacklevel, max_hole_area_um2=None, bin_size_um=1.0):
    """Shared component-labeling core of identify_analysis_buffer_from_filtered/
    correct_layer: split into spatially-connected grid
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

    With max_hole_area_um2, small enclosed holes of the n_counts > 0
    footprint are filled first (see _fill_small_holes). Returns
    (row_aug, col_aug, footprint_aug, components_aug, n_real): arrays over
    the real nodes followed by the virtual hole nodes; only the first n_real
    entries correspond to adata.obs rows.
    """
    row = adata.obs[row_key].to_numpy()
    col = adata.obs[col_key].to_numpy()
    n_counts = adata.obs[n_counts_key].to_numpy() if n_counts_key is not None else None
    row_aug, col_aug, footprint_aug, n_real = _fill_small_holes(
        row, col, n_counts, grid_type, max_hole_area_um2, bin_size_um,
    )

    current_params = {
        "row_key": row_key, "col_key": col_key, "grid_type": grid_type,
        "n_counts_key": n_counts_key, "min_component_size": min_component_size,
    }
    if max_hole_area_um2 is not None:
        current_params.update(max_hole_area_um2=max_hole_area_um2, bin_size_um=bin_size_um)
    if components_key is not None and _reuse_if_matching_params(adata, components_key, current_params, stacklevel + 1):
        try:
            components = adata.obs[components_key].to_numpy().astype(int)
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"components_key={components_key!r} column isn't usable as integer "
                f"component labels: {e}"
            ) from e
        components_aug = np.concatenate([components, _label_virtual_nodes(row_aug, col_aug, components, n_real)])
    else:
        components_aug = split_into_connected_components(
            row_aug, col_aug, n_counts=footprint_aug, grid_type=grid_type, min_size=min_component_size,
        )
        components = components_aug[:n_real]

    if components_key is not None:
        # Always (re)write the int-cast array back, even when reused from an
        # existing column -- otherwise a column stored in some other dtype
        # (e.g. str, from adata.obs["components"] = components.astype(str))
        # keeps that dtype on disk while every fit ran against the in-memory
        # int-cast copy, so a later caller re-reading adata.obs[components_key]
        # directly (e.g. plot_border_effect) gets values that
        # never match the int component labels stored in *_fit["per_component"].
        adata.obs[components_key] = components
        adata.uns[f"{components_key}_bosperrus_params"] = current_params

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

    return row_aug, col_aug, footprint_aug, components_aug, n_real


def _label_virtual_nodes(row_aug, col_aug, components, n_real):
    """Component label for each virtual hole node when components come from
    a reused/user-supplied column: the label of the nearest real node that
    belongs to a component (holes are enclosed, so that's the surrounding
    tissue's component)."""
    if len(row_aug) == n_real:
        return np.array([], dtype=int)
    from scipy.spatial import cKDTree
    labelled = np.flatnonzero(components >= 0)
    if len(labelled) == 0:
        return np.full(len(row_aug) - n_real, -1, dtype=int)
    tree = cKDTree(np.stack([row_aug[labelled], col_aug[labelled]], axis=1))
    _, nearest = tree.query(np.stack([row_aug[n_real:], col_aug[n_real:]], axis=1))
    return components[labelled[nearest]]


def _components_and_distance(adata, row_key, col_key, grid_type, n_counts_key, bin_size_um,
                              min_component_size, components_key, distance_key, max_hole_area_um2=None):
    """Shared core of identify_analysis_buffer_from_filtered/correct_layer: components (see
    _get_components), border flags (see find_grid_border), and each node's
    distance to its own component's border (see distance_to_grid_border).

    distance_key is dual-purpose exactly like components_key (see
    _get_components) -- reused if already present (sanity-checked: numeric,
    non-negative), computed and written otherwise.

    is_border is masked to False wherever components < 0 (n_counts <= 0 and
    not in a filled hole, or a component dropped by min_component_size),
    matching the same exclusion semantics as the buffer/correction outputs:
    nothing meaningful is reported for spots outside the actual analysis.

    Returns (components, distance, is_border, virtual): the first three over
    adata.obs rows; virtual = {"array_row", "array_col", "components"} of the
    virtual hole nodes (grid positions inside a filled hole with no row).
    """
    row_aug, col_aug, footprint_aug, components_aug, n_real = _get_components(
        adata, row_key, col_key, grid_type, n_counts_key, min_component_size, components_key, stacklevel=4,
        max_hole_area_um2=max_hole_area_um2, bin_size_um=bin_size_um,
    )
    components = components_aug[:n_real]
    is_border = (find_grid_border(row_aug, col_aug, n_counts=footprint_aug, grid_type=grid_type)
                 & (components_aug >= 0))[:n_real]

    current_params = {
        "row_key": row_key, "col_key": col_key, "grid_type": grid_type, "bin_size_um": bin_size_um,
        "n_counts_key": n_counts_key, "min_component_size": min_component_size, "components_key": components_key,
    }
    if max_hole_area_um2 is not None:
        current_params["max_hole_area_um2"] = max_hole_area_um2
    if distance_key is not None and _reuse_if_matching_params(adata, distance_key, current_params, stacklevel=4):
        distance = adata.obs[distance_key].to_numpy()
        _validate_numeric_nonneg_column(distance, f"distance_key={distance_key!r}")
    else:
        distance = distance_to_grid_border(
            row_aug, col_aug, bin_size_um=bin_size_um, n_counts=footprint_aug,
            grid_type=grid_type, component_labels=components_aug,
        ).to_numpy()[:n_real]
        if distance_key is not None:
            adata.obs[distance_key] = distance
            adata.uns[f"{distance_key}_bosperrus_params"] = current_params

    virtual = {
        "array_row": row_aug[n_real:], "array_col": col_aug[n_real:], "components": components_aug[n_real:],
    }
    return components, distance, is_border, virtual


def load_filtered(path, technology, resolution=8.0):
    """Per-bin total counts of a pipeline's own *tissue-filtered* output, as a
    genes-free AnnData ready for `identify_analysis_buffer_from_filtered`.

    - Visium HD (`technology="visium_hd"`): `path` is Space Ranger's
      `binned_outputs/` directory; reads
      `square_{resolution:03d}um/filtered_feature_bc_matrix.h5` (summed over
      genes) and `spatial/tissue_positions.parquet` (grid indices and
      full-resolution pixel coordinates).
    - Stereo-seq (`technology="stereo-seq"`): `path` is a SAW run folder
      (containing `outs/`); reads `outs/feature_expression/{SN}.tissue.gef`
      at bin1 and pools it to `resolution` by floor division -- identical to
      SAW's own binN (e.g. 10um reproduces SAW's bin20 exactly). Partial bins
      at the far chip edge are dropped. Only bins with counts exist in a gef,
      so every returned bin has `n_counts > 0`.

    Only per-bin totals are read, never the gene x bin matrix.

    Parameters
    ----------
    path : str or Path
    technology : {"visium_hd", "stereo-seq"}
    resolution : float, default 8.0
        Bin size (um). Visium HD: 2, 8 or 16. Stereo-seq: any integer
        multiple of the 0.5um DNB pitch.

    Returns
    -------
    AnnData
        `n_vars == 0`; `.obs` has `array_row`, `array_col` (integer grid
        indices) and `n_counts`; `.obsm["spatial"]` holds (x, y) bin centres
        in the pipeline's native pixel space (Visium full-resolution pixels,
        Stereo-seq bin1/DNB units); `.uns["bosperrus"]` records
        `technology`, `bin_size_um` and `path`.
    """
    _require_anndata()
    import anndata

    technology, resolution, dnbs_per_bin = _readers.validate_technology_and_resolution(technology, resolution)

    if technology == "visium_hd":
        bin_dir = _readers.visium_bin_dir(path, resolution)
        barcodes, n_counts = _readers.read_10x_h5_total_counts(bin_dir / "filtered_feature_bc_matrix.h5")
        positions = _readers.read_visium_positions(bin_dir, barcodes)
        obs = pd.DataFrame({
            # Space Ranger stores these as uint32 -- cast to signed so neighbour
            # arithmetic (row - 1) can't silently wrap around
            "array_row": positions["array_row"].to_numpy().astype(np.int64),
            "array_col": positions["array_col"].to_numpy().astype(np.int64),
            "n_counts": n_counts,
        }, index=pd.Index(barcodes, name="barcode"))
        spatial = positions[["pxl_col_in_fullres", "pxl_row_in_fullres"]].to_numpy(dtype=float)
    else:
        outs, sn = _readers.saw_outs_and_sn(path)
        extent = _readers.saw_chip_extent(outs, sn)
        dense = _readers.pool_gef_counts(outs / "feature_expression" / f"{sn}.tissue.gef", dnbs_per_bin, extent)
        row, col = np.nonzero(dense)
        obs = pd.DataFrame({
            "array_row": row, "array_col": col, "n_counts": dense[row, col].astype(np.float64),
        }, index=pd.Index(pd.Series(row).astype(str).str.cat(pd.Series(col).astype(str), sep="_"), name="bin"))
        half = (dnbs_per_bin - 1) / 2
        min_y, min_x = extent[0], extent[1]
        spatial = np.stack([col * dnbs_per_bin + half + min_x, row * dnbs_per_bin + half + min_y], axis=1).astype(float)

    adata = anndata.AnnData(obs=obs, obsm={"spatial": spatial})
    adata.uns["bosperrus"] = {"technology": technology, "bin_size_um": resolution, "path": str(path)}
    return adata


def fit_border_profile(distance, score, rising_only=False, score_name="score"):
    """Fit one border profile: `ConstantFit` vs. `PiecewiseLinearFit` (or
    `RisingPiecewiseLinearFit`) of `score` against `distance`, best by AIC.

    The fit that `identify_analysis_buffer_from_filtered` runs per component and
    `interior_window_test` runs per interior window. A border effect is a winning
    piecewise fit that rises into the tissue (m > 0); a falling one is a trend,
    not a border depression.

    Returns the per-component record: `best_fit_type`, `border_effect`,
    `elbow_um` (knot, None without a border effect), `params`,
    `observed_effect_strength`, `observed_half_life` (normalised by the deepest
    fitted distance), `half_life_um`, `max_fitted_distance_um`,
    `affected_fraction` and `n_spots`.
    """
    distance = np.asarray(distance, dtype=float)
    flow = Flow.from_distances_and_scores(
        distances=pd.Series(distance, name="distance_to_border"),
        scores=pd.DataFrame({score_name: np.asarray(score, dtype=float)}),
    )
    flow.flow(fits=[ConstantFit, RisingPiecewiseLinearFit if rising_only else PiecewiseLinearFit])
    best_fit = flow.best_fits[score_name]
    border_effect = isinstance(best_fit, PiecewiseLinearFit) and best_fit.params["piecewise_linear_m"] > 0
    d_max = float(np.nanmax(distance))
    return {
        "best_fit_type": best_fit.name,
        "border_effect": bool(border_effect),
        "elbow_um": float(best_fit.params["piecewise_linear_b"]) if border_effect else None,
        "params": dict(best_fit.params),
        "observed_effect_strength": best_fit.observed_effect_strength,
        "observed_half_life": best_fit.observed_half_life,
        "half_life_um": float(best_fit.observed_half_life * d_max) if border_effect else None,
        "max_fitted_distance_um": d_max,
        "affected_fraction": best_fit.fraction_not_converged,
        "n_spots": int(len(distance)),
    }


def identify_analysis_buffer_from_filtered(
    adata,
    score="n_counts",
    row_key="array_row",
    col_key="array_col",
    grid_type="rect",
    n_counts_key="n_counts",
    bin_size_um=None,
    min_component_size=0,
    components_key="components",
    distance_key="distance_to_border",
    key_added="analysis_buffer",
    border_key="border",
    max_hole_area_um2=1024.0,
    rising_only=False,
    max_fit_distance_um=None,
    copy=False,
):
    """Flag spots within the piecewise-linear-fit elbow of the tissue border,
    fit independently per spatially-connected grid component (see
    `split_into_connected_components`) -- e.g. a TMA's individual cores are
    never pooled into one global elbow.

    Meant for a pipeline's *tissue-filtered* bins (see `load_filtered`):
    the tissue footprint is the `n_counts > 0` bins with small enclosed holes
    filled in (see `max_hole_area_um2`), border bins are footprint bins
    missing at least one footprint grid neighbour, and components are the
    connected pieces of the footprint.

    Fits `PiecewiseLinearFit` (vs. the `ConstantFit` null) of `score` against
    distance to the nearest border point, separately within each component.
    Writes a boolean column to `adata.obs[key_added]`: True for spots closer
    to their own component's border than that component's fitted breakpoint.
    A component gets an all-False buffer if `ConstantFit` wins (no detected
    effect), or if the winning piecewise fit has slope `m <= 0` -- i.e. the
    score *falls* into the tissue, which is a trend but not a border
    depression to buffer. Each component's `per_component` entry records
    `border_effect` (True only for a winning piecewise fit with `m > 0`) and
    `elbow_um` (its breakpoint, else None) alongside the raw fit `params`.

    Parameters
    ----------
    adata : AnnData
    score : str or array-like, default "n_counts"
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
    n_counts_key : str or None, default "n_counts"
        `.obs` column of per-spot counts. Spots with `n_counts <= 0` are
        excluded before splitting into components and computing distances
        (see `split_into_connected_components`) -- an empty grid position
        never counts as real tissue. Pass None to keep every spot.
    bin_size_um : float, optional
        Physical size (um) of one grid step, passed to
        `distance_to_grid_border`. Defaults to
        `adata.uns["bosperrus"]["bin_size_um"]` (set by `load_filtered`) if
        present, else 1.0 (distances/`params` in raw grid-step units).
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
        per-component). The parameters used to compute it are stored
        alongside it (`adata.uns[f"{components_key}_bosperrus_params"]`); if
        a later call changes `row_key`/`n_counts_key`/`grid_type`/
        `min_component_size`, that mismatch is detected and the column is
        recomputed (with a warning) rather than silently reused stale. This
        check only applies to a column bosperrus itself computed before --
        an externally-supplied column (no such `.uns` entry) is always
        trusted as-is, unconditionally.
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
    max_hole_area_um2 : float or None, default 1024.0
        Enclosed holes of the `n_counts > 0` footprint up to this area (in
        `bin_size_um` units squared; 1024 um^2 = 16 bins at 8um) are filled
        before finding borders, so isolated empty bins inside the tissue
        (capture dropout) don't each become a ring of internal border. Bins
        inside a filled hole that have a row (`n_counts <= 0`) are kept and
        fit with their observed counts; hole positions without any row only
        count as tissue for adjacency. Larger holes (vessels, lumens, tears)
        stay borders. See `small_grid_holes`. "rect" grids only; pass None to
        disable.
    rising_only : bool, default False
        Fit `RisingPiecewiseLinearFit` (m >= 0) instead of the unconstrained
        `PiecewiseLinearFit`: the piecewise model can only rise from the border
        and plateau, so a falling interior trend can't win and its best rising
        shape (e.g. a narrow edge dip) is fitted instead.
    max_fit_distance_um : float, optional
        Fit only spots at most this far from the border (in `bin_size_um`
        units), so a narrow edge effect isn't swamped by tissue-scale gradients
        deeper inside. Spots farther in are never in the buffer. None (default)
        fits every spot. `interior_window_test` needs it: its interior windows
        are squares of side 2 * `max_fit_distance_um`, so they span the same
        range of distances as the fit.
    copy : bool, default False
        If True, return a modified copy of `adata` instead of mutating in place.

    Returns
    -------
    AnnData or None
        The modified AnnData if `copy=True`, else None (`adata` is mutated in place).
        `adata.uns[f"{key_added}_fit"]` also records `max_hole_area_um2`,
        `n_filled_hole_bins` (rows kept only because they lie in a filled
        hole) and `filled_hole_positions` (`array_row`/`array_col`/
        `components` of the hole positions without a row), and the call's
        parameters (`params`) so `interior_window_test` can fit its interior
        windows the same way.
    """
    _require_anndata()
    adata = adata.copy() if copy else adata
    if bin_size_um is None:
        bin_size_um = float(adata.uns.get("bosperrus", {}).get("bin_size_um", 1.0))

    components, distance, is_border, virtual = _components_and_distance(
        adata, row_key, col_key, grid_type, n_counts_key, bin_size_um,
        min_component_size, components_key, distance_key, max_hole_area_um2=max_hole_area_um2,
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
        if max_fit_distance_um is not None:
            mask &= distance <= max_fit_distance_um
        if not mask.any():
            continue
        info = fit_border_profile(distance[mask], score_values[mask], rising_only=rising_only, score_name=score_name)
        if info["border_effect"]:
            buffer[mask] = distance[mask] < info["elbow_um"]
        per_component[int(label)] = info

    adata.obs[key_added] = buffer
    if n_counts_key is not None:
        n_filled = int(((adata.obs[n_counts_key].to_numpy() <= 0) & (components >= 0)).sum())
    else:
        n_filled = 0
    adata.uns[f"{key_added}_fit"] = {
        "grid_type": grid_type, "bin_size_um": bin_size_um, "per_component": per_component,
        "max_hole_area_um2": max_hole_area_um2, "n_filled_hole_bins": n_filled,
        "filled_hole_positions": virtual,
        "params": {
            "score": score_name if isinstance(score, str) else None, "row_key": row_key, "col_key": col_key,
            "grid_type": grid_type, "n_counts_key": n_counts_key, "bin_size_um": bin_size_um,
            "components_key": components_key, "distance_key": distance_key, "border_key": border_key,
            "max_hole_area_um2": max_hole_area_um2, "rising_only": rising_only,
            "max_fit_distance_um": max_fit_distance_um,
        },
    }

    return adata if copy else None


def interior_window_test(
    adata,
    key_added="analysis_buffer",
    score=None,
    margin_um=None,
    n_windows=200,
    min_windows=19,
    max_overlap=0.5,
    min_half_life_um=0.0,
    alpha=0.05,
    drop_non_significant=True,
    seed=0,
    copy=False,
):
    """Test each component's border effect against interior windows of the same component.

    Run after `identify_analysis_buffer_from_filtered` (with `max_fit_distance_um`).
    For every component with a border effect (a rising piecewise-linear elbow),
    square windows are placed inside that same component, away from its border,
    and each window is fitted exactly like the border: the score against the
    distance to the window's own edge, with the same fit models. A window has no
    physical border, so any elbow found there comes from tissue structure. The
    component's elbow is significant if it is steeper than the windows' elbows:
    the statistic is -effect strength / half-life, the drop per um, which is
    large only for a dip that is both deep and narrow (windows without a rising
    elbow have steepness 0); p = (1 + #windows at least as steep) / (1 + #windows)
    (see `bosperrus._border_test`).

    Windows are squares of side 2 * `max_fit_distance_um` (rounded to whole
    bins), so their distances span the same range as the border fit. A window
    is valid if every grid position in it belongs to the component (positions
    in filled holes count) and every spot in it is at least `margin_um` from the
    component's border (by `distance_key`). Up to `n_windows` are drawn at
    random from the valid positions such that no two share more than
    `max_overlap` of their area.

    Parameters
    ----------
    adata : AnnData
        After `identify_analysis_buffer_from_filtered(..., max_fit_distance_um=...)`
        with the same `key_added`; its components, distances and fit settings
        are reused.
    key_added : str, default "analysis_buffer"
        The buffer to test (`adata.obs[key_added]`, `adata.uns[f"{key_added}_fit"]`).
    score : array-like, optional
        Only needed if the buffer was fitted on an array rather than an `.obs`
        column: the same array.
    margin_um : float, optional
        Minimum distance of every window spot from the component's border.
        Default: `max_fit_distance_um`, i.e. windows lie in the interior that
        the border fit does not use.
    n_windows : int, default 200
        Maximum number of windows per component.
    min_windows : int, default 19
        Components where fewer windows fit (under the overlap cap) get
        `p_value` NaN and a warning, and their buffer is kept as fitted. The
        smallest reachable p-value is 1 / (n + 1), so 19 windows allow p = 0.05.
        With the defaults (windows of side 2D for fit depth D, margin D,
        max_overlap 0.5) a square component needs a side of roughly 8 D
        (about 2 mm for D = 250 um).
    max_overlap : float, default 0.5
        Largest shared area fraction of any two windows (0: disjoint windows,
        1: no cap).
    min_half_life_um : float, default 0.0
        Floor of the half-life in the steepness statistic, for the component
        and its windows alike. A fit whose knot sits at the first bin has a
        half-life of half a bin, so at fine bin sizes shallow one-bin steps in
        the windows get extreme steepness. Set it to the precision of the
        border (e.g. the tissue mask's pixel size).
    alpha : float, default 0.05
        Significance level.
    drop_non_significant : bool, default True
        Clear the buffer (`adata.obs[key_added]` = False) of components whose
        border effect has `p_value > alpha`.
    seed : int, default 0
        Seeds the window placement (per component: `[seed, label]`).
    copy : bool, default False
        If True, return a modified copy of `adata` instead of mutating in place.

    Returns
    -------
    AnnData or None
        The modified AnnData if `copy=True`, else None. Each entry of
        `adata.uns[f"{key_added}_fit"]["per_component"]` gains `p_value` (NaN
        without a border effect or without enough windows), `significant`,
        `buffer_dropped`, `steepness`, `n_windows`, `n_window_positions` (all
        valid positions, before the overlap cap), `p_min` (1 / (n_windows + 1))
        and `windows` (per window: grid offset `row0`/`col0`,
        `effect_strength`, `half_life_um`, `steepness`). The test settings are stored in
        `adata.uns[f"{key_added}_test"]`.
    """
    _require_anndata()
    adata = adata.copy() if copy else adata
    fit_key = f"{key_added}_fit"
    if fit_key not in adata.uns or "params" not in adata.uns[fit_key]:
        raise ValueError(f"adata.uns[{fit_key!r}] with fit params not found -- run "
                         f"identify_analysis_buffer_from_filtered(..., key_added={key_added!r}) first.")
    fit = adata.uns[fit_key]
    prm = fit["params"]
    if prm["grid_type"] != "rect":
        raise NotImplementedError("Interior windows are only implemented for grid_type='rect'.")
    if prm["max_fit_distance_um"] is None:
        raise ValueError("The buffer was fitted without max_fit_distance_um; the interior windows need it "
                         "(window side = 2 * max_fit_distance_um). Rerun identify_analysis_buffer_from_filtered "
                         "with max_fit_distance_um.")
    if prm["components_key"] is None or prm["distance_key"] is None:
        raise ValueError("The buffer was fitted with components_key=None or distance_key=None; the test needs "
                         "both columns in adata.obs.")
    if score is None:
        if prm["score"] is None:
            raise ValueError("The buffer was fitted on an array score; pass the same array as `score`.")
        score_values = adata.obs[prm["score"]].to_numpy(dtype=float)
    else:
        score_values = np.asarray(score, dtype=float)

    bin_size = float(prm["bin_size_um"])
    max_fit = float(prm["max_fit_distance_um"])
    margin_um = max_fit if margin_um is None else float(margin_um)
    if not 0 <= max_overlap <= 1:
        raise ValueError(f"max_overlap must lie in [0, 1], got {max_overlap}.")
    size = int(round(2 * max_fit / bin_size))
    offsets = np.arange(size)
    edge_steps = np.minimum(offsets, size - 1 - offsets)
    window_distance = np.minimum.outer(edge_steps, edge_steps) * bin_size  # distance to the window's own edge

    row = adata.obs[prm["row_key"]].to_numpy(dtype=np.int64)
    col = adata.obs[prm["col_key"]].to_numpy(dtype=np.int64)
    components = adata.obs[prm["components_key"]].to_numpy()
    distance = adata.obs[prm["distance_key"]].to_numpy(dtype=float)
    virtual = fit["filled_hole_positions"]
    v_row = np.asarray(virtual["array_row"], dtype=np.int64)
    v_col = np.asarray(virtual["array_col"], dtype=np.int64)
    v_comp = np.asarray(virtual["components"])
    buffer = adata.obs[key_added].to_numpy().copy()

    for label, info in fit["per_component"].items():
        label = int(label)
        info.update(p_value=np.nan, p_min=np.nan, significant=False, buffer_dropped=False,
                    steepness=_border_test.steepness(info, min_half_life_um), n_windows=0, n_window_positions=0,
                    windows=[])
        if not info["border_effect"]:
            continue
        member = np.flatnonzero(components == label)
        vr, vc = v_row[v_comp == label], v_col[v_comp == label]
        all_r, all_c = np.concatenate([row[member], vr]), np.concatenate([col[member], vc])
        r0, c0 = all_r.min(), all_c.min()
        shape = (all_r.max() - r0 + 1, all_c.max() - c0 + 1)
        index = np.full(shape, -1, dtype=np.int64)  # adata row of each grid position (-1: none)
        index[row[member] - r0, col[member] - c0] = member
        eligible = np.zeros(shape, dtype=bool)
        eligible[row[member] - r0, col[member] - c0] = distance[member] >= margin_um  # NaN compares False
        eligible[vr - r0, vc - c0] = True  # filled hole positions without a spot
        corners = _border_test.window_corners(eligible, size)
        info["n_window_positions"] = int(len(corners))
        rng = np.random.default_rng([seed, label])
        chosen = _border_test.spread_corners(corners, size, n_windows, max_overlap, rng)
        if len(chosen) < min_windows:
            warnings.warn(
                f"Component {label}: only {len(chosen)} interior windows of {size} bins fit at least "
                f"{margin_um:g} um from the border with max_overlap={max_overlap:g} (< min_windows={min_windows}) "
                f"-- p_value NaN, buffer kept as fitted.",
                UserWarning, stacklevel=2,
            )
            continue
        windows = []
        for wr, wc in chosen:
            idx = index[wr:wr + size, wc:wc + size]
            spot = idx >= 0
            values = score_values[idx[spot]]
            ok = np.isfinite(values)
            w = fit_border_profile(window_distance[spot][ok], values[ok], rising_only=prm["rising_only"])
            eff, half = (float(w["observed_effect_strength"]), float(w["half_life_um"])) if w["border_effect"] \
                else (np.nan, np.nan)
            windows.append({"row0": int(wr + r0), "col0": int(wc + c0), "effect_strength": eff,
                            "half_life_um": half, "steepness": _border_test.steepness(w, min_half_life_um)})
        p = _border_test.window_p_value(info["steepness"], [w["steepness"] for w in windows])
        p_min = 1 / (1 + len(windows))
        if p_min > alpha:
            warnings.warn(
                f"Component {label}: with {len(windows)} windows the smallest reachable p-value is {p_min:.3g} "
                f"> alpha={alpha:g}, so the border effect cannot be significant.",
                UserWarning, stacklevel=2,
            )
        info.update(p_value=p, p_min=p_min, significant=bool(p <= alpha), n_windows=len(windows), windows=windows)
        if drop_non_significant and p > alpha:
            buffer[components == label] = False
            info["buffer_dropped"] = True

    adata.obs[key_added] = buffer
    adata.uns[f"{key_added}_test"] = {
        "method": "interior_windows", "window_size_um": size * bin_size, "margin_um": margin_um,
        "statistic": "steepness", "min_half_life_um": min_half_life_um, "n_windows": n_windows,
        "min_windows": min_windows, "max_overlap": max_overlap,
        "alpha": alpha,
        "drop_non_significant": drop_non_significant, "seed": seed,
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
    `identify_analysis_buffer_from_filtered`) -- e.g. a TMA's individual cores each get
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
    min_component_size, components_key, distance_key : see `identify_analysis_buffer_from_filtered`.
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

    components, distance, is_border, _ = _components_and_distance(
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


# Above this many bins, quantify_diffusion_* refuses to build the dense
# zero-filled grid (every bin is fit individually; e.g. Stereo-seq at 0.5um
# is ~5e8 bins).
_DEFAULT_MAX_BINS = 200_000_000


def _fit_diffusion(distance_um, n_counts, resolution_um):
    """Shared fitting core of quantify_diffusion_visium_HD/_stereoseq:
    `ExponentialDecayFit` vs. the `ConstantFit` null (AIC, via `Flow`) of
    per-bin total counts against distance outside the mask, on every bin
    with distance > 0 -- zero-count bins included (both callers pass the
    full, dense capture-area grid).

    Returns (result dict, best Fit instance)."""
    distance_um = np.asarray(distance_um, dtype=float)
    n_counts = np.asarray(n_counts, dtype=float)
    outside = distance_um > 0
    if not outside.any():
        raise ValueError("No bins outside the tissue mask -- nothing to fit.")

    flow = Flow.from_distances_and_scores(
        distances=pd.Series(distance_um[outside], name="distance_outside_mask"),
        scores=pd.DataFrame({"n_counts": n_counts[outside]}),
    )
    flow.flow(fits=[ConstantFit, ExponentialDecayFit])
    best_fit = flow.best_fits["n_counts"]

    alpha = beta = decay_length_um = None
    if isinstance(best_fit, ExponentialDecayFit) and best_fit._converged:
        # alpha: amplitude per unit area (counts/um^2), so differently sized
        # bins are comparable; beta: decay rate (1/um), distances are in um.
        alpha = float(best_fit.params["exponential_decay_a"]) / resolution_um ** 2
        beta = float(best_fit.params["exponential_decay_b"])
        decay_length_um = 1.0 / beta

    n_counts_total = float(n_counts.sum())
    n_counts_outside = float(n_counts[outside].sum())
    result = {
        "resolution_um": float(resolution_um),
        "best_fit_type": best_fit.name,
        "params": {k: float(v) for k, v in best_fit.params.items()},
        "aic": float(best_fit.AIC),
        "alpha": alpha,
        "beta": beta,
        "decay_length_um": decay_length_um,
        # counts-weighted: fraction of all counts (UMIs/MIDs) in the capture
        # area that landed outside the mask -- not a fraction of bins
        "perc_counts_outside": 100.0 * n_counts_outside / n_counts_total if n_counts_total > 0 else float("nan"),
        "n_bins_total": int(len(n_counts)),
        "n_bins_outside": int(outside.sum()),
        "n_counts_total": n_counts_total,
        "n_counts_outside": n_counts_outside,
    }
    return result, best_fit


def _check_n_bins(n_bins, max_bins, resolution_um):
    if n_bins > max_bins:
        raise ValueError(
            f"{n_bins:,} bins at {resolution_um}um exceeds max_bins={max_bins:,} -- every bin of the "
            f"capture area (zeros included) is fit individually. Use a coarser resolution, or raise "
            f"max_bins if you really have the memory."
        )


def quantify_diffusion_visium_HD(path, resolution=8.0, mask_kwargs=None, return_data=False,
                                 max_bins=_DEFAULT_MAX_BINS):
    """RNA diffusion outside an image-only tissue mask, for one Visium HD
    sample (see `quantify_diffusion_from_raw` for the returned quantities).

    - Mask: `segment_tissue_from_rgb` on Space Ranger's own hires H&E image
      (`square_{res}um/spatial/tissue_hires_image.png`) -- purely
      image-derived, unlike `tissue_positions.parquet`'s `in_tissue`.
    - Counts: `square_{res}um/raw_feature_bc_matrix.h5` summed over genes --
      every bin of the capture area, zero-count bins included.
    - Distance: each bin centre (`pxl_*_in_fullres * tissue_hires_scalef`)
      looked up in the mask's distance transform (`distance_to_mask`), in um
      (`microns_per_pixel / tissue_hires_scalef` per hires pixel). The mask
      is zero-padded first so bins beyond the image edge get their true
      distance instead of being clipped onto the image border.

    Parameters
    ----------
    path : str or Path
        Space Ranger `binned_outputs/` directory.
    resolution : {2, 8, 16}, default 8
    mask_kwargs : dict, optional
        Forwarded to `segment_tissue_from_rgb` (default: its own defaults --
        single Otsu threshold after a sigma=8 blur, closing radius 10, holes
        < 50000 px filled, objects < 3000 px removed).
    return_data : bool, default False
        See `quantify_diffusion_from_raw`.
    max_bins : int, default 2e8
        See `quantify_diffusion_from_raw`.
    """
    technology, resolution, _ = _readers.validate_technology_and_resolution("visium_hd", resolution)
    mask_kwargs = dict(mask_kwargs or {})
    bin_dir = _readers.visium_bin_dir(path, resolution)

    image = _readers.read_visium_hires_image(bin_dir)
    mask = segment_tissue_from_rgb(image, **mask_kwargs)
    scalefactors = _readers.read_visium_scalefactors(bin_dir)
    scalef = float(scalefactors["tissue_hires_scalef"])
    mask_pixel_size_um = float(scalefactors["microns_per_pixel"]) / scalef

    barcodes, n_counts = _readers.read_10x_h5_total_counts(bin_dir / "raw_feature_bc_matrix.h5")
    _check_n_bins(len(barcodes), max_bins, resolution)
    positions = _readers.read_visium_positions(bin_dir, barcodes)
    mask_row = positions["pxl_row_in_fullres"].to_numpy(dtype=float) * scalef
    mask_col = positions["pxl_col_in_fullres"].to_numpy(dtype=float) * scalef

    # pad so every bin centre falls inside the (padded) mask: distance_to_mask
    # clips out-of-range coordinates onto the edge, which would give bins
    # beyond the image the distance of the image border instead of their own
    pad_top = max(0, int(np.ceil(-mask_row.min()))) + 1
    pad_left = max(0, int(np.ceil(-mask_col.min()))) + 1
    pad_bottom = max(0, int(np.ceil(mask_row.max())) - (mask.shape[0] - 1)) + 1
    pad_right = max(0, int(np.ceil(mask_col.max())) - (mask.shape[1] - 1)) + 1
    padded = np.pad(mask, ((pad_top, pad_bottom), (pad_left, pad_right)), constant_values=False)
    distance_um = distance_to_mask(
        np.stack([mask_row + pad_top, mask_col + pad_left], axis=1), padded, pixel_size_um=mask_pixel_size_um,
    ).to_numpy()
    del padded

    result, best_fit = _fit_diffusion(distance_um, n_counts, resolution)
    result = {"technology": technology, "path": str(path), **result,
              "mask_kwargs": mask_kwargs, "mask_pixel_size_um": mask_pixel_size_um,
              "mask_tissue_fraction": float(mask.mean())}
    if not return_data:
        return result
    data = {
        "image": image, "mask": mask, "mask_pixel_size_um": mask_pixel_size_um,
        "mask_row": mask_row, "mask_col": mask_col,
        "array_row": positions["array_row"].to_numpy().astype(np.int64),
        "array_col": positions["array_col"].to_numpy().astype(np.int64),
        "distance_um": distance_um, "n_counts": n_counts, "best_fit": best_fit,
    }
    return result, data


def quantify_diffusion_stereoseq(path, resolution=8.0, mask_kwargs=None, return_data=False,
                                 max_bins=_DEFAULT_MAX_BINS):
    """RNA diffusion outside an image-only tissue mask, for one Stereo-seq
    (STOmics) sample (see `quantify_diffusion_from_raw` for the returned
    quantities).

    - Mask: `segment_tissue_from_ssdna` on SAW's registered ssDNA image
      (`outs/image/{SN}_ssDNA_regist.tif`) -- purely image-derived, unlike
      SAW's own `*_tissue_cut.tif`, whose tissuecut step also takes per-DNB
      read counts as input.
    - Counts: `outs/feature_expression/{SN}.raw.gef` (bin1, the whole chip),
      pooled to `resolution` by floor division like SAW's own binN, on the
      dense chip grid -- zero-count bins included (a gef only lists DNBs
      with counts; the rest are filled in as 0). Partial bins at the far chip
      edge are dropped.
    - Distance: each bin centre looked up in the (downscaled) mask's
      distance transform (`distance_to_mask`), in um (0.5um * downscale
      factor per mask pixel). The registered image spans exactly the chip's
      DNB grid (checked), so no padding is needed.

    Parameters
    ----------
    path : str or Path
        SAW run folder (containing `outs/`).
    resolution : float, default 8
        Any integer multiple of the 0.5um DNB pitch (e.g. 8 = 16x16 DNBs,
        10 = SAW bin20).
    mask_kwargs : dict, optional
        Forwarded to `segment_tissue_from_ssdna` (default: its own defaults --
        single Otsu on a ~2000 px downscale, closing disk 6, holes filled).
    return_data : bool, default False
        See `quantify_diffusion_from_raw`.
    max_bins : int, default 2e8
        See `quantify_diffusion_from_raw`.
    """
    technology, resolution, dnbs_per_bin = _readers.validate_technology_and_resolution("stereo-seq", resolution)
    mask_kwargs = dict(mask_kwargs or {})
    outs, sn = _readers.saw_outs_and_sn(path)
    extent = _readers.saw_chip_extent(outs, sn)
    min_y, min_x, len_y, len_x = extent
    _check_n_bins((len_y // dnbs_per_bin) * (len_x // dnbs_per_bin), max_bins, resolution)

    image = _readers.read_ssdna_image(outs, sn)
    if image.shape != (len_y, len_x):
        raise ValueError(
            f"{sn}_ssDNA_regist.tif has shape {image.shape}, but the chip's DNB grid is "
            f"{(len_y, len_x)} -- can't map bins onto the image."
        )
    mask, factor = segment_tissue_from_ssdna(image, **mask_kwargs)
    small_image = None
    if return_data:
        from skimage.transform import downscale_local_mean
        small_image = downscale_local_mean(image, (factor, factor))
    del image
    mask_pixel_size_um = _readers.STEREOSEQ_DNB_PITCH_UM * factor

    dense = _readers.pool_gef_counts(outs / "feature_expression" / f"{sn}.raw.gef", dnbs_per_bin, extent)
    n_rows, n_cols = dense.shape
    array_row, array_col = np.divmod(np.arange(n_rows * n_cols), n_cols)
    n_counts = dense.ravel().astype(np.float64)
    del dense

    # bin centre in DNB units -> mask pixel coordinate: mask pixel i covers DNBs
    # [i*factor, (i+1)*factor), i.e. is centred on DNB i*factor + (factor-1)/2
    half_bin, half_px = (dnbs_per_bin - 1) / 2, (factor - 1) / 2
    mask_row = (array_row * dnbs_per_bin + half_bin - half_px) / factor
    mask_col = (array_col * dnbs_per_bin + half_bin - half_px) / factor
    distance_um = distance_to_mask(
        np.stack([mask_row, mask_col], axis=1), mask, pixel_size_um=mask_pixel_size_um,
    ).to_numpy()

    result, best_fit = _fit_diffusion(distance_um, n_counts, resolution)
    result = {"technology": technology, "path": str(path), "sn": sn, **result,
              "mask_kwargs": mask_kwargs, "mask_pixel_size_um": mask_pixel_size_um,
              "mask_tissue_fraction": float(mask.mean())}
    if not return_data:
        return result
    data = {
        "image": small_image, "mask": mask, "mask_pixel_size_um": mask_pixel_size_um,
        "mask_row": mask_row, "mask_col": mask_col, "array_row": array_row, "array_col": array_col,
        "distance_um": distance_um, "n_counts": n_counts, "best_fit": best_fit,
    }
    return result, data


def quantify_diffusion_from_raw(path, technology, resolution=8.0, mask_kwargs=None, return_data=False,
                                max_bins=_DEFAULT_MAX_BINS):
    """Quantify RNA diffusion outside the tissue from a pipeline's *raw*
    (whole capture area, unfiltered) output and an image-only tissue mask.

    Dispatches to `quantify_diffusion_visium_HD` (Space Ranger
    `binned_outputs/`) or `quantify_diffusion_stereoseq` (SAW run folder) --
    see those for how each finds its image, mask and raw counts. Both then do
    the same thing: every bin of the capture area -- zero-count bins
    included -- gets its distance (um) outside the mask, and per-bin total
    counts of the bins outside the mask (distance > 0) are fit with
    `ExponentialDecayFit` (`a * exp(-b * d)`) vs. the `ConstantFit` null,
    selected by AIC (via `Flow`).

    Parameters
    ----------
    path : str or Path
    technology : {"visium_hd", "stereo-seq"}
    resolution : float, default 8.0
        Bin size (um). Visium HD: 2, 8 or 16. Stereo-seq: any integer
        multiple of 0.5 (pooled from bin1).
    mask_kwargs : dict, optional
        Segmentation parameters, forwarded to `segment_tissue_from_rgb`
        (Visium HD) or `segment_tissue_from_ssdna` (Stereo-seq).
    return_data : bool, default False
        Also return the per-bin data behind the fit (see Returns).
    max_bins : int, default 2e8
        Refuse to build a capture-area grid larger than this (every bin is
        held in memory and fit individually).

    Returns
    -------
    result : dict
        JSON-serializable: `technology`, `path`, `resolution_um`,
        `best_fit_type`, `params`, `aic`,
        `alpha` (fitted amplitude `a` / bin area, counts/um^2),
        `beta` (fitted decay rate `b`, 1/um), `decay_length_um` (1/beta) --
        all three None unless `ExponentialDecayFit` wins and converged --
        `perc_counts_outside` (counts-weighted: % of all counts outside the
        mask), `n_bins_total`, `n_bins_outside`, `n_counts_total`,
        `n_counts_outside`, `mask_kwargs`, `mask_pixel_size_um`,
        `mask_tissue_fraction` (plus `sn` for Stereo-seq).
    data : dict, only if `return_data=True`
        `image` (the image exactly as segmented: Visium hires RGB, or the
        downscaled Stereo-seq ssDNA), `mask`, `mask_pixel_size_um`, per-bin
        `mask_row`/`mask_col` (bin centres in mask pixel coordinates; can
        lie outside the image for Visium), `array_row`/`array_col` (bin grid
        indices), `distance_um`, `n_counts`, and `best_fit` (the fitted `Fit`
        instance, e.g. for `best_fit.predict(d)`).
    """
    technology, _, _ = _readers.validate_technology_and_resolution(technology, resolution)
    fn = quantify_diffusion_visium_HD if technology == "visium_hd" else quantify_diffusion_stereoseq
    return fn(path, resolution=resolution, mask_kwargs=mask_kwargs, return_data=return_data, max_bins=max_bins)


_PREDICT_FORMULAS = {
    "Constant Fit": lambda d, p: np.full_like(np.asarray(d, dtype=float), p["constant_c"]),
    "Piecewise Linear Fit": lambda d, p: PiecewiseLinearFit.piecewise_plateau(
        d, p["piecewise_linear_b"], p["piecewise_linear_m"], p["piecewise_linear_c"]),
    "Exponential Saturation Fit": lambda d, p: ExponentialSaturationFit.exp_sat(
        d, p["exponential_saturation_a"], p["exponential_saturation_b"], p["exponential_saturation_c"]),
    "Exponential Decay Fit": lambda d, p: ExponentialDecayFit.exp_decay(
        d, p["exponential_decay_a"], p["exponential_decay_b"]),
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
    elbow curve overlaid on top (see `identify_analysis_buffer_from_filtered`, which must
    be run first -- this reads its stored results, it doesn't fit anything
    itself). One subplot per component.

    Parameters
    ----------
    adata : AnnData
    score : str or array-like
        Same score `identify_analysis_buffer_from_filtered` was run with.
    distance_key, components_key, key_added : see `identify_analysis_buffer_from_filtered`
        -- must match the values it was actually called with, since this
        reads `adata.obs[distance_key]`/`adata.obs[components_key]` and
        `adata.uns[f"{key_added}_fit"]["per_component"]`.
    ncols : int, default 4
        Subplot grid width.
    figsize : (float, float), optional
        Defaults to `(4 * ncols, 3 * nrows)`.
    **plot_fit_kwargs
        Forwarded to `plotting.plot_fit` (e.g. `bins`, `quantiles`, `line_kwargs`).

    Returns
    -------
    matplotlib.figure.Figure
    """
    _require_anndata()
    fit_key = f"{key_added}_fit"
    if fit_key not in adata.uns:
        raise KeyError(f"{fit_key!r} not found in adata.uns -- run identify_analysis_buffer_from_filtered first.")

    score_values = adata.obs[score].to_numpy() if isinstance(score, str) else np.asarray(score)
    distance = adata.obs[distance_key].to_numpy()
    components = adata.obs[components_key].to_numpy()
    per_component = adata.uns[fit_key]["per_component"]

    return _plot_per_component_fit(
        score_values, distance, components, per_component,
        xlabel="distance to border", ncols=ncols, figsize=figsize, **plot_fit_kwargs,
    )
