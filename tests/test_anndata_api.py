import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

anndata = pytest.importorskip("anndata", reason="anndata not installed")

from bosperrus.anndata_api import (
    identify_analysis_buffer, correct_layer, quantify_diffusion, plot_border_effect, plot_diffusion,
)
from bosperrus.fit import PiecewiseLinearFit, ExponentialSaturationFit, ExponentialDecayFit



def _rect_grid_adata(n_side=10):
    """n_side x n_side square grid AnnData, deterministic. Returns the AnnData plus
    the row/col arrays for building known-shape test scores/layers against."""
    rows, cols = np.meshgrid(np.arange(n_side), np.arange(n_side), indexing="ij")
    row = rows.ravel()
    col = cols.ravel()
    n = len(row)
    adata = anndata.AnnData(
        X=np.zeros((n, 2)),
        obs=pd.DataFrame({"array_row": row, "array_col": col}, index=[f"spot_{i}" for i in range(n)]),
        var=pd.DataFrame(index=["gene_a", "gene_b"]),
    )
    return adata, row, col


def _two_block_adata(n_side=30, gap=50):
    """Two disconnected n_side x n_side blocks (block b offset by `gap` on the
    row axis, so grid_edges can never connect them) -- for testing that
    components are fit independently rather than pooled into one global fit."""
    rows_a, cols_a = np.meshgrid(np.arange(n_side), np.arange(n_side), indexing="ij")
    rows_b, cols_b = np.meshgrid(np.arange(n_side), np.arange(n_side), indexing="ij")
    row = np.concatenate([rows_a.ravel(), rows_b.ravel() + gap])
    col = np.concatenate([cols_a.ravel(), cols_b.ravel()])
    n = len(row)
    adata = anndata.AnnData(
        X=np.zeros((n, 2)),
        obs=pd.DataFrame({"array_row": row, "array_col": col}, index=[f"spot_{i}" for i in range(n)]),
        var=pd.DataFrame(index=["gene_a", "gene_b"]),
    )
    block_a_mask = np.arange(n) < n_side * n_side
    return adata, row, col, block_a_mask, n_side, gap


def _distance_to_nearest_edge(row, col, n_side, row_offset=0):
    """Ground-truth distance-to-border for an n_side x n_side square grid
    (optionally offset on the row axis), used only to build synthetic scores
    with a KNOWN elbow -- independent of grid_edges/distance_to_pointset, so
    the test isn't circular."""
    local_row = row - row_offset
    return np.minimum.reduce([local_row, col, n_side - 1 - local_row, n_side - 1 - col]).astype(float)


# ---------------------------------------------------------------------------
# identify_analysis_buffer
# ---------------------------------------------------------------------------

def test_identify_analysis_buffer_no_effect_flags_nothing():
    """Pure noise, unrelated to distance -> ConstantFit should win -> buffer all False."""
    RNG = np.random.default_rng(42)
    adata, row, col = _rect_grid_adata()
    adata.obs["score"] = RNG.normal(5.0, 0.1, size=len(row))

    identify_analysis_buffer(adata, score="score", grid_type="rect")

    assert "analysis_buffer" in adata.obs
    assert adata.obs["analysis_buffer"].dtype == bool
    assert not adata.obs["analysis_buffer"].any()
    fit_info = adata.uns["analysis_buffer_fit"]
    assert 0 in fit_info["per_component"]
    assert fit_info["per_component"][0]["best_fit_type"] == "Constant Fit"


def test_identify_analysis_buffer_detects_known_elbow():
    """Score built directly from PiecewiseLinearFit's own formula with a known
    breakpoint b=5 -> fitted buffer should closely match distance < 5.

    Uses a 30x30 grid (not 10x10): distance-to-edge on a square grid only takes
    n_side//2 unique integer values, and curve_fit needs enough distinct x-values
    around the breakpoint to localize it -- a 10x10 grid (only 5 unique distances)
    left the breakpoint essentially unidentifiable, unrelated to the implementation
    (confirmed by test_fit.py::test_recovers_known_parameters already passing with
    adequate resolution)."""
    RNG = np.random.default_rng(42)
    n_side = 30
    adata, row, col = _rect_grid_adata(n_side)
    d_true = _distance_to_nearest_edge(row, col, n_side)
    b_true, m_true, c_true = 5.0, -1.0, 10.0
    signal = PiecewiseLinearFit.piecewise_plateau(d_true, b_true, m_true, c_true)
    adata.obs["score"] = signal + RNG.normal(0, 0.05, size=len(row))

    identify_analysis_buffer(adata, score="score", grid_type="rect")

    fit_info = adata.uns["analysis_buffer_fit"]["per_component"][0]
    assert fit_info["best_fit_type"] == "Piecewise Linear Fit"
    fitted_b = fit_info["params"]["piecewise_linear_b"]
    assert fitted_b == pytest.approx(b_true, abs=1.0)

    # buffer should agree with the ground-truth threshold for the vast majority of spots
    ground_truth_buffer = d_true < b_true
    agreement = (adata.obs["analysis_buffer"].to_numpy() == ground_truth_buffer).mean()
    assert agreement > 0.9


def test_identify_analysis_buffer_array_score():
    """score can be a raw array, not just an .obs column name."""
    RNG = np.random.default_rng(42)
    adata, row, col = _rect_grid_adata()
    values = RNG.normal(5.0, 0.1, size=len(row))

    identify_analysis_buffer(adata, score=values, grid_type="rect")
    assert "analysis_buffer" in adata.obs


def test_identify_analysis_buffer_copy_does_not_mutate_original():
    RNG = np.random.default_rng(42)
    adata, row, col = _rect_grid_adata()
    adata.obs["score"] = RNG.normal(5.0, 0.1, size=len(row))

    result = identify_analysis_buffer(adata, score="score", grid_type="rect", copy=True)
    assert result is not None
    assert "analysis_buffer" not in adata.obs
    assert "analysis_buffer" in result.obs


def test_identify_analysis_buffer_distance_key_overrides_computed_distance():
    """distance_key reuses that .obs column directly instead of computing via
    distance_to_grid_border -- verified by injecting an artificial distance
    column with a DIFFERENT known elbow than the true grid-based distance
    would produce, and checking the fit responds to the artificial one."""
    RNG = np.random.default_rng(42)
    n_side = 30
    adata, row, col = _rect_grid_adata(n_side)
    fake_distance = RNG.uniform(0, 20, size=len(row))  # unrelated to the real grid geometry
    adata.obs["my_distance"] = fake_distance

    b_true, m_true, c_true = 8.0, -1.0, 10.0
    signal = PiecewiseLinearFit.piecewise_plateau(fake_distance, b_true, m_true, c_true)
    adata.obs["score"] = signal + RNG.normal(0, 0.05, size=len(row))

    identify_analysis_buffer(adata, score="score", grid_type="rect", distance_key="my_distance")

    fit_info = adata.uns["analysis_buffer_fit"]["per_component"][0]
    assert fit_info["best_fit_type"] == "Piecewise Linear Fit"
    assert fit_info["params"]["piecewise_linear_b"] == pytest.approx(b_true, abs=1.5)
    np.testing.assert_array_equal(
        adata.obs["analysis_buffer"].to_numpy(), fake_distance < fit_info["params"]["piecewise_linear_b"]
    )


def test_identify_analysis_buffer_new_distance_key_gets_computed_and_written():
    """A distance_key that doesn't yet exist is computed (not an error) and
    written to that column -- distance_key is reuse-if-present, not a
    strict require-if-given override."""
    RNG = np.random.default_rng(42)
    adata, row, col = _rect_grid_adata()
    adata.obs["score"] = RNG.normal(5.0, 0.1, size=len(row))
    assert "my_new_distance" not in adata.obs

    identify_analysis_buffer(adata, score="score", grid_type="rect", distance_key="my_new_distance")

    assert "my_new_distance" in adata.obs
    assert (adata.obs["my_new_distance"] >= 0).all()


def test_identify_analysis_buffer_writes_components_and_distance_by_default():
    """Every intermediate value computed along the way (components, distance
    to border) lands in .obs by default, not just the final buffer/border
    outputs."""
    RNG = np.random.default_rng(42)
    adata, row, col = _rect_grid_adata()
    adata.obs["score"] = RNG.normal(5.0, 0.1, size=len(row))

    identify_analysis_buffer(adata, score="score", grid_type="rect")

    assert "components" in adata.obs
    assert "distance_to_border" in adata.obs
    assert (adata.obs["components"] == 0).all()  # single connected block
    assert (adata.obs["distance_to_border"] >= 0).all()


def test_identify_analysis_buffer_components_key_reuses_existing_column():
    """An existing components_key column is reused as-is, overriding
    split_into_connected_components -- verified by supplying a deliberately
    wrong labeling (splits the single connected block into two fake halves)
    and checking the fit is actually computed per that fake split."""
    RNG = np.random.default_rng(42)
    n_side = 20
    adata, row, col = _rect_grid_adata(n_side)
    adata.obs["score"] = RNG.normal(5.0, 0.1, size=len(row))
    fake_components = np.where(col < n_side // 2, 0, 1)
    adata.obs["my_components"] = fake_components

    identify_analysis_buffer(adata, score="score", grid_type="rect", components_key="my_components")

    per_component = adata.uns["analysis_buffer_fit"]["per_component"]
    assert set(per_component) == {0, 1}
    np.testing.assert_array_equal(adata.obs["my_components"].to_numpy(), fake_components)


def test_identify_analysis_buffer_normalizes_reused_components_key_dtype():
    """A reused components_key column stored as strings (e.g. from an
    earlier `adata.obs["components"] = components.astype(str)` for
    plotting) must come back out as int afterward -- otherwise a later
    re-read of adata.obs[components_key] (e.g. by plot_border_effect) can
    never match per_component's int keys via `components == label`."""
    RNG = np.random.default_rng(42)
    adata, row, col = _rect_grid_adata()
    adata.obs["score"] = RNG.normal(5.0, 0.1, size=len(row))
    adata.obs["components"] = np.zeros(len(row), dtype=int).astype(str)

    identify_analysis_buffer(adata, score="score", grid_type="rect")

    stored = adata.obs["components"].to_numpy()
    assert np.issubdtype(stored.dtype, np.integer)
    assert (stored == 0).all()


def test_identify_analysis_buffer_non_numeric_components_key_raises_clear_error():
    RNG = np.random.default_rng(42)
    adata, row, col = _rect_grid_adata()
    adata.obs["score"] = RNG.normal(5.0, 0.1, size=len(row))
    adata.obs["my_components"] = [f"core_{i}" for i in range(len(row))]

    with pytest.raises(ValueError, match="isn't usable as integer component labels"):
        identify_analysis_buffer(adata, score="score", grid_type="rect", components_key="my_components")


def test_identify_analysis_buffer_non_numeric_distance_key_raises_clear_error():
    RNG = np.random.default_rng(42)
    adata, row, col = _rect_grid_adata()
    adata.obs["score"] = RNG.normal(5.0, 0.1, size=len(row))
    adata.obs["my_distance"] = [f"far_{i}" for i in range(len(row))]

    with pytest.raises(ValueError, match="must be numeric"):
        identify_analysis_buffer(adata, score="score", grid_type="rect", distance_key="my_distance")


def test_identify_analysis_buffer_negative_distance_key_raises_clear_error():
    RNG = np.random.default_rng(42)
    adata, row, col = _rect_grid_adata()
    adata.obs["score"] = RNG.normal(5.0, 0.1, size=len(row))
    fake_distance = RNG.uniform(0, 5, size=len(row))
    fake_distance[0] = -1.0
    adata.obs["my_distance"] = fake_distance

    with pytest.raises(ValueError, match="contains negative values"):
        identify_analysis_buffer(adata, score="score", grid_type="rect", distance_key="my_distance")


def test_identify_analysis_buffer_warns_on_unreasonably_many_components():
    """A components_key that's really just per-spot noise (unique label per
    spot) should trigger the over-fragmentation warning."""
    RNG = np.random.default_rng(42)
    adata, row, col = _rect_grid_adata()
    adata.obs["score"] = RNG.normal(5.0, 0.1, size=len(row))
    adata.obs["my_components"] = np.arange(len(row))  # every spot its own "component"

    with pytest.warns(UserWarning, match="unusually fragmented"):
        identify_analysis_buffer(adata, score="score", grid_type="rect", components_key="my_components")


def test_identify_analysis_buffer_no_warning_for_reasonable_component_count():
    RNG = np.random.default_rng(42)
    adata, row, col = _rect_grid_adata()
    adata.obs["score"] = RNG.normal(5.0, 0.1, size=len(row))

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning fails the test
        identify_analysis_buffer(adata, score="score", grid_type="rect")


def test_identify_analysis_buffer_n_counts_key_excludes_zero_count_spots():
    RNG = np.random.default_rng(42)
    adata, row, col = _rect_grid_adata()
    adata.obs["score"] = RNG.normal(5.0, 0.1, size=len(row))
    n_counts = np.ones(len(row))
    n_counts[0] = 0  # exclude one spot
    adata.obs["n_counts"] = n_counts

    identify_analysis_buffer(adata, score="score", grid_type="rect", n_counts_key="n_counts")

    assert not adata.obs["analysis_buffer"].iloc[0]  # excluded spot is never in the buffer
    fit_info = adata.uns["analysis_buffer_fit"]["per_component"]
    assert fit_info[0]["n_spots"] == len(row) - 1


def test_identify_analysis_buffer_fits_components_independently():
    """Two disconnected blocks with DIFFERENT true elbows -- pooling them into
    one global fit would blur both; per-component fitting should recover
    each block's own breakpoint."""
    RNG = np.random.default_rng(42)
    adata, row, col, block_a_mask, n_side, gap = _two_block_adata()
    d_true = np.where(
        block_a_mask,
        _distance_to_nearest_edge(row, col, n_side, row_offset=0),
        _distance_to_nearest_edge(row, col, n_side, row_offset=gap),
    )
    b_a, b_b, m_true, c_true = 4.0, 10.0, -1.0, 10.0
    signal = np.where(
        block_a_mask,
        PiecewiseLinearFit.piecewise_plateau(d_true, b_a, m_true, c_true),
        PiecewiseLinearFit.piecewise_plateau(d_true, b_b, m_true, c_true),
    )
    adata.obs["score"] = signal + RNG.normal(0, 0.05, size=len(row))

    identify_analysis_buffer(adata, score="score", grid_type="rect")

    per_component = adata.uns["analysis_buffer_fit"]["per_component"]
    assert len(per_component) == 2
    fitted_bs = sorted(
        info["params"]["piecewise_linear_b"]
        for info in per_component.values()
        if info["best_fit_type"] == "Piecewise Linear Fit"
    )
    assert len(fitted_bs) == 2
    assert fitted_bs[0] == pytest.approx(min(b_a, b_b), abs=1.5)
    assert fitted_bs[1] == pytest.approx(max(b_a, b_b), abs=1.5)


def test_identify_analysis_buffer_no_surviving_components_raises():
    RNG = np.random.default_rng(42)
    adata, row, col = _rect_grid_adata()
    adata.obs["score"] = RNG.normal(5.0, 0.1, size=len(row))
    adata.obs["n_counts"] = np.zeros(len(row))
    with pytest.raises(ValueError, match="No connected components"):
        identify_analysis_buffer(adata, score="score", grid_type="rect", n_counts_key="n_counts")


def test_identify_analysis_buffer_writes_border_column():
    """Corner spots (degree 2 on a rect grid) are border; the center-ish
    interior spot (degree 4) is not; a spot excluded via n_counts_key is
    never flagged as border even if it would otherwise be one."""
    RNG = np.random.default_rng(42)
    n_side = 10
    adata, row, col = _rect_grid_adata(n_side)
    adata.obs["score"] = RNG.normal(5.0, 0.1, size=len(row))
    n_counts = np.ones(len(row))
    corner_idx = 0  # (row=0, col=0)
    interior_idx = np.flatnonzero((row == n_side // 2) & (col == n_side // 2))[0]
    n_counts[corner_idx] = 0  # this corner is excluded -- should read False despite being a true border node
    adata.obs["n_counts"] = n_counts

    identify_analysis_buffer(adata, score="score", grid_type="rect", n_counts_key="n_counts")

    assert "border" in adata.obs
    assert adata.obs["border"].dtype == bool
    assert not adata.obs["border"].iloc[corner_idx]  # excluded, despite true topological border status
    assert not adata.obs["border"].iloc[interior_idx]
    other_corner = np.flatnonzero((row == n_side - 1) & (col == n_side - 1))[0]
    assert adata.obs["border"].iloc[other_corner]


def test_identify_analysis_buffer_border_key_is_configurable():
    RNG = np.random.default_rng(42)
    adata, row, col = _rect_grid_adata()
    adata.obs["score"] = RNG.normal(5.0, 0.1, size=len(row))

    identify_analysis_buffer(adata, score="score", grid_type="rect", border_key="is_border_spot")
    assert "is_border_spot" in adata.obs
    assert "border" not in adata.obs


# ---------------------------------------------------------------------------
# correct_layer
# ---------------------------------------------------------------------------

def test_correct_layer_no_effect_passes_through_unchanged():
    RNG = np.random.default_rng(42)
    adata, row, col = _rect_grid_adata()
    adata.X = np.column_stack([
        RNG.normal(5.0, 0.05, size=len(row)),
        RNG.normal(3.0, 0.05, size=len(row)),
    ])

    correct_layer(adata, grid_type="rect")

    assert "bosperrus_corrected" in adata.layers
    np.testing.assert_allclose(adata.layers["bosperrus_corrected"], adata.X, atol=1e-8)
    fit_quality = adata.uns["bosperrus_corrected_fit_quality"][0]
    assert (fit_quality.loc["best_fit_type"] == "Constant Fit").all()


def test_correct_layer_shifts_detected_effect_toward_asymptote():
    RNG = np.random.default_rng(42)
    from bosperrus.fit import ExponentialSaturationFit

    n_side = 10
    adata, row, col = _rect_grid_adata(n_side)
    d_true = _distance_to_nearest_edge(row, col, n_side)
    a_true, b_true, c_true = 4.0, 0.5, 1.0
    effect_col = ExponentialSaturationFit.exp_sat(d_true, a_true, b_true, c_true) + RNG.normal(0, 0.05, len(row))
    flat_col = RNG.normal(5.0, 0.05, size=len(row))
    adata.X = np.column_stack([effect_col, flat_col])

    correct_layer(adata, grid_type="rect")

    corrected = adata.layers["bosperrus_corrected"]
    fit_quality = adata.uns["bosperrus_corrected_fit_quality"][0]
    # effect column (gene_a) should have been detected and actually changed
    assert fit_quality.loc["best_fit_type", "gene_a"] == "Exponential Saturation Fit"
    assert not np.allclose(corrected[:, 0], adata.X[:, 0])
    # near-border spots (small d_true) should be shifted toward the asymptote a+c,
    # i.e. their corrected value should move further from their raw value than
    # already-converged interior spots
    border_mask = d_true < 1.0
    interior_mask = d_true > d_true.max() - 1.0
    border_shift = np.abs(corrected[border_mask, 0] - adata.X[border_mask, 0]).mean()
    interior_shift = np.abs(corrected[interior_mask, 0] - adata.X[interior_mask, 0]).mean()
    assert border_shift > interior_shift
    # flat column (gene_b) should be untouched
    assert fit_quality.loc["best_fit_type", "gene_b"] == "Constant Fit"
    np.testing.assert_allclose(corrected[:, 1], adata.X[:, 1], atol=1e-8)


def test_correct_layer_named_layer_not_X():
    RNG = np.random.default_rng(42)
    adata, row, col = _rect_grid_adata()
    adata.layers["counts"] = np.column_stack([
        RNG.normal(5.0, 0.05, size=len(row)),
        RNG.normal(3.0, 0.05, size=len(row)),
    ])

    correct_layer(adata, layer="counts", grid_type="rect")
    assert "bosperrus_corrected" in adata.layers
    np.testing.assert_allclose(adata.layers["bosperrus_corrected"], adata.layers["counts"], atol=1e-8)


def test_correct_layer_copy_does_not_mutate_original():
    RNG = np.random.default_rng(42)
    adata, row, col = _rect_grid_adata()
    adata.X = np.column_stack([
        RNG.normal(5.0, 0.05, size=len(row)),
        RNG.normal(3.0, 0.05, size=len(row)),
    ])

    result = correct_layer(adata, grid_type="rect", copy=True)
    assert result is not None
    assert "bosperrus_corrected" not in adata.layers
    assert "bosperrus_corrected" in result.layers


def test_correct_layer_writes_border_column():
    RNG = np.random.default_rng(42)
    n_side = 10
    adata, row, col = _rect_grid_adata(n_side)
    adata.X = np.column_stack([
        RNG.normal(5.0, 0.05, size=len(row)),
        RNG.normal(3.0, 0.05, size=len(row)),
    ])

    correct_layer(adata, grid_type="rect")

    assert "border" in adata.obs
    assert adata.obs["border"].dtype == bool
    assert adata.obs["border"].iloc[0]  # corner (0,0), degree 2
    interior_idx = np.flatnonzero((row == n_side // 2) & (col == n_side // 2))[0]
    assert not adata.obs["border"].iloc[interior_idx]


def test_correct_layer_writes_components_and_distance_by_default():
    RNG = np.random.default_rng(42)
    adata, row, col = _rect_grid_adata()
    adata.X = np.column_stack([
        RNG.normal(5.0, 0.05, size=len(row)),
        RNG.normal(3.0, 0.05, size=len(row)),
    ])

    correct_layer(adata, grid_type="rect")

    assert "components" in adata.obs
    assert "distance_to_border" in adata.obs
    assert (adata.obs["components"] == 0).all()


def test_correct_layer_fits_components_independently():
    """Two disconnected blocks: gene_a has a real effect only in block a,
    gene_b has a real effect only in block b -- per-component fitting should
    detect each independently instead of one pooled (and wrong) verdict."""
    RNG = np.random.default_rng(42)
    from bosperrus.fit import ExponentialSaturationFit

    adata, row, col, block_a_mask, n_side, gap = _two_block_adata(n_side=15)
    d_true = np.where(
        block_a_mask,
        _distance_to_nearest_edge(row, col, n_side, row_offset=0),
        _distance_to_nearest_edge(row, col, n_side, row_offset=gap),
    )
    a_true, b_true, c_true = 4.0, 0.5, 1.0
    effect = ExponentialSaturationFit.exp_sat(d_true, a_true, b_true, c_true)
    flat = np.full(len(row), 5.0)

    gene_a = np.where(block_a_mask, effect, flat) + RNG.normal(0, 0.05, len(row))
    gene_b = np.where(block_a_mask, flat, effect) + RNG.normal(0, 0.05, len(row))
    adata.X = np.column_stack([gene_a, gene_b])

    correct_layer(adata, grid_type="rect")

    fit_quality = adata.uns["bosperrus_corrected_fit_quality"]
    assert len(fit_quality) == 2
    # identify which label corresponds to which block by checking which fit detected an effect
    detected_effect_genes = {
        label: {g for g in ["gene_a", "gene_b"] if fq.loc["best_fit_type", g] == "Exponential Saturation Fit"}
        for label, fq in fit_quality.items()
    }
    all_detected = set().union(*detected_effect_genes.values())
    assert all_detected == {"gene_a", "gene_b"}
    # each component should have detected an effect for exactly one gene, not both/neither
    for genes in detected_effect_genes.values():
        assert len(genes) == 1


# ---------------------------------------------------------------------------
# quantify_diffusion
# ---------------------------------------------------------------------------

def test_quantify_diffusion_mask_distance_key_overrides_computed_distance():
    """mask_distance_key reuses that .obs column directly instead of
    computing via image_masks.get_tissue_mask/distances.distance_to_mask --
    verified with a known ExponentialSaturationFit signal (a < 0, since
    counts decay AWAY from tissue) against an injected distance column, and
    checked against the alpha=-a/bin_size_um**2, beta=b convention."""
    RNG = np.random.default_rng(42)
    n_side = 30
    adata, row, col = _rect_grid_adata(n_side)
    fake_distance = RNG.uniform(0, 20, size=len(row))
    adata.obs["my_mask_distance"] = fake_distance

    a_true, b_true, c_true = -4.0, 0.5, 10.0
    signal = ExponentialSaturationFit.exp_sat(fake_distance, a_true, b_true, c_true)
    adata.obs["n_counts"] = signal + RNG.normal(0, 0.05, size=len(row))

    bin_size_um = 8.0
    quantify_diffusion(
        adata, library_id="unused", score="n_counts", grid_type="rect",
        bin_size_um=bin_size_um, mask_distance_key="my_mask_distance",
    )

    fit_info = adata.uns["diffusion_fit"]["per_component"][0]
    assert fit_info["best_fit_type"] == "Exponential Saturation Fit"
    assert fit_info["alpha"] == pytest.approx(-a_true / bin_size_um**2, rel=0.2)
    assert fit_info["beta"] == pytest.approx(b_true, rel=0.3)


def test_quantify_diffusion_picks_exponential_decay_when_floor_is_zero():
    """When the true far-field value genuinely is 0 (no c offset),
    ExponentialDecayFit's one-fewer-parameter should win over
    ExponentialSaturationFit on AIC -- and alpha should come out positive
    directly from its `a` (no sign flip needed, unlike ExponentialSaturationFit's).

    Needs low noise AND a modest n to be decisive: log_likelihood is
    -0.5*n*log(sigma2), so at large n (e.g. 900 spots) even a tiny relative
    SSE reduction from ExponentialSaturationFit's extra, nearly-redundant c
    (its curve_fit still finds a+c very close to 0, i.e. essentially the
    same curve as ExponentialDecayFit -- but "nearly the same" still isn't
    "exactly the same" on any single finite noisy sample) gets amplified
    into a real AIC edge that overcomes the fixed 2-unit parameter-count
    penalty, regardless of how low the noise is -- an honest, expected
    large-N property of AIC here, not a bug, and not something a smaller
    per-component sample (typical for this kind of per-component fit) runs
    into: confirmed empirically, decay wins reliably once n <~ 225."""
    RNG = np.random.default_rng(42)
    n_side = 15
    adata, row, col = _rect_grid_adata(n_side)
    fake_distance = RNG.uniform(0, 20, size=len(row))
    adata.obs["my_mask_distance"] = fake_distance

    a_true, b_true = 4.0, 0.5
    signal = ExponentialDecayFit.exp_decay(fake_distance, a_true, b_true)
    adata.obs["n_counts"] = signal + RNG.normal(0, 0.001, size=len(row))

    bin_size_um = 8.0
    quantify_diffusion(
        adata, library_id="unused", score="n_counts", grid_type="rect",
        bin_size_um=bin_size_um, mask_distance_key="my_mask_distance",
    )

    fit_info = adata.uns["diffusion_fit"]["per_component"][0]
    assert fit_info["best_fit_type"] == "Exponential Decay Fit"
    assert fit_info["alpha"] == pytest.approx(a_true / bin_size_um**2, rel=0.2)
    assert fit_info["beta"] == pytest.approx(b_true, rel=0.3)


def test_quantify_diffusion_no_effect_gives_none_alpha_beta():
    RNG = np.random.default_rng(42)
    adata, row, col = _rect_grid_adata()
    adata.obs["my_mask_distance"] = RNG.uniform(0, 10, size=len(row))
    adata.obs["n_counts"] = RNG.normal(5.0, 0.1, size=len(row))

    quantify_diffusion(
        adata, library_id="unused", score="n_counts", grid_type="rect", mask_distance_key="my_mask_distance",
    )

    fit_info = adata.uns["diffusion_fit"]["per_component"][0]
    assert fit_info["best_fit_type"] == "Constant Fit"
    assert fit_info["alpha"] is None
    assert fit_info["beta"] is None


def test_quantify_diffusion_writes_components_and_distance_by_default():
    RNG = np.random.default_rng(42)
    adata, row, col = _rect_grid_adata()
    adata.obs["n_counts"] = RNG.normal(5.0, 0.1, size=len(row))
    adata.obs["my_mask_distance"] = RNG.uniform(0, 10, size=len(row))

    quantify_diffusion(
        adata, library_id="unused", score="n_counts", grid_type="rect", mask_distance_key="my_mask_distance",
    )

    assert "components" in adata.obs
    assert (adata.obs["components"] == 0).all()


def test_quantify_diffusion_non_numeric_mask_distance_key_raises_clear_error():
    RNG = np.random.default_rng(42)
    adata, row, col = _rect_grid_adata()
    adata.obs["n_counts"] = RNG.normal(5.0, 0.1, size=len(row))
    adata.obs["my_mask_distance"] = [f"far_{i}" for i in range(len(row))]

    with pytest.raises(ValueError, match="must be numeric"):
        quantify_diffusion(
            adata, library_id="unused", score="n_counts", grid_type="rect", mask_distance_key="my_mask_distance",
        )


def test_quantify_diffusion_negative_mask_distance_key_raises_clear_error():
    RNG = np.random.default_rng(42)
    adata, row, col = _rect_grid_adata()
    adata.obs["n_counts"] = RNG.normal(5.0, 0.1, size=len(row))
    fake_distance = RNG.uniform(0, 5, size=len(row))
    fake_distance[0] = -1.0
    adata.obs["my_mask_distance"] = fake_distance

    with pytest.raises(ValueError, match="contains negative values"):
        quantify_diffusion(
            adata, library_id="unused", score="n_counts", grid_type="rect", mask_distance_key="my_mask_distance",
        )


def test_quantify_diffusion_fits_components_independently():
    """Two disconnected blocks with DIFFERENT true diffusion amplitudes --
    pooling them into one global fit would blur both; per-component fitting
    should recover each block's own alpha."""
    RNG = np.random.default_rng(42)
    adata, row, col, block_a_mask, n_side, gap = _two_block_adata()
    fake_distance = RNG.uniform(0, 20, size=len(row))
    adata.obs["my_mask_distance"] = fake_distance

    a_a, a_b, b_true, c_true = -4.0, -8.0, 0.5, 10.0
    signal = np.where(
        block_a_mask,
        ExponentialSaturationFit.exp_sat(fake_distance, a_a, b_true, c_true),
        ExponentialSaturationFit.exp_sat(fake_distance, a_b, b_true, c_true),
    )
    adata.obs["n_counts"] = signal + RNG.normal(0, 0.05, size=len(row))

    quantify_diffusion(
        adata, library_id="unused", score="n_counts", grid_type="rect",
        bin_size_um=1.0, mask_distance_key="my_mask_distance",
    )

    per_component = adata.uns["diffusion_fit"]["per_component"]
    assert len(per_component) == 2
    fitted_alphas = sorted(
        info["alpha"] for info in per_component.values() if info["alpha"] is not None
    )
    assert len(fitted_alphas) == 2
    assert fitted_alphas[0] == pytest.approx(min(-a_a, -a_b), rel=0.3)
    assert fitted_alphas[1] == pytest.approx(max(-a_a, -a_b), rel=0.3)


def test_quantify_diffusion_computes_mask_distance_from_real_image():
    """End-to-end smoke test of the real path (no mask_distance_key
    override): get_tissue_mask segments a synthetic image, native pixel
    size is calibrated from the grid, and distance_to_mask is computed --
    checks the result is sane (non-negative, some spots genuinely outside),
    not exact parameter recovery (covered by the override-based tests above)."""
    pytest.importorskip("skimage", reason="scikit-image not installed")

    RNG = np.random.default_rng(42)
    n_side = 20
    adata, row, col = _rect_grid_adata(n_side)
    pixel_pitch = 10.0  # native px per grid step
    adata.obsm["spatial"] = np.column_stack([col * pixel_pitch, row * pixel_pitch]).astype(float)

    size = int(n_side * pixel_pitch) + 20
    image = np.full((size, size, 3), 255, dtype=np.uint8)
    lo, hi = int(size * 0.25), int(size * 0.75)
    image[lo:hi, lo:hi, :] = 30  # dark "tissue" square in the middle
    adata.uns["spatial"] = {
        "sample1": {"images": {"hires": image}, "scalefactors": {"tissue_hires_scalef": 1.0}}
    }
    adata.obs["n_counts"] = RNG.normal(5.0, 0.5, size=len(row))

    quantify_diffusion(
        adata, library_id="sample1", score="n_counts", grid_type="rect", bin_size_um=8.0,
        segment_kwargs=dict(sigma=2, close_radius=3, min_hole_area=100, min_object_area=100),
    )

    assert "distance_to_mask" in adata.obs
    assert (adata.obs["distance_to_mask"] >= 0).all()
    assert adata.obs["distance_to_mask"].max() > 0


def test_quantify_diffusion_writes_mask_as_plottable_spatial_image():
    """The computed mask lands in adata.uns["spatial"][library_id]["images"]
    under mask_image_key, with a matching scalefactor -- the exact shape
    sc.pl.spatial(adata, library_id=..., img_key=mask_image_key) reads."""
    pytest.importorskip("skimage", reason="scikit-image not installed")

    RNG = np.random.default_rng(42)
    n_side = 20
    adata, row, col = _rect_grid_adata(n_side)
    pixel_pitch = 10.0
    adata.obsm["spatial"] = np.column_stack([col * pixel_pitch, row * pixel_pitch]).astype(float)

    size = int(n_side * pixel_pitch) + 20
    image = np.full((size, size, 3), 255, dtype=np.uint8)
    lo, hi = int(size * 0.25), int(size * 0.75)
    image[lo:hi, lo:hi, :] = 30
    adata.uns["spatial"] = {
        "sample1": {"images": {"hires": image}, "scalefactors": {"tissue_hires_scalef": 1.0}}
    }
    adata.obs["n_counts"] = RNG.normal(5.0, 0.5, size=len(row))

    quantify_diffusion(
        adata, library_id="sample1", score="n_counts", grid_type="rect", bin_size_um=8.0,
        segment_kwargs=dict(sigma=2, close_radius=3, min_hole_area=100, min_object_area=100),
    )

    spatial_meta = adata.uns["spatial"]["sample1"]
    assert "mask" in spatial_meta["images"]
    mask_image = spatial_meta["images"]["mask"]
    assert mask_image.shape == image.shape[:2]
    assert mask_image.dtype == np.uint8
    assert set(np.unique(mask_image)) <= {0, 255}
    assert "tissue_mask_scalef" in spatial_meta["scalefactors"]
    assert spatial_meta["scalefactors"]["tissue_mask_scalef"] == pytest.approx(1.0)


def test_quantify_diffusion_no_mask_image_written_when_mask_distance_key_reused():
    """No mask is computed at all when mask_distance_key is reused, so
    nothing new should be written to uns["spatial"]."""
    RNG = np.random.default_rng(42)
    adata, row, col = _rect_grid_adata()
    adata.obs["n_counts"] = RNG.normal(5.0, 0.1, size=len(row))
    adata.obs["my_mask_distance"] = RNG.uniform(0, 10, size=len(row))

    quantify_diffusion(
        adata, library_id="unused", score="n_counts", grid_type="rect", mask_distance_key="my_mask_distance",
    )

    assert "spatial" not in adata.uns
    assert "diffusion_fit" in adata.uns


# ---------------------------------------------------------------------------
# plot_border_effect / plot_diffusion
# ---------------------------------------------------------------------------

def test_plot_border_effect_one_subplot_per_component():
    RNG = np.random.default_rng(42)
    adata, row, col, block_a_mask, n_side, gap = _two_block_adata(n_side=30)
    d_true = np.where(
        block_a_mask,
        _distance_to_nearest_edge(row, col, n_side, row_offset=0),
        _distance_to_nearest_edge(row, col, n_side, row_offset=gap),
    )
    b_true, m_true, c_true = 5.0, -1.0, 10.0
    signal = PiecewiseLinearFit.piecewise_plateau(d_true, b_true, m_true, c_true)
    adata.obs["score"] = signal + RNG.normal(0, 0.05, size=len(row))

    identify_analysis_buffer(adata, score="score", grid_type="rect")

    fig = plot_border_effect(adata, score="score", ncols=4)
    n_components = len(adata.uns["analysis_buffer_fit"]["per_component"])
    assert n_components == 2
    lines_per_axes = [len(ax.get_lines()) for ax in fig.get_axes()]
    assert sum(1 for n in lines_per_axes if n == 1) == n_components
    plt.close(fig)


def test_plot_border_effect_raises_without_prior_run():
    adata, row, col = _rect_grid_adata()
    adata.obs["score"] = np.zeros(len(row))
    with pytest.raises(KeyError, match="run identify_analysis_buffer first"):
        plot_border_effect(adata, score="score")


def test_plot_border_effect_works_after_reused_string_components_key():
    """Regression test for the exact bug this fixed: a pre-existing
    components column stored as strings must not break plot_border_effect's
    later re-read of adata.obs["components"]."""
    RNG = np.random.default_rng(42)
    n_side = 30
    adata, row, col = _rect_grid_adata(n_side)
    d_true = _distance_to_nearest_edge(row, col, n_side)
    b_true, m_true, c_true = 5.0, -1.0, 10.0
    signal = PiecewiseLinearFit.piecewise_plateau(d_true, b_true, m_true, c_true)
    adata.obs["score"] = signal + RNG.normal(0, 0.05, size=len(row))
    adata.obs["components"] = np.zeros(len(row), dtype=int).astype(str)

    identify_analysis_buffer(adata, score="score", grid_type="rect")
    fig = plot_border_effect(adata, score="score")

    assert len(fig.get_axes()[0].get_lines()) == 1
    plt.close(fig)


def test_plot_diffusion_one_subplot_per_component():
    RNG = np.random.default_rng(42)
    adata, row, col, block_a_mask, n_side, gap = _two_block_adata()
    fake_distance = RNG.uniform(0, 20, size=len(row))
    adata.obs["my_mask_distance"] = fake_distance

    a_a, a_b, b_true, c_true = -4.0, -8.0, 0.5, 10.0
    signal = np.where(
        block_a_mask,
        ExponentialSaturationFit.exp_sat(fake_distance, a_a, b_true, c_true),
        ExponentialSaturationFit.exp_sat(fake_distance, a_b, b_true, c_true),
    )
    adata.obs["n_counts"] = signal + RNG.normal(0, 0.05, size=len(row))

    quantify_diffusion(
        adata, library_id="unused", score="n_counts", grid_type="rect",
        bin_size_um=1.0, mask_distance_key="my_mask_distance",
    )

    fig = plot_diffusion(adata, score="n_counts", mask_distance_key="my_mask_distance")
    n_components = len(adata.uns["diffusion_fit"]["per_component"])
    assert n_components == 2
    lines_per_axes = [len(ax.get_lines()) for ax in fig.get_axes()]
    assert sum(1 for n in lines_per_axes if n == 1) == n_components
    plt.close(fig)


def test_plot_diffusion_raises_without_prior_run():
    adata, row, col = _rect_grid_adata()
    adata.obs["n_counts"] = np.zeros(len(row))
    with pytest.raises(KeyError, match="run quantify_diffusion first"):
        plot_diffusion(adata, score="n_counts")
