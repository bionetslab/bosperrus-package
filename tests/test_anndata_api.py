import numpy as np
import pandas as pd
import pytest

anndata = pytest.importorskip("anndata", reason="anndata not installed")

from bosperrus.anndata_api import identify_analysis_buffer, correct_layer
from bosperrus.fit import PiecewiseLinearFit


RNG = np.random.default_rng(42)


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


def _distance_to_nearest_edge(row, col, n_side):
    """Ground-truth distance-to-border for an n_side x n_side square grid, used only
    to build synthetic scores with a KNOWN elbow -- independent of grid_edges/
    distance_to_pointset, so the test isn't circular."""
    return np.minimum.reduce([row, col, n_side - 1 - row, n_side - 1 - col]).astype(float)


# ---------------------------------------------------------------------------
# identify_analysis_buffer
# ---------------------------------------------------------------------------

def test_identify_analysis_buffer_no_effect_flags_nothing():
    """Pure noise, unrelated to distance -> ConstantFit should win -> buffer all False."""
    adata, row, col = _rect_grid_adata()
    adata.obs["score"] = RNG.normal(5.0, 0.1, size=len(row))

    identify_analysis_buffer(adata, score="score", grid_type="rect")

    assert "analysis_buffer" in adata.obs
    assert adata.obs["analysis_buffer"].dtype == bool
    assert not adata.obs["analysis_buffer"].any()
    assert "analysis_buffer_fit" in adata.uns


def test_identify_analysis_buffer_detects_known_elbow():
    """Score built directly from PiecewiseLinearFit's own formula with a known
    breakpoint b=5 -> fitted buffer should closely match distance < 5.

    Uses a 30x30 grid (not 10x10): distance-to-edge on a square grid only takes
    n_side//2 unique integer values, and curve_fit needs enough distinct x-values
    around the breakpoint to localize it -- a 10x10 grid (only 5 unique distances)
    left the breakpoint essentially unidentifiable, unrelated to the implementation
    (confirmed by test_fit.py::test_recovers_known_parameters already passing with
    adequate resolution)."""
    n_side = 30
    adata, row, col = _rect_grid_adata(n_side)
    d_true = _distance_to_nearest_edge(row, col, n_side)
    b_true, m_true, c_true = 5.0, -1.0, 10.0
    signal = PiecewiseLinearFit.piecewise_plateau(d_true, b_true, m_true, c_true)
    adata.obs["score"] = signal + RNG.normal(0, 0.05, size=len(row))

    identify_analysis_buffer(adata, score="score", grid_type="rect")

    fit_info = adata.uns["analysis_buffer_fit"]
    assert fit_info["best_fit_type"] == "Piecewise Linear Fit"
    fitted_b = fit_info["params"]["piecewise_linear_b"]
    assert fitted_b == pytest.approx(b_true, abs=1.0)

    # buffer should agree with the ground-truth threshold for the vast majority of spots
    ground_truth_buffer = d_true < b_true
    agreement = (adata.obs["analysis_buffer"].to_numpy() == ground_truth_buffer).mean()
    assert agreement > 0.9


def test_identify_analysis_buffer_array_score():
    """score can be a raw array, not just an .obs column name."""
    adata, row, col = _rect_grid_adata()
    values = RNG.normal(5.0, 0.1, size=len(row))

    identify_analysis_buffer(adata, score=values, grid_type="rect")
    assert "analysis_buffer" in adata.obs


def test_identify_analysis_buffer_copy_does_not_mutate_original():
    adata, row, col = _rect_grid_adata()
    adata.obs["score"] = RNG.normal(5.0, 0.1, size=len(row))

    result = identify_analysis_buffer(adata, score="score", grid_type="rect", copy=True)
    assert result is not None
    assert "analysis_buffer" not in adata.obs
    assert "analysis_buffer" in result.obs


def test_identify_analysis_buffer_distance_key_skips_grid_computation():
    """Passing distance_key should skip grid construction entirely -- verified by
    deliberately pointing row_key/col_key at nonexistent columns, which would raise
    if the grid path were reached."""
    adata, row, col = _rect_grid_adata()
    n_side = 10
    d_true = _distance_to_nearest_edge(row, col, n_side)
    adata.obs["my_distance"] = d_true
    adata.obs["score"] = RNG.normal(5.0, 0.1, size=len(row))

    identify_analysis_buffer(
        adata, score="score", row_key="nonexistent_row", col_key="nonexistent_col",
        distance_key="my_distance",
    )
    assert "analysis_buffer" in adata.obs


def test_identify_analysis_buffer_missing_distance_key_raises():
    adata, row, col = _rect_grid_adata()
    adata.obs["score"] = RNG.normal(5.0, 0.1, size=len(row))
    with pytest.raises(KeyError, match="not found in adata.obs"):
        identify_analysis_buffer(adata, score="score", distance_key="does_not_exist")


# ---------------------------------------------------------------------------
# correct_layer
# ---------------------------------------------------------------------------

def test_correct_layer_no_effect_passes_through_unchanged():
    adata, row, col = _rect_grid_adata()
    adata.X = np.column_stack([
        RNG.normal(5.0, 0.05, size=len(row)),
        RNG.normal(3.0, 0.05, size=len(row)),
    ])

    correct_layer(adata, grid_type="rect")

    assert "bosperrus_corrected" in adata.layers
    np.testing.assert_allclose(adata.layers["bosperrus_corrected"], adata.X, atol=1e-8)
    assert (adata.var["bosperrus_corrected_best_fit_type"] == "Constant Fit").all()


def test_correct_layer_shifts_detected_effect_toward_asymptote():
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
    # effect column (gene_a) should have been detected and actually changed
    assert adata.var.loc["gene_a", "bosperrus_corrected_best_fit_type"] == "Exponential Saturation Fit"
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
    assert adata.var.loc["gene_b", "bosperrus_corrected_best_fit_type"] == "Constant Fit"
    np.testing.assert_allclose(corrected[:, 1], adata.X[:, 1], atol=1e-8)


def test_correct_layer_named_layer_not_X():
    adata, row, col = _rect_grid_adata()
    adata.layers["counts"] = np.column_stack([
        RNG.normal(5.0, 0.05, size=len(row)),
        RNG.normal(3.0, 0.05, size=len(row)),
    ])

    correct_layer(adata, layer="counts", grid_type="rect")
    assert "bosperrus_corrected" in adata.layers
    np.testing.assert_allclose(adata.layers["bosperrus_corrected"], adata.layers["counts"], atol=1e-8)


def test_correct_layer_copy_does_not_mutate_original():
    adata, row, col = _rect_grid_adata()
    adata.X = np.column_stack([
        RNG.normal(5.0, 0.05, size=len(row)),
        RNG.normal(3.0, 0.05, size=len(row)),
    ])

    result = correct_layer(adata, grid_type="rect", copy=True)
    assert result is not None
    assert "bosperrus_corrected" not in adata.layers
    assert "bosperrus_corrected" in result.layers
