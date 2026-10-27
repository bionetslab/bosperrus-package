import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from bosperrus.fit import (
    ConstantFit, PiecewiseLinearFit, ExponentialSaturationFit, ExponentialDecayFit, MichaelisMentenFit,
)
from bosperrus.plotting import plot_fit, FIT_PALETTE


def test_fit_palette_matches_each_subclasss_own_color():
    """FIT_PALETTE is derived from each subclass's own `color` class
    attribute -- must never drift out of sync with it."""
    expected = {
        "Constant Fit": ConstantFit.color,
        "Piecewise Linear Fit": PiecewiseLinearFit.color,
        "Exponential Saturation Fit": ExponentialSaturationFit.color,
        "Exponential Decay Fit": ExponentialDecayFit.color,
        "Michaelis-Menten Fit": MichaelisMentenFit.color,
    }
    assert FIT_PALETTE == expected


def test_plot_fit_draws_curve_matching_predict_fn():
    rng = np.random.default_rng(42)
    d = rng.uniform(0, 10, 200)
    s = ExponentialSaturationFit.exp_sat(d, 3.0, 0.5, 1.0) + rng.normal(0, 0.05, len(d))

    fit = ExponentialSaturationFit(pd.Series(s), pd.Series(d))
    fit.fit()

    fig, ax = plt.subplots()
    returned_ax = plot_fit(ax, d, s, fit.predict)
    assert returned_ax is ax

    lines = ax.get_lines()
    assert len(lines) == 2  # per-bin mean, then the fitted curve on top
    line_x, line_y = lines[-1].get_data()
    np.testing.assert_allclose(line_y, fit.predict(np.asarray(line_x)))
    plt.close(fig)


def test_plot_fit_ignores_non_finite_points():
    d = np.array([1.0, 2.0, np.nan, 4.0, np.inf])
    s = np.array([1.0, 2.0, 3.0, np.nan, 5.0])

    fig, ax = plt.subplots()
    plot_fit(ax, d, s, lambda x: np.zeros_like(x))
    assert len(ax.collections) > 0  # quantile bands drawn without error
    plt.close(fig)


def test_plot_fit_mismatched_lengths_raises():
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="same length"):
        plot_fit(ax, np.zeros(5), np.zeros(4), lambda x: x)
    plt.close(fig)


def test_plot_fit_all_non_finite_raises():
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="No finite"):
        plot_fit(ax, np.full(5, np.nan), np.zeros(5), lambda x: x)
    plt.close(fig)


def test_plot_fit_default_d_grid_spans_data_range():
    d = np.array([2.0, 5.0, 8.0])
    s = np.array([1.0, 2.0, 3.0])
    fig, ax = plt.subplots()
    plot_fit(ax, d, s, lambda x: x)
    line_x, _ = ax.get_lines()[-1].get_data()
    assert line_x.min() == pytest.approx(2.0)
    assert line_x.max() == pytest.approx(8.0)
    plt.close(fig)


def test_plot_fit_explicit_d_grid_used_verbatim():
    d = np.array([2.0, 5.0, 8.0])
    s = np.array([1.0, 2.0, 3.0])
    d_grid = np.array([0.0, 100.0])
    fig, ax = plt.subplots()
    plot_fit(ax, d, s, lambda x: x, d_grid=d_grid)
    line_x, _ = ax.get_lines()[-1].get_data()
    np.testing.assert_allclose(line_x, d_grid)
    plt.close(fig)


def test_plot_fit_sets_axis_labels():
    fig, ax = plt.subplots()
    plot_fit(ax, np.array([1.0, 2.0]), np.array([1.0, 2.0]), lambda x: x)
    assert ax.get_xlabel() == "distance"
    assert ax.get_ylabel() == "score"
    plt.close(fig)


def test_plot_fit_mean_line_is_per_bin_mean():
    d = np.array([0.1, 0.2, 0.9, 1.1, 1.8, 1.9])
    s = np.array([1.0, 3.0, 2.0, 10.0, 20.0, 30.0])
    fig, ax = plt.subplots()
    plot_fit(ax, d, s, lambda x: np.zeros_like(x), bins=[0.0, 1.0, 2.0])
    mean_x, mean_y = ax.get_lines()[0].get_data()
    np.testing.assert_allclose(mean_x, [0.5, 1.5])
    np.testing.assert_allclose(mean_y, [2.0, 20.0])
    plt.close(fig)


def test_plot_fit_mean_differs_from_median_on_skewed_bin():
    d = np.array([0.1, 0.2, 0.3, 0.4])
    s = np.array([1.0, 1.0, 1.0, 101.0])
    fig, ax = plt.subplots()
    plot_fit(ax, d, s, lambda x: np.zeros_like(x), bins=[0.0, 1.0])
    _, mean_y = ax.get_lines()[0].get_data()
    np.testing.assert_allclose(mean_y, [26.0])
    plt.close(fig)


def test_plot_fit_one_band_per_quantile_pair():
    rng = np.random.default_rng(0)
    d = rng.uniform(0, 10, 500)
    s = rng.normal(0, 1, 500)
    fig, ax = plt.subplots()
    plot_fit(ax, d, s, lambda x: np.zeros_like(x), bins=10)
    assert len(ax.collections) == 2
    plot_fit(ax, d, s, lambda x: np.zeros_like(x), bins=10, quantiles=[(0.1, 0.9)], band_colors=["0.8"])
    assert len(ax.collections) == 3
    plt.close(fig)


def test_plot_fit_band_colors_must_match_quantiles():
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="band color"):
        plot_fit(ax, np.array([1.0, 2.0]), np.array([1.0, 2.0]), lambda x: x, band_colors=["0.8"])
    plt.close(fig)


def test_plot_fit_rejects_2d_bins():
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="1-D"):
        plot_fit(ax, np.array([1.0, 2.0]), np.array([1.0, 2.0]), lambda x: x,
                 bins=[np.array([0.0, 1.0, 2.0]), np.array([0.0, 5.0])])
    plt.close(fig)
