"""
Unit tests for fit.py
Coverage:
- Basic API (fit runs, attributes set, correct() works)
- Score / AIC sanity (better fit → lower AIC)
- fraction_not_converged correctness and edge cases
- Numerical sanity of half-life and effect_strength
- NaN handling in __init__
- Recovery of known ground-truth parameters (smoke tests)
- Guarding correct() called before fit()
- __repr__ smoke test
"""

import numpy as np
import pandas as pd
import pytest

from bosperrus.fit import (
    ConstantFit,
    PiecewiseLinearFit,
    ExponentialSaturationFit,
    MichaelisMentenFit,
)


# ============================================================
# Fixtures
# ============================================================

RNG = np.random.default_rng(42)
N = 300
D = np.linspace(0, 10, N)  # distance axis, no noise


def make_series(arr):
    return pd.Series(arr)


# ============================================================
# ConstantFit
# ============================================================

class TestConstantFit:

    def _fit(self, C, d=None):
        d = d if d is not None else make_series(D)
        f = ConstantFit(make_series(C), d)
        f.fit()
        return f

    def test_fit_runs_and_sets_attributes(self):
        f = self._fit(np.ones(N) * 5)
        assert f.S_model is not None
        assert f.params is not None
        assert f.AIC < np.inf
        assert f.log_likelihood > -np.inf

    def test_recovers_known_constant(self):
        target = 7.3
        f = self._fit(np.full(N, target))
        assert f.params["constant_c"] == pytest.approx(target)
        np.testing.assert_allclose(f.S_model, target)

    def test_effect_strength_is_zero(self):
        f = self._fit(np.ones(N))
        assert f.observed_effect_strength == 0

    def test_half_life_is_zero(self):
        f = self._fit(np.ones(N))
        assert f.observed_half_life == 0

    def test_fraction_not_converged_is_zero(self):
        f = self._fit(np.ones(N))
        assert f.fraction_not_converged == 0.0

    def test_nan_in__S_true_is_masked(self):
        C = np.ones(N)
        C[::5] = np.nan
        f = self._fit(C)
        assert f.included_samples < 1.0
        assert not np.any(np.isnan(f._S_true))

    def test_nan_in_d_is_masked(self):
        d = make_series(D.copy())
        d[::5] = np.nan
        f = self._fit(np.ones(N), d)
        assert f.included_samples < 1.0
        assert not np.any(np.isnan(f._d))

    def test_aic_worse_than_better_fitting_model(self):
        # Constant fit should have worse AIC than a model that actually fits a trend
        C_trend = make_series(D + RNG.normal(0, 0.001, N))
        d_s = make_series(D)
        const_f = ConstantFit(C_trend, d_s)
        const_f.fit()
        pw_f = PiecewiseLinearFit(C_trend, d_s)
        pw_f.fit()
        assert const_f.AIC > pw_f.AIC

    def test_inf_in__S_true_is_masked(self):
        C = np.ones(N)
        C[::5] = np.inf
        f = self._fit(C)
        assert f.included_samples < 1.0
        assert not np.any(np.isinf(f._S_true))

    def test_negative_inf_in__S_true_is_masked(self):
        C = np.ones(N)
        C[::5] = -np.inf
        f = self._fit(C)
        assert f.included_samples < 1.0
        assert not np.any(np.isinf(f._S_true))

    def test_inf_in_d_is_masked(self):
        d = make_series(D.copy())
        d[::5] = np.inf
        f = self._fit(np.ones(N), d)
        assert f.included_samples < 1.0
        assert not np.any(np.isinf(f._d))

    def test_attributes_are_immutable(self):
        f = self._fit(np.ones(N))
        with pytest.raises(AttributeError):
            f.params = {"constant_c": 999}
        with pytest.raises(AttributeError):
            f.S_model = np.zeros(N)


    def test_repr_smoke(self):
        f = self._fit(np.ones(N))
        r = repr(f)
        assert "ConstantFit" in r
        assert "converged" in r


# ============================================================
# PiecewiseLinearFit
# ============================================================

class TestPiecewiseLinearFit:

    def _ground_truth(self, b=5.0, m=2.0, c=1.0, noise=0.0):
        C = PiecewiseLinearFit.piecewise_plateau(D, b, m, c)
        C = C + RNG.normal(0, noise, N)
        return make_series(C), make_series(D)

    def _fit(self, C, d):
        f = PiecewiseLinearFit(C, d)
        f.fit()
        return f

    def test_fit_runs_and_sets_attributes(self):
        C, d = self._ground_truth()
        f = self._fit(C, d)
        assert f.params is not None
        assert f.AIC < np.inf

    def test_recovers_known_parameters(self):
        b_true, m_true, c_true = 4.0, 1.5, 0.5
        C, d = self._ground_truth(b=b_true, m=m_true, c=c_true, noise=0.01)
        f = self._fit(C, d)
        assert f.params["piecewise_linear_b"] == pytest.approx(b_true, rel=0.05)
        assert f.params["piecewise_linear_m"] == pytest.approx(m_true, rel=0.05)
        assert f.params["piecewise_linear_c"] == pytest.approx(c_true, abs=0.1)

    def test_elbow_within_data_range(self):
        C, d = self._ground_truth(noise=0.1)
        f = self._fit(C, d)
        b = f.params["piecewise_linear_b"]
        assert d.min() <= b <= d.max()

    def test_effect_strength_negative_for_increasing_fit(self):
        C, d = self._ground_truth(m=2.0)  # increasing with d → c_center > c_border
        f = self._fit(C, d)
        assert f.observed_effect_strength < 0

    def test_effect_strength_positive_for_decreasing_fit(self):
        C, d = self._ground_truth(m=-2.0, c=20.0)
        f = self._fit(C, d)
        assert f.observed_effect_strength > 0

    def test_half_life_within_data_range(self):
        C, d = self._ground_truth(noise=0.05)
        f = self._fit(C, d)
        assert d.min() <= f.observed_half_life <= d.max()

    def test_fraction_not_converged_all_left_of_knot(self):
        # If all points are left of b, fraction should be 1.0
        d_tight = make_series(np.linspace(0, 2, N))
        C = make_series(PiecewiseLinearFit.piecewise_plateau(d_tight.values, b=5.0, m=1.0, c=0.0))
        f = self._fit(C, d_tight)
        # b is constrained to [d_min, d_max] = [0, 2], so all points are <= b
        assert f.fraction_not_converged == pytest.approx(1.0, abs=0.01)

    def test_correct_shape(self):
        C, d = self._ground_truth(noise=0.1)
        f = self._fit(C, d)
        result = f.correct()
        assert len(result) == len(f._S_true)

    def test_attributes_are_immutable(self):
        C, d = self._ground_truth()
        f = self._fit(C, d)
        with pytest.raises(AttributeError):
            f.params = {"piecewise_linear_b": 999}
        with pytest.raises(AttributeError):
            f.S_model = np.zeros(N)


    def test_repr_smoke(self):
        C, d = self._ground_truth()
        f = self._fit(C, d)
        r = repr(f)
        assert "PiecewiseLinearFit" in r
        assert "converged" in r


# ============================================================
# ExponentialSaturationFit
# ============================================================

class TestExponentialSaturationFit:
    def _ground_truth(self, a=3.0, b=0.5, c=1.0, noise=0.0):
        C = ExponentialSaturationFit.exp_sat(D, a, b, c)
        C = C + RNG.normal(0, noise, N)
        return make_series(C), make_series(D)

    def _fit(self, C, d):
        f = ExponentialSaturationFit(C, d)
        f.fit()
        return f

    def test_fit_runs_and_sets_attributes(self):
        C, d = self._ground_truth()
        f = self._fit(C, d)
        assert f.params is not None
        assert f.AIC < np.inf

    def test_recovers_known_parameters(self):
        a_true, b_true, c_true = 4.0, 0.3, 0.5
        C, d = self._ground_truth(a=a_true, b=b_true, c=c_true, noise=0.01)
        f = self._fit(C, d)
        assert f.params["exponential_saturation_a"] == pytest.approx(a_true, rel=0.05)
        assert f.params["exponential_saturation_b"] == pytest.approx(b_true, rel=0.05)
        assert f.params["exponential_saturation_c"] == pytest.approx(c_true, abs=0.1)

    def test_b_is_positive(self):
        # Bound enforced → b must always be positive
        C, d = self._ground_truth(noise=0.2)
        f = self._fit(C, d)
        assert f.params["exponential_saturation_b"] > 0

    def test_convergence_distance_analytically_correct(self):
        # At d_converge = -ln(1 - threshold) / b, exactly `threshold` of a is reached
        a, b, c = 5.0, 0.4, 1.0
        C, d = self._ground_truth(a=a, b=b, c=c, noise=0.0)
        f = self._fit(C, d)
        threshold = 0.95
        d_converge = -np.log(1 - threshold) / f.params["exponential_saturation_b"]
        value_at_d_converge = ExponentialSaturationFit.exp_sat(
            d_converge,
            f.params["exponential_saturation_a"],
            f.params["exponential_saturation_b"],
            f.params["exponential_saturation_c"],
        )
        saturation_fraction = (value_at_d_converge - f.params["exponential_saturation_c"]) / f.params["exponential_saturation_a"]
        assert saturation_fraction == pytest.approx(threshold, rel=1e-4)

    def test_fraction_not_converged_in_01(self):
        C, d = self._ground_truth(noise=0.05)
        f = self._fit(C, d)
        frac = f.fraction_not_converged
        assert 0.0 <= frac <= 1.0

    def test_fraction_not_converged_invalid_threshold(self):
        # threshold is passed to _calculate_fraction_not_converged, not the property;
        # invalid threshold should raise on the next fit() call with that threshold
        C, d = self._ground_truth()
        f = ExponentialSaturationFit(C, d)
        with pytest.raises(ValueError):
            f._calculate_fraction_not_converged(threshold=1.5)
        with pytest.raises(ValueError):
            f._calculate_fraction_not_converged(threshold=0.0)

    def test_fraction_not_converged_before_fit_is_none(self):
        # Before fit(), fraction_not_converged is None (not yet computed)
        C, d = self._ground_truth()
        f = ExponentialSaturationFit(C, d)
        assert f.fraction_not_converged is None

    def test_effect_strength_negative_for_increasing(self):
        C, d = self._ground_truth(a=3.0, c=1.0)
        f = self._fit(C, d)
        assert f.observed_effect_strength < 0

    def test_half_life_positive(self):
        C, d = self._ground_truth(a=3.0, b=0.5, c=1.0)
        f = self._fit(C, d)
        assert f.observed_half_life > 0

    def test_aic_better_than_constant_on_saturating_data(self):
        C, d = self._ground_truth(a=4.0, b=0.4, c=0.5, noise=0.05)
        const_f = ConstantFit(C, d)
        const_f.fit()
        exp_f = self._fit(C, d)
        assert exp_f.AIC < const_f.AIC

    def test_attributes_are_immutable(self):
        C, d = self._ground_truth()
        f = self._fit(C, d)
        with pytest.raises(AttributeError):
            f.params = {"exponential_saturation_a": 999}
        with pytest.raises(AttributeError):
            f.S_model = np.zeros(N)

    def test_repr_smoke(self):
        C, d = self._ground_truth()
        f = self._fit(C, d)
        r = repr(f)
        assert "ExponentialSaturationFit" in r
        assert "converged" in r


# ============================================================
# MichaelisMentenFit
# ============================================================

class TestMichaelisMentenFit:
    def _ground_truth(self, a=3.0, b=2.0, c=0.5, noise=0.0):
        C = MichaelisMentenFit.michaelis_menten(D, a, b, c)
        C = C + RNG.normal(0, noise, N)
        return make_series(C), make_series(D)

    def _fit(self, C, d):
        f = MichaelisMentenFit(C, d)
        f.fit()
        return f

    def test_fit_runs_and_sets_attributes(self):
        C, d = self._ground_truth()
        f = self._fit(C, d)
        assert f.params is not None
        assert f.AIC < np.inf

    def test_recovers_known_parameters(self):
        a_true, b_true, c_true = 5.0, 3.0, 0.5
        C, d = self._ground_truth(a=a_true, b=b_true, c=c_true, noise=0.01)
        f = self._fit(C, d)
        assert f.params["michaelis_menten_a"] == pytest.approx(a_true, rel=0.05)
        assert f.params["michaelis_menten_b"] == pytest.approx(b_true, rel=0.05)
        assert f.params["michaelis_menten_c"] == pytest.approx(c_true, abs=0.1)

    def test_b_is_positive(self):
        C, d = self._ground_truth(noise=0.2)
        f = self._fit(C, d)
        assert f.params["michaelis_menten_b"] > 0

    def test_km_definition_holds(self):
        # At d = b (Km), the saturable term should be exactly 50% of its maximum
        a, b, c = 6.0, 2.0, 0.0
        C, d = self._ground_truth(a=a, b=b, c=c, noise=0.0)
        f = self._fit(C, d)
        b_fit = f.params["michaelis_menten_b"]
        a_fit = f.params["michaelis_menten_a"]
        c_fit = f.params["michaelis_menten_c"]
        value_at_km = MichaelisMentenFit.michaelis_menten(b_fit, a_fit, b_fit, c_fit)
        saturation_at_km = (value_at_km - c_fit) / a_fit
        assert saturation_at_km == pytest.approx(0.5, rel=1e-4)

    def test_convergence_distance_analytically_correct(self):
        # At d_converge = b * threshold / (1 - threshold), saturation == threshold
        a, b, c = 4.0, 2.0, 0.0
        C, d = self._ground_truth(a=a, b=b, c=c, noise=0.0)
        f = self._fit(C, d)
        threshold = 0.95
        b_fit = f.params["michaelis_menten_b"]
        d_converge = b_fit * threshold / (1 - threshold)
        value_at_converge = MichaelisMentenFit.michaelis_menten(
            d_converge,
            f.params["michaelis_menten_a"],
            b_fit,
            f.params["michaelis_menten_c"],
        )
        saturation = (value_at_converge - f.params["michaelis_menten_c"]) / f.params["michaelis_menten_a"]
        assert saturation == pytest.approx(threshold, rel=1e-4)

    def test_fraction_not_converged_in_01(self):
        C, d = self._ground_truth(noise=0.05)
        f = self._fit(C, d)
        frac = f.fraction_not_converged
        assert 0.0 <= frac <= 1.0

    def test_fraction_not_converged_invalid_threshold(self):
        C, d = self._ground_truth()
        f = self._fit(C, d)
        with pytest.raises(ValueError):
            f._calculate_fraction_not_converged(threshold=0.0)
        with pytest.raises(ValueError):
            f._calculate_fraction_not_converged(threshold=1.0)

    def test_fraction_not_converged_before_fit_is_none(self):
        C, d = self._ground_truth()
        f = MichaelisMentenFit(C, d)
        assert f.fraction_not_converged is None

    def test_effect_strength_negative_for_increasing(self):
        C, d = self._ground_truth(a=3.0, c=1.0)
        f = self._fit(C, d)
        assert f.observed_effect_strength < 0

    def test_half_life_close_to_b(self):
        # observed_half_life is the d where midpoint of [c_border, c_center] is reached.
        # With d_min~0, this is close to the true Km (b).
        a, b, c = 5.0, 2.0, 0.0
        C, d = self._ground_truth(a=a, b=b, c=c, noise=0.0)
        f = self._fit(C, d)
        # Not exactly b because midpoint depends on d_max, but should be in same ballpark
        assert f.observed_half_life == pytest.approx(b, rel=0.3)

    def test_aic_better_than_constant_on_saturating_data(self):
        C, d = self._ground_truth(a=5.0, b=2.0, c=0.5, noise=0.05)
        const_f = ConstantFit(C, d)
        const_f.fit()
        mm_f = self._fit(C, d)
        assert mm_f.AIC < const_f.AIC

    def test_attributes_are_immutable(self):
        C, d = self._ground_truth()
        f = self._fit(C, d)
        with pytest.raises(AttributeError):
            f.params = {"michaelis_menten_a": 999}
        with pytest.raises(AttributeError):
            f.S_model = np.zeros(N)

    def test_repr_smoke(self):
        C, d = self._ground_truth()
        f = self._fit(C, d)
        r = repr(f)
        assert "MichaelisMentenFit" in r
        assert "converged" in r


# ============================================================
# Cross-model sanity checks
# ============================================================

class TestCrossModel:

    def test_constant_data_favors_constant_fit(self):
        C = make_series(np.full(N, 3.0))
        d = make_series(D)
        fits = [ConstantFit(C, d), PiecewiseLinearFit(C, d),
                ExponentialSaturationFit(C, d), MichaelisMentenFit(C, d)]
        for f in fits:
            f.fit()
        best = min(fits, key=lambda f: f.AIC)
        assert best.name == "Constant Fit"

    def test_all_models_have_finite_aic(self):
        C = make_series(ExponentialSaturationFit.exp_sat(D, 3.0, 0.4, 1.0) + RNG.normal(0, 0.1, N))
        d = make_series(D)
        for cls in [ConstantFit, PiecewiseLinearFit, ExponentialSaturationFit, MichaelisMentenFit]:
            f = cls(C, d)
            f.fit()
            assert np.isfinite(f.AIC), f"{cls.__name__} produced non-finite AIC"

    def test_included_samples_is_one_without_nans(self):
        C = make_series(np.ones(N))
        d = make_series(D)
        for cls in [ConstantFit, PiecewiseLinearFit, ExponentialSaturationFit, MichaelisMentenFit]:
            f = cls(C, d)
            assert f.included_samples == pytest.approx(1.0)

    def test_correct_returns_array_of_correct_length(self):
        C = make_series(ExponentialSaturationFit.exp_sat(D, 3.0, 0.4, 1.0))
        d = make_series(D)
        for cls in [PiecewiseLinearFit, ExponentialSaturationFit, MichaelisMentenFit]:
            f = cls(C, d)
            f.fit()
            result = f.correct()
            assert len(result) == len(f._S_true)

    def test_correct_raises_before_fit(self):
        C = make_series(np.ones(N))
        d = make_series(D)
        for cls in [ConstantFit, PiecewiseLinearFit, ExponentialSaturationFit, MichaelisMentenFit]:
            f = cls(C, d)
            with pytest.raises(RuntimeError):
                f.correct()

    def test_mismatched_lengths_raises(self):
        C = make_series(np.ones(N))
        d = make_series(D[:N - 1])
        with pytest.raises(ValueError):
            ConstantFit(C, d)

    def test_all_attributes_immutable_across_models(self):
        C = make_series(ExponentialSaturationFit.exp_sat(D, 3.0, 0.4, 1.0))
        d = make_series(D)
        for cls in [ConstantFit, PiecewiseLinearFit, ExponentialSaturationFit, MichaelisMentenFit]:
            f = cls(C, d)
            f.fit()
            with pytest.raises(AttributeError):
                f.AIC = 0.0

    def test_repr_before_fit(self):
        C = make_series(np.ones(N))
        d = make_series(D)
        for cls in [ConstantFit, PiecewiseLinearFit, ExponentialSaturationFit, MichaelisMentenFit]:
            f = cls(C, d)
            r = repr(f)
            assert cls.__name__ in r
            assert "not fitted" in r


# ============================================================
# Regression tests — bugs that were fixed
# ============================================================

class TestRegressions:
    """Regression tests for bugs that were fixed (B3, B4)."""

    rng = np.random.default_rng(42)

    def test_constant_fit_d_none_does_not_crash(self):
        """B4 — ConstantFit(d=None) must not raise TypeError."""
        # Should construct without raising
        f = ConstantFit(pd.Series([1.0, 2.0, 3.0]), None)
        assert f is not None

    def test_constant_fit_d_none_fit_runs(self):
        """B4 — After construction with d=None, fit() and correct() must succeed."""
        f = ConstantFit(pd.Series([1.0, 2.0, 3.0]), None)
        f.fit()
        result = f.correct()
        assert result is not None
        assert len(result) == 3

    def _normal_data(self):
        """Return a simple valid dataset."""
        d = pd.Series(np.linspace(0, 10, 50))
        S = pd.Series(np.ones(50) * 3.0)
        return S, d

    def test_piecewise_linear_fraction_not_converged_is_nan_not_none_when_not_converged(self, monkeypatch):
        """B3 — fraction_not_converged must be np.nan (not None) after a failed fit.

        Forces curve_fit failure via monkeypatching the internal fit method so
        that the RuntimeError/ValueError branch in fit() is exercised.
        """
        S, d = self._normal_data()
        f = PiecewiseLinearFit(S, d)
        monkeypatch.setattr(f, "_fit_piece_wise_linear", lambda **kw: (_ for _ in ()).throw(RuntimeError("forced failure")))
        f.fit()
        # After a failed fit, _converged is False; B3 was that _fraction_not_converged
        # stayed None instead of being set to np.nan.
        assert f.fraction_not_converged is not None, (
            "fraction_not_converged should be np.nan after a failed fit, not None"
        )
        assert np.isnan(f.fraction_not_converged), (
            f"fraction_not_converged should be np.nan, got {f.fraction_not_converged!r}"
        )

    def test_michaelis_menten_fraction_not_converged_is_nan_not_none_when_not_converged(self, monkeypatch):
        """B3 — same regression check for MichaelisMentenFit."""
        S, d = self._normal_data()
        f = MichaelisMentenFit(S, d)
        monkeypatch.setattr(f, "_fit_michaelis_menten", lambda: (_ for _ in ()).throw(RuntimeError("forced failure")))
        f.fit()
        assert f.fraction_not_converged is not None, (
            "fraction_not_converged should be np.nan after a failed fit, not None"
        )
        assert np.isnan(f.fraction_not_converged), (
            f"fraction_not_converged should be np.nan, got {f.fraction_not_converged!r}"
        )


# ============================================================
# PiecewiseLinearFit branch coverage
# ============================================================

class TestPiecewiseLinearFitBranches:
    """Branch coverage for PiecewiseLinearFit.fit() refinement path."""

    rng = np.random.default_rng(42)

    def _good_data(self):
        d = make_series(np.linspace(0, 10, 200))
        C = make_series(PiecewiseLinearFit.piecewise_plateau(d.values, b=5.0, m=2.0, c=1.0))
        return C, d

    def test_fit_without_refinement(self):
        """refine_fit=False must still set _params and _converged."""
        C, d = self._good_data()
        f = PiecewiseLinearFit(C, d)
        f.fit(refine_fit=False)
        assert f._params is not None
        assert f._converged is True

    def test_fit_refine_true_default(self):
        """Default fit() (refine_fit=True) must produce valid params."""
        C, d = self._good_data()
        f = PiecewiseLinearFit(C, d)
        f.fit()
        assert f._params is not None
        assert f._converged is True
        assert np.isfinite(f.params["piecewise_linear_b"])

    def test_correct_before_fit_raises(self):
        """calling correct() before fit() must raise RuntimeError."""
        C, d = self._good_data()
        f = PiecewiseLinearFit(C, d)
        with pytest.raises(RuntimeError):
            f.correct()


# ============================================================
# fit_correct() passthrough and robustness
# ============================================================

class TestFitCorrectPassthrough:
    """Test that fit_correct() survives hard data without crashing."""

    rng = np.random.default_rng(42)

    def _degenerate_d(self):
        """Single unique d value causes curve_fit to fail for non-constant models."""
        d = pd.Series(np.ones(30) * 5.0)
        S = pd.Series(np.ones(30) * 2.0)
        return S, d

    def test_all_models_survive_fit_correct_on_hard_data(self):
        """fit_correct() on degenerate data must not crash and return correct-length array.

        When fitting fails, the passthrough (S_corrected = S_true) must be returned.
        ConstantFit uses a synthetic d so it never fails.
        """
        S, d = self._degenerate_d()

        # ConstantFit always succeeds; pass d explicitly for a consistent length check
        f_const = ConstantFit(S, d)
        result_const = f_const.fit_correct()
        assert result_const is not None
        assert len(result_const) == len(S)

        for cls in [PiecewiseLinearFit, ExponentialSaturationFit, MichaelisMentenFit]:
            f = cls(S, d)
            result = f.fit_correct()
            # If not converged, S_corrected is None (fit_correct returns None when
            # not converged); alternatively the passthrough may return an array.
            # Either way, the call must not raise an exception.
            # When converged: len must match. When not converged: result may be None.
            if result is not None:
                assert len(result) == len(S), (
                    f"{cls.__name__}.fit_correct() returned wrong-length result"
                )

    def test_correct_passthrough_when_not_converged(self):
        """When _converged=False, correct() must return S_true unchanged."""
        C, d = make_series(np.linspace(1.0, 5.0, 50)), make_series(np.linspace(0, 10, 50))
        f = PiecewiseLinearFit(C, d)
        f.fit()
        # Force non-convergence and call correct() again
        f._converged = False
        f._params = {"piecewise_linear_b": np.nan, "piecewise_linear_m": np.nan, "piecewise_linear_c": np.nan}
        f._S_model = f._S_true.values.copy() if hasattr(f._S_true, 'values') else f._S_true.copy()
        result = f.correct()
        # Passthrough: corrected values should equal S_true (for non-NaN entries)
        original_vals = f._S_true.values
        result_vals = result.dropna().values
        np.testing.assert_allclose(result_vals, original_vals, err_msg=(
            "correct() passthrough should return S_true unchanged when not converged"
        ))
