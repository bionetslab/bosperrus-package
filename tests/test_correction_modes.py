"""
Unit tests for the additive / multiplicative correction modes and the correction-mode diagnostic.

Coverage:
- correct() default and correct("additive") are identical to correct_additive()
- plateau of every model equals the value the additive correction targets
- multiplicative correction formula, guards (curve/plateau of opposite sign or zero -> NaN + warning)
- ConstantFit: identity in both modes
- Synthetic truncation with known true scores: the matching correction recovers the ranking better,
  and suggest_correction_mode() picks the generating model
- proportional_log_likelihood against a direct computation
- Flow.flow(correction=...): additive default unchanged, invalid mode raises, fit_quality reporting
"""

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm, spearmanr

from bosperrus.evaluate_fit import proportional_log_likelihood
from bosperrus.fit import (
    ConstantFit,
    ExponentialDecayFit,
    ExponentialSaturationFit,
    MichaelisMentenFit,
    PiecewiseLinearFit,
)
from bosperrus.pipeline import Flow

N = 2000


def truncated_scores(kind, seed=42):
    """Heavy-tailed true scores T (lognormal) and distances d in [0, 100]; the border keeps a
    fraction g(d) = 1 - 0.8 * exp(-d / 20) of each score ("multiplicative") or removes the same
    absolute amount mean(T) * (1 - g(d)) from every node ("additive")."""
    rng = np.random.default_rng(seed)
    d = pd.Series(rng.uniform(0, 100, N))
    T = pd.Series(rng.lognormal(mean=4.0, sigma=0.6, size=N))
    g = 1 - 0.8 * np.exp(-d / 20)
    S = T * g if kind == "multiplicative" else T - T.mean() * (1 - g)
    return S, d, T


def fitted(cls, S, d):
    f = cls(S, d)
    f.fit()
    return f


class TestModeDispatch:

    @pytest.mark.parametrize("cls", [PiecewiseLinearFit, ExponentialSaturationFit, MichaelisMentenFit,
                                     ExponentialDecayFit, ConstantFit])
    def test_default_is_additive(self, cls):
        S, d, _ = truncated_scores("multiplicative")
        a, b = fitted(cls, S, d), fitted(cls, S, d)
        pd.testing.assert_series_equal(a.correct(), b.correct_additive())
        assert a.correction_mode == "additive"

    def test_invalid_mode_raises(self):
        S, d, _ = truncated_scores("additive")
        with pytest.raises(ValueError):
            fitted(ExponentialSaturationFit, S, d).correct("quadratic")

    def test_correct_before_fit_raises(self):
        S, d, _ = truncated_scores("additive")
        with pytest.raises(RuntimeError):
            ExponentialSaturationFit(S, d).correct("multiplicative")

    def test_fit_correct_passes_mode(self):
        S, d, _ = truncated_scores("multiplicative")
        f = ExponentialSaturationFit(S, d)
        out = f.fit_correct(mode="multiplicative")
        assert f.correction_mode == "multiplicative"
        np.testing.assert_allclose(out, S * f.plateau / f.S_model)


class TestPlateau:

    @pytest.mark.parametrize("cls", [PiecewiseLinearFit, ExponentialSaturationFit, MichaelisMentenFit,
                                     ExponentialDecayFit, ConstantFit])
    def test_additive_correction_targets_plateau(self, cls):
        """S_add - S_true + S_model must equal the plateau for every node."""
        S, d, _ = truncated_scores("additive")
        f = fitted(cls, S, d)
        target = f.correct_additive() - S + f.S_model
        np.testing.assert_allclose(target, f.plateau, rtol=1e-12, atol=1e-9)

    def test_exponential_decay_plateau_is_zero(self):
        d = pd.Series(np.linspace(0, 10, 300))
        f = fitted(ExponentialDecayFit, 5 * np.exp(-d), d)
        assert f.plateau == 0.0

    def test_plateau_before_fit_raises(self):
        S, d, _ = truncated_scores("additive")
        with pytest.raises(RuntimeError):
            _ = ExponentialSaturationFit(S, d).plateau


class TestMultiplicative:

    def test_formula(self):
        S, d, _ = truncated_scores("multiplicative")
        f = fitted(ExponentialSaturationFit, S, d)
        np.testing.assert_allclose(f.correct("multiplicative"), S * f.plateau / f.S_model)
        assert f.multiplicative_n_undefined == 0
        assert f.multiplicative_max_factor == pytest.approx(np.max(f.plateau / f.S_model))

    def test_undefined_where_curve_and_plateau_differ_in_sign(self):
        """A fit whose curve crosses zero: values on the wrong side become NaN, with a warning."""
        d = pd.Series(np.linspace(0, 10, 500))
        S = pd.Series(-2 + 4 * (1 - np.exp(-d)))          # curve from -2 up to plateau +2
        f = fitted(ExponentialSaturationFit, S, d)
        with pytest.warns(RuntimeWarning, match="undefined"):
            out = f.correct("multiplicative")
        wrong_side = (f.S_model / f.plateau) <= 0
        assert wrong_side.any()
        assert out[wrong_side].isna().all()
        assert out[~wrong_side].notna().all()
        assert f.multiplicative_n_undefined == int(wrong_side.sum())

    def test_negative_curve_and_plateau_allowed(self):
        """Both negative: the factor is positive and the correction is defined everywhere."""
        S, d, _ = truncated_scores("multiplicative")
        f = fitted(ExponentialSaturationFit, -S, d)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            out = f.correct("multiplicative")
        assert out.notna().all()
        np.testing.assert_allclose(out, -S * f.plateau / f.S_model)

    def test_zero_plateau_is_undefined(self):
        d = pd.Series(np.linspace(0, 10, 300))
        f = fitted(ExponentialDecayFit, pd.Series(5 * np.exp(-d)), d)
        with pytest.warns(RuntimeWarning):
            out = f.correct("multiplicative")
        assert out.isna().all()

    def test_constant_fit_is_identity(self):
        S, d, _ = truncated_scores("multiplicative")
        f = fitted(ConstantFit, S, d)
        pd.testing.assert_series_equal(f.correct("multiplicative"), S, check_names=False)

    def test_nan_input_rows_stay_nan(self):
        S, d, _ = truncated_scores("multiplicative")
        S = S.copy()
        S.iloc[:10] = np.nan
        out = fitted(ExponentialSaturationFit, S, d).correct("multiplicative")
        assert out.iloc[:10].isna().all() and out.iloc[10:].notna().all()


class TestRecoveryAndDiagnostic:

    @pytest.mark.parametrize("kind", ["multiplicative", "additive"])
    def test_matching_correction_recovers_ranking_better(self, kind):
        S, d, T = truncated_scores(kind)
        f = fitted(ExponentialSaturationFit, S, d)
        rho = {m: spearmanr(f.correct(m), T)[0] for m in ["additive", "multiplicative"]}
        other = "additive" if kind == "multiplicative" else "multiplicative"
        assert rho[kind] > rho[other]

    @pytest.mark.parametrize("kind", ["multiplicative", "additive"])
    def test_suggestion_matches_generating_model(self, kind):
        S, d, _ = truncated_scores(kind)
        diag = fitted(ExponentialSaturationFit, S, d).suggest_correction_mode()
        assert diag["correction_mode_suggested"] == kind
        assert diag["fraction_excluded_from_diagnostic"] == 0.0
        if kind == "multiplicative":
            assert diag["variance_model_dAIC"] < 0 and diag["gamma"] > 0.5
        else:
            assert diag["variance_model_dAIC"] > 0 and diag["gamma"] < 0.5

    def test_constant_fit_suggests_additive(self):
        S, d, _ = truncated_scores("multiplicative")
        diag = fitted(ConstantFit, S, d).suggest_correction_mode()
        assert diag["correction_mode_suggested"] == "additive"
        assert np.isnan(diag["variance_model_dAIC"])

    def test_too_many_undefined_points_fall_back_to_additive(self):
        d = pd.Series(np.linspace(0, 10, 500))
        S = pd.Series(-2 + 4 * (1 - np.exp(-0.3 * d)))   # long stretch on the wrong side of 0
        f = fitted(ExponentialSaturationFit, S, d)
        diag = f.suggest_correction_mode(max_excluded_fraction=0.1)
        assert diag["fraction_excluded_from_diagnostic"] > 0.1
        assert diag["correction_mode_suggested"] == "additive"
        assert "undefined" in diag["diagnostic_note"]


class TestProportionalLogLikelihood:

    def test_matches_direct_gaussian_sum(self):
        rng = np.random.default_rng(1)
        f = rng.uniform(1, 10, 400)
        y = f * (1 + 0.2 * rng.standard_normal(400))
        sigma0 = np.sqrt(np.mean(((y - f) / f) ** 2))
        direct = norm.logpdf(y, loc=f, scale=sigma0 * np.abs(f)).sum()
        assert proportional_log_likelihood(y, f) == pytest.approx(direct, rel=1e-10)


class TestFlowCorrection:

    def _flow(self, correction):
        S, d, _ = truncated_scores("multiplicative")
        flow = Flow.from_distances_and_scores(d.rename("distance"), pd.DataFrame({"score": S}))
        flow.flow(fits=[ConstantFit, ExponentialSaturationFit], correction=correction)
        return flow

    def test_default_equals_explicit_additive(self):
        S, d, _ = truncated_scores("multiplicative")
        a = Flow.from_distances_and_scores(d.rename("distance"), pd.DataFrame({"score": S}))
        a.flow(fits=[ConstantFit, ExponentialSaturationFit])
        b = self._flow("additive")
        pd.testing.assert_series_equal(a.observations["BOSPERRUS corrected score"],
                                       b.observations["BOSPERRUS corrected score"])

    def test_auto_applies_suggestion_and_reports_it(self):
        flow = self._flow("auto")
        fq = flow.fit_quality["score"]
        assert fq["correction_mode_suggested"] == "multiplicative"
        assert fq["correction_mode"] == "multiplicative"
        best = flow.best_fits["score"]
        np.testing.assert_allclose(flow.observations["BOSPERRUS corrected score"],
                                   flow.observations["score"] * best.plateau / best.S_model)

    def test_diagnostic_reported_in_additive_mode(self):
        fq = self._flow("additive").fit_quality["score"]
        assert fq["correction_mode"] == "additive"
        assert fq["correction_mode_suggested"] == "multiplicative"
        assert np.isfinite(fq["gamma"]) and np.isfinite(fq["variance_model_dAIC"])

    def test_invalid_correction_raises(self):
        with pytest.raises(ValueError):
            self._flow("both")
