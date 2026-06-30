"""
Unit tests for evaluate_fit.py

Coverage:
- log_likelihood: finite result for noisy data; very large finite positive for perfect fit (sigma2 floored)
- akaike_information_criterion: exact formula check (2*k - 2*ll, no +1 here — +1 is
  added by _score() in Fit, not by the bare AIC function)
- relative_likelihood: same AIC → 1.0; lower AIC model → > 1.0
- calculate_AIC_weight_entropy: single element → 0.0, uniform → ~1.0, skewed → near 0.0
"""

import numpy as np
import pytest

from bosperrus.evaluate_fit import (
    log_likelihood,
    akaike_information_criterion,
    relative_likelihood,
    calculate_AIC_weight_entropy,
)


# ============================================================
# log_likelihood
# ============================================================

class TestLogLikelihood:

    def test_log_likelihood_finite_for_good_data(self):
        """Typical noisy data should give a finite log-likelihood."""
        rng = np.random.default_rng(42)
        C_true = rng.normal(loc=5.0, scale=1.0, size=100)
        C_pred = C_true + rng.normal(loc=0.0, scale=0.5, size=100)
        ll = log_likelihood(C_true, C_pred)
        assert np.isfinite(ll), f"Expected finite log-likelihood, got {ll}"

    def test_log_likelihood_finite_and_large_for_perfect_fit(self):
        """When C_true == C_pred, sigma2=0; the floor ensures a very large finite positive value
        (rather than +inf or -inf), so AIC-based model selection remains well-defined."""
        C_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        C_pred = C_true.copy()
        ll = log_likelihood(C_true, C_pred)
        assert np.isfinite(ll), f"Expected finite log-likelihood for perfect fit, got {ll}"
        assert ll > 0, f"Perfect fit should have positive log-likelihood (very small sigma2), got {ll}"

    def test_log_likelihood_decreases_with_larger_residuals(self):
        """Better-fitting predictions should have higher (less negative) log-likelihood."""
        rng = np.random.default_rng(0)
        C_true = rng.normal(5.0, 1.0, 200)
        C_pred_good = C_true + rng.normal(0, 0.1, 200)
        C_pred_bad = C_true + rng.normal(0, 2.0, 200)
        ll_good = log_likelihood(C_true, C_pred_good)
        ll_bad = log_likelihood(C_true, C_pred_bad)
        assert ll_good > ll_bad, "Good fit should have higher log-likelihood than bad fit"

    def test_log_likelihood_is_negative(self):
        """Gaussian log-likelihood should be negative for imperfect fits."""
        rng = np.random.default_rng(7)
        C_true = rng.normal(0, 1, 50)
        C_pred = C_true + rng.normal(0, 0.5, 50)
        ll = log_likelihood(C_true, C_pred)
        assert ll < 0, f"Expected negative log-likelihood, got {ll}"


# ============================================================
# akaike_information_criterion
# ============================================================

class TestAkaikInformationCriterion:

    def test_akaike_information_criterion_formula(self):
        """
        AIC formula is 2*num_params - 2*log_likelihood.
        With num_params=2, log_likelihood=-10:
          AIC = 2*2 - 2*(-10) = 4 + 20 = 24
        Note: the +1 for the estimated variance is added by Fit._score(), not here.
        """
        aic = akaike_information_criterion(num_params=2, log_likelihood_model=-10)
        assert aic == 24, f"Expected AIC=24, got {aic}"

    def test_aic_increases_with_num_params(self):
        """More parameters → higher AIC (penalizes complexity)."""
        ll = -50.0
        aic_2 = akaike_information_criterion(2, ll)
        aic_5 = akaike_information_criterion(5, ll)
        assert aic_5 > aic_2

    def test_aic_decreases_with_better_likelihood(self):
        """Better fit (higher ll) → lower AIC."""
        k = 3
        aic_bad = akaike_information_criterion(k, -100.0)
        aic_good = akaike_information_criterion(k, -20.0)
        assert aic_good < aic_bad

    def test_aic_zero_params(self):
        """Edge case: 0 parameters."""
        aic = akaike_information_criterion(0, -5.0)
        assert aic == pytest.approx(2 * 0 - 2 * (-5.0))  # = 10


# ============================================================
# relative_likelihood
# ============================================================

class TestRelativeLikelihood:

    def test_relative_likelihood_same_aic(self):
        """Same model vs itself: relative likelihood == 1.0."""
        aic = 42.0
        N = 100
        rl = relative_likelihood(aic_model=aic, aic_baseline=aic, N=N)
        assert rl == pytest.approx(1.0), f"Expected 1.0, got {rl}"

    def test_relative_likelihood_lower_aic_is_better(self):
        """A model with AIC 10 lower than the baseline should have rl > 1.0."""
        aic_baseline = 50.0
        aic_model = 40.0  # 10 lower = better
        N = 100
        rl = relative_likelihood(aic_model=aic_model, aic_baseline=aic_baseline, N=N)
        assert rl > 1.0, f"Better model should have relative likelihood > 1.0, got {rl}"

    def test_relative_likelihood_higher_aic_is_worse(self):
        """A model with AIC higher than baseline should have rl < 1.0."""
        aic_baseline = 50.0
        aic_model = 60.0  # worse
        N = 100
        rl = relative_likelihood(aic_model=aic_model, aic_baseline=aic_baseline, N=N)
        assert rl < 1.0, f"Worse model should have relative likelihood < 1.0, got {rl}"

    def test_relative_likelihood_scaled_by_N(self):
        """Larger N reduces the per-sample AIC difference effect."""
        delta = 10.0
        rl_small_N = relative_likelihood(0.0, delta, N=10)
        rl_large_N = relative_likelihood(0.0, delta, N=1000)
        # With large N the exponent shrinks: exp(delta/2N) is smaller
        assert rl_small_N > rl_large_N


# ============================================================
# calculate_AIC_weight_entropy
# ============================================================

class TestCalculateAICWeightEntropy:

    def test_calculate_AIC_weight_entropy_single_model(self):
        """Single-element array → entropy is 0.0 (undefined/trivial)."""
        entropy = calculate_AIC_weight_entropy(np.array([1.0]))
        assert entropy == 0.0, f"Expected 0.0 for single element, got {entropy}"

    def test_calculate_AIC_weight_entropy_uniform(self):
        """
        Uniform relative likelihoods → maximum entropy (close to 1.0 after normalisation).
        For n=10 equal weights the normalised entropy should be very close to 1.
        """
        n = 10
        uniform = np.ones(n)
        entropy = calculate_AIC_weight_entropy(uniform)
        assert entropy == pytest.approx(1.0, abs=1e-6), (
            f"Uniform weights should give entropy ~1.0, got {entropy}"
        )

    def test_calculate_AIC_weight_entropy_all_weight_one_model(self):
        """
        Heavily skewed weights (one model dominates) → entropy near 0.
        We simulate this by giving one model a very large relative likelihood.
        """
        skewed = np.array([1e6, 1.0, 1.0, 1.0, 1.0])
        entropy = calculate_AIC_weight_entropy(skewed)
        assert entropy < 0.1, (
            f"Heavily skewed weights should give entropy near 0, got {entropy}"
        )

    def test_calculate_AIC_weight_entropy_two_equal(self):
        """Two equal weights → normalised entropy == 1.0 (log(2)/log(2))."""
        two = np.array([1.0, 1.0])
        entropy = calculate_AIC_weight_entropy(two)
        assert entropy == pytest.approx(1.0, abs=1e-6)

    def test_calculate_AIC_weight_entropy_in_unit_interval(self):
        """Entropy should always be in [0, 1]."""
        rng = np.random.default_rng(99)
        for _ in range(20):
            n = rng.integers(2, 15)
            weights = rng.exponential(1.0, size=n)
            entropy = calculate_AIC_weight_entropy(weights)
            assert 0.0 <= entropy <= 1.0 + 1e-9, (
                f"Entropy out of [0,1]: {entropy}"
            )
