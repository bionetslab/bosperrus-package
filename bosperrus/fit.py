import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, differential_evolution
from .evaluate_fit import log_likelihood, akaike_information_criterion

_EPS = 1e-10

class Fit():
    def __init__(self, S_true: pd.DataFrame | pd.Series, d: pd.DataFrame | pd.Series):
        """
        Base class for all curve-fitting models applied to S_true as a function of d.

        Filters out non-finite values from both S_true and d before fitting, storing
        the mask so corrected outputs can be expanded back to the original index.

        Parameters
        ----------
        S_true : pd.DataFrame | pd.Series
            Observed scores (e.g. measured pixel intensities or centrality metrics).
        d : pd.DataFrame | pd.Series
            Independent variable (e.g. distance from border), aligned with S_true.

        Attributes set here are read-only via properties. Subclasses write
        directly to the private attributes (e.g. self._AIC, self._params) rather
        than going through the properties, since the properties are read-only.

        Subclasses must implement:
            fit()                          -- estimate model parameters from data
            correct()                      -- apply the fitted model to produce S_corrected
            _rate_observed_metrics()       -- populate _observed_effect_strength and
                                             _observed_half_life from fitted params
            _calculate_fraction_not_converged() -- populate _fraction_not_converged;
                                             the threshold argument may be ignored by
                                             implementations where convergence is a
                                             hard boundary rather than asymptotic
        """
        
        if len(S_true) != len(d):
            raise ValueError(f"S_true and d must have the same length, got {len(S_true)} and {len(d)}")

        self._S_true_original = S_true.copy()  # Original data with full index
        
        mask = np.isfinite(S_true) & np.isfinite(d)
        self._S_true = S_true[mask].copy()
        self._d = d[mask].copy()
        
        if len(S_true[mask]) == 0:
            raise ValueError("No valid data points to fit after filtering non-finite values.")
        
        self._mask = mask.copy()  # store the mask, not really meant to be visible from outside
        self._included_samples = np.sum(mask) / len(mask)
        
        self._AIC = np.inf
        self._log_likelihood = -np.inf
        self._observed_effect_strength = None
        self._observed_half_life = None
        self._fraction_not_converged = None
        
        self._params = None
        self._S_model = None
        self._S_corrected = None
        self._converged = False # not really meaningful to outside
        self._name = None
        
        # These are only modified when fits are compared against each other from the Flow class
        self.entropy_AIC_weights = None
        self.scaled_relative_loglikelihood_over_baseline = None

    def __repr__(self):
        status = "converged" if self._converged else ("not fitted" if self._params is None else "failed to converge")
        n = len(self._S_true)
        pct = f"{self._included_samples * 100:.1f}%"
        aic = f"{self._AIC:.2f}" if np.isfinite(self._AIC) else "—"
        params = ", ".join(f"{k}={v:.3g}" for k, v in self._params.items()) if self._params else "—"
        return (
            f"{self.__class__.__name__}("
            f"status={status}, n={n} [{pct} included], "
            f"AIC={aic}, params=[{params}])"
        )
        
    def params_summary(self):
        summary = {
            "best_fit_type": self._name,
            "entropy_AIC_weights": self.entropy_AIC_weights,
            "observed_half_life": self.observed_half_life,
            "observed_effect_strength": self.observed_effect_strength,
            "included samples": self.included_samples,
            "affected samples": self.fraction_not_converged,
            "scaled_relative_likelihood_over_baseline": self.scaled_relative_loglikelihood_over_baseline,
        }
        summary.update(self.params)
        return summary

    @property
    def included_samples(self):
        """Read-only access to included_samples (set only during construction)."""
        return self._included_samples

    @property
    def AIC(self):
        """Read-only access to AIC."""
        return self._AIC

    @property
    def log_likelihood(self):
        """Read-only access to log_likelihood."""
        return self._log_likelihood

    @property
    def observed_effect_strength(self):
        """Read-only access to observed_effect_strength."""
        return self._observed_effect_strength

    @property
    def observed_half_life(self):
        """Read-only access to observed_half_life."""
        return self._observed_half_life
    
    @property
    def fraction_not_converged(self):
        """Read-only access to fraction_not_converged."""
        return self._fraction_not_converged

    @property
    def params(self):
        """Read-only access to params."""
        return self._params

    @property
    def S_model(self):
        """Read-only access to S_model."""
        return self._S_model

    @property
    def S_corrected(self):
        """Read-only access to S_corrected."""
        if self._S_corrected is None:
            return None
        return self._expand_to_original_index(self._S_corrected)

    @property
    def name(self):
        """Read-only access to name."""
        return self._name

    def _expand_to_original_index(self, filtered_data):
        """Expand filtered data back to original index."""
        if isinstance(self._S_true_original, pd.DataFrame):
            result = pd.DataFrame(np.nan, index=range(len(self._S_true_original)), columns=self._S_true_original.columns)
            result[self._mask] = filtered_data
            return result
        else:  # pd.Series
            result = pd.Series(np.nan, index=range(len(self._S_true_original)), dtype=filtered_data.dtype)
            result[self._mask] = filtered_data
            return result

    def fit(self):
        """
        Estimate model parameters from S_true and d.

        Must set self._params, self._S_model, and self._converged.
        Should call self._rate_observed_metrics(), self._calculate_fraction_not_converged(),
        and self._score() on success.
        """
        raise NotImplementedError("Subclasses should implement this method")

    def correct(self):
        """
        Apply the fitted model to produce a border-effect-corrected signal.

        Must set self._S_corrected and return self.S_corrected (the property),
        which automatically expands the result back to the original index.
        """
        raise NotImplementedError("Subclasses should implement this method")
    
    def fit_correct(self):
        """Convenience method: fit the model and, if converged, return the corrected signal."""
        self.fit()
        if self._converged:
            self.correct()
        return self.S_corrected

    def _rate_observed_metrics(self):
        """
        Compute and store observed_effect_strength and observed_half_life from fitted params.

        Must set self._observed_effect_strength and self._observed_half_life.
        Should handle the non-converged case by assigning np.nan to both.
        """
        raise NotImplementedError("Subclasses should implement this method")

    def _calculate_fraction_not_converged(self, threshold: float = 0.95) -> float:
        """
        Compute and store the fraction of nodes not yet converged.

        Must set self._fraction_not_converged.
        The threshold parameter defines the saturation level considered 'converged'
        (e.g. 0.95 = 95% of asymptote). Implementations with a hard convergence
        boundary (e.g. PiecewiseLinearFit) may ignore threshold and should document
        this behaviour in their docstring.
        """
        raise NotImplementedError("Subclasses should implement this method")

    def _score(self):
        """
        Compute log-likelihood and AIC for the fitted model.

        Reads self.S_model and self._params. Should only be called after a
        successful fit, i.e. when self._converged is True.
        """
        if self.S_model is None:
            raise ValueError("Model has not been fitted yet")
        self._log_likelihood = log_likelihood(self._S_true, self.S_model)
        self._AIC = akaike_information_criterion(len(self.params) + 1, self.log_likelihood) # +1 for the variance term in the likelihood, which is estimated from the residuals and thus counts as an additional parameter
        return


class ConstantFit(Fit):
    def __init__(self, S_true: pd.DataFrame | pd.Series, d: pd.DataFrame | pd.Series = None):
        """
        Fits a horizontal constant (the mean of S_true) as the null/baseline model.

        A constant fit is trivially always converged and has no border effect, so
        observed_effect_strength and observed_half_life are both 0, and
        fraction_not_converged is 0. This model is primarily used as the AIC
        baseline against which more complex fits are compared.

        Parameters
        ----------
        S_true : pd.DataFrame | pd.Series
            Observed signal values.
        d : pd.DataFrame | pd.Series, optional
            Not used for fitting but accepted for API consistency with other subclasses.
        """
        super().__init__(S_true, d)
        self._name = "Constant Fit"

    def _fit_constant(self, C):
        c = np.mean(C)
        S_model = np.full_like(C, c, dtype=float)
        return c, S_model

    def _rate_observed_metrics(self):
        self._observed_effect_strength = 0
        self._observed_half_life = 0

    def _calculate_fraction_not_converged(self, threshold: float = 0.95) -> float:
        """
        A constant function is trivially converged everywhere.
        Returns 0.0 regardless of threshold.
        """
        self._fraction_not_converged = 0.0

    def fit(self):
        c, S_model = self._fit_constant(self._S_true)
        self._S_model = S_model
        self._S_corrected = self._S_true
        self._params = {"constant_c": c}
        self._rate_observed_metrics()
        self._calculate_fraction_not_converged()
        self._score()
        self._converged = True

    def correct(self):
        if self._params is None:
            raise RuntimeError("fit() must be called before correct()")
        self._S_corrected = self._S_true
        return self.S_corrected


class PiecewiseLinearFit(Fit):
    def __init__(self, S_true: pd.DataFrame | pd.Series, d: pd.DataFrame | pd.Series):
        """
        Fits a piecewise linear model with a plateau: S(d) = m*d + c for d <= b,
        and S(d) = m*b + c for d > b, where b is the knot (breakpoint).

        The fit is initialised with scipy's curve_fit and optionally refined with
        differential evolution seeded around the curve_fit solution, to avoid
        local minima near the knot location.

        Convergence is structurally defined by the knot b: nodes with d > b are
        on the plateau and considered fully converged; nodes with d <= b are in
        the border-affected linear regime.

        Parameters
        ----------
        S_true : pd.DataFrame | pd.Series
            Observed signal values.
        d : pd.DataFrame | pd.Series
            Distance from border, aligned with S_true.
        """
        super().__init__(S_true, d)
        self._name = "Piecewise Linear Fit"

    @staticmethod
    def piecewise_plateau(d, b, m, c):
        return np.where(d <= b, m * d + c, m * b + c)

    def _refine_fit(self, p_opt):
        def residuals(params):
            return np.sum((self.piecewise_plateau(self._d, *params) - self._S_true) ** 2)
        
        # Build a search region around the curve_fit solution
        spread = np.abs(p_opt) * 0.5 + 1e-6  # 50% spread around solution, avoid zero
        de_bounds = [
            (max(p_opt[0] - spread[0], np.min(self._d)), min(p_opt[0] + spread[0], np.max(self._d))),  # b
            (p_opt[1] - spread[1], p_opt[1] + spread[1]),  # m
            (p_opt[2] - spread[2], p_opt[2] + spread[2]),  # c
        ]
        
        # Seed the population around the curve_fit solution
        rng = np.random.default_rng(42)
        popsize = 15
        population = p_opt + rng.uniform(-0.5, 0.5, size=(popsize, len(p_opt))) * spread
        
        de_result = differential_evolution(
            residuals,
            bounds=de_bounds,
            init=population,
            seed=42,
            tol=1e-10,
        )
        
        # Only accept if better (guaranteed not to be worse due to init)
        if de_result.fun < residuals(p_opt):
            p_opt = de_result.x
        return p_opt

    def _fit_piece_wise_linear(self, refine_fit):
        p0 = [np.median(self._d), 1.0, np.mean(self._S_true)]
        lower_bounds = [np.min(self._d), -np.inf, -np.inf]
        upper_bounds = [np.max(self._d), np.inf, np.inf]
        p_opt, _ = curve_fit(self.piecewise_plateau, self._d, self._S_true, p0=p0, bounds=(lower_bounds, upper_bounds), maxfev=2000)
        
        if refine_fit:
            p_opt = self._refine_fit(p_opt)
        
        b_opt, m_opt, c_opt = p_opt
        C_fit = self.piecewise_plateau(self._d, b_opt, m_opt, c_opt)
        return b_opt, m_opt, c_opt, C_fit
    
    def _rate_observed_metrics(self):
        if not self._converged:
            self._observed_effect_strength = np.nan
            self._observed_half_life = np.nan
            return
        d_min, d_max = np.min(self._d), np.max(self._d)
        c_border = self.piecewise_plateau(d_min, self.params["piecewise_linear_b"], self.params["piecewise_linear_m"], self.params["piecewise_linear_c"])
        c_center = self.piecewise_plateau(d_max, self.params["piecewise_linear_b"], self.params["piecewise_linear_m"], self.params["piecewise_linear_c"])
        
        self._observed_effect_strength = (c_border - c_center) / (c_border + c_center + _EPS)
        
        y_mid = (c_border + c_center) / 2
        self._observed_half_life = (y_mid - self.params["piecewise_linear_c"]) / (self.params["piecewise_linear_m"] + _EPS)

    def _calculate_fraction_not_converged(self, threshold: float = 0.95) -> float:
        """
        For the piecewise linear model, convergence is structurally defined by the
        knot b: nodes at d > b are on the plateau and fully converged. Nodes at d <= b
        are still in the linear (border-affected) regime.

        The threshold parameter is accepted for API consistency but not used, because
        the transition is a hard boundary (not asymptotic). To reflect this, the method
        returns the fraction of nodes strictly left of the knot.
        """
        if self.params is None:
            raise ValueError("Model has not been fitted yet")
        if not self._converged:
            return np.nan
        b = self.params["piecewise_linear_b"]
        self._fraction_not_converged = float(np.mean(self._d <= b))

    def fit(self, refine_fit=True):
        try:
            b_opt, m_opt, c_opt, C_fit = self._fit_piece_wise_linear(refine_fit=refine_fit)
            self._params = {"piecewise_linear_b": b_opt, "piecewise_linear_m": m_opt, "piecewise_linear_c": c_opt}
            self._S_model = C_fit
            self._converged = True
        except RuntimeError:
            self._params = {"piecewise_linear_b": np.nan, "piecewise_linear_m": np.nan, "piecewise_linear_c": np.nan}
            self._S_model = self._S_true
            self._converged = False
        
        if self._converged:
            self._rate_observed_metrics()
            self._calculate_fraction_not_converged()
            self._score()

    def correct(self):
        if self._params is None:
            raise RuntimeError("fit() must be called before correct()")
        if self._converged:
            self._S_corrected = self._S_true + self.params["piecewise_linear_m"] * self.params["piecewise_linear_b"] + self.params["piecewise_linear_c"] - self.S_model
        else:
            self._S_corrected = self._S_true
        return self.S_corrected # this calls the getter function of the property and extends to original index


class ExponentialSaturationFit(Fit):
    def __init__(self, S_true: pd.DataFrame | pd.Series, d: pd.DataFrame | pd.Series):
        """
        Fits an exponential saturation model: S(d) = a*(1 - exp(-b*d)) + c,
        which asymptotes to a + c as d -> infinity.

        Parameters a, b, c represent amplitude, rate, and offset respectively.
        b is constrained to be positive (monotonic approach to saturation).

        A node is considered converged once it has reached `threshold` of the
        total range [c, a+c], i.e. when d >= -ln(1 - threshold) / b.

        Parameters
        ----------
        S_true : pd.DataFrame | pd.Series
            Observed signal values.
        d : pd.DataFrame | pd.Series
            Distance from border, aligned with S_true.
        """
        super().__init__(S_true, d)
        self._name = "Exponential Saturation Fit"

    @staticmethod
    def exp_sat(d, a, b, c):
        return a * (1 - np.exp(-b * d)) + c

    def _fit_exponential_saturation(self):
        p0 = [max(self._S_true) - min(self._S_true), 1 / (np.mean(self._d) + 1e-6), min(self._S_true)]
        bounds = ([-np.inf, 1e-6, -np.inf], [np.inf, np.inf, np.inf])
        popt, _ = curve_fit(self.exp_sat, self._d, self._S_true, p0=p0, bounds=bounds, maxfev=2000)
        a_opt, b_opt, c_opt = popt
        C_fit = self.exp_sat(self._d, a_opt, b_opt, c_opt)
        return a_opt, b_opt, c_opt, C_fit

    def _rate_observed_metrics(self):
        if not self._converged:
            self._observed_effect_strength = np.nan
            self._observed_half_life = np.nan
            return
        d_min = np.min(self._d)
        d_max = np.max(self._d)
        c_border = self.exp_sat(d_min, self.params["exponential_saturation_a"], self.params["exponential_saturation_b"], self.params["exponential_saturation_c"])
        c_center = self.exp_sat(d_max, self.params["exponential_saturation_a"], self.params["exponential_saturation_b"], self.params["exponential_saturation_c"])
        
        self._observed_effect_strength = (c_border - c_center) / (c_border + c_center + _EPS)
        
        C_mid = (c_border + c_center) / 2
        self._observed_half_life = -(1 / self.params["exponential_saturation_b"]) * np.log(1 - (C_mid - self.params["exponential_saturation_c"]) / (self.params["exponential_saturation_a"] + _EPS))

    def _calculate_fraction_not_converged(self, threshold: float = 0.95) -> float:
        """
        The exponential saturation a*(1 - exp(-b*d)) + c asymptotes to a + c.
        A point is considered converged once it has reached `threshold` of the
        total range [c, a+c], i.e. when:

            a * (1 - exp(-b*d)) >= threshold * a
            => d >= -ln(1 - threshold) / b

        Returns the fraction of observed nodes below this convergence distance.
        """
        if self.params is None:
            raise ValueError("Model has not been fitted yet")
        if not self._converged:
            self._fraction_not_converged = np.nan
            return
        if not (0 < threshold < 1):
            raise ValueError("threshold must be in (0, 1)")
        b = self.params["exponential_saturation_b"]
        d_converge = -np.log(1 - threshold) / b
        self._fraction_not_converged = float(np.mean(self._d < d_converge))

    def fit(self):
        try:
            a_opt, b_opt, c_opt, C_fit = self._fit_exponential_saturation()
            self._params = {"exponential_saturation_a": a_opt, "exponential_saturation_b": b_opt, "exponential_saturation_c": c_opt}
            self._S_model = C_fit
            self._converged = True
        except RuntimeError:
            self._converged = False
            self._params = {"exponential_saturation_a": np.nan, "exponential_saturation_b": np.nan, "exponential_saturation_c": np.nan}
            self._S_model = self._S_true
        
        if self._converged:
            self._rate_observed_metrics()
            self._calculate_fraction_not_converged()
            self._score()

    def correct(self):
        if self._params is None:
            raise RuntimeError("fit() must be called before correct()")
        if self._converged:
            self._S_corrected = self._S_true + self.params["exponential_saturation_a"] + self.params["exponential_saturation_c"] - self.S_model
        else:
            self._S_corrected = self._S_true
        return self.S_corrected


class MichaelisMentenFit(Fit):
    def __init__(self, S_true: pd.DataFrame | pd.Series, d: pd.DataFrame | pd.Series):
        """
        Fits a Michaelis-Menten (hyperbolic saturation) model: S(d) = a*d/(b+d) + c,
        which asymptotes to a + c as d -> infinity.

        Parameters a, b, c represent maximum amplitude, half-saturation constant,
        and offset respectively. b is the distance at which 50% of the maximum
        effect is reached (analogous to the biochemical Km).

        A node is considered converged once the saturable term a*d/(b+d) has
        reached `threshold` of its maximum a, i.e. when d >= b*threshold/(1-threshold).

        Parameters
        ----------
        S_true : pd.DataFrame | pd.Series
            Observed signal values.
        d : pd.DataFrame | pd.Series
            Distance from border, aligned with S_true.
        """
        super().__init__(S_true, d)
        self._name = "Michaelis-Menten Fit"

    @staticmethod
    def michaelis_menten(d, a, b, c):
        return a * d / (b + d) + c

    def _fit_michaelis_menten(self):
        p0 = [np.max(self._S_true) - np.min(self._S_true), np.median(self._d), np.min(self._S_true)]
        bounds = ([-np.inf, 1e-6, -np.inf], [np.inf, np.inf, np.inf])
        popt, _ = curve_fit(self.michaelis_menten, self._d, self._S_true, p0=p0, bounds=bounds, maxfev=2000)
        a_opt, b_opt, c_opt = popt
        C_fit = self.michaelis_menten(self._d, a_opt, b_opt, c_opt)
        return a_opt, b_opt, c_opt, C_fit

    def _rate_observed_metrics(self):
        if not self._converged:
            self._observed_effect_strength = np.nan
            self._observed_half_life = np.nan
            return
        d_max = np.max(self._d)
        d_min = np.min(self._d)
        c_border = self.michaelis_menten(d_min, self.params["michaelis_menten_a"], self.params["michaelis_menten_b"], self.params["michaelis_menten_c"])
        c_center = self.michaelis_menten(d_max, self.params["michaelis_menten_a"], self.params["michaelis_menten_b"], self.params["michaelis_menten_c"])
        
        self._observed_effect_strength = (c_border - c_center) / (c_border + c_center + _EPS)
        
        C_mid = (c_border + c_center) / 2
        self._observed_half_life = self.params["michaelis_menten_b"] * (C_mid - self.params["michaelis_menten_c"]) / (self.params["michaelis_menten_a"] - C_mid + self.params["michaelis_menten_c"] + _EPS)

    def _calculate_fraction_not_converged(self, threshold: float = 0.95) -> float:
        """
        The Michaelis-Menten function a*d/(b+d) + c asymptotes to a + c.
        A point is considered converged once the saturable term a*d/(b+d)
        has reached `threshold` of its maximum a:

            d / (b + d) >= threshold
            => d >= b * threshold / (1 - threshold)

        This is analogous to the standard Km definition: at d = b (the
        half-saturation constant), exactly 50% of the maximum is reached.

        Returns the fraction of observed nodes below this convergence distance.
        """
        if self.params is None:
            raise ValueError("Model has not been fitted yet")
        if not self._converged:
            return np.nan
        if not (0 < threshold < 1):
            raise ValueError("threshold must be in (0, 1)")
        b = self.params["michaelis_menten_b"]
        d_converge = b * threshold / (1 - threshold)
        self._fraction_not_converged = float(np.mean(self._d < d_converge))

    def fit(self):
        try:
            a_opt, b_opt, c_opt, C_fit = self._fit_michaelis_menten()
            self._params = {"michaelis_menten_a": a_opt, "michaelis_menten_b": b_opt, "michaelis_menten_c": c_opt}
            self._S_model = C_fit
            self._converged = True
        except RuntimeError:
            self._params = {"michaelis_menten_a": np.nan, "michaelis_menten_b": np.nan, "michaelis_menten_c": np.nan}
            self._S_model = self._S_true
            self._converged = False
        
        if self._converged:
            self._rate_observed_metrics()
            self._calculate_fraction_not_converged()
            self._score()


    def correct(self):
        if self._params is None:
            raise RuntimeError("fit() must be called before correct()")
        if self._converged:
            self._S_corrected = self._S_true + self.params["michaelis_menten_a"] + self.params["michaelis_menten_c"] - self.S_model
        else:
            self._S_corrected = self._S_true
        return self.S_corrected
    
