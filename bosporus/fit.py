import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from .evaluate_fit import log_likelihood, akaike_information_criterion


class Fit():
    def __init__(self, C_true: pd.DataFrame | pd.Series, d: pd.DataFrame | pd.Series):
        mask = np.isfinite(C_true) & np.isfinite(d)
        self._C_true = C_true[mask]
        self._d = d[mask]
        self._included_samples = np.sum(mask) / len(mask)
        self._AIC = np.inf
        self._log_likelihood = -np.inf
        self._params = None
        self._C_model = None
        self._effect_strength = None
        self._relative_support = None
        self._name = None
        self._converged = True

    @property
    def C_true(self):
        """Read-only access to C_true (set only during construction)."""
        return self._C_true

    @property
    def d(self):
        """Read-only access to d (set only during construction)."""
        return self._d

    @property
    def included_samples(self):
        """Read-only access to included_samples."""
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
    def params(self):
        """Read-only access to params."""
        return self._params

    @property
    def C_model(self):
        """Read-only access to C_model."""
        return self._C_model

    @property
    def effect_strength(self):
        """Read-only access to effect_strength."""
        return self._effect_strength

    @property
    def relative_support(self):
        """Read-only access to relative_support."""
        return self._relative_support

    @property
    def name(self):
        """Read-only access to name."""
        return self._name

    @property
    def converged(self):
        """Read-only access to converged."""
        return self._converged

    @property
    def observed_effect_strength(self):
        """Read-only access to observed_effect_strength."""
        return getattr(self, '_observed_effect_strength', None)

    @property
    def observed_half_life(self):
        """Read-only access to observed_half_life."""
        return getattr(self, '_observed_half_life', None)

    def _rate_observed_effect_strength(self):
        raise NotImplementedError("Subclasses should implement this method")

    def _rate_observed_half_life(self):
        raise NotImplementedError("Subclasses should implement this method")

    def fit(self):
        raise NotImplementedError("Subclasses should implement this method")

    def fraction_not_converged(self, threshold: float = 0.95) -> float:
        raise NotImplementedError("Subclasses should implement this method")

    def correct(self):
        if self.C_model is None:
            raise ValueError("Model has not been fitted yet")
        return self.C_true - self.C_model

    def score(self):
        if self.C_model is None:
            raise ValueError("Model has not been fitted yet")
        self._log_likelihood = log_likelihood(self.C_true, self.C_model)
        self._AIC = akaike_information_criterion(len(self._params) + 1, self._log_likelihood)
        return


class ConstantFit(Fit):
    def __init__(self, C_true: pd.DataFrame | pd.Series, d: pd.DataFrame | pd.Series = None):
        super().__init__(C_true, d)
        self._name = "Constant Fit"

    def _fit_constant(self, C):
        c = np.mean(C)
        C_model = np.full_like(C, c, dtype=float)
        return c, C_model

    def _rate_observed_effect_strength(self):
        self._observed_effect_strength = 0

    def _rate_observed_half_life(self):
        self._observed_half_life = 0

    def fit(self):
        c, C_model = self._fit_constant(self.C_true)
        self._C_model = C_model
        self._params = {"constant_c": c}
        self._rate_observed_half_life()
        self._rate_observed_effect_strength()
        self.score()

    def fraction_not_converged(self, threshold: float = 0.95) -> float:
        """
        A constant function is trivially converged everywhere.
        Returns 0.0 regardless of threshold.
        """
        return 0.0

    def correct(self):
        return


class PiecewiseLinearFit(Fit):
    def __init__(self, C_true: pd.DataFrame | pd.Series, d: pd.DataFrame | pd.Series):
        super().__init__(C_true, d)
        self._name = "Piecewise Linear Fit"

    @staticmethod
    def piecewise_plateau(d, b, m, c):
        return np.where(d <= b, m * d + c, m * b + c)

    def _rate_observed_effect_strength(self):
        if not self.converged:
            self._observed_effect_strength = np.nan
            return
        d_min, d_max = np.min(self.d), np.max(self.d)
        c_border = self.piecewise_plateau(d_min, self.params["piecewise_linear_b"], self.params["piecewise_linear_m"], self.params["piecewise_linear_c"])
        c_center = self.piecewise_plateau(d_max, self.params["piecewise_linear_b"], self.params["piecewise_linear_m"], self.params["piecewise_linear_c"])
        self._observed_effect_strength = (c_center - c_border) / (c_center + 1e-10)

    def _rate_observed_half_life(self):
        if not self.converged:
            self._observed_half_life = np.nan
            return
        d_min, d_max = np.min(self.d), np.max(self.d)
        c_border = self.piecewise_plateau(d_min, self.params["piecewise_linear_b"], self.params["piecewise_linear_m"], self.params["piecewise_linear_c"])
        c_center = self.piecewise_plateau(d_max, self.params["piecewise_linear_b"], self.params["piecewise_linear_m"], self.params["piecewise_linear_c"])
        y_mid = (c_border + c_center) / 2
        self._observed_half_life = (y_mid - self.params["piecewise_linear_c"]) / (self.params["piecewise_linear_m"] + 1e-10)

    def _fit_piece_wise_linear(self):
        p0 = [np.median(self.d), 1.0, np.mean(self.C_true)]
        lower_bounds = [np.min(self.d), -np.inf, -np.inf]
        upper_bounds = [np.max(self.d), np.inf, np.inf]
        p_opt, _ = curve_fit(self.piecewise_plateau, self.d, self.C_true, p0=p0, bounds=(lower_bounds, upper_bounds), maxfev=2000)
        b_opt, m_opt, c_opt = p_opt
        C_fit = self.piecewise_plateau(self.d, b_opt, m_opt, c_opt)
        return m_opt, c_opt, b_opt, C_fit

    def fit(self):
        try:
            m_opt, c_opt, b_opt, C_fit = self._fit_piece_wise_linear()
            self._params = {"piecewise_linear_b": b_opt, "piecewise_linear_m": m_opt, "piecewise_linear_c": c_opt}
            self._C_model = C_fit
        except RuntimeError:
            self._converged = False
            self._params = {"piecewise_linear_b": np.nan, "piecewise_linear_m": np.nan, "piecewise_linear_c": np.nan}
        self._rate_observed_half_life()
        self._rate_observed_effect_strength()
        if self.converged:
            self.score()

    def fraction_not_converged(self, threshold: float = 0.95) -> float:
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
        if not self.converged:
            return np.nan
        b = self.params["piecewise_linear_b"]
        return float(np.mean(self.d <= b))


class ExponentialSaturationFit(Fit):
    def __init__(self, C_true: pd.DataFrame | pd.Series, d: pd.DataFrame | pd.Series):
        super().__init__(C_true, d)
        self._name = "Exponential Saturation Fit"

    @staticmethod
    def exp_sat(d, a, b, c):
        return a * (1 - np.exp(-b * d)) + c

    def _rate_observed_effect_strength(self):
        if not self.converged:
            self._observed_effect_strength = np.nan
            return
        d_min, d_max = np.min(self.d), np.max(self.d)
        c_border = self.exp_sat(d_min, self.params["exponential_saturation_a"], self.params["exponential_saturation_b"], self.params["exponential_saturation_c"])
        c_center = self.exp_sat(d_max, self.params["exponential_saturation_a"], self.params["exponential_saturation_b"], self.params["exponential_saturation_c"])
        self._observed_effect_strength = (c_center - c_border) / (c_center + 1e-10)

    def _rate_observed_half_life(self):
        if not self.converged:
            self._observed_half_life = np.nan
            return
        d_min = np.min(self.d)
        d_max = np.max(self.d)
        c_border = self.exp_sat(d_min, self.params["exponential_saturation_a"], self.params["exponential_saturation_b"], self.params["exponential_saturation_c"])
        c_center = self.exp_sat(d_max, self.params["exponential_saturation_a"], self.params["exponential_saturation_b"], self.params["exponential_saturation_c"])
        C_mid = (c_border + c_center) / 2
        self._observed_half_life = -(1 / self.params["exponential_saturation_b"]) * np.log(1 - (C_mid - self.params["exponential_saturation_c"]) / self.params["exponential_saturation_a"])

    def _fit_exponential_saturation(self):
        p0 = [max(self.C_true) - min(self.C_true), 1 / (np.mean(self.d) + 1e-6), min(self.C_true)]
        bounds = ([-np.inf, 1e-6, -np.inf], [np.inf, np.inf, np.inf])
        popt, _ = curve_fit(self.exp_sat, self.d, self.C_true, p0=p0, bounds=bounds, maxfev=2000)
        a_opt, b_opt, c_opt = popt
        C_fit = self.exp_sat(self.d, a_opt, b_opt, c_opt)
        return a_opt, b_opt, c_opt, C_fit

    def fit(self):
        try:
            a_opt, b_opt, c_opt, C_fit = self._fit_exponential_saturation()
            self._params = {"exponential_saturation_a": a_opt, "exponential_saturation_b": b_opt, "exponential_saturation_c": c_opt}
            self._C_model = C_fit
        except RuntimeError:
            self._converged = False
            self._params = {"exponential_saturation_a": np.nan, "exponential_saturation_b": np.nan, "exponential_saturation_c": np.nan}
        self._rate_observed_half_life()
        self._rate_observed_effect_strength()
        if self.converged:
            self.score()

    def fraction_not_converged(self, threshold: float = 0.95) -> float:
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
        if not self.converged:
            return np.nan
        if not (0 < threshold < 1):
            raise ValueError("threshold must be in (0, 1)")
        b = self.params["exponential_saturation_b"]
        d_converge = -np.log(1 - threshold) / b
        return float(np.mean(self.d < d_converge))


class MichaelisMentenFit(Fit):
    def __init__(self, C_true: pd.DataFrame | pd.Series, d: pd.DataFrame | pd.Series):
        super().__init__(C_true, d)
        self._name = "Michaelis-Menten Fit"

    @staticmethod
    def michaelis_menten(d, a, b, c):
        return a * d / (b + d) + c

    def _rate_observed_effect_strength(self):
        if not self.converged:
            self._observed_effect_strength = np.nan
            return
        d_max = np.max(self.d)
        d_min = np.min(self.d)
        c_border = self.michaelis_menten(d_min, self.params["michaelis_menten_a"], self.params["michaelis_menten_b"], self.params["michaelis_menten_c"])
        c_center = self.michaelis_menten(d_max, self.params["michaelis_menten_a"], self.params["michaelis_menten_b"], self.params["michaelis_menten_c"])
        self._observed_effect_strength = (c_center - c_border) / (c_center + 1e-10)

    def _rate_observed_half_life(self):
        if not self.converged:
            self._observed_half_life = np.nan
            return
        d_max = np.max(self.d)
        d_min = np.min(self.d)
        c_border = self.michaelis_menten(d_min, self.params["michaelis_menten_a"], self.params["michaelis_menten_b"], self.params["michaelis_menten_c"])
        c_center = self.michaelis_menten(d_max, self.params["michaelis_menten_a"], self.params["michaelis_menten_b"], self.params["michaelis_menten_c"])
        C_mid = (c_border + c_center) / 2
        self._observed_half_life = self.params["michaelis_menten_b"] * (C_mid - self.params["michaelis_menten_c"]) / (self.params["michaelis_menten_a"] - C_mid + self.params["michaelis_menten_c"])

    def _fit_michaelis_menten(self):
        p0 = [np.max(self.C_true) - np.min(self.C_true), np.median(self.d), np.min(self.C_true)]
        bounds = ([-np.inf, 1e-6, -np.inf], [np.inf, np.inf, np.inf])
        popt, _ = curve_fit(self.michaelis_menten, self.d, self.C_true, p0=p0, bounds=bounds, maxfev=2000)
        a_opt, b_opt, c_opt = popt
        C_fit = self.michaelis_menten(self.d, a_opt, b_opt, c_opt)
        return a_opt, b_opt, c_opt, C_fit

    def fit(self):
        try:
            a_opt, b_opt, c_opt, C_fit = self._fit_michaelis_menten()
            self._params = {"michaelis_menten_a": a_opt, "michaelis_menten_b": b_opt, "michaelis_menten_c": c_opt}
            self._C_model = C_fit
        except RuntimeError:
            self._converged = False
            self._params = {"michaelis_menten_a": np.nan, "michaelis_menten_b": np.nan, "michaelis_menten_c": np.nan}
        self._rate_observed_half_life()
        self._rate_observed_effect_strength()
        if self.converged:
            self.score()

    def fraction_not_converged(self, threshold: float = 0.95) -> float:
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
        if not self.converged:
            return np.nan
        if not (0 < threshold < 1):
            raise ValueError("threshold must be in (0, 1)")
        b = self.params["michaelis_menten_b"]
        d_converge = b * threshold / (1 - threshold)
        return float(np.mean(self.d < d_converge))