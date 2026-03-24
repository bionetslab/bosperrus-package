import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from evaluate_fit import log_likelihood, akaike_information_criterion


class Fit():
    def __init__(self, C_true: pd.DataFrame | pd.Series, d: pd.DataFrame | pd.Series):
        self.C_true = C_true 
        self.d = d
        self.AIC = np.inf
        self.log_likelihood = -np.inf
        self.params = None
        self.C_model = None
        self.effect_strength = 0
        self.relative_support = 0
        self.sign_border_effect = 0

    
    def _fit_properties(self):
        raise NotImplementedError("Subclasses should implement this method")

        
    def fit(self):
        raise NotImplementedError("Subclasses should implement this method")


    def correct(self):
        if self.C_model is None:
            raise ValueError("Model has not been fitted yet")
        return self.C_true - self.C_model
    

    def score(self):
        if self.C_model is None:
            raise ValueError("Model has not been fitted yet")
        
        self.log_likelihood = log_likelihood(self.C_true, self.C_model)
        self.AIC = akaike_information_criterion(len(self.params) + 1, self.log_likelihood)
        return

    
class ConstantFit(Fit):
    def __init__(self, C_true: pd.DataFrame | pd.Series, d: pd.DataFrame | pd.Series):
        super().__init__(C_true, d)
        
    
    def _fit_properties(self):
        return

    def _fit_constant(self, C):
        c = np.mean(C)
        C_model = np.full_like(C, c, dtype=float)
        return c, C_model

    def fit(self):
        c, C_model = self._fit_constant(self.C_true)
        self.C_model = C_model
        self.params = {"c": c}
        self.score()
        return
        
        
    def correct(self):
        return
    
    

class PiecewiseLinearFit(Fit):
    def __init__(self, C_true: pd.DataFrame | pd.Series, d: pd.DataFrame | pd.Series):
        super().__init__(C_true, d)
    
    @staticmethod
    def piecewise_plateau(d, b, m, c0):
        return np.where(
                d <= b,
                m * d + c0,
                m * b + c0 
            )
    
    
    def _fit_properties(self):
        self.sign_border_effect = 1 if self.params["m"] > 0 else -1
        self.effect_strength = -(self.params["m"] * self.params["b"]) / ((self.params["m"] * self.params["b"]) + self.params["c0"])
        # Support is simply the breakpoint where it hits the plateau
        self.relative_support = self.params["b"] / np.max(self.d)
        return 
    
    
    def _fit_piece_wise_linear(self):        
        # initial guesses
        p0 = [np.median(self.d), 1.0, np.mean(self.C_true)]

        lower_bounds = [0, -np.inf, -np.inf]  # b >= 0 # b smaller than max?
        upper_bounds = [np.inf, np.inf, np.inf]
        
        p_opt, _ = curve_fit(self.piecewise_plateau, self.d, self.C_true, p0=p0, bounds=(lower_bounds, upper_bounds))
        b_opt, m_opt, c0_opt = p_opt
        C_fit = self.piecewise_plateau(self.d, b_opt, m_opt, c0_opt)
        return m_opt, c0_opt, b_opt, C_fit


    def fit(self):
        # initial guesses
        m_opt, c0_opt, b_opt, C_fit = self._fit_piece_wise_linear()
        self.params = {"b": b_opt, "m": m_opt, "c0": c0_opt}
        self.C_model = C_fit
        self.score()
        return
    

class ExponentialSaturationFit(Fit):
    def __init__(self, C_true: pd.DataFrame | pd.Series, d: pd.DataFrame | pd.Series):
        super().__init__(C_true, d)
        
    @staticmethod
    def exp_sat(d, a, b, c):
        return a * (1 - np.exp(-b * d)) + c
    
        
    def _fit_properties(self):
        self.sign_border_effect = 1 if self.params["m"] > 0 else -1
        self.effect_strength = -self.params["a"] / (self.params["a"] + self.params["c"])
        self.relative_support = (-np.log(0.05) / self.params["b"]) / np.max(self.d) # point where it reaches 95% of the saturation level
        return 
    
    
    def _fit_exponential_saturation(self):
        # initial guesses
        p0 = [max(self.C_true) - min(self.C_true), 1 / (np.mean(self.d) + 1e-6), min(self.C_true)]

        popt, _ = curve_fit(self.exp_sat, self.d, self.C_true, p0=p0, maxfev=5000)
        a_opt, b_opt, c_opt = popt
        C_fit = self.exp_sat(self.d, a_opt, b_opt, c_opt)
        return a_opt, b_opt, c_opt, C_fit


    def fit(self):
        a_opt, b_opt, c_opt, C_fit = self._fit_exponential_saturation()
        self.params = {"a": a_opt, "b": b_opt, "c": c_opt}
        self.C_model = C_fit
        self.score()
        return
    
    
# TODO: 2D versions of these models for future work
       
    
def exp_sat_2D(d0, d1, a0, a1, b0, b1, c):
    return a0 * (1 - np.exp(-b0 * d0)) + a1 * (1 - np.exp(-b1 * d1)) + c 


def piecewise_plateau_2D(d0, d1, b0, b1, m0, m1, c):
    d0_term = np.where(
            d0 <= b0,
            m0 * d0,
            m0 * b0
        )
    d1_term = np.where(
            d1 <= b1,
            m1 * d1,
            m1 * b1
        )
    return d0_term + d1_term + c
    



