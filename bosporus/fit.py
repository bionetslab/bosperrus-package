import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from evaluate_fit import log_likelihood

def fit_constant(C):
    c = np.mean(C)
    C_model = np.full_like(C, c, dtype=float)
    return c, C_model


def exp_sat(d, a, b, c):
    return a * (1 - np.exp(-b * d)) + c

    
def fit_exponential_saturation(d, C):
    d = np.array(d, dtype=float)
    C = np.array(C, dtype=float)

    # initial guesses
    p0 = [max(C) - min(C), 1 / (np.mean(d) + 1e-6), min(C)]

    popt, _ = curve_fit(exp_sat, d, C, p0=p0, maxfev=5000)
    a, b, c = popt
    C_model = exp_sat(d, a, b, c)
    return a, b, c, C_model


def piecewise_plateau(x, b, m, c0):
    return np.where(
            x <= b,
            m * x + c0,
            m * b + c0 
        )
        
     
def fit_piece_wise_linear(d, C):
    d = np.array(d, dtype=float)
    C = np.array(C, dtype=float)
    
    # initial guesses
    p0 = [np.median(d), 1.0, np.mean(C)]

    lower_bounds = [0, -np.inf, -np.inf]  # b >= 0
    upper_bounds = [np.inf, np.inf, np.inf]
    
    p_opt, _ = curve_fit(piecewise_plateau, d, C, p0=p0, bounds=(lower_bounds, upper_bounds))
    b_opt, m_opt, c0_opt = p_opt
    C_fit = piecewise_plateau(d, b_opt, m_opt, c0_opt)
    return m_opt, c0_opt, b_opt, C_fit
    

def get_fits(dataset, dataset_df, measures = ["degree", "closeness", "betweenness", "harmonic", "clustering", "pagerank"]):
    d = dataset_df["distance_to_border"].values
    result_dfs = list()
    result = dict()
    for measure in measures:
        result[measure] = list()
        C_true = dataset_df[measure].values
        
        a, C_const = fit_constant(C_true)        
        ll = log_likelihood(C_true, C_const)
        result[measure] += [a, ll]
        
        m, c0, b, C_pieli = fit_piece_wise_linear(d, C_true)
        ll = log_likelihood(C_true, C_pieli)
        result[measure] += [m, c0, b, ll]
        
        a, b, c, C_exp = fit_exponential_saturation(d, C_true)
        ll = log_likelihood(C_true, C_exp)
        result[measure] += [a, b, c, ll]
        result_dfs.append(pd.DataFrame(result[measure], columns=[measure], index=["const_a", "const_ll", "pieli_m", "pieli_c", "pieli_b", "pieli_ll", "exp_a", "exp_b", "exp_c", "exp_ll"]).T)
    result = pd.concat(result_dfs)
    result["dataset"] = dataset
    result["num_nodes"] = len(C_true)
    return result

