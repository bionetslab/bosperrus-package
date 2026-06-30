import numpy as np

__all__ = ['log_likelihood', 'akaike_information_criterion', 'relative_likelihood', 'calculate_AIC_weight_entropy']

def log_likelihood(C_true, C_pred):
    n = len(C_true)
    residuals = C_true - C_pred
    sigma2 = np.mean(residuals**2)
    # Floor sigma2 so a perfect fit (sigma2=0) gets a very large finite log-likelihood
    # rather than +inf or -inf, keeping AIC-based model selection well-defined.
    sigma2 = max(sigma2, np.finfo(float).tiny)
    return -0.5 * n * (np.log(2 * np.pi * sigma2) + 1)

def akaike_information_criterion(num_params, log_likelihood_model):
    return 2 * num_params - 2 * log_likelihood_model

def relative_likelihood(aic_model, aic_baseline, N):
    return np.exp((aic_baseline - aic_model) / (2 * N))

def calculate_AIC_weight_entropy(rel_ll_values):
    if len(rel_ll_values) == 1:
        return 0.0
    weights = rel_ll_values / np.sum(rel_ll_values)
    entropy = -(weights * np.log(weights + 1e-15)).sum()
    entropy = entropy / np.log(len(weights))  # Normalize to [0,1]
    return entropy
    