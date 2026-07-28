import numpy as np

__all__ = ['log_likelihood', 'akaike_information_criterion', 'relative_likelihood',
           'scaled_relative_likelihood', 'calculate_AIC_weight_entropy']

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

def relative_likelihood(aic_model, aic_baseline):
    """Standard Akaike-weight relative likelihood: exp(-ΔAIC / 2), where
    ΔAIC = aic_model - aic_baseline. This is the correct input for AIC
    weights/entropy across a fixed set of models fit to the same N
    observations. Do not further divide by N for that purpose — see
    scaled_relative_likelihood, which does that for a different reason.

    Can legitimately overflow to +inf when aic_model is far better than
    aic_baseline (a real, meaningful "infinitely more likely" result, not an
    error) -- silenced here rather than left to warn on every such call.
    """
    with np.errstate(over="ignore"):
        return np.exp((aic_baseline - aic_model) / 2)

def scaled_relative_likelihood(aic_model, aic_baseline, N):
    """Sample-size-normalized relative likelihood: exp(-ΔAIC / (2N)). Puts the
    comparison on a roughly per-observation scale, so it's meaningful to
    compare across datasets of different N (e.g. reporting effect strength
    for different measures/graphs side by side).

    NOT a valid input for AIC weights/entropy: dividing by N acts as a
    softmax temperature. For any fixed ΔAIC, exp(-ΔAIC/(2N)) -> 1 as N grows,
    so normalized weights drift toward uniform (entropy toward 1) regardless
    of how decisively AIC actually favors one model. Use relative_likelihood
    (no N) for weight/entropy computation instead.
    """
    return np.exp((aic_baseline - aic_model) / (2 * N))

def calculate_AIC_weight_entropy(rel_ll_values):
    if len(rel_ll_values) == 1:
        return 0.0
    weights = rel_ll_values / np.sum(rel_ll_values)
    entropy = -(weights * np.log(weights + 1e-15)).sum()
    entropy = entropy / np.log(len(weights))  # Normalize to [0,1]
    return entropy
    