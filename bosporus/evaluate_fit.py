import numpy as np 
import pandas as pd

def log_likelihood(C_true, C_pred):
    n = len(C_true)
    residuals = C_true - C_pred
    sigma2 = np.mean(residuals**2)  # MSE as variance estimate, THIS IS ANOTHER PARAMETER THAT NEEDS TO BE ACCOUNTED FOR IN AIC CALCULATION
    return -0.5 * n * (np.log(2 * np.pi * sigma2) + 1)


def akaike_information_criterion(num_params, log_likelihood_model):
    return 2 * num_params - 2 * log_likelihood_model


def relative_likelihood(aic_model, aic_baseline, N):
    return np.exp((aic_baseline - aic_model) / (2 * N))
