import numpy as np 
import pandas as pd

def log_likelihood(C_true, C_pred):
    n = len(C_true)
    residuals = C_true - C_pred
    sigma2 = np.mean(residuals**2)  # MSE as variance estimate, THIS IS ANOTHER PARAMETER THAT NEEDS TO BE ACCOUNTED FOR IN AIC CALCULATION
    return -0.5 * n * (np.log(2 * np.pi * sigma2) + 1)


def akaike_information_criterion(num_params, log_likelihood_model):
    return 2 * num_params - 2 * log_likelihood_model


def calculate_fit_properties(conc):
    conc["pieli_mon_incr"] = conc["pieli_m"] > 0
    conc["exp_mon_incr"] = (conc["exp_a"] > 0) == (conc["exp_b"] > 0)
    conc["const_mon_incr"] = False

    conc["measure"] = conc.index
    conc["dataset_type"] = conc["dataset"].apply(lambda x: x.split(":")[0])

    
    conc["const_aic"] = conc["const_ll"].apply(lambda x: akaike_information_criterion(2, x))
    conc["exp_aic"] = conc["exp_ll"].apply(lambda x: akaike_information_criterion(4, x))
    conc["pieli_aic"] = conc["pieli_ll"].apply(lambda x: akaike_information_criterion(4, x))

    aics = [c for c in conc.columns if c.endswith("aic")] 
    best_fits = conc[aics].idxmin(axis=1)
    heatmap_matrix = pd.crosstab(best_fits.index, best_fits.values)

    conc["best_fit"] = best_fits
    conc["best_fit"] = conc["best_fit"].apply(lambda x: x.split("_aic")[0])

    conc["sign_border_effect"] = conc.apply(
    lambda row: 
            1
            if row[f"{row['best_fit']}_mon_incr"]
            else -1,
        axis=1
    )
    conc["sign_border_effect"] = np.where(conc["best_fit"] == "const", 0,  conc["sign_border_effect"])

    delta = - conc[["exp_aic", "pieli_aic"]].sub(conc["const_aic"], axis=0)
    
    divisor = 2 * conc["num_nodes"]
    conc[["rel_ll_exp_aic", "rel_ll_pieli_aic"]] = np.exp(
        delta[["exp_aic", "pieli_aic"]].div(divisor, axis=0)
        )
    conc["rel_ll_best"] = conc[["rel_ll_exp_aic", "rel_ll_pieli_aic"]].max(axis=1)

    weight_sum = conc[["rel_ll_exp_aic", "rel_ll_pieli_aic"]].sum(axis=1)
    weights = conc[["rel_ll_exp_aic", "rel_ll_pieli_aic"]].div(weight_sum, axis=0)

    conc["exp_akaike_weight"] = weights["rel_ll_exp_aic"]
    conc["pieli_akaike_weight"] = weights["rel_ll_pieli_aic"]
    
    eps = 1e-15
    entropy = -(weights * np.log(weights + eps)).sum(axis=1)

    num_models = 2
    conc["akaike_weights_entropy"] = entropy / np.log(num_models)
    
    return heatmap_matrix, conc
