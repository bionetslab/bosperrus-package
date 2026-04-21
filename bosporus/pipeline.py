from .graph_construction import delaunay_edges, knn_edges, rnn_edges
from .centrality_measures import compute_centrality_measures
from .distances import distance_to_convex_hull, distance_to_pointset, distance_to_mask, distance_to_rectangular_border
import numpy as np
import pandas as pd
from .fit import ConstantFit, MichaelisMentenFit, PiecewiseLinearFit, ExponentialSaturationFit


class BosporusFlow():
    def __init__(self, coordinates: np.ndarray):
        self.coordinates = coordinates
        self.df = pd.DataFrame(index=range(len(coordinates)))
        self.fits = list()
        self.fit_quality = None
    
    

    def run_all(self, graph_type, params, distance_function, distance_key=None, measures=["degree", "closeness", "betweenness", "harmonic", "clustering", "pagerank"], fits=[ConstantFit, PiecewiseLinearFit, ExponentialSaturationFit, MichaelisMentenFit], calculate_rel_ll_to_baseline=ConstantFit):
        self.construct_graph(graph_type, params)
        self.compute_centralities(measures=measures)
        distance_key = distance_function(distance_key=distance_key)
        self.fit_models(measures=measures, distance_key=distance_key, fits=fits, calculate_rel_ll_to_baseline=calculate_rel_ll_to_baseline)
        return
    
    
    def construct_graph(self, graph_type, params=None):
        if graph_type == "delaunay":
            edge_list = delaunay_edges(self.coordinates)
        elif graph_type == "knn":
            if params is None or "k" not in params:
                raise ValueError("For knn graph construction, 'params' must be provided with a key 'k'.")

            edge_list = knn_edges(self.coordinates, k=params["k"])
        elif graph_type == "rnn":
            if params is None or "r" not in params:
                raise ValueError("For rnn graph construction, 'params' must be provided with a key 'r'.")   
            edge_list = rnn_edges(self.coordinates, r=params["r"])
        else:
            raise ValueError(f"Unknown graph type: {graph_type}")

        self.edge_list = edge_list
        return     
    
    def compute_centralities(self, measures):
        centralities = pd.DataFrame(compute_centrality_measures(self.edge_list, N=len(self.coordinates), measures=measures))
        self.df = pd.concat([self.df, centralities], axis=1)
        return
    
    def compute_distance_to_convex_hull(self, distance_key=None):
        distance = distance_to_convex_hull(self.coordinates)
        if distance_key is not None:
            distance.name = distance_key
        self.df = pd.concat([self.df, distance], axis=1)
        return distance.name
    
    def compute_distance_to_pointset(self, pointset, distance_key=None):
        distance = distance_to_pointset(self.coordinates, pointset)
        if distance_key is not None:
            distance.name = distance_key
        self.df = pd.concat([self.df, distance], axis=1)
        return distance.name
    
    def compute_distance_to_mask(self, mask, distance_key=None):
        distance = distance_to_mask(self.coordinates, mask)
        if distance_key is not None:
            distance.name = distance_key
        self.df = pd.concat([self.df, distance], axis=1)
        return distance.name
    
    def compute_distance_to_rectangular_border(self, distance_key=None):
        distance = distance_to_rectangular_border(self.coordinates)
        if distance_key is not None:
            distance.name = distance_key
        self.df = pd.concat([self.df, distance], axis=1)
        return distance.name
    
        
    def fit_models(
        self,
        measures,
        distance_key,
        fits=[ConstantFit, PiecewiseLinearFit, ExponentialSaturationFit, MichaelisMentenFit],
        calculate_rel_ll_to_baseline=ConstantFit
    ):
        self.fits = dict()
        fit_quality_data = []

        d = self.df[distance_key].values

        for measure in measures:
            try:
                C = self.df[measure].values
                measure_fits = []
                aic_dict = {}

                baseline_name = None
                baseline_aic = None

                # Fit all models
                for fit_class in fits:
                    # Remove NaN values from both C and d
                    fit_instance = fit_class(C, d)
                    fit_instance.fit()

                    name = fit_instance.name
                    measure_fits.append((name, fit_instance))
                    aic_dict[name] = fit_instance.AIC

                    if fit_class == calculate_rel_ll_to_baseline:
                        baseline_name = name
                        baseline_aic = fit_instance.AIC

                self.fits[measure] = measure_fits

                # Best fit (still among all models)
                best_fit_name, best_fit = min(measure_fits, key=lambda x: x[1].AIC)

                # --- relative likelihoods vs baseline (exclude baseline itself) ---
                rel_ll = {
                    name: np.exp((baseline_aic - aic) / (2 * len(self.coordinates)))
                    for name, aic in aic_dict.items()
                    if name != baseline_name
                }
                
                # --- Akaike weights ONLY among non-baseline models ---
                rel_ll_values = np.array(list(rel_ll.values()))
                weights = rel_ll_values / np.sum(rel_ll_values)
                weight_dict = dict(zip(rel_ll.keys(), weights))

                # Entropy over competing (non-baseline) models
                entropy = -(weights * np.log(weights + 1e-15)).sum()

                # Build result row
                row = {
                    "measure": measure,
                    "best_fit_type": best_fit_name,
                    "entropy_AIC_weights": entropy,
                    "observed_half_life": best_fit.observed_half_life,
                    "observed_effect_strength": best_fit.observed_effect_strength,
                    "included samples": best_fit.included_samples,
                    "affected samples": best_fit.fraction_not_converged()
                }
                
                row.update(best_fit.params)  # Add best fit parameters to the row
                
                # Add per-model metrics (order-independent)
                for name in rel_ll:
                    row[f"{name}_scaled_relative_likelihood_over_{baseline_name}"] = rel_ll[name]
                    row[f"{name}_AIC_weight"] = weight_dict[name]

                fit_quality_data.append(row)
            except Exception as e:
                print(f"Error fitting models for measure '{measure}': {e}")

        self.fit_quality = pd.DataFrame(fit_quality_data)