from .graph_construction import delaunay_edges, knn_edges, rnn_edges
from .distances import distance_to_convex_hull, distance_to_pointset, distance_to_mask, distance_to_rectangular_border
import numpy as np
import pandas as pd
from .fit import ConstantFit, MichaelisMentenFit, PiecewiseLinearFit, ExponentialSaturationFit
from .centrality_measures import compute_centrality_measures

class Flow():
    def __init__(self, coordinates: np.ndarray = None):
        self.coordinates = coordinates
        if coordinates is not None:
            self.observations = pd.DataFrame(index=range(len(coordinates)))
        else:
            self.observations = pd.DataFrame()
            
        self.fit_quality = None
        self.best_fits = dict()
                
        if self.coordinates is not None:
            self.N = len(self.coordinates)
        else:
            self.N = None
    

    def flow(self, graph_type, params, distance_function, distance_key=None, measures=None, fits=None, calculate_rel_ll_to_baseline=None):
        if measures is None:
            measures = ["degree", "betweenness", "closeness", "clustering", "pagerank"]
        self.construct_graph(graph_type, params)
        self.compute_centralities(measures=measures)
        distance_key = distance_function(distance_key=distance_key)
        self.fit_models(measures=measures, distance_key=distance_key, fits=fits, calculate_rel_ll_to_baseline=calculate_rel_ll_to_baseline)
        return self.best_fits
    
    def set_edge_list(self, edge_list):
        self.edge_list = edge_list
        self.N = len(self.observations)
    
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
        centralities = pd.DataFrame(compute_centrality_measures(self.edge_list, N=self.N, measures=measures))
        self.observations = pd.concat([self.observations, centralities], axis=1)
        return
    
    def set_custom_centralities(self, centralities: pd.DataFrame):
        self.observations = pd.concat([self.observations, centralities], axis=1)
        return
    
    def compute_distance_to_convex_hull(self, distance_key=None):
        distance = distance_to_convex_hull(self.coordinates)
        if distance_key is not None:
            distance.name = distance_key
        self.observations = pd.concat([self.observations, distance], axis=1)
        return distance.name
    
    def compute_distance_to_pointset(self, pointset, distance_key=None):
        distance = distance_to_pointset(self.coordinates, pointset)
        if distance_key is not None:
            distance.name = distance_key
        self.observations = pd.concat([self.observations, distance], axis=1)
        return distance.name
    
    def compute_distance_to_mask(self, mask, distance_key=None):
        distance = distance_to_mask(self.coordinates, mask)
        if distance_key is not None:
            distance.name = distance_key
        self.observations = pd.concat([self.observations, distance], axis=1)
        return distance.name
    
    def set_custom_distances(self, distance_series: pd.Series):
        self.observations = pd.concat([self.observations, distance_series], axis=1)
        return distance_series.name
    
    def compute_distance_to_rectangular_border(self, distance_key=None):
        distance = distance_to_rectangular_border(self.coordinates)
        if distance_key is not None:
            distance.name = distance_key
        self.observations = pd.concat([self.observations, distance], axis=1)
        return distance.name
    
        
    def fit_models(
        self,
        measures,
        distance_key,
        fits=None, 
        calculate_rel_ll_to_baseline=None
    ):
        
        if fits is None:
            fits = [ConstantFit, PiecewiseLinearFit, ExponentialSaturationFit, MichaelisMentenFit]
        if calculate_rel_ll_to_baseline is None:
            calculate_rel_ll_to_baseline = ConstantFit
        if calculate_rel_ll_to_baseline not in fits:
            raise ValueError("baseline fit class must be included in fits")
            
        self.best_fits = dict()
        fit_quality_data = []

        d = self.observations[distance_key].values

        for measure in measures:
            S = self.observations[measure].values
            measure_fits = []
            aic_dict = {}

            baseline_name = None
            baseline_aic = None

            # Fit all models
            for fit_class in fits:
                fit_instance = fit_class(S, d)
                fit_instance.fit()

                name = fit_instance.name
                measure_fits.append((name, fit_instance))
                aic_dict[name] = fit_instance.AIC

                if fit_class == calculate_rel_ll_to_baseline:
                    baseline_name = name
                    baseline_aic = fit_instance.AIC


            # Best fit (still among all models)
            best_fit_name, best_fit = min(measure_fits, key=lambda x: x[1].AIC)

            # --- relative likelihoods vs baseline (exclude baseline itself) ---
            rel_ll = {
                name: np.exp((baseline_aic - aic) / (2 * len(d)))
                for name, aic in aic_dict.items()
                if name != baseline_name
            }
            
            # --- Akaike weights ONLY among non-baseline models ---
            rel_ll_values = np.array(list(rel_ll.values()))
            weights = rel_ll_values / np.sum(rel_ll_values)
            weight_dict = dict(zip(rel_ll.keys(), weights))

            # Entropy over competing (non-baseline) models
            entropy = -(weights * np.log(weights + 1e-15)).sum()
            entropy = entropy / np.log(len(weights))  # Normalize to [0,1]

            row = {
                "measure": measure,
                "best_fit_type": best_fit_name,
                "entropy_AIC_weights": entropy,
                "observed_half_life": best_fit.observed_half_life,
                "observed_effect_strength": best_fit.observed_effect_strength,
                "included samples": best_fit.included_samples,
                "affected samples": best_fit.fraction_not_converged,
                f"scaled_relative_likelihood_over_{baseline_name}": rel_ll[best_fit_name] if best_fit_name in rel_ll else 1.0,  # If best fit is baseline, set to 1.0
                "AIC_weight": weight_dict[best_fit_name] if best_fit_name in weight_dict else np.nan  # If best fit is baseline, set to nan
            }
            
            row.update(best_fit.params)  # Add best fit parameters to the row
            fit_quality_data.append(row)
            
            self.best_fits[measure] = best_fit
            self.observations[f"BOSPORUS corrected {measure}"] = best_fit.S_corrected

        self.fit_quality = pd.DataFrame(fit_quality_data)