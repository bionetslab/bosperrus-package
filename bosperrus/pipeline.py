import numpy as np
import pandas as pd
from .fit import ConstantFit, MichaelisMentenFit, PiecewiseLinearFit, ExponentialSaturationFit
from .evaluate_fit import relative_likelihood, calculate_AIC_weight_entropy
from .graph_construction import construct_graph
from .centrality_measures import compute_centrality_measures

class Flow():
    def __init__(self, scores: pd.DataFrame, distances: pd.Series):
        self._distance_key = distances.name
        self._measures = list(scores.columns)
        self.observations = scores
        self.observations[self._distance_key] = distances
        self.best_fit = dict()

    @classmethod
    def from_coords(cls, coordinates, distance_fn, measures, graph_type, distance_kwargs=None, graph_kwargs=None):
        """Path 1: full pipeline — coords + distance fn + graph construction"""
        distances = distance_fn(coordinates, **(distance_kwargs or {}))
        edge_list = construct_graph(coordinates, graph_type, **(graph_kwargs or {}))
        scores = compute_centrality_measures(edge_list, N=len(coordinates), measures=measures)
        obj = cls(scores=scores, distances=distances)
        obj._edge_list = edge_list
        return obj

    @classmethod
    def from_coords_and_edgelist(cls, coordinates, distance_fn, measures, edge_list, distance_kwargs=None):
        """Path 2: coords + distance fn + pre-built edge list"""
        distances = distance_fn(coordinates, **(distance_kwargs or {}))
        scores = compute_centrality_measures(edge_list, N=len(coordinates), measures=measures)
        obj = cls(scores=scores, distances=distances)
        obj._edge_list = edge_list
        return obj

    @classmethod
    def from_coords_and_scores(cls, coordinates, distance_fn, scores: pd.DataFrame, distance_kwargs=None):
        """Path 3: coords + distance fn + pre-computed scores"""
        distances = distance_fn(coordinates, **(distance_kwargs or {}))
        obj = cls(scores=scores, distances=distances)
        return obj

    @classmethod
    def from_distances_and_scores(cls, distances: pd.Series, scores: pd.DataFrame):
        """Path 4: no coords — pre-computed distances and scores only"""
        obj = cls(scores=scores, distances=distances)
        return obj
    
    @staticmethod
    def _set_entropy_weights(fit_instances, baseline_fit):
        rel_ll = [fit_instance.scaled_relative_loglikelihood_over_baseline for fit_instance in fit_instances if fit_instance != baseline_fit]
        entropy = calculate_AIC_weight_entropy(np.array(rel_ll))
        
        i = 0
        for fit in fit_instances:
            if fit == baseline_fit:
                continue
            else:
                fit.entropy_AIC_weights = entropy
                i += 1
        
    def flow(
        self,
        measures=None,
        fits=None, 
        calculate_rel_ll_to_baseline=None
    ):
        
        if fits is None:
            fits = [ConstantFit, PiecewiseLinearFit, ExponentialSaturationFit, MichaelisMentenFit]
        if calculate_rel_ll_to_baseline is None:
            calculate_rel_ll_to_baseline = ConstantFit
        if calculate_rel_ll_to_baseline not in fits:
            raise ValueError("baseline fit class must be included in fits")
        
        if measures is None:
            measures = self._measures
        
        for m in measures:
            assert m in self._measures, f"Measure '{m}' not found in scores. Available measures: {self._measures}"
        
        self.best_fits = dict()
        fit_quality_data = dict()

        d = self.observations[self._distance_key].values

        for measure in measures:
            S = self.observations[measure].values
            
            baseline_fit = calculate_rel_ll_to_baseline(S, d)
            baseline_fit.fit()
            baseline_aic = baseline_fit.AIC
            
            fit_instances = []
            for fit_class in fits:
                if fit_class == calculate_rel_ll_to_baseline:
                    continue
                fit_instance = fit_class(S, d)
                fit_instance.fit_correct()
                fit_instance.scaled_relative_loglikelihood_over_baseline = relative_likelihood(fit_instance.AIC, baseline_aic,  len(d))
                fit_instances.append(fit_instance)
            
            fit_instances.append(baseline_fit)
            best_fit = min(fit_instances, key=lambda x: x.AIC)
            self._set_entropy_weights(fit_instances, baseline_fit)
            
            fit_quality_data[measure] = best_fit.params_summary()
            self.observations[f"BOSPERRUS corrected {measure}"] = best_fit.S_corrected
            
        self.fit_quality = pd.DataFrame(fit_quality_data)