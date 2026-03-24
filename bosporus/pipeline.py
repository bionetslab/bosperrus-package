from .graph_construction import delaunay_edges, knn_edges, rnn_edges
from .centrality_measures import compute_centrality_measures
from .distances import distance_to_convex_hull, distance_to_pointset, distance_to_mask, distance_to_rectangular_border
import numpy as np
import pandas as pd
from .fit import ConstantFit, PiecewiseLinearFit, ExponentialSaturationFit


class BosporusFlow():
    def __init__(self, coordinates: np.ndarray):
        self.coordinates = coordinates
        self.df = pd.DataFrame(index=range(len(coordinates)))
        self.fits = list()
        self.fit_quality = None

    def construct_graph(self, graph_type, r=None, k=None):
        if graph_type == "delaunay":
            edge_list = delaunay_edges(self.coordinates)
        elif graph_type == "knn":
            edge_list = knn_edges(self.coordinates, k=k)
        elif graph_type == "rnn":
            edge_list = rnn_edges(self.coordinates, r=r)
        else:
            raise ValueError(f"Unknown graph type: {graph_type}")
        # Placeholder for the actual flow logic
        self.edge_list = edge_list
        return     
    
    def compute_centralities(self, measures):
        self.df = pd.concat([self.df, compute_centrality_measures(self.edge_list, N=len(self.coordinates), measures=measures)], axis=1)
        return
    
    def compute_distance_to_convex_hull(self, distance_key=None):
        distance = distance_to_convex_hull(self.coordinates)
        if distance_key is not None:
            distance.name = distance_key
        self.df = pd.concat([self.df, distance], axis=1)
        return  
    
    def compute_distance_to_pointset(self, pointset, distance_key=None):
        distance = distance_to_pointset(self.coordinates, pointset)
        if distance_key is not None:
            distance.name = distance_key
        self.df = pd.concat([self.df, distance], axis=1)
        return
    
    def compute_distance_to_mask(self, mask, distance_key=None):
        distance = distance_to_mask(self.coordinates, mask)
        if distance_key is not None:
            distance.name = distance_key
        self.df = pd.concat([self.df, distance], axis=1)
        return
    
    def compute_distance_to_rectangular_border(self, distance_key=None):
        distance = distance_to_rectangular_border(self.coordinates)
        if distance_key is not None:
            distance.name = distance_key
        self.df = pd.concat([self.df, distance], axis=1)
        return
    
    def fit_models(self, measures, distance_key, fits=[ConstantFit, PiecewiseLinearFit, ExponentialSaturationFit]):
        self.fits = dict()
        fit_quality_data = []
        
        d = self.df[distance_key].values
        
        for measure in measures:
            C = self.df[measure].values
            measure_fits = []
            
            # Fit all models for this measure
            for fit_class in fits:
                fit_instance = fit_class(C, d)
                fit_instance.fit()
                measure_fits.append((fit_class.__name__, fit_instance))
            
            self.fits[measure] = measure_fits
            
            # Find best fit by AIC
            best_fit_name, best_fit = min(measure_fits, key=lambda x: x[1].AIC)
            
            aics = np.array([f[1].AIC for f in measure_fits])
            rel_ll_over_const = np.exp((aics[1:] - aics[0]) / (2 * len(self.coordinates)))  # relative to the const model
            
            
            # Akaike weights (relative likelihood)
            weights = rel_ll_over_const / np.sum(rel_ll_over_const)
            entropy = -(weights * np.log(weights + 1e-15)).sum(axis=1)
            
            fit_quality_data.append({
                'measure': measure,
                'best_fit_type': best_fit_name,
                'piecewise_linear_relative_likelihood_over_const': rel_ll_over_const[0],
                'exp_relative_likelihood_over_const': rel_ll_over_const[1],
                'piecewise_linear_AIC_weight': weights[0],
                'exp_AIC_weight': weights[1],
                'entropy_AIC_weights': entropy,
                'effect_strength': best_fit.effect_strength,
                'relative_support': best_fit.relative_support
            })
        
        self.fit_quality = pd.DataFrame(fit_quality_data)
        return
    
    