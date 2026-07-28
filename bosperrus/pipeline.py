import numpy as np
import pandas as pd
from .fit import ConstantFit, MichaelisMentenFit, PiecewiseLinearFit, ExponentialSaturationFit
from .evaluate_fit import relative_likelihood, scaled_relative_likelihood, calculate_AIC_weight_entropy
from .graph_construction import construct_graph
from .centrality_measures import compute_centrality_measures

__all__ = ['Flow']

class Flow():
    def __init__(self, scores, distances):
        """
        Parameters
        ----------
        scores : pd.DataFrame | pd.Series | array-like
            One or more measures to correct for boundary effects, one column per
            measure. A Series or 1-D array is treated as a single measure (named
            after the Series, or "score" if unnamed/a plain array). A 2-D array
            is treated as multiple unnamed measures ("score_0", "score_1", ...).
        distances : pd.Series | array-like
            Distance-from-border values, aligned with `scores`. An unnamed
            Series or a plain array is named "distance".
        """
        scores = self._coerce_scores(scores)
        distances = self._coerce_distances(distances)

        self._distance_key = distances.name
        self._score_names = list(scores.columns)
        self.observations = scores.copy()
        self.observations[self._distance_key] = distances

    @staticmethod
    def _coerce_scores(scores):
        if isinstance(scores, pd.DataFrame):
            return scores
        if isinstance(scores, pd.Series):
            return scores.to_frame(scores.name if scores.name is not None else "score")
        arr = np.asarray(scores)
        if arr.ndim == 1:
            return pd.DataFrame({"score": arr})
        return pd.DataFrame(arr, columns=[f"score_{i}" for i in range(arr.shape[1])])

    @staticmethod
    def _coerce_distances(distances):
        if isinstance(distances, pd.Series):
            return distances if distances.name is not None else distances.rename("distance")
        return pd.Series(np.asarray(distances), name="distance")

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
    def from_coords_and_scores(cls, coordinates, distance_fn, scores, distance_kwargs=None):
        """Path 3: coords + distance fn + pre-computed scores"""
        distances = distance_fn(coordinates, **(distance_kwargs or {}))
        obj = cls(scores=scores, distances=distances)
        return obj

    @classmethod
    def from_distances_and_scores(cls, distances, scores):
        """Path 4: no coords — pre-computed distances and scores only"""
        obj = cls(scores=scores, distances=distances)
        return obj

    @staticmethod
    def _set_entropy_weights(fit_instances, baseline_fit):
        # Uses the unscaled relative_likelihood (not scaled_relative_likelihood):
        # AIC weights/entropy compare models fit to the same N observations, so
        # there's no cross-dataset-comparability need here, and dividing by N
        # would artificially flatten the weights for large N. See
        # evaluate_fit.scaled_relative_likelihood's docstring.
        rel_ll = [fit_instance.relative_likelihood_over_baseline for fit_instance in fit_instances if fit_instance != baseline_fit]
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
        score_names=None,
        fits=None,
        baseline_fit_class=None
    ):
        """
        Parameters
        ----------
        score_names : list of str, optional
            Which columns of `self.observations` (i.e. which of the `scores`
            passed at construction) to fit/correct. Defaults to all of them.
            Note this is *not* the same vocabulary as
            `compute_centrality_measures`'s `measures` argument — these are
            arbitrary score column names (e.g. "log1p_total_counts", a gene
            name, ...), not restricted to a fixed set of centrality measures.
        """
        if fits is None:
            fits = [ConstantFit, PiecewiseLinearFit, ExponentialSaturationFit, MichaelisMentenFit]
        if baseline_fit_class is None:
            baseline_fit_class = ConstantFit
        if baseline_fit_class not in fits:
            raise ValueError("baseline fit class must be included in fits")

        if score_names is None:
            score_names = self._score_names

        for name in score_names:
            if name not in self._score_names:
                raise ValueError(f"Score '{name}' not found in scores. Available scores: {self._score_names}")

        self.best_fits = dict()
        fit_quality_data = dict()

        # Deliberately NOT .values here: Fit._expand_to_original_index() rebuilds
        # S_corrected against S_true's own index, so passing the real
        # (possibly non-default, e.g. adata.obs_names) index through lets
        # `self.observations[col] = best_fit.S_corrected` below label-align
        # correctly instead of silently coming back all-NaN.
        d = self.observations[self._distance_key]

        for score_name in score_names:
            S = self.observations[score_name]

            baseline_fit = baseline_fit_class(S, d)
            baseline_fit.fit()
            baseline_aic = baseline_fit.AIC

            fit_instances = []
            for fit_class in fits:
                if fit_class == baseline_fit_class:
                    continue
                fit_instance = fit_class(S, d)
                fit_instance.fit_correct()
                fit_instance.relative_likelihood_over_baseline = relative_likelihood(fit_instance.AIC, baseline_aic)
                fit_instance.scaled_relative_loglikelihood_over_baseline = scaled_relative_likelihood(fit_instance.AIC, baseline_aic, len(d))
                fit_instances.append(fit_instance)

            fit_instances.append(baseline_fit)
            best_fit = min(fit_instances, key=lambda x: x.AIC)
            self.best_fits[score_name] = best_fit
            self._set_entropy_weights(fit_instances, baseline_fit)

            fit_quality_data[score_name] = best_fit.params_summary()
            self.observations[f"BOSPERRUS corrected {score_name}"] = best_fit.S_corrected

        self.fit_quality = pd.DataFrame(fit_quality_data)