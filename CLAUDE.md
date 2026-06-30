# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this package does

**BOSPERRUS** (BOundary effects in SPatial graphs: errOR modeling and Untangling Strategies) detects and corrects boundary effects in spatial graphs. Nodes near the border of a tissue sample tend to have systematically lower centrality scores than interior nodes — not because of biology, but because they have fewer neighbours. BOSPERRUS models this distance-dependent bias with parametric curves and corrects for it.

## Commands

**Install (editable):**
```bash
pip install -e /data/bionets/je30bery/bosperrus-package/
```

**Centrality measures require graph-tool, which must be installed via conda (not pip):**
```bash
conda install graph-tool -c conda-forge
```

**Run all tests:**
```bash
cd /data/bionets/je30bery/bosperrus-package
pytest tests/
```

**Run a single test class or test:**
```bash
pytest tests/test_fit.py::TestPiecewiseLinearFit
pytest tests/test_fit.py::TestPiecewiseLinearFit::test_recovers_known_parameters
```

## Architecture

### `Flow` (pipeline.py) — the main pipeline object

`Flow` is the central orchestrator. It does **not** take coordinates in `__init__`; instead it takes pre-computed `scores` (centrality DataFrame) and `distances` (Series). Four factory classmethods cover different entry points:

| Factory method | When to use |
|---|---|
| `Flow.from_coords(coordinates, distance_fn, measures, graph_type, ...)` | Full pipeline from scratch |
| `Flow.from_coords_and_edgelist(coordinates, distance_fn, measures, edge_list)` | Skip graph construction |
| `Flow.from_coords_and_scores(coordinates, distance_fn, scores)` | Pre-computed centralities |
| `Flow.from_distances_and_scores(distances, scores)` | No coords at all |

After construction, call `flow.flow()` to run AIC-based model selection for each measure. Results land in:
- `flow.observations` — DataFrame with all raw centralities + `"BOSPERRUS corrected {measure}"` columns
- `flow.fit_quality` — DataFrame summary of the winning model per measure (params, effect strength, half-life, AIC weights)
- `flow.best_fits` — `dict[str, Fit]` of the winning `Fit` instance per measure


### `Fit` base class (fit.py) — the extension point

All fit models subclass `Fit`. The base class:
- Filters non-finite values from both `S_true` and `d` at construction time and stores a boolean `_mask`
- Exposes read-only properties backed by private `_` attributes (subclasses must write to `_AIC`, `_params`, etc. directly — not through the properties)
- `S_corrected` property automatically expands filtered results back to the original index via `_expand_to_original_index()`

**Subclass contract** — every subclass must implement:

| Method | What it must do |
|---|---|
| `fit()` | Estimate params; set `_params`, `_S_model`, `_converged`; call `_rate_observed_metrics()`, `_calculate_fraction_not_converged()`, `_score()` on success |
| `correct()` | Set `_S_corrected`; return `self.S_corrected` (the property, not the raw array) |
| `_rate_observed_metrics()` | Set `_observed_effect_strength` and `_observed_half_life` from fitted params |
| `_calculate_fraction_not_converged(threshold)` | Set `_fraction_not_converged`; `threshold` may be ignored for hard-boundary models |

`fit_correct()` is the convenience method: calls `fit()` then `correct()` in sequence.

### The four fit models

| Class | Functional form | Params dict keys |
|---|---|---|
| `ConstantFit` | `c` (mean; null/baseline model) | `constant_c` |
| `PiecewiseLinearFit` | `m*d + c` for `d ≤ b`, plateau at `m*b + c` for `d > b` | `piecewise_linear_b`, `piecewise_linear_m`, `piecewise_linear_c` |
| `ExponentialSaturationFit` | `a*(1 - exp(-b*d)) + c`, `b > 0` | `exponential_saturation_a`, `exponential_saturation_b`, `exponential_saturation_c` |
| `MichaelisMentenFit` | `a*d/(b+d) + c`, `b > 0` (Km = `b`) | `michaelis_menten_a`, `michaelis_menten_b`, `michaelis_menten_c` |

**Correction formula**: all saturation models shift raw values to the asymptote: `S_corrected = S_true + (asymptote - S_model)`. If fitting fails (`_converged = False`), `S_corrected = S_true` (passthrough).

**PiecewiseLinearFit** is special: it uses `scipy.optimize.curve_fit` followed by optional `differential_evolution` refinement seeded around the `curve_fit` solution. `fraction_not_converged` is the fraction of nodes with `d ≤ b` (hard boundary, not asymptotic).

### Model selection (`flow.flow()`)

1. `ConstantFit` is always the AIC baseline (the "no effect" null model).
2. All other fits call `fit_correct()`.
3. AIC is computed as `2*(k+1) - 2*log_likelihood`, where `+1` accounts for the variance estimated from residuals.
4. The best fit (lowest AIC) is selected. `entropy_AIC_weights` is the normalised entropy of relative likelihoods vs. baseline, stored on each non-baseline fit — higher entropy = more ambiguous model selection.

### `evaluate_fit.py`

Stateless functions:
- `log_likelihood(C_true, C_pred)` — Gaussian log-likelihood with residual-estimated variance
- `akaike_information_criterion(num_params, log_likelihood)` — standard AIC
- `relative_likelihood(aic_model, aic_baseline, N)` — sample-size-scaled relative likelihood
- `calculate_AIC_weight_entropy(rel_ll_values)` — normalised entropy of AIC weight vector

### `graph_construction.py`

- `knn_edges(coords, k)` → directed edges as `set` of `(u, v)` tuples (asymmetric)
- `rnn_edges(coords, r)` → undirected edges as `set` of `frozenset({u, v})`
- `delaunay_edges(coords)` → undirected edges as `set` of `frozenset({u, v})`
- `construct_graph(coords, graph_type, k=None, r=None)` → dispatches to the above

kNN edges are directed tuples; Delaunay and rNN edges are undirected frozensets. This asymmetry matters anywhere edge sets are compared or passed to downstream functions.

### `distances.py`

All functions return a named `pd.Series` with one distance value per node:
- `distance_to_convex_hull(coords)` — works for 2-D and 3-D
- `distance_to_rectangular_border(coords)` — 2-D only; min distance to any of the four sides
- `distance_to_pointset(coords, pointset)` — nearest-neighbour distance via `cKDTree`
- `distance_to_mask(coords, mask)` — `distance_transform_edt` on inverted binary mask

### `centrality_measures.py`

`compute_centrality_measures(edge_list, N, measures)` — requires `graph-tool` at import time. If `graph-tool` is absent the module imports silently but every call raises `ImportError`. Supported measures: `"degree"`, `"pagerank"`, `"betweenness"`, `"closeness"`, `"harmonic"`, `"clustering"`. Isolated nodes (present in `N` but not in `edge_list`) are zero-padded so the output always has exactly `N` rows.

## Tests

Tests live in `tests/test_fit.py` and cover all four `Fit` subclasses plus cross-model sanity checks. They do **not** require `graph-tool`. Test fixtures use `np.random.default_rng(42)` for reproducibility.

## Adding a new fit model

1. Subclass `Fit` in `fit.py`.
2. Implement `fit()`, `correct()`, `_rate_observed_metrics()`, `_calculate_fraction_not_converged()`.
3. Set `self._name` in `__init__`.
4. Name param dict keys as `{snake_case_model_name}_{param}` (e.g. `my_model_a`).
5. Add to the default `fits` list in `Flow.flow()` if it should run by default.
6. Add a `TestMyModelFit` class in `tests/test_fit.py` mirroring the existing test structure.
