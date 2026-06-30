# BOSPERRUS: BOundary effects in SPatial graphs: ERRor modeling and Untangling Strategies

## Tutorial — Getting Started with `Flow`

BOSPERRUS detects and corrects **boundary effects** in spatial graphs. Nodes close to the border of a tissue sample or point cloud tend to have systematically lower centrality scores than nodes in the center — not because they are biologically different, but because they have fewer neighbours. BOSPERRUS models this distance-dependent bias and optionally corrects for it.

---

## Table of Contents

1. [Installation](#installation)
2. [Key concepts](#key-concepts)
3. [Quick-start](#quick-start)
4. [Step-by-step walkthrough](#step-by-step-walkthrough)
   - [Step 1 – Create a toy dataset](#step-1--create-a-toy-dataset)
   - [Step 2 – Initialise `Flow`](#step-2--initialise-flow)
   - [Step 3 – Run model fitting and correction](#step-3--run-model-fitting-and-correction)
5. [Reading the results](#reading-the-results)
6. [Choosing a graph type](#choosing-a-graph-type)
7. [Choosing a distance function](#choosing-a-distance-function)
8. [Visualising the correction (optional)](#visualising-the-correction-optional)
9. [API reference summary](#api-reference-summary)

---

## Installation
Within a conda environment, run

```bash
pip install git+https://github.com/bionetslab/bosperrus-package.git
```

Pip dependencies (`scikit-learn`, `scipy`, `numpy`, `pandas`) are installed automatically. Within the environment, you also need to 
```bash
conda install graph-tool -c conda-forge
```
---

## Key concepts

Entry points / input types:
![Entry points](https://github.com/bionetslab/bosperrus-package/blob/master/plots_readme/bosperrus_flow.svg?raw=true)

`Flow` enables multiple entry points depending on what you already have:   
🟣 Given coordinates, a distance function, a graph type (Delaunay, k-nearest neighbor, or radius nearest neighbor), and centrality measures to compute.   
🩷 Given coordinates, a distance function, and a pre-built edge list.   
🟡 Given coordinates, a distance function, and pre-computed centrality scores.   
☘️ Given pre-computed distances and scores only (no coordinates needed).   


Distance functions:   
![Distance functions](https://github.com/bionetslab/bosperrus-package/blob/master/plots_readme/distance_functions.svg?raw=true)

---
## Quick-start

```python
import numpy as np
from bosperrus import Flow
from bosperrus.distances import distance_to_convex_hull

rng = np.random.default_rng(42)
coords = rng.uniform(0, 100, size=(500, 2))   # 500 random 2-D points

flow = Flow.from_coords(
    coordinates=coords,
    distance_fn=distance_to_convex_hull,
    measures=["degree", "closeness", "betweenness", "clustering", "pagerank"],
    graph_type="delaunay",
)
flow.flow()

# Inspect results
print(flow.fit_quality)
print(flow.observations.head())   # contains raw + corrected centrality columns
```

---

## Step-by-step walkthrough

### Step 1 – Create a toy dataset

We simulate a disc-shaped point cloud to produce an obvious boundary effect: nodes at the
perimeter will always have fewer neighbours than nodes at the centre.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bosperrus import Flow
from bosperrus.distances import distance_to_convex_hull

rng = np.random.default_rng(0)

# Sample uniformly inside a disc of radius 50
n = 600
angles = rng.uniform(0, 2 * np.pi, n)
radii  = 50 * np.sqrt(rng.uniform(0, 1, n))   # sqrt for uniform area sampling
coords = np.stack([radii * np.cos(angles),
                   radii * np.sin(angles)], axis=1)
```

### Step 2 – Initialise `Flow`

`Flow` is initialised via one of four factory classmethods depending on what you already have. For a full pipeline from coordinates:

```python
measures = ["degree", "closeness", "betweenness", "harmonic", "clustering", "pagerank"]

flow = Flow.from_coords(
    coordinates=coords,
    distance_fn=distance_to_convex_hull,
    measures=measures,
    graph_type="delaunay",   # or "knn" / "rnn", see below
)
```

If you already have an edge list, use `Flow.from_coords_and_edgelist`. If you already have centrality scores, use `Flow.from_coords_and_scores`. If you have neither coordinates nor a graph, use `Flow.from_distances_and_scores`.

All factory methods populate `flow.observations`: a DataFrame with one row per node containing the centrality scores and the distance column.

### Step 3 – Run model fitting and correction

```python
from bosperrus.fit import ConstantFit, PiecewiseLinearFit, ExponentialSaturationFit, MichaelisMentenFit

flow.flow(
    measures=measures,
    fits=[ConstantFit, PiecewiseLinearFit, ExponentialSaturationFit, MichaelisMentenFit],
    calculate_rel_ll_to_baseline=ConstantFit,   # ConstantFit = "no boundary effect" null model
)
```

`flow()` selects the best model per measure by AIC and writes corrected values back into `flow.observations`. All arguments are optional — omitting them runs all four models on all measures with `ConstantFit` as baseline.

After this step:

- `flow.fit_quality` — a DataFrame with one row per measure summarising the best model,
  AIC-based statistics, effect strength, and half-life.
- `flow.observations` — now also contains columns named `"BOSPERRUS corrected <measure>"` for each
  centrality.

---

## Reading the results

```python
# Overview table
print(flow.fit_quality.T[[
    "best_fit_type",
    "observed_effect_strength",   # signed relative drop from border to centre
    "observed_half_life",         # distance at which ~50% of the effect has decayed
    "affected samples",           # fraction of nodes still in the boundary-affected zone
]])
```

**`observed_effect_strength`** is defined as `(c_border − c_centre) / (c_border + c_centre)`.
A value close to 0 means negligible boundary bias; a value near ±1 means strong bias.

**`best_fit_type`** tells you which model won the AIC comparison:

| Model | Interpretation |
|---|---|
| `Constant Fit` | No detectable boundary effect |
| `Piecewise Linear Fit` | Linear drop up to a breakpoint, then flat plateau |
| `Exponential Saturation Fit` | Smooth exponential approach to a plateau |
| `Michaelis-Menten Fit` | Hyperbolic saturation (analogous to enzyme kinetics) |

To access the corrected values:

```python
# All corrected columns at once
corrected_cols = [c for c in flow.observations.columns if c.startswith("BOSPERRUS corrected")]
print(flow.observations[corrected_cols].head())

# Single measure
flow.observations[["degree", "BOSPERRUS corrected degree"]].head()
```

To access fit parameters directly:

```python
degree_fit = flow.best_fits["degree"]
print(degree_fit.name)      # e.g. "Exponential Saturation Fit"
print(degree_fit.params)    # fitted parameter dict
print(degree_fit.AIC)
```

---

## Choosing a graph type

| Graph type | Key parameter | Notes |
|---|---|---|
| `"delaunay"` | none | Default choice; produces a natural triangulation without isolated nodes |
| `"knn"` | `k` (int) | Directed, asymmetric — pass via `graph_kwargs={"k": 6}` |
| `"rnn"` | `r` (float, same units as coords) | Undirected — pass via `graph_kwargs={"r": 15.0}` |

```python
# k-nearest-neighbour example
flow = Flow.from_coords(
    coordinates=coords,
    distance_fn=distance_to_convex_hull,
    measures=measures,
    graph_type="knn",
    graph_kwargs={"k": 6},
)
```

---

## Choosing a distance function

| Function | Use case |
|---|---|
| `distance_to_convex_hull(coords)` | General default; works for any convex or near-convex tissue shape (2-D and 3-D) |
| `distance_to_rectangular_border(coords)` | Rectangular imaging window / biopsy (2-D only) |
| `distance_to_pointset(coords, pointset)` | Border defined by a set of explicit landmark coordinates |
| `distance_to_mask(coords, mask)` | Binary image mask defining the tissue boundary |

All functions are importable from `bosperrus.distances` and return a named `pd.Series` with one value per node.

---

## Visualising the correction (optional)

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sc = axes[0].scatter(coords[:, 0], coords[:, 1],
                     c=flow.observations["degree"], cmap="viridis", s=8)
axes[0].set_title("Raw degree centrality")
plt.colorbar(sc, ax=axes[0])

sc2 = axes[1].scatter(coords[:, 0], coords[:, 1],
                      c=flow.observations["BOSPERRUS corrected degree"], cmap="viridis", s=8)
axes[1].set_title("BOSPERRUS-corrected degree centrality")
plt.colorbar(sc2, ax=axes[1])

plt.tight_layout()
plt.savefig("bosperrus_correction.png", dpi=150)
plt.show()
```

Nodes near the border should appear more uniform after correction.

---

## API reference summary

### `Flow` — construction

| Factory method | When to use |
|---|---|
| `Flow.from_coords(coordinates, distance_fn, measures, graph_type, distance_kwargs=None, graph_kwargs=None)` | Full pipeline from scratch |
| `Flow.from_coords_and_edgelist(coordinates, distance_fn, measures, edge_list, distance_kwargs=None)` | Skip graph construction |
| `Flow.from_coords_and_scores(coordinates, distance_fn, scores, distance_kwargs=None)` | Pre-computed centralities |
| `Flow.from_distances_and_scores(distances, scores)` | No coords needed |

### `Flow` — methods

| Method | Signature | Description |
|---|---|---|
| `flow` | `(measures=None, fits=None, calculate_rel_ll_to_baseline=None)` | Fit all models, select best by AIC, write corrected columns to `observations` |

### Key attributes after a full run

| Attribute | Type | Content |
|---|---|---|
| `flow.observations` | `pd.DataFrame` | All node-level data: centralities, distance column, corrected values |
| `flow.fit_quality` | `pd.DataFrame` | Per-measure model selection summary (columns = measures) |
| `flow.best_fits` | `dict[str, Fit]` | Best `Fit` object per measure |
| `flow._edge_list` | `set` | Set of edges built during construction (available for Paths 1 and 2) |
