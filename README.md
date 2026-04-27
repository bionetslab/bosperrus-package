## BOSPORUS: BOundary effects in SPatial graphs: errOR modeling and Untangling Strategies

Distance functions:

![Distance functions](https://github.com/bionetslab/bosporus-package/blob/master/plots_readme/distance_functions.svg?raw=true)

# BOSPORUS: Tutorial — Getting Started with `BosporusFlow`

BOSPORUS (**BO**undary effects in **S**patial graphs: error modeling and **U**ntangling **S**trategies) detects and corrects **boundary effects** in spatial graphs. Nodes close to the border of a tissue sample or point cloud tend to have systematically lower centrality scores than nodes in the center — not because they are biologically different, but because they have fewer neighbours. BOSPORUS models this distance-dependent bias and optionally corrects for it.

---

## Table of Contents

1. [Installation](#installation)
2. [Key concepts](#key-concepts)
3. [Quick-start: one-liner with `run_all`](#quick-start-one-liner-with-run_all)
4. [Step-by-step walkthrough](#step-by-step-walkthrough)
   - [Step 1 – Create a toy dataset](#step-1--create-a-toy-dataset)
   - [Step 2 – Initialise `BosporusFlow`](#step-2--initialise-bosporusflow)
   - [Step 3 – Build the spatial graph](#step-3--build-the-spatial-graph)
   - [Step 4 – Compute centrality measures](#step-4--compute-centrality-measures)
   - [Step 5 – Compute a boundary distance](#step-5--compute-a-boundary-distance)
   - [Step 6 – Fit models and correct](#step-6--fit-models-and-correct)
5. [Reading the results](#reading-the-results)
6. [Choosing a graph type](#choosing-a-graph-type)
7. [Choosing a distance function](#choosing-a-distance-function)
8. [Visualising the correction (optional)](#visualising-the-correction-optional)
9. [API reference summary](#api-reference-summary)

---

## Installation

```bash
pip install git+https://github.com/bionetslab/bosporus-package.git
```

Dependencies (`scikit-learn`, `scipy`, `numpy`, `pandas`) are installed automatically.

---

## Key concepts

| Term | Meaning |
|---|---|
| **Spatial graph** | A graph whose nodes are 2-D (or 3-D) coordinates, connected by edges derived from spatial proximity. |
| **Boundary effect** | The systematic drop in centrality values for nodes that are near the border of the point cloud. |
| **Distance function** | A function that assigns every node a scalar "distance to the boundary". BOSPORUS ships four variants. |
| **Fit model** | A parametric curve (constant, piecewise-linear, exponential saturation, Michaelis-Menten) fitted to `centrality ~ distance`. The best model is selected by AIC. |
| **Correction** | The best-fit model is used to normalize each node's centrality to what it would be in the interior. |

---

## Quick-start: one-liner with `run_all`

```python
import numpy as np
from bosporus import BosporusFlow

rng = np.random.default_rng(42)
coords = rng.uniform(0, 100, size=(500, 2))   # 500 random 2-D points

flow = BosporusFlow(coords)

# run_all builds the graph, computes centralities, fits models, and corrects
best_fits = flow.run_all(
    graph_type="delaunay",
    params=None,
    distance_function=flow.compute_distance_to_convex_hull,
)

# Inspect results
print(flow.fit_quality[["measure", "best_fit_type", "observed_effect_strength"]])
print(flow.df.head())   # contains raw + corrected centrality columns
```

`run_all` returns a dictionary of the best `Fit` object for each centrality measure.

---

## Step-by-step walkthrough

### Step 1 – Create a toy dataset

We simulate a disc-shaped point cloud to produce an obvious boundary effect: nodes at the
perimeter will always have fewer neighbours than nodes at the centre.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bosporus import BosporusFlow

rng = np.random.default_rng(0)

# Sample uniformly inside a disc of radius 50
n = 600
angles = rng.uniform(0, 2 * np.pi, n)
radii  = 50 * np.sqrt(rng.uniform(0, 1, n))   # sqrt for uniform area sampling
coords = np.stack([radii * np.cos(angles),
                   radii * np.sin(angles)], axis=1)
```

### Step 2 – Initialise `BosporusFlow`

```python
flow = BosporusFlow(coords)
```

`BosporusFlow` takes a single argument: an `(N, 2)` or `(N, 3)` NumPy array of spatial
coordinates. Internally it creates an empty DataFrame `flow.df` indexed by node ID.

### Step 3 – Build the spatial graph

Choose one of three graph construction methods:

```python
# Option A – Delaunay triangulation (recommended default, no hyperparameter)
flow.construct_graph("delaunay", params=None)

# Option B – k-nearest-neighbour graph (directed)
# flow.construct_graph("knn", params={"k": 6})

# Option C – radius-nearest-neighbour graph (undirected)
# flow.construct_graph("rnn", params={"r": 15.0})
```

The edge list is stored in `flow.edge_list`.

### Step 4 – Compute centrality measures

```python
measures = ["degree", "closeness", "betweenness", "harmonic", "clustering", "pagerank"]
flow.compute_centralities(measures=measures)

print(flow.df.head())
```

All six measures are now columns in `flow.df`.

### Step 5 – Compute a boundary distance

BOSPORUS needs to know how far each node is from the boundary. The method returns the
column key that was added to `flow.df` and that will be used for fitting.

```python
# Distance to the convex hull of the point cloud
distance_key = flow.compute_distance_to_convex_hull()

# Alternatives (use at most one):
# distance_key = flow.compute_distance_to_rectangular_border()
# distance_key = flow.compute_distance_to_pointset(pointset=boundary_coords)
# distance_key = flow.compute_distance_to_mask(mask=binary_2d_array)

print(f"Distance column: '{distance_key}'")
print(flow.df[[distance_key]].describe())
```

### Step 6 – Fit models and correct

```python
from bosporus.fit import ConstantFit, PiecewiseLinearFit, ExponentialSaturationFit, MichaelisMentenFit

flow.fit_models(
    measures=measures,
    distance_key=distance_key,
    fits=[ConstantFit, PiecewiseLinearFit, ExponentialSaturationFit, MichaelisMentenFit],
    calculate_rel_ll_to_baseline=ConstantFit,   # ConstantFit = "no boundary effect" baseline
)
```

After this step:

- `flow.fit_quality` — a DataFrame with one row per measure summarising the best model,
  AIC-based statistics, effect strength, and half-life.
- `flow.df` — now also contains columns named `"BOSPORUS corrected <measure>"` for each
  centrality.

---

## Reading the results

```python
# Overview table
print(flow.fit_quality[[
    "measure",
    "best_fit_type",
    "observed_effect_strength",   # signed relative drop from border to centre
    "observed_half_life",         # distance at which ~50% of the effect has decayed
    "affected samples",           # fraction of nodes still in the boundary-affected zone
]].to_string(index=False))
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
corrected_cols = [c for c in flow.df.columns if c.startswith("BOSPORUS corrected")]
print(flow.df[corrected_cols].head())

# Single measure
flow.df[["degree", "BOSPORUS corrected degree"]].head()
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

| Graph type | Key parameter | When to use |
|---|---|---|
| `"delaunay"` | none | Default choice; produces a natural triangulation without isolated nodes |
| `"knn"` | `k` (int) | When you want a fixed connectivity per node |
| `"rnn"` | `r` (float, same units as coords) | When biological interaction radius is known |

---

## Choosing a distance function

| Method | Use case |
|---|---|
| `compute_distance_to_convex_hull()` | General default; works for any convex or near-convex tissue shape |
| `compute_distance_to_rectangular_border()` | Rectangular imaging window / biopsy |
| `compute_distance_to_pointset(pointset)` | Border defined by a set of explicit landmark coordinates |
| `compute_distance_to_mask(mask)` | Binary image mask defining the tissue boundary |

All four methods accept an optional `distance_key` string argument to rename the distance
column in `flow.df`.

---

## Visualising the correction (optional)

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sc = axes[0].scatter(coords[:, 0], coords[:, 1],
                     c=flow.df["degree"], cmap="viridis", s=8)
axes[0].set_title("Raw degree centrality")
plt.colorbar(sc, ax=axes[0])

sc2 = axes[1].scatter(coords[:, 0], coords[:, 1],
                      c=flow.df["BOSPORUS corrected degree"], cmap="viridis", s=8)
axes[1].set_title("BOSPORUS-corrected degree centrality")
plt.colorbar(sc2, ax=axes[1])

plt.tight_layout()
plt.savefig("bosporus_correction.png", dpi=150)
plt.show()
```

Nodes near the border should appear more uniform after correction.

---

## API reference summary

### `BosporusFlow(coordinates)`

| Method | Signature | Description |
|---|---|---|
| `run_all` | `(graph_type, params, distance_function, ...)` | Single-call convenience wrapper |
| `construct_graph` | `(graph_type, params)` | Build edge list |
| `compute_centralities` | `(measures)` | Populate `flow.df` with centrality columns |
| `compute_distance_to_convex_hull` | `(distance_key=None)` | Add convex-hull distance column |
| `compute_distance_to_rectangular_border` | `(distance_key=None)` | Add rectangular-border distance column |
| `compute_distance_to_pointset` | `(pointset, distance_key=None)` | Add distance-to-pointset column |
| `compute_distance_to_mask` | `(mask, distance_key=None)` | Add distance-to-mask column |
| `fit_models` | `(measures, distance_key, fits, ...)` | Fit all models, select best by AIC, correct centralities |

### Key attributes after a full run

| Attribute | Type | Content |
|---|---|---|
| `flow.df` | `pd.DataFrame` | All node-level data: coordinates index, centralities, distances, corrected values |
| `flow.fit_quality` | `pd.DataFrame` | Per-measure model selection summary |
| `flow.best_fits` | `dict[str, Fit]` | Best `Fit` object per measure |
| `flow.edge_list` | `set` | Set of edges (frozensets or tuples) |
