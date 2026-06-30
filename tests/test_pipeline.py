"""
Unit tests for pipeline.py (Flow class).

All tests use Flow.from_distances_and_scores so that graph-tool is NOT required.
Synthetic data is generated with numpy rng for reproducibility.
"""

import numpy as np
import pandas as pd
import pytest

from bosperrus.pipeline import Flow
from bosperrus.fit import ConstantFit
from bosperrus.distances import distance_to_rectangular_border


# ============================================================
# Shared fixtures
# ============================================================

RNG = np.random.default_rng(42)
N = 80  # number of "nodes"

# Distances: monotone, in [0, 10]
DISTANCES = pd.Series(np.linspace(0.5, 10.0, N), name="distance")

# Scores: two centrality-like measures correlated with distance (border effect)
_noise = RNG.normal(0, 0.3, size=(N, 2))
_signal = np.column_stack([
    2.0 * (1 - np.exp(-0.5 * DISTANCES.values)) + 1.0,  # exponential saturation pattern
    DISTANCES.values * 0.4 + 0.5,                         # linear-ish pattern
])
SCORES = pd.DataFrame(_signal + _noise, columns=["measure_A", "measure_B"])

# Constant scores (no distance effect) for testing ConstantFit selection
_const_noise = RNG.normal(0, 0.05, size=(N, 2))
CONSTANT_SCORES = pd.DataFrame(
    np.column_stack([
        np.full(N, 5.0) + _const_noise[:, 0],
        np.full(N, 3.0) + _const_noise[:, 1],
    ]),
    columns=["const_A", "const_B"],
)


# ============================================================
# Construction tests
# ============================================================

class TestFromDistancesAndScores:

    def test_from_distances_and_scores_creates_flow(self):
        """Basic construction succeeds and observations has the distance column."""
        flow = Flow.from_distances_and_scores(distances=DISTANCES, scores=SCORES)
        assert isinstance(flow, Flow)
        assert "distance" in flow.observations.columns
        assert "measure_A" in flow.observations.columns
        assert "measure_B" in flow.observations.columns

    def test_observations_contains_all_score_columns(self):
        """All score columns appear in observations."""
        flow = Flow.from_distances_and_scores(distances=DISTANCES, scores=SCORES)
        for col in SCORES.columns:
            assert col in flow.observations.columns

    def test_observations_row_count_matches_input(self):
        """observations has the same number of rows as the input."""
        flow = Flow.from_distances_and_scores(distances=DISTANCES, scores=SCORES)
        assert len(flow.observations) == N


# ============================================================
# flow() execution tests
# ============================================================

class TestFlowExecution:

    @pytest.fixture(scope="class")
    def run_flow(self):
        """Run flow() once and return the Flow object."""
        flow = Flow.from_distances_and_scores(distances=DISTANCES, scores=SCORES)
        flow.flow()
        return flow

    def test_flow_runs_and_produces_outputs(self, run_flow):
        """After flow(), observations has corrected columns, fit_quality is a DataFrame,
        best_fits is a dict."""
        flow = run_flow
        assert hasattr(flow, "fit_quality"), "flow.fit_quality not set"
        assert hasattr(flow, "best_fits"), "flow.best_fits not set"
        assert isinstance(flow.fit_quality, pd.DataFrame)
        assert isinstance(flow.best_fits, dict)

    def test_best_fits_populated_after_flow(self, run_flow):
        """
        Regression: best_fits was previously always empty.
        Assert len(best_fits) == number of measures.
        """
        flow = run_flow
        measures = list(SCORES.columns)
        assert len(flow.best_fits) == len(measures), (
            f"Expected {len(measures)} entries in best_fits, got {len(flow.best_fits)}"
        )

    def test_observations_has_corrected_columns_for_each_measure(self, run_flow):
        """Each measure gets a 'BOSPERRUS corrected {measure}' column in observations."""
        flow = run_flow
        for measure in SCORES.columns:
            expected_col = f"BOSPERRUS corrected {measure}"
            assert expected_col in flow.observations.columns, (
                f"Missing corrected column: '{expected_col}'"
            )

    def test_fit_quality_columns_match_measures(self, run_flow):
        """fit_quality.columns should equal the list of measures that were fitted."""
        flow = run_flow
        expected = set(SCORES.columns)
        actual = set(flow.fit_quality.columns)
        assert actual == expected, f"fit_quality columns mismatch: {actual} != {expected}"

    def test_corrected_columns_have_correct_length(self, run_flow):
        """Corrected columns should have the same length as the input."""
        flow = run_flow
        for measure in SCORES.columns:
            col = f"BOSPERRUS corrected {measure}"
            assert len(flow.observations[col]) == N


# ============================================================
# Mutation / isolation tests
# ============================================================

class TestDataImmutability:

    def test_input_dataframe_not_mutated(self):
        """
        Regression: original scores DataFrame must NOT have 'BOSPERRUS corrected' columns
        added to it after flow() is called.
        """
        scores_copy = SCORES.copy()
        original_columns = list(scores_copy.columns)

        flow = Flow.from_distances_and_scores(distances=DISTANCES, scores=scores_copy)
        flow.flow()

        # The original DataFrame (scores_copy) should be unchanged
        assert list(scores_copy.columns) == original_columns, (
            "flow() mutated the input scores DataFrame by adding corrected columns: "
            f"{[c for c in scores_copy.columns if c not in original_columns]}"
        )

    def test_input_series_not_mutated(self):
        """distances Series should not be mutated."""
        distances_copy = DISTANCES.copy()
        original_values = distances_copy.values.copy()

        flow = Flow.from_distances_and_scores(distances=distances_copy, scores=SCORES)
        flow.flow()

        np.testing.assert_array_equal(distances_copy.values, original_values)


# ============================================================
# Error handling tests
# ============================================================

class TestErrorHandling:

    def test_invalid_measure_raises_value_error(self):
        """Calling flow(measures=['nonexistent']) should raise ValueError, not AssertionError."""
        flow = Flow.from_distances_and_scores(distances=DISTANCES, scores=SCORES)
        with pytest.raises(ValueError, match="nonexistent"):
            flow.flow(measures=["nonexistent"])

    def test_baseline_not_in_fits_raises_value_error(self):
        """baseline_fit_class not in fits should raise ValueError."""
        flow = Flow.from_distances_and_scores(distances=DISTANCES, scores=SCORES)
        from bosperrus.fit import ExponentialSaturationFit, PiecewiseLinearFit
        with pytest.raises(ValueError):
            flow.flow(fits=[PiecewiseLinearFit], baseline_fit_class=ExponentialSaturationFit)


# ============================================================
# Constant data → ConstantFit selection
# ============================================================

class TestConstantDataFitSelection:

    def test_constant_data_selects_constant_fit(self):
        """
        When scores are essentially constant (no distance effect), the best fit
        should be ConstantFit for all measures because the AIC of the constant
        model will be lowest (no extra parameters needed).
        """
        flow = Flow.from_distances_and_scores(
            distances=DISTANCES,
            scores=CONSTANT_SCORES,
        )
        flow.flow()

        for measure, best_fit in flow.best_fits.items():
            assert isinstance(best_fit, ConstantFit), (
                f"Expected ConstantFit for nearly-constant measure '{measure}', "
                f"got {type(best_fit).__name__}"
            )


# ============================================================
# Subset of measures
# ============================================================

class TestSubsetMeasures:

    def test_flow_with_single_measure(self):
        """Passing a single measure to flow() should work and only populate that measure."""
        flow = Flow.from_distances_and_scores(distances=DISTANCES, scores=SCORES)
        flow.flow(measures=["measure_A"])

        assert "measure_A" in flow.best_fits
        assert "measure_B" not in flow.best_fits
        assert "BOSPERRUS corrected measure_A" in flow.observations.columns
        assert "BOSPERRUS corrected measure_B" not in flow.observations.columns

    def test_fit_quality_columns_match_subset(self):
        """fit_quality should only contain columns for the requested measures."""
        flow = Flow.from_distances_and_scores(distances=DISTANCES, scores=SCORES)
        flow.flow(measures=["measure_B"])

        assert list(flow.fit_quality.columns) == ["measure_B"]


# ============================================================
# from_coords_and_scores
# ============================================================

class TestFromCoordsAndScores:

    def test_from_coords_and_scores_constructs_correctly(self):
        """
        Create 2-D coordinates, use distance_to_rectangular_border as distance_fn,
        pre-compute scores, and verify Flow.from_coords_and_scores constructs correctly.
        """
        rng = np.random.default_rng(123)
        n = 60
        coords = rng.uniform(0, 10, size=(n, 2))

        # Pre-compute distances manually for reference
        expected_distances = distance_to_rectangular_border(coords)

        # Pre-compute synthetic scores
        noise = rng.normal(0, 0.2, size=(n, 2))
        scores = pd.DataFrame(
            np.column_stack([
                expected_distances.values * 0.5 + noise[:, 0],
                expected_distances.values * 0.3 + 1.0 + noise[:, 1],
            ]),
            columns=["score_X", "score_Y"],
        )

        flow = Flow.from_coords_and_scores(
            coordinates=coords,
            distance_fn=distance_to_rectangular_border,
            scores=scores,
        )

        # Construction should succeed
        assert isinstance(flow, Flow)
        assert "distance_to_rectangular_border" in flow.observations.columns
        assert "score_X" in flow.observations.columns
        assert "score_Y" in flow.observations.columns
        assert len(flow.observations) == n

        # Distances should match what distance_to_rectangular_border returns
        np.testing.assert_allclose(
            flow.observations["distance_to_rectangular_border"].values,
            expected_distances.values,
            rtol=1e-10,
        )

    def test_from_coords_and_scores_flow_runs(self):
        """Flow constructed from coords_and_scores can run flow() without errors."""
        rng = np.random.default_rng(456)
        n = 50
        coords = rng.uniform(0, 5, size=(n, 2))
        distances = distance_to_rectangular_border(coords)

        noise = rng.normal(0, 0.1, size=n)
        scores = pd.DataFrame(
            {"metric": distances.values * 0.7 + 2.0 + noise},
        )

        flow = Flow.from_coords_and_scores(
            coordinates=coords,
            distance_fn=distance_to_rectangular_border,
            scores=scores,
        )
        flow.flow()

        assert "metric" in flow.best_fits
        assert "BOSPERRUS corrected metric" in flow.observations.columns
