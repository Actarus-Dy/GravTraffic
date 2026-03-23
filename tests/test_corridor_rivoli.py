"""Tests for the Rivoli corridor benchmark scenario.

Validates the scenario infrastructure for Milestone S10:
"Gain debit >= +15% vs feux fixes sur corridor (12 carrefours)"

Tests use SHORT durations (60 s = 600 steps) for fast CI execution.
The 15% throughput gain target may not be met in a 60 s test; that is
expected.  These tests validate correctness of the simulation
infrastructure, not the final benchmark result.

Author: Agent #23 Scientific Validation Tester
Date: 2026-03-22
"""

from __future__ import annotations

import numpy as np
import pytest

from gravtraffic.scenarios.rivoli import RivoliCorridor

# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def corridor():
    """Default RivoliCorridor with 12 intersections."""
    return RivoliCorridor(seed=42)


@pytest.fixture
def small_corridor():
    """Smaller corridor (4 intersections) for faster tests."""
    return RivoliCorridor(n_intersections=4, spacing=150.0, seed=42)


# ======================================================================
# Test 1: Construction
# ======================================================================


class TestConstruction:
    """RivoliCorridor creates without error and has correct geometry."""

    def test_creates_without_error(self):
        corridor = RivoliCorridor()
        assert corridor is not None

    def test_default_parameters(self, corridor):
        assert corridor.n_intersections == 12
        assert corridor.spacing == 150.0
        assert corridor.v_max == pytest.approx(13.9)
        assert corridor.corridor_length == pytest.approx(1650.0)

    def test_intersection_positions(self, corridor):
        assert len(corridor.intersection_x) == 12
        assert corridor.intersection_x[0] == pytest.approx(0.0)
        assert corridor.intersection_x[1] == pytest.approx(150.0)
        assert corridor.intersection_x[-1] == pytest.approx(1650.0)

    def test_custom_parameters(self):
        c = RivoliCorridor(n_intersections=6, spacing=200.0, v_max=10.0)
        assert c.n_intersections == 6
        assert c.corridor_length == pytest.approx(1000.0)
        assert c.v_max == pytest.approx(10.0)


# ======================================================================
# Test 2: Fixed timing run
# ======================================================================


class TestFixedTiming:
    """run_fixed_timing runs for 60 s and returns valid KPIs."""

    def test_runs_and_returns_dict(self, corridor):
        result = corridor.run_fixed_timing(duration_s=60.0)
        assert isinstance(result, dict)
        assert result["mode"] == "fixed"

    def test_required_keys_present(self, corridor):
        result = corridor.run_fixed_timing(duration_s=60.0)
        required = {
            "mode",
            "duration_s",
            "mean_speed_ms",
            "mean_speed_kmh",
            "mean_stops",
            "total_throughput",
            "final_n_vehicles",
            "total_steps",
        }
        assert required.issubset(result.keys())

    def test_duration_matches(self, corridor):
        result = corridor.run_fixed_timing(duration_s=60.0)
        assert result["duration_s"] == pytest.approx(60.0)
        assert result["total_steps"] == 600


# ======================================================================
# Test 3: Optimized timing run
# ======================================================================


class TestOptimizedTiming:
    """run_optimized runs for 60 s and returns valid KPIs."""

    def test_runs_and_returns_dict(self, corridor):
        result = corridor.run_optimized(duration_s=60.0)
        assert isinstance(result, dict)
        assert result["mode"] == "optimized"

    def test_required_keys_present(self, corridor):
        result = corridor.run_optimized(duration_s=60.0)
        required = {
            "mode",
            "duration_s",
            "mean_speed_ms",
            "mean_speed_kmh",
            "mean_stops",
            "total_throughput",
            "final_n_vehicles",
            "total_steps",
        }
        assert required.issubset(result.keys())


# ======================================================================
# Test 4: compare() runs both and returns gain metrics
# ======================================================================


class TestCompare:
    """compare() returns both results and computed gain percentages."""

    def test_compare_returns_all_keys(self, small_corridor):
        result = small_corridor.compare(duration_s=60.0)
        assert "fixed" in result
        assert "optimized" in result
        assert "speed_gain_pct" in result
        assert "stops_reduction_pct" in result
        assert "throughput_gain_pct" in result

    def test_compare_fixed_and_optimized_are_dicts(self, small_corridor):
        result = small_corridor.compare(duration_s=60.0)
        assert isinstance(result["fixed"], dict)
        assert isinstance(result["optimized"], dict)
        assert result["fixed"]["mode"] == "fixed"
        assert result["optimized"]["mode"] == "optimized"

    def test_gain_is_finite(self, small_corridor):
        result = small_corridor.compare(duration_s=60.0)
        assert np.isfinite(result["speed_gain_pct"])
        assert np.isfinite(result["stops_reduction_pct"])
        assert np.isfinite(result["throughput_gain_pct"])


# ======================================================================
# Test 5: Mean speed is positive and bounded
# ======================================================================


class TestSpeedBounds:
    """Mean speed is positive and below v_max * 3.6 km/h."""

    def test_fixed_speed_positive(self, corridor):
        result = corridor.run_fixed_timing(duration_s=60.0)
        assert result["mean_speed_ms"] > 0.0, "Mean speed must be positive"

    def test_fixed_speed_below_vmax(self, corridor):
        result = corridor.run_fixed_timing(duration_s=60.0)
        max_kmh = corridor.v_max * 3.6
        assert result["mean_speed_kmh"] < max_kmh * 1.5, (
            f"Mean speed {result['mean_speed_kmh']:.1f} km/h exceeds "
            f"1.5 * v_max = {max_kmh * 1.5:.1f} km/h"
        )

    def test_optimized_speed_positive(self, corridor):
        result = corridor.run_optimized(duration_s=60.0)
        assert result["mean_speed_ms"] > 0.0, "Mean speed must be positive"

    def test_optimized_speed_below_vmax(self, corridor):
        result = corridor.run_optimized(duration_s=60.0)
        max_kmh = corridor.v_max * 3.6
        assert result["mean_speed_kmh"] < max_kmh * 1.5, (
            f"Mean speed {result['mean_speed_kmh']:.1f} km/h exceeds "
            f"1.5 * v_max = {max_kmh * 1.5:.1f} km/h"
        )

    def test_speed_consistency_ms_vs_kmh(self, corridor):
        result = corridor.run_fixed_timing(duration_s=60.0)
        assert result["mean_speed_kmh"] == pytest.approx(result["mean_speed_ms"] * 3.6, rel=1e-10)


# ======================================================================
# Test 6: Final vehicle count is reasonable
# ======================================================================


class TestVehicleCount:
    """Number of final vehicles is reasonable (not 0, not exploding)."""

    def test_fixed_has_vehicles(self, corridor):
        result = corridor.run_fixed_timing(duration_s=60.0)
        assert result["final_n_vehicles"] > 0, (
            "Simulation ended with 0 vehicles -- all escaped or none injected"
        )

    def test_fixed_not_exploding(self, corridor):
        result = corridor.run_fixed_timing(duration_s=60.0)
        # With injection_rate=0.5/s * 2 directions * 60s + ~82 initial
        # we expect roughly < 200 vehicles (some exit the corridor)
        assert result["final_n_vehicles"] < 500, (
            f"Vehicle count {result['final_n_vehicles']} seems too high"
        )

    def test_optimized_has_vehicles(self, corridor):
        result = corridor.run_optimized(duration_s=60.0)
        assert result["final_n_vehicles"] > 0

    def test_optimized_not_exploding(self, corridor):
        result = corridor.run_optimized(duration_s=60.0)
        assert result["final_n_vehicles"] < 500


# ======================================================================
# Test 7: Optimized mode produces different results from fixed
# ======================================================================


class TestModeDifference:
    """The optimized mode must produce DIFFERENT results from fixed.

    Because the green wave changes phase offsets and the optimizer may
    adjust timings, the two modes should not produce identical KPIs.
    """

    def test_results_differ(self, corridor):
        fixed = corridor.run_fixed_timing(duration_s=60.0)
        optimized = corridor.run_optimized(duration_s=60.0)

        # At least one KPI must differ (they use different signal logic)
        speeds_differ = abs(fixed["mean_speed_ms"] - optimized["mean_speed_ms"]) > 1e-6
        stops_differ = abs(fixed["mean_stops"] - optimized["mean_stops"]) > 1e-6
        throughput_differs = abs(fixed["total_throughput"] - optimized["total_throughput"]) > 1e-6
        vehicles_differ = fixed["final_n_vehicles"] != optimized["final_n_vehicles"]

        assert speeds_differ or stops_differ or throughput_differs or vehicles_differ, (
            "Fixed and optimized modes produced identical results -- "
            "green wave coordination had no effect"
        )


# ======================================================================
# Additional validation: dimensional analysis & edge cases
# ======================================================================


class TestDimensionalConsistency:
    """Verify units and dimensional relationships."""

    def test_total_steps_equals_duration_over_dt(self, corridor):
        result = corridor.run_fixed_timing(duration_s=60.0)
        expected_steps = int(60.0 / 0.1)
        assert result["total_steps"] == expected_steps

    def test_stops_non_negative(self, corridor):
        result = corridor.run_fixed_timing(duration_s=60.0)
        assert result["mean_stops"] >= 0.0

    def test_throughput_non_negative(self, corridor):
        result = corridor.run_fixed_timing(duration_s=60.0)
        assert result["total_throughput"] >= 0.0


class TestReproducibility:
    """Seed-based reproducibility."""

    def test_same_seed_same_result(self):
        c1 = RivoliCorridor(seed=42)
        c2 = RivoliCorridor(seed=42)
        r1 = c1.run_fixed_timing(duration_s=30.0)
        r2 = c2.run_fixed_timing(duration_s=30.0)
        assert r1["mean_speed_ms"] == pytest.approx(r2["mean_speed_ms"], rel=1e-12)
        assert r1["mean_stops"] == pytest.approx(r2["mean_stops"], rel=1e-12)

    def test_different_seed_different_result(self):
        c1 = RivoliCorridor(seed=42)
        c2 = RivoliCorridor(seed=99)
        r1 = c1.run_fixed_timing(duration_s=30.0)
        r2 = c2.run_fixed_timing(duration_s=30.0)
        # With different seeds, initial conditions differ, so results
        # should differ (extremely unlikely to be identical)
        assert (
            abs(r1["mean_speed_ms"] - r2["mean_speed_ms"]) > 1e-10
            or r1["final_n_vehicles"] != r2["final_n_vehicles"]
        )
