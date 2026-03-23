"""Tests for the scientific validation module.

Uses quick mode for fast CI execution (~10s).

Author: Agent #23 Scientific Validation Tester
Date: 2026-03-23
"""

from __future__ import annotations

import numpy as np
import pytest

from gravtraffic.validation.emergence import (
    gini_coefficient,
    run_emergence_analysis,
)
from gravtraffic.validation.fundamental_diagram import (
    greenshields_speed,
    run_fd_sweep,
)
from gravtraffic.validation.report import run_validation_suite

# ---------------------------------------------------------------------------
# Fundamental Diagram
# ---------------------------------------------------------------------------


class TestGreenshieldsSpeed:
    def test_free_flow(self) -> None:
        assert greenshields_speed(0.0) == pytest.approx(33.33)

    def test_jam(self) -> None:
        assert greenshields_speed(150.0) == pytest.approx(0.0)

    def test_half_density(self) -> None:
        assert greenshields_speed(75.0) == pytest.approx(33.33 / 2, rel=0.01)

    def test_over_jam(self) -> None:
        assert greenshields_speed(200.0) == 0.0


class TestFDSweep:
    def test_sweep_returns_expected_keys(self) -> None:
        result = run_fd_sweep(
            densities=[20, 60, 100],
            n_steps=50,
            warmup_steps=20,
            seed=42,
        )
        assert "r_squared" in result
        assert "rmse" in result
        assert "densities" in result
        assert "measured_speeds" in result
        assert len(result["densities"]) == 3

    def test_sweep_r_squared_above_threshold(self) -> None:
        """With calibrated params and sufficient warmup, R² should exceed 0.5."""
        result = run_fd_sweep(
            densities=[20, 40, 60, 80, 100],
            n_steps=400,
            warmup_steps=250,
            seed=42,
        )
        assert result["r_squared"] > 0.5, (
            f"R²={result['r_squared']:.4f} too low — model may not converge "
            f"to Greenshields from v_free initial conditions"
        )

    def test_speed_decreases_with_density(self) -> None:
        """Mean speed should generally decrease with density."""
        result = run_fd_sweep(
            densities=[20, 80, 120],
            n_steps=100,
            warmup_steps=30,
            seed=42,
        )
        speeds = result["measured_speeds"]
        assert speeds[0] > speeds[-1], (
            f"Speed at rho=20 ({speeds[0]:.1f}) should exceed speed at rho=120 ({speeds[-1]:.1f})"
        )


# ---------------------------------------------------------------------------
# Emergence
# ---------------------------------------------------------------------------


class TestGiniCoefficient:
    def test_equal_values(self) -> None:
        assert gini_coefficient(np.array([10.0, 10.0, 10.0])) == pytest.approx(0.0)

    def test_maximal_inequality(self) -> None:
        g = gini_coefficient(np.array([0.0, 0.0, 100.0]))
        assert g > 0.5

    def test_empty(self) -> None:
        assert gini_coefficient(np.array([])) == 0.0


class TestEmergenceAnalysis:
    def test_returns_expected_structure(self) -> None:
        result = run_emergence_analysis(n_steps=50, seed=42)
        assert "gravity_on" in result
        assert "gravity_off" in result
        assert "emergence_score" in result
        assert isinstance(result["emergence_score"], float)

    def test_gravity_on_has_metrics(self) -> None:
        result = run_emergence_analysis(n_steps=50, seed=42)
        g = result["gravity_on"]
        for key in (
            "upstream_deceleration_ms",
            "variance_ratio",
            "gini_initial",
            "gini_final",
            "wave_speed_ms",
        ):
            assert key in g, f"Missing key {key}"


# ---------------------------------------------------------------------------
# Validation Report
# ---------------------------------------------------------------------------


class TestValidationReport:
    def test_quick_report_generates(self) -> None:
        """Quick validation should complete without errors."""
        report = run_validation_suite(quick=True, seed=42)
        assert "overall_verdict" in report
        assert report["mode"] == "quick"
        assert report["fundamental_diagram"]["verdict"] in ("PASS", "FAIL")
        assert report["emergence"]["verdict"] in ("PASS", "FAIL")

    def test_quick_report_deterministic(self) -> None:
        """Same seed should produce identical results across all fields."""
        r1 = run_validation_suite(quick=True, seed=42)
        r2 = run_validation_suite(quick=True, seed=42)
        # Aggregates
        assert r1["fundamental_diagram"]["r_squared"] == r2["fundamental_diagram"]["r_squared"]
        assert r1["emergence"]["score"] == r2["emergence"]["score"]
        assert r1["overall_verdict"] == r2["overall_verdict"]
        # Detailed data points
        for d1, d2 in zip(r1["fundamental_diagram"]["data"], r2["fundamental_diagram"]["data"]):
            assert d1["measured_speed"] == d2["measured_speed"]
        # Emergence detail
        for key in r1["emergence"]["gravity_on"]:
            assert r1["emergence"]["gravity_on"][key] == r2["emergence"]["gravity_on"][key]
