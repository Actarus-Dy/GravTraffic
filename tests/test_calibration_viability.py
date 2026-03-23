"""Calibration viability tests -- go/no-go gate for GravTraffic (C-01).

These tests verify that the gravitational traffic model can reproduce
a plausible speed-density relationship when compared against Greenshields'
fundamental diagram.

This is a feasibility study, NOT production calibration.

NOTE: These tests intentionally use legacy G_s values (9.8, 2.0) that differ
from the unified calibration (G_s=5.0). They validate the original parameter
exploration that led to the final calibration, not the production parameters.

Agent #23 Scientific Validation Tester
"""

from __future__ import annotations

import numpy as np

from gravtraffic.core.calibration import (
    RHO_JAM,
    V_FREE_MS,
    _compute_r_squared,
    _compute_rmse,
    _greenshields_speed,
    calibration_viability_report,
    run_calibration_test,
)

# ======================================================================
# Analytical validation of helper functions
# ======================================================================


class TestGreenshieldsFormula:
    """Validate Greenshields v(rho) = v_free * (1 - rho/rho_jam)."""

    def test_zero_density(self) -> None:
        """At zero density, speed equals free-flow speed."""
        rho = np.array([0.0], dtype=np.float64)
        v = _greenshields_speed(rho)
        assert np.isclose(v[0], V_FREE_MS, rtol=1e-12)

    def test_jam_density(self) -> None:
        """At jam density, speed is zero."""
        rho = np.array([RHO_JAM], dtype=np.float64)
        v = _greenshields_speed(rho)
        assert np.isclose(v[0], 0.0, atol=1e-12)

    def test_half_density(self) -> None:
        """At half jam density, speed is half free-flow."""
        rho = np.array([RHO_JAM / 2.0], dtype=np.float64)
        v = _greenshields_speed(rho)
        assert np.isclose(v[0], V_FREE_MS / 2.0, rtol=1e-12)

    def test_monotonically_decreasing(self) -> None:
        """Speed must decrease monotonically with density."""
        rho = np.linspace(0.0, RHO_JAM, 200, dtype=np.float64)
        v = _greenshields_speed(rho)
        diffs = np.diff(v)
        assert np.all(diffs <= 0.0), "Greenshields speed is not monotonically decreasing"

    def test_vectorized_output_dtype(self) -> None:
        """Output must be float64."""
        rho = np.array([10.0, 50.0, 100.0], dtype=np.float64)
        v = _greenshields_speed(rho)
        assert v.dtype == np.float64


class TestRSquared:
    """Validate R^2 computation: 1 - SS_res / SS_tot."""

    def test_perfect_fit(self) -> None:
        """R^2 = 1.0 when prediction equals truth exactly."""
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        r2 = _compute_r_squared(y, y)
        assert np.isclose(r2, 1.0, atol=1e-14)

    def test_mean_prediction(self) -> None:
        """R^2 = 0.0 when prediction is the mean of truth."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        y_pred = np.full(5, np.mean(y_true), dtype=np.float64)
        r2 = _compute_r_squared(y_true, y_pred)
        assert np.isclose(r2, 0.0, atol=1e-14)

    def test_negative_r_squared(self) -> None:
        """R^2 < 0 when prediction is worse than mean."""
        y_true = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        y_pred = np.array([10.0, 20.0, 30.0], dtype=np.float64)
        r2 = _compute_r_squared(y_true, y_pred)
        assert r2 < 0.0

    def test_known_value(self) -> None:
        """R^2 for a known analytical case."""
        y_true = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        y_pred = np.array([1.1, 2.1, 2.9], dtype=np.float64)
        ss_res = 0.01 + 0.01 + 0.01  # 0.03
        ss_tot = 1.0 + 0.0 + 1.0  # 2.0
        expected = 1.0 - ss_res / ss_tot  # 0.985
        r2 = _compute_r_squared(y_true, y_pred)
        assert np.isclose(r2, expected, rtol=1e-10)


class TestRMSE:
    """Validate RMSE computation."""

    def test_zero_error(self) -> None:
        y = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        rmse = _compute_rmse(y, y)
        assert np.isclose(rmse, 0.0, atol=1e-14)

    def test_known_rmse(self) -> None:
        y_true = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        y_pred = np.array([2.0, 3.0, 4.0], dtype=np.float64)
        # MSE = (1+1+1)/3 = 1.0, RMSE = 1.0
        assert np.isclose(_compute_rmse(y_true, y_pred), 1.0, rtol=1e-12)


# ======================================================================
# Integration tests for run_calibration_test
# ======================================================================


class TestRunCalibrationTest:
    """Validate run_calibration_test returns well-formed results."""

    REQUIRED_KEYS = {"G_s", "beta", "r_squared", "rmse_ms", "final_speeds", "densities"}

    def test_returns_valid_dict_with_all_keys(self) -> None:
        """Test 1: run_calibration_test returns valid dict with all keys."""
        result = run_calibration_test(G_s=9.8, beta=1.0, n_vehicles=30, n_steps=10)
        assert isinstance(result, dict)
        assert self.REQUIRED_KEYS == set(result.keys()), (
            f"Missing keys: {self.REQUIRED_KEYS - set(result.keys())}"
        )

    def test_r_squared_is_float(self) -> None:
        """Test 2: r_squared is a float (can be negative for terrible fit)."""
        result = run_calibration_test(G_s=9.8, beta=1.0, n_vehicles=30, n_steps=10)
        assert isinstance(result["r_squared"], float)
        # R^2 can be negative but should not be NaN or Inf
        assert np.isfinite(result["r_squared"])

    def test_rmse_is_non_negative_float(self) -> None:
        """RMSE must be a non-negative finite float."""
        result = run_calibration_test(G_s=9.8, beta=1.0, n_vehicles=30, n_steps=10)
        assert isinstance(result["rmse_ms"], float)
        assert result["rmse_ms"] >= 0.0
        assert np.isfinite(result["rmse_ms"])

    def test_output_array_shapes(self) -> None:
        """final_speeds and densities must have shape (n_vehicles,)."""
        n = 50
        result = run_calibration_test(G_s=2.0, beta=0.5, n_vehicles=n, n_steps=5)
        assert result["final_speeds"].shape == (n,)
        assert result["densities"].shape == (n,)

    def test_output_dtype_float64(self) -> None:
        """All arrays must be float64."""
        result = run_calibration_test(G_s=2.0, beta=0.5, n_vehicles=30, n_steps=5)
        assert result["final_speeds"].dtype == np.float64
        assert result["densities"].dtype == np.float64

    def test_speeds_physically_bounded(self) -> None:
        """Final speeds must be in [0, v_free]."""
        result = run_calibration_test(G_s=15.0, beta=1.5, n_vehicles=50, n_steps=50)
        assert np.all(result["final_speeds"] >= 0.0)
        assert np.all(result["final_speeds"] <= V_FREE_MS + 1e-10)

    def test_densities_in_expected_range(self) -> None:
        """Sampled densities must be in [5, 140] veh/km."""
        result = run_calibration_test(G_s=9.8, beta=1.0, n_vehicles=100, n_steps=10)
        assert np.all(result["densities"] >= 5.0)
        assert np.all(result["densities"] <= 140.0)

    def test_g_s_and_beta_echoed(self) -> None:
        """Returned G_s and beta match input."""
        result = run_calibration_test(G_s=2.0, beta=0.5, n_vehicles=20, n_steps=5)
        assert result["G_s"] == 2.0
        assert result["beta"] == 0.5

    def test_reproducibility_with_seed(self) -> None:
        """Same seed produces identical results."""
        r1 = run_calibration_test(G_s=9.8, beta=1.0, n_vehicles=30, n_steps=10, seed=42)
        r2 = run_calibration_test(G_s=9.8, beta=1.0, n_vehicles=30, n_steps=10, seed=42)
        assert r1["r_squared"] == r2["r_squared"]
        np.testing.assert_array_equal(r1["final_speeds"], r2["final_speeds"])

    def test_different_seeds_differ(self) -> None:
        """Different seeds produce different results."""
        r1 = run_calibration_test(G_s=9.8, beta=1.0, n_vehicles=50, n_steps=10, seed=42)
        r2 = run_calibration_test(G_s=9.8, beta=1.0, n_vehicles=50, n_steps=10, seed=99)
        # Extremely unlikely to be identical with different seeds
        assert not np.array_equal(r1["final_speeds"], r2["final_speeds"])


# ======================================================================
# Integration tests for calibration_viability_report
# ======================================================================


class TestCalibrationViabilityReport:
    """Validate calibration_viability_report."""

    def test_returns_three_results(self, capsys) -> None:
        """Test 3: calibration_viability_report returns 3 results."""
        results = calibration_viability_report()
        assert len(results) == 3
        captured = capsys.readouterr()
        assert "GRAVTRAFFIC CALIBRATION VIABILITY REPORT" in captured.out

    def test_results_sorted_by_r_squared_descending(self) -> None:
        """Results must be sorted best-first by R^2."""
        results = calibration_viability_report()
        r2_values = [r["r_squared"] for r in results]
        assert r2_values == sorted(r2_values, reverse=True)

    def test_all_configs_present(self) -> None:
        """All three named configs appear in results."""
        results = calibration_viability_report()
        names = {r["name"] for r in results}
        assert names == {"Helbing", "GravJanus", "NonLinear"}

    def test_each_result_has_required_keys(self) -> None:
        """Each result dict has all required keys plus 'name'."""
        required = {"name", "G_s", "beta", "r_squared", "rmse_ms", "final_speeds", "densities"}
        results = calibration_viability_report()
        for r in results:
            assert required <= set(r.keys()), (
                f"Config {r.get('name', '?')} missing keys: {required - set(r.keys())}"
            )


# ======================================================================
# Sanity check: at least one config does something useful
# ======================================================================


class TestModelSanity:
    """Test 4 & 5: Verify the model produces meaningful output."""

    def test_at_least_one_positive_r_squared(self, capsys) -> None:
        """Test 4: At least one config achieves R^2 > 0.0."""
        results = calibration_viability_report()
        r2_values = [r["r_squared"] for r in results]

        # Test 5: Print R^2 values for human inspection
        print("\n--- R^2 values for human inspection ---")
        for r in results:
            print(f"  {r['name']:<12}: R^2 = {r['r_squared']:.6f}, RMSE = {r['rmse_ms']:.4f} m/s")
        print("---------------------------------------")

        assert any(r2 > 0.0 for r2 in r2_values), (
            f"All R^2 values are <= 0: {r2_values}. "
            "The gravitational model does not produce any meaningful "
            "speed-density relationship."
        )

    def test_speeds_vary_with_density(self) -> None:
        """Final speeds should show variation (not collapsed to one value)."""
        result = run_calibration_test(G_s=9.8, beta=1.0, n_vehicles=100, n_steps=50)
        speed_std = np.std(result["final_speeds"])
        assert speed_std > 0.5, (
            f"Speed standard deviation = {speed_std:.4f} m/s is too small. "
            "The model may have collapsed all vehicles to one speed."
        )


# ======================================================================
# Edge cases and boundary conditions
# ======================================================================


class TestEdgeCases:
    """Boundary conditions per formula validation requirements."""

    def test_single_vehicle(self) -> None:
        """With 1 vehicle, no N-body forces act; speed changes only via relaxation."""
        result = run_calibration_test(G_s=9.8, beta=1.0, n_vehicles=1, n_steps=10)
        assert result["final_speeds"].shape == (1,)
        assert np.isfinite(result["final_speeds"][0])

    def test_two_vehicles(self) -> None:
        """Minimal N-body: 2 vehicles interact."""
        result = run_calibration_test(G_s=9.8, beta=1.0, n_vehicles=2, n_steps=10)
        assert result["final_speeds"].shape == (2,)
        assert np.all(np.isfinite(result["final_speeds"]))

    def test_zero_g_s(self) -> None:
        """G_s=0: no gravitational force. Only relaxation drives dynamics."""
        result = run_calibration_test(G_s=0.0, beta=1.0, n_vehicles=50, n_steps=50)
        # With only relaxation, should converge toward Greenshields
        assert np.isfinite(result["r_squared"])
        # With damping toward desired speed and no gravitational perturbation,
        # R^2 should actually be decent
        assert result["r_squared"] > 0.5, (
            f"G_s=0 (pure relaxation) gave R^2={result['r_squared']:.4f}; "
            "expected > 0.5 since relaxation drives toward Greenshields"
        )

    def test_very_large_g_s(self) -> None:
        """Very large G_s should not produce NaN or Inf."""
        result = run_calibration_test(G_s=1000.0, beta=1.0, n_vehicles=30, n_steps=10)
        assert np.all(np.isfinite(result["final_speeds"]))

    def test_beta_zero(self) -> None:
        """beta=0: all |delta|^0 = 1, mass depends only on sign and density."""
        result = run_calibration_test(G_s=9.8, beta=0.0, n_vehicles=30, n_steps=10)
        assert np.all(np.isfinite(result["final_speeds"]))
        assert np.isfinite(result["r_squared"])
