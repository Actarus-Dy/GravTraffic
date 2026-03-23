"""Tests for gravtraffic.core.signal_optimizer -- time-integrated potential optimizer.

Validates the two public functions:
    - estimate_phi_integral: temporal potential integration
    - optimize_signal_timing: green-phase sweep and selection

Test cases cover correctness, boundary conditions, and physical plausibility
(asymmetric traffic should favour longer green for the heavier direction).

Author: Agent #01 Python Scientific Developer
Date: 2026-03-22
"""

from __future__ import annotations

import numpy as np
import pytest

from gravtraffic.core.signal_optimizer import (
    estimate_phi_integral,
    optimize_signal_timing,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def intersection_pos() -> np.ndarray:
    """Intersection at the origin."""
    return np.array([0.0, 0.0], dtype=np.float64)


@pytest.fixture()
def symmetric_traffic(intersection_pos: np.ndarray) -> dict:
    """10 NS vehicles + 10 EW vehicles, symmetric about the intersection.

    All vehicles are slow (positive mass) and approaching at 10 m/s.
    """
    rng = np.random.default_rng(42)

    # NS vehicles: approaching from +y and -y
    ns_pos = np.column_stack([
        rng.uniform(-5, 5, 10),
        rng.uniform(50, 150, 10) * rng.choice([-1, 1], 10),
    ])
    ns_vel = np.column_stack([
        np.zeros(10),
        -np.sign(ns_pos[:, 1]) * 10.0,
    ])

    # EW vehicles: approaching from +x and -x
    ew_pos = np.column_stack([
        rng.uniform(50, 150, 10) * rng.choice([-1, 1], 10),
        rng.uniform(-5, 5, 10),
    ])
    ew_vel = np.column_stack([
        -np.sign(ew_pos[:, 0]) * 10.0,
        np.zeros(10),
    ])

    positions = np.vstack([ns_pos, ew_pos]).astype(np.float64)
    velocities = np.vstack([ns_vel, ew_vel]).astype(np.float64)
    masses = np.ones(20, dtype=np.float64) * 5.0  # positive = slow = congestion

    return {
        "positions": positions,
        "velocities": velocities,
        "masses": masses,
    }


# ---------------------------------------------------------------------------
# Test 1: estimate_phi_integral returns a finite float
# ---------------------------------------------------------------------------

class TestEstimatePhiIntegral:
    """Tests for estimate_phi_integral."""

    def test_returns_finite_float(
        self, symmetric_traffic: dict, intersection_pos: np.ndarray
    ) -> None:
        """The integral must be a finite Python float."""
        phi = estimate_phi_integral(
            symmetric_traffic["positions"],
            symmetric_traffic["velocities"],
            symmetric_traffic["masses"],
            intersection_pos,
            green_ns=60.0,
            green_ew=50.0,
        )
        assert isinstance(phi, float)
        assert np.isfinite(phi)

    # -------------------------------------------------------------------
    # Test 2: Different timings produce different phi values
    # -------------------------------------------------------------------

    def test_different_timings_different_phi(
        self, symmetric_traffic: dict, intersection_pos: np.ndarray
    ) -> None:
        """Two different green_ns values must produce distinct phi integrals."""
        phi_a = estimate_phi_integral(
            symmetric_traffic["positions"],
            symmetric_traffic["velocities"],
            symmetric_traffic["masses"],
            intersection_pos,
            green_ns=20.0,
            green_ew=90.0,
        )
        phi_b = estimate_phi_integral(
            symmetric_traffic["positions"],
            symmetric_traffic["velocities"],
            symmetric_traffic["masses"],
            intersection_pos,
            green_ns=80.0,
            green_ew=30.0,
        )
        assert phi_a != phi_b, (
            f"Expected different phi values for different timings, "
            f"got phi_a={phi_a}, phi_b={phi_b}"
        )

    # -------------------------------------------------------------------
    # Test 6: No vehicles in radius -> phi = 0
    # -------------------------------------------------------------------

    def test_no_vehicles_in_radius(self, intersection_pos: np.ndarray) -> None:
        """When no vehicles are within the radius, phi integral must be 0.0."""
        # Vehicles are 500 m away, radius is 200 m
        far_positions = np.array([[500.0, 500.0], [-500.0, -500.0]], dtype=np.float64)
        far_velocities = np.array([[1.0, 0.0], [-1.0, 0.0]], dtype=np.float64)
        far_masses = np.array([5.0, 5.0], dtype=np.float64)

        phi = estimate_phi_integral(
            far_positions,
            far_velocities,
            far_masses,
            intersection_pos,
            green_ns=60.0,
            green_ew=50.0,
            radius=200.0,
        )
        assert phi == 0.0

    def test_no_vehicles_at_all(self, intersection_pos: np.ndarray) -> None:
        """Empty vehicle arrays must produce phi = 0.0."""
        phi = estimate_phi_integral(
            np.empty((0, 2), dtype=np.float64),
            np.empty((0, 2), dtype=np.float64),
            np.empty(0, dtype=np.float64),
            intersection_pos,
            green_ns=60.0,
            green_ew=50.0,
        )
        assert phi == 0.0


# ---------------------------------------------------------------------------
# Test 3-5: optimize_signal_timing
# ---------------------------------------------------------------------------

class TestOptimizeSignalTiming:
    """Tests for optimize_signal_timing."""

    def test_returns_all_expected_keys(
        self, symmetric_traffic: dict, intersection_pos: np.ndarray
    ) -> None:
        """Result dict must contain all documented keys."""
        result = optimize_signal_timing(
            symmetric_traffic["positions"],
            symmetric_traffic["velocities"],
            symmetric_traffic["masses"],
            intersection_pos,
        )
        expected_keys = {
            "green_ns",
            "green_ew",
            "cycle_s",
            "phi_integral",
            "phi_fixed_60",
            "improvement_pct",
        }
        assert set(result.keys()) == expected_keys
        # All values must be finite floats
        for key, val in result.items():
            assert isinstance(val, float), f"result['{key}'] is {type(val)}, expected float"
            assert np.isfinite(val), f"result['{key}'] = {val} is not finite"

    def test_green_ns_in_valid_range(
        self, symmetric_traffic: dict, intersection_pos: np.ndarray
    ) -> None:
        """green_ns must be in [15, 90] and green_ew >= 10."""
        result = optimize_signal_timing(
            symmetric_traffic["positions"],
            symmetric_traffic["velocities"],
            symmetric_traffic["masses"],
            intersection_pos,
        )
        assert 15.0 <= result["green_ns"] <= 90.0
        assert result["green_ew"] >= 10.0
        # Cycle consistency: cycle = green_ns + green_ew + 10 (amber)
        assert result["cycle_s"] == pytest.approx(
            result["green_ns"] + result["green_ew"] + 10.0
        )

    def test_asymmetric_ns_heavy_favours_longer_green_ns(
        self, intersection_pos: np.ndarray
    ) -> None:
        """When NS has much more congestion traffic, optimizer should pick
        a larger green_ns than green_ew.

        We place 30 slow (positive mass) NS vehicles and only 2 EW vehicles.
        """
        rng = np.random.default_rng(123)

        # 30 NS vehicles approaching from +y
        ns_pos = np.column_stack([
            rng.uniform(-5, 5, 30),
            rng.uniform(30, 150, 30),
        ])
        ns_vel = np.column_stack([
            np.zeros(30),
            np.full(30, -8.0),
        ])

        # 2 EW vehicles (light traffic)
        ew_pos = np.array([[100.0, 2.0], [-100.0, -2.0]], dtype=np.float64)
        ew_vel = np.array([[-8.0, 0.0], [8.0, 0.0]], dtype=np.float64)

        positions = np.vstack([ns_pos, ew_pos]).astype(np.float64)
        velocities = np.vstack([ns_vel, ew_vel]).astype(np.float64)
        masses = np.ones(32, dtype=np.float64) * 5.0

        result = optimize_signal_timing(
            positions, velocities, masses, intersection_pos
        )

        assert result["green_ns"] > result["green_ew"], (
            f"Expected green_ns > green_ew for NS-heavy traffic, "
            f"got green_ns={result['green_ns']}, green_ew={result['green_ew']}"
        )

    def test_asymmetric_ew_heavy_favours_longer_green_ew(
        self, intersection_pos: np.ndarray
    ) -> None:
        """Mirror test: heavy EW traffic should yield green_ew > green_ns."""
        rng = np.random.default_rng(456)

        # 2 NS vehicles (light)
        ns_pos = np.array([[2.0, 100.0], [-2.0, -100.0]], dtype=np.float64)
        ns_vel = np.array([[0.0, -8.0], [0.0, 8.0]], dtype=np.float64)

        # 30 EW vehicles approaching from +x
        ew_pos = np.column_stack([
            rng.uniform(30, 150, 30),
            rng.uniform(-5, 5, 30),
        ])
        ew_vel = np.column_stack([
            np.full(30, -8.0),
            np.zeros(30),
        ])

        positions = np.vstack([ns_pos, ew_pos]).astype(np.float64)
        velocities = np.vstack([ns_vel, ew_vel]).astype(np.float64)
        masses = np.ones(32, dtype=np.float64) * 5.0

        result = optimize_signal_timing(
            positions, velocities, masses, intersection_pos
        )

        assert result["green_ew"] > result["green_ns"], (
            f"Expected green_ew > green_ns for EW-heavy traffic, "
            f"got green_ns={result['green_ns']}, green_ew={result['green_ew']}"
        )

    def test_no_nearby_vehicles(self, intersection_pos: np.ndarray) -> None:
        """With no vehicles in radius, phi values should be zero and
        improvement_pct should be zero."""
        far = np.array([[999.0, 999.0]], dtype=np.float64)
        result = optimize_signal_timing(
            far,
            np.array([[1.0, 0.0]], dtype=np.float64),
            np.array([5.0], dtype=np.float64),
            intersection_pos,
            radius=200.0,
        )
        assert result["phi_integral"] == 0.0
        assert result["phi_fixed_60"] == 0.0
        assert result["improvement_pct"] == 0.0


# ---------------------------------------------------------------------------
# Test 7-9: Velocity-dependent behaviour (DA audit C-06)
# ---------------------------------------------------------------------------

class TestVelocityDependence:
    """Verify that real velocities change optimizer results vs zero velocities.

    The signal_optimizer uses linear extrapolation of vehicle positions.
    With zero velocities, vehicles never move and the extrapolation is
    trivially static.  With real approaching velocities, vehicles reach
    the intersection during the horizon, producing different potential
    integrals and potentially different optimal timings.
    """

    def test_phi_integral_differs_with_real_vs_zero_velocities(
        self, intersection_pos: np.ndarray
    ) -> None:
        """estimate_phi_integral must give different values for approaching
        vehicles vs stationary vehicles."""
        # Vehicles approaching the intersection from +y at 10 m/s
        positions = np.array([
            [0.0, 80.0],
            [0.0, 120.0],
            [0.0, 160.0],
        ], dtype=np.float64)
        masses = np.array([5.0, 5.0, 5.0], dtype=np.float64)

        vel_real = np.array([
            [0.0, -10.0],
            [0.0, -10.0],
            [0.0, -10.0],
        ], dtype=np.float64)
        vel_zero = np.zeros((3, 2), dtype=np.float64)

        phi_real = estimate_phi_integral(
            positions, vel_real, masses, intersection_pos,
            green_ns=60.0, green_ew=50.0,
        )
        phi_zero = estimate_phi_integral(
            positions, vel_zero, masses, intersection_pos,
            green_ns=60.0, green_ew=50.0,
        )

        assert phi_real != phi_zero, (
            f"Expected different phi with real velocities vs zero, "
            f"got phi_real={phi_real}, phi_zero={phi_zero}"
        )

    def test_approaching_vehicles_produce_stronger_potential(
        self, intersection_pos: np.ndarray
    ) -> None:
        """Vehicles moving toward the intersection should produce a more
        negative (stronger congestion) phi integral than stationary ones,
        because they get closer to the red-light obstacles over time."""
        positions = np.array([
            [0.0, 80.0],
            [0.0, 120.0],
        ], dtype=np.float64)
        masses = np.array([5.0, 5.0], dtype=np.float64)

        vel_approach = np.array([[0.0, -10.0], [0.0, -10.0]], dtype=np.float64)
        vel_zero = np.zeros((2, 2), dtype=np.float64)

        phi_approach = estimate_phi_integral(
            positions, vel_approach, masses, intersection_pos,
            green_ns=60.0, green_ew=50.0,
        )
        phi_static = estimate_phi_integral(
            positions, vel_zero, masses, intersection_pos,
            green_ns=60.0, green_ew=50.0,
        )

        # Approaching vehicles get closer to obstacles -> stronger (more negative) phi
        assert phi_approach < phi_static, (
            f"Approaching vehicles should produce more negative phi integral, "
            f"got phi_approach={phi_approach}, phi_static={phi_static}"
        )

    def test_optimizer_uses_velocity_information(
        self, intersection_pos: np.ndarray
    ) -> None:
        """optimize_signal_timing must produce different results with
        real approaching velocities vs zero velocities for asymmetric traffic."""
        rng = np.random.default_rng(789)

        # 20 NS vehicles approaching fast, 5 EW vehicles stationary
        ns_pos = np.column_stack([
            rng.uniform(-5, 5, 20),
            rng.uniform(50, 180, 20),
        ])
        ns_vel_real = np.column_stack([
            np.zeros(20),
            np.full(20, -12.0),  # approaching at 12 m/s
        ])

        ew_pos = np.column_stack([
            rng.uniform(50, 100, 5),
            rng.uniform(-5, 5, 5),
        ])
        ew_vel_real = np.column_stack([
            np.full(5, -5.0),  # slow approach
            np.zeros(5),
        ])

        positions = np.vstack([ns_pos, ew_pos]).astype(np.float64)
        vel_real = np.vstack([ns_vel_real, ew_vel_real]).astype(np.float64)
        vel_zero = np.zeros_like(vel_real)
        masses = np.ones(25, dtype=np.float64) * 5.0

        result_real = optimize_signal_timing(
            positions, vel_real, masses, intersection_pos
        )
        result_zero = optimize_signal_timing(
            positions, vel_zero, masses, intersection_pos
        )

        # The phi integrals should differ since vehicle trajectories differ
        assert result_real["phi_integral"] != result_zero["phi_integral"], (
            f"Expected different phi_integral with real vs zero velocities, "
            f"got real={result_real['phi_integral']}, zero={result_zero['phi_integral']}"
        )
