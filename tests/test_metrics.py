"""Tests for gravtraffic.core.metrics -- traffic KPI calculator.

Every test uses float64 arrays and asserts against known analytical values.
"""

from __future__ import annotations

import numpy as np
import pytest

from gravtraffic.core.metrics import (
    compute_congestion_index,
    compute_delay,
    compute_level_of_service,
    compute_mean_speed,
    compute_snapshot_kpis,
    compute_stops,
    compute_throughput,
    compute_travel_time,
)


# -----------------------------------------------------------------------
# 1. compute_throughput
# -----------------------------------------------------------------------
class TestComputeThroughput:
    def test_known_crossings(self):
        """3 vehicles cross gate, 2 do not -> correct count."""
        gate_x = 100.0
        dt = 1.0  # 1 second timestep
        prev = np.array(
            [
                [90.0, 0.0],  # crosses left-to-right
                [95.0, 5.0],  # crosses left-to-right
                [110.0, 0.0],  # crosses right-to-left
                [50.0, 0.0],  # stays left
                [150.0, 0.0],  # stays right
            ],
            dtype=np.float64,
        )
        curr = np.array(
            [
                [105.0, 0.0],
                [100.0, 5.0],  # lands exactly on gate -> counts
                [90.0, 0.0],
                [60.0, 0.0],
                [160.0, 0.0],
            ],
            dtype=np.float64,
        )
        result = compute_throughput(curr, prev, gate_x, dt)
        expected = 3.0 / dt * 3600.0  # 10800 veh/hr
        assert result == pytest.approx(expected, rel=1e-12)

    def test_no_crossings(self):
        prev = np.array([[50.0, 0.0], [60.0, 0.0]], dtype=np.float64)
        curr = np.array([[55.0, 0.0], [65.0, 0.0]], dtype=np.float64)
        assert compute_throughput(curr, prev, 100.0, 1.0) == 0.0


# -----------------------------------------------------------------------
# 2. compute_mean_speed
# -----------------------------------------------------------------------
class TestComputeMeanSpeed:
    def test_known_velocities(self):
        """Vehicles with speeds 3-4-5 triangle -> known speeds."""
        vels = np.array(
            [[3.0, 4.0], [5.0, 0.0], [0.0, 10.0]],
            dtype=np.float64,
        )
        # speeds: 5.0, 5.0, 10.0 -> mean = 20/3
        expected = (5.0 + 5.0 + 10.0) / 3.0
        assert compute_mean_speed(vels) == pytest.approx(expected, rel=1e-12)

    def test_uniform_speed(self):
        vels = np.array([[10.0, 0.0]] * 100, dtype=np.float64)
        assert compute_mean_speed(vels) == pytest.approx(10.0, rel=1e-12)


# -----------------------------------------------------------------------
# 3-4. compute_delay
# -----------------------------------------------------------------------
class TestComputeDelay:
    def test_at_free_flow_zero_delay(self):
        """All vehicles at v_free -> delay is 0."""
        v_free = 33.33
        vels = np.array([[v_free, 0.0]] * 50, dtype=np.float64)
        delay = compute_delay(vels, v_free)
        assert delay == pytest.approx(0.0, abs=1e-10)

    def test_half_free_flow_positive_delay(self):
        """All vehicles at half v_free -> positive delay."""
        v_free = 33.33
        v_half = v_free / 2.0
        vels = np.array([[v_half, 0.0]] * 20, dtype=np.float64)
        delay = compute_delay(vels, v_free)
        expected = 1000.0 / v_half - 1000.0 / v_free
        assert delay == pytest.approx(expected, rel=1e-12)
        assert delay > 0.0


# -----------------------------------------------------------------------
# 5. compute_stops
# -----------------------------------------------------------------------
class TestComputeStops:
    def test_known_count(self):
        """5 vehicles below threshold, 10 above -> returns 5."""
        threshold = 2.0
        slow = np.array([[0.5, 0.0]] * 5, dtype=np.float64)
        fast = np.array([[10.0, 0.0]] * 10, dtype=np.float64)
        vels = np.vstack([slow, fast])
        assert compute_stops(vels, threshold) == 5

    def test_none_stopped(self):
        vels = np.array([[5.0, 5.0]] * 10, dtype=np.float64)
        assert compute_stops(vels, 2.0) == 0

    def test_all_stopped(self):
        vels = np.array([[0.1, 0.0]] * 8, dtype=np.float64)
        assert compute_stops(vels, 2.0) == 8


# -----------------------------------------------------------------------
# 6. compute_congestion_index
# -----------------------------------------------------------------------
class TestComputeCongestionIndex:
    def test_known_distribution(self):
        """3 congested (mass > 0.1), 7 not -> index = 0.3."""
        masses = np.array(
            [1.0, 2.0, 0.5, 0.0, -1.0, 0.05, -0.5, 0.0, 0.0, -2.0],
            dtype=np.float64,
        )
        # masses > 0.1: 1.0, 2.0, 0.5 -> 3 out of 10
        assert compute_congestion_index(masses) == pytest.approx(0.3, rel=1e-12)

    def test_all_congested(self):
        masses = np.array([1.0, 2.0, 5.0], dtype=np.float64)
        assert compute_congestion_index(masses) == pytest.approx(1.0, rel=1e-12)

    def test_none_congested(self):
        masses = np.array([-1.0, -2.0, 0.0, 0.05], dtype=np.float64)
        assert compute_congestion_index(masses) == pytest.approx(0.0, rel=1e-12)


# -----------------------------------------------------------------------
# 7. compute_level_of_service
# -----------------------------------------------------------------------
class TestComputeLevelOfService:
    @pytest.mark.parametrize(
        "speed_fraction, expected_los",
        [
            (0.95, "A"),
            (0.91, "A"),
            (0.80, "B"),
            (0.71, "B"),
            (0.60, "C"),
            (0.51, "C"),
            (0.45, "D"),
            (0.41, "D"),
            (0.30, "E"),
            (0.26, "E"),
            (0.20, "F"),
            (0.10, "F"),
            (0.00, "F"),
        ],
    )
    def test_los_levels(self, speed_fraction: float, expected_los: str):
        v_free = 30.0
        speed = v_free * speed_fraction
        # Single vehicle moving in x direction at the target speed
        vels = np.array([[speed, 0.0]], dtype=np.float64)
        assert compute_level_of_service(vels, v_free) == expected_los

    def test_boundary_090(self):
        """Exactly 0.90 ratio -> B (not A, since condition is > 0.90)."""
        v_free = 100.0
        vels = np.array([[90.0, 0.0]], dtype=np.float64)
        assert compute_level_of_service(vels, v_free) == "B"


# -----------------------------------------------------------------------
# 8-9. compute_travel_time
# -----------------------------------------------------------------------
class TestComputeTravelTime:
    def test_known_trajectory(self):
        """Vehicle moves 10 m/step over 10 steps to cover 100 m corridor."""
        dt = 0.5
        start_x = 20.0
        end_x = 120.0
        # 1 vehicle, linearly increasing x from 0 to 150
        n_steps = 16
        history = []
        for t in range(n_steps):
            x = t * 10.0
            history.append(np.array([[x, 0.0]], dtype=np.float64))

        tt = compute_travel_time(history, start_x, end_x, dt)
        assert tt is not None

        # Entry at step 2 (x=20), exit at step 12 (x=120) -> 10 steps * 0.5 s
        assert tt == pytest.approx(5.0, rel=1e-12)

    def test_multiple_vehicles(self):
        """Two vehicles with different entry times."""
        dt = 1.0
        start_x = 0.0
        end_x = 50.0
        history = []
        for t in range(20):
            # Vehicle 0: starts at x=-10, moves +5/step
            # Vehicle 1: starts at x=-30, moves +5/step
            x0 = -10.0 + t * 5.0
            x1 = -30.0 + t * 5.0
            history.append(np.array([[x0, 0.0], [x1, 0.0]], dtype=np.float64))

        tt = compute_travel_time(history, start_x, end_x, dt)
        assert tt is not None
        # Both vehicles take 10 steps to cross 50 m at 5 m/step -> 10 s
        assert tt == pytest.approx(10.0, rel=1e-12)

    def test_no_completion_returns_none(self):
        """No vehicle reaches end_x -> returns None."""
        dt = 1.0
        history = [
            np.array([[5.0, 0.0]], dtype=np.float64),
            np.array([[10.0, 0.0]], dtype=np.float64),
            np.array([[15.0, 0.0]], dtype=np.float64),
        ]
        result = compute_travel_time(history, 0.0, 1000.0, dt)
        assert result is None

    def test_empty_history(self):
        assert compute_travel_time([], 0.0, 100.0, 1.0) is None


# -----------------------------------------------------------------------
# 10. compute_snapshot_kpis
# -----------------------------------------------------------------------
class TestComputeSnapshotKPIs:
    def test_returns_all_keys(self):
        pos = np.array([[0.0, 0.0], [10.0, 10.0]], dtype=np.float64)
        vel = np.array([[30.0, 0.0], [25.0, 0.0]], dtype=np.float64)
        mass = np.array([0.5, -0.3], dtype=np.float64)

        kpis = compute_snapshot_kpis(pos, vel, mass)

        expected_keys = {
            "mean_speed_ms",
            "mean_speed_kmh",
            "n_stops",
            "congestion_index",
            "delay_s_per_km",
            "level_of_service",
            "n_vehicles",
        }
        assert set(kpis.keys()) == expected_keys

    def test_values_consistent(self):
        vel = np.array([[33.33, 0.0]] * 10, dtype=np.float64)
        pos = np.zeros((10, 2), dtype=np.float64)
        mass = np.zeros(10, dtype=np.float64)

        kpis = compute_snapshot_kpis(pos, vel, mass, v_free=33.33)

        assert kpis["mean_speed_ms"] == pytest.approx(33.33, rel=1e-10)
        assert kpis["mean_speed_kmh"] == pytest.approx(33.33 * 3.6, rel=1e-10)
        assert kpis["n_stops"] == 0
        assert kpis["congestion_index"] == pytest.approx(0.0)
        assert kpis["delay_s_per_km"] == pytest.approx(0.0, abs=1e-10)
        assert kpis["level_of_service"] == "A"
        assert kpis["n_vehicles"] == 10


# -----------------------------------------------------------------------
# 11. Empty array handling
# -----------------------------------------------------------------------
class TestEmptyArrays:
    def test_throughput_empty(self):
        empty2d = np.empty((0, 2), dtype=np.float64)
        assert compute_throughput(empty2d, empty2d, 100.0, 1.0) == 0.0

    def test_mean_speed_empty(self):
        empty2d = np.empty((0, 2), dtype=np.float64)
        assert compute_mean_speed(empty2d) == 0.0

    def test_delay_empty(self):
        empty2d = np.empty((0, 2), dtype=np.float64)
        assert compute_delay(empty2d) == 0.0

    def test_stops_empty(self):
        empty2d = np.empty((0, 2), dtype=np.float64)
        assert compute_stops(empty2d) == 0

    def test_congestion_index_empty(self):
        empty1d = np.empty((0,), dtype=np.float64)
        assert compute_congestion_index(empty1d) == 0.0

    def test_level_of_service_empty(self):
        empty2d = np.empty((0, 2), dtype=np.float64)
        # mean speed = 0 -> ratio = 0 -> LOS F
        assert compute_level_of_service(empty2d) == "F"

    def test_travel_time_empty_vehicles(self):
        history = [np.empty((0, 2), dtype=np.float64)]
        assert compute_travel_time(history, 0.0, 100.0, 1.0) is None

    def test_snapshot_kpis_empty(self):
        empty2d = np.empty((0, 2), dtype=np.float64)
        empty1d = np.empty((0,), dtype=np.float64)
        kpis = compute_snapshot_kpis(empty2d, empty2d, empty1d)
        assert kpis["n_vehicles"] == 0
        assert kpis["mean_speed_ms"] == 0.0
        assert kpis["n_stops"] == 0
