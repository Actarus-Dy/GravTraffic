"""Tests for gravtraffic.core.green_wave -- Green Wave Coordinator.

Validates offset computation, offset application to intersection agents,
bandwidth-optimal speed search, and edge cases.

All computations use float64.  Tolerances are set for exact analytical
solutions where applicable.

Author: Agent #01 Python Scientific Developer
Date: 2026-03-22
"""

from __future__ import annotations

import mesa
import numpy as np
import pytest

from gravtraffic.agents.intersection_agent import IntersectionAgent
from gravtraffic.core.green_wave import GreenWaveCoordinator

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_model() -> mesa.Model:
    """Create a minimal Mesa model for agent construction."""
    return mesa.Model()


def _make_intersection(
    model: mesa.Model,
    x: float,
    y: float,
    node_id: int,
    n_phases: int = 2,
    cycle_s: float = 120.0,
) -> IntersectionAgent:
    """Create an IntersectionAgent at the given position."""
    return IntersectionAgent(
        model=model,
        position=np.array([x, y], dtype=np.float64),
        node_id=node_id,
        n_phases=n_phases,
        cycle_s=cycle_s,
    )


# ------------------------------------------------------------------
# Test: compute_offsets -- 5 intersections spaced 200 m apart
# ------------------------------------------------------------------


class TestComputeOffsets:
    """Tests for GreenWaveCoordinator.compute_offsets."""

    def test_uniform_spacing_200m(self):
        """5 intersections at 200 m spacing, v = 50 km/h = 13.888... m/s.

        Expected offsets: 0, 200/v, 400/v, 600/v, 800/v
        = 0, 14.4, 28.8, 43.2, 57.6 seconds.

        Derivation: v = 50 km/h = 50000/3600 m/s.
        200 / (50000/3600) = 200 * 3600 / 50000 = 720000 / 50000 = 14.4 s.
        """
        positions = np.array(
            [
                [0.0, 0.0],
                [200.0, 0.0],
                [400.0, 0.0],
                [600.0, 0.0],
                [800.0, 0.0],
            ],
            dtype=np.float64,
        )

        gw = GreenWaveCoordinator(wave_speed=50.0 / 3.6)
        offsets = gw.compute_offsets(positions)

        expected = np.array([0.0, 14.4, 28.8, 43.2, 57.6], dtype=np.float64)

        assert offsets.dtype == np.float64
        assert offsets.shape == (5,)
        np.testing.assert_allclose(offsets, expected, rtol=1e-12)

    def test_custom_corridor_direction_diagonal(self):
        """Intersections along the diagonal (1, 1); only the projection matters.

        Position (d, d) projected onto (1/sqrt2, 1/sqrt2) gives
        d * 2 / sqrt(2) = d * sqrt(2).

        For d = 0, 100, 200:
            projections = 0, 100*sqrt(2), 200*sqrt(2)
            offsets = projections / v_wave
        """
        positions = np.array(
            [
                [0.0, 0.0],
                [100.0, 100.0],
                [200.0, 200.0],
            ],
            dtype=np.float64,
        )

        direction = np.array([1.0, 1.0], dtype=np.float64)
        v_wave = 10.0  # m/s for simple arithmetic
        gw = GreenWaveCoordinator(wave_speed=v_wave)
        offsets = gw.compute_offsets(positions, corridor_direction=direction)

        sqrt2 = np.sqrt(2.0)
        expected = np.array(
            [
                0.0,
                100.0 * sqrt2 / v_wave,
                200.0 * sqrt2 / v_wave,
            ],
            dtype=np.float64,
        )

        assert offsets.dtype == np.float64
        np.testing.assert_allclose(offsets, expected, rtol=1e-12)

    def test_single_intersection_offset_zero(self):
        """A single intersection must have offset exactly 0."""
        positions = np.array([[500.0, 300.0]], dtype=np.float64)
        gw = GreenWaveCoordinator(wave_speed=15.0)
        offsets = gw.compute_offsets(positions)

        assert offsets.shape == (1,)
        assert offsets[0] == 0.0


# ------------------------------------------------------------------
# Test: apply_offsets
# ------------------------------------------------------------------


class TestApplyOffsets:
    """Tests for GreenWaveCoordinator.apply_offsets."""

    def test_offsets_change_phase_and_time(self):
        """After applying offsets, intersections should have different
        current_phase and/or time_in_phase values."""
        model = _make_model()

        # 3 intersections, each with cycle = 120 s (60 + 60, 2 phases)
        agents = [_make_intersection(model, i * 200.0, 0.0, node_id=i) for i in range(3)]

        # Offsets: 0, 30, 90 seconds
        offsets = np.array([0.0, 30.0, 90.0], dtype=np.float64)

        gw = GreenWaveCoordinator()
        gw.apply_offsets(agents, offsets)

        # Agent 0: offset 0 -> phase 0, time 0
        assert agents[0].current_phase == 0
        assert agents[0].time_in_phase == pytest.approx(0.0)

        # Agent 1: offset 30 -> phase 0 (green_times[0]=60, 30<60), time 30
        assert agents[1].current_phase == 0
        assert agents[1].time_in_phase == pytest.approx(30.0)

        # Agent 2: offset 90 -> 90 mod 120 = 90;  phase 0 covers [0,60),
        #   phase 1 covers [60,120) -> phase 1, time = 90-60 = 30
        assert agents[2].current_phase == 1
        assert agents[2].time_in_phase == pytest.approx(30.0)

    def test_zero_offset_starts_at_phase0_time0(self):
        """An intersection with offset = 0 must start at phase 0, time_in_phase 0."""
        model = _make_model()
        agent = _make_intersection(model, 0.0, 0.0, node_id=0)

        # Manually set agent to a non-zero state first
        agent.current_phase = 1
        agent.time_in_phase = 42.0

        offsets = np.array([0.0], dtype=np.float64)
        gw = GreenWaveCoordinator()
        gw.apply_offsets([agent], offsets)

        assert agent.current_phase == 0
        assert agent.time_in_phase == pytest.approx(0.0)


# ------------------------------------------------------------------
# Test: optimize_wave_speed
# ------------------------------------------------------------------


class TestOptimizeWaveSpeed:
    """Tests for GreenWaveCoordinator.optimize_wave_speed."""

    def test_result_in_valid_range(self):
        """Optimal speed must be within the specified search range."""
        positions = np.array(
            [
                [0.0, 0.0],
                [200.0, 0.0],
                [400.0, 0.0],
                [600.0, 0.0],
            ],
            dtype=np.float64,
        )

        green_times = [60.0, 60.0]
        speed_range = (8.0, 20.0)

        gw = GreenWaveCoordinator()
        optimal = gw.optimize_wave_speed(
            positions, green_times, speed_range=speed_range, n_candidates=50
        )

        assert speed_range[0] <= optimal <= speed_range[1]

    def test_result_is_positive_float(self):
        """Optimal speed must be a positive float."""
        positions = np.array(
            [
                [0.0, 0.0],
                [300.0, 0.0],
            ],
            dtype=np.float64,
        )

        green_times = [45.0, 35.0]

        gw = GreenWaveCoordinator()
        optimal = gw.optimize_wave_speed(positions, green_times)

        assert isinstance(optimal, float)
        assert optimal > 0.0


# ------------------------------------------------------------------
# Test: edge cases and validation
# ------------------------------------------------------------------


class TestEdgeCases:
    """Validation and edge-case tests."""

    def test_wave_speed_must_be_positive(self):
        """Constructor rejects non-positive wave speed."""
        with pytest.raises(ValueError, match="positive"):
            GreenWaveCoordinator(wave_speed=0.0)
        with pytest.raises(ValueError, match="positive"):
            GreenWaveCoordinator(wave_speed=-5.0)

    def test_offsets_dtype_is_float64(self):
        """All returned offsets must be float64."""
        positions = np.array(
            [
                [0.0, 0.0],
                [100.0, 0.0],
            ],
            dtype=np.float64,
        )
        gw = GreenWaveCoordinator()
        offsets = gw.compute_offsets(positions)
        assert offsets.dtype == np.float64

    def test_apply_offsets_length_mismatch_raises(self):
        """apply_offsets raises ValueError on length mismatch."""
        model = _make_model()
        agents = [_make_intersection(model, 0.0, 0.0, node_id=0)]
        offsets = np.array([0.0, 10.0], dtype=np.float64)

        gw = GreenWaveCoordinator()
        with pytest.raises(ValueError, match="Length mismatch"):
            gw.apply_offsets(agents, offsets)
