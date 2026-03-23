"""Tests for IntersectionAgent -- GravTraffic C-01.

Covers:
    1. Agent creation with correct defaults
    2. Phase cycling via repeated step() calls
    3. is_green property reflects current phase
    4. get_red_light_masses returns masses for non-green phases
    5. Red-light mass values are positive (congestion wells)
    6. try_optimize updates green_times when interval elapses
    7. try_optimize is a no-op before interval elapses
    8. to_dict returns all expected keys

Mesa 3.x: Agent constructor takes (model) only, no unique_id.

Author: Agent #01 Python Scientific Developer
Date: 2026-03-22
"""

from __future__ import annotations

import mesa
import numpy as np
import pytest

from gravtraffic.agents.intersection_agent import IntersectionAgent

# ======================================================================
# Stub model
# ======================================================================


class StubModel(mesa.Model):
    """Minimal Mesa model for unit-testing agents in isolation."""

    def __init__(self, seed: int = 42):
        super().__init__(seed=seed)


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def model() -> StubModel:
    return StubModel(seed=42)


@pytest.fixture
def agent(model: StubModel) -> IntersectionAgent:
    """Default 2-phase intersection at the origin."""
    return IntersectionAgent(
        model=model,
        position=np.array([100.0, 200.0]),
        node_id=7,
        n_phases=2,
        cycle_s=120.0,
        red_light_mass=50.0,
        optimize_interval_steps=300,
    )


# ======================================================================
# 1. Agent creation with correct defaults
# ======================================================================


class TestCreation:
    def test_position_dtype_and_value(self, agent: IntersectionAgent):
        assert agent.position.dtype == np.float64
        np.testing.assert_array_equal(agent.position, [100.0, 200.0])

    def test_node_id(self, agent: IntersectionAgent):
        assert agent.node_id == 7

    def test_n_phases(self, agent: IntersectionAgent):
        assert agent.n_phases == 2

    def test_cycle_s(self, agent: IntersectionAgent):
        assert agent.cycle_s == 120.0

    def test_red_light_mass_value(self, agent: IntersectionAgent):
        assert agent.red_light_mass == 50.0

    def test_initial_phase_zero(self, agent: IntersectionAgent):
        assert agent.current_phase == 0

    def test_initial_green_times_equal_split(self, agent: IntersectionAgent):
        expected = [60.0, 60.0]
        assert agent.green_times == pytest.approx(expected)

    def test_initial_time_in_phase_zero(self, agent: IntersectionAgent):
        assert agent.time_in_phase == 0.0

    def test_initial_steps_since_optimize_zero(self, agent: IntersectionAgent):
        assert agent.steps_since_optimize == 0

    def test_position_shape_validation(self, model: StubModel):
        with pytest.raises(ValueError, match="shape"):
            IntersectionAgent(
                model=model,
                position=np.array([1.0, 2.0, 3.0]),
                node_id=0,
            )

    def test_negative_red_light_mass_rejected(self, model: StubModel):
        with pytest.raises(ValueError, match="positive"):
            IntersectionAgent(
                model=model,
                position=np.array([0.0, 0.0]),
                node_id=0,
                red_light_mass=-10.0,
            )


# ======================================================================
# 2. Phase cycling
# ======================================================================


class TestPhaseCycling:
    def test_phase_advances_after_green_time(self, agent: IntersectionAgent):
        """Step enough times to exhaust phase 0 green (60 s at dt=0.1)."""
        dt = 0.1
        steps_for_phase = int(agent.green_times[0] / dt)
        for _ in range(steps_for_phase):
            agent.step(dt=dt)
        assert agent.current_phase == 1
        assert agent.time_in_phase == pytest.approx(0.0, abs=1e-12)

    def test_full_cycle_returns_to_phase_zero(self, agent: IntersectionAgent):
        """Two full phases should bring us back to phase 0."""
        dt = 0.1
        total_steps = int(agent.cycle_s / dt)
        for _ in range(total_steps):
            agent.step(dt=dt)
        assert agent.current_phase == 0

    def test_time_in_phase_accumulates(self, agent: IntersectionAgent):
        agent.step(dt=0.5)
        agent.step(dt=0.5)
        assert agent.time_in_phase == pytest.approx(1.0, rel=1e-12)

    def test_no_phase_change_before_green_expires(self, agent: IntersectionAgent):
        dt = 0.1
        for _ in range(10):  # 1 second, well below 60 s
            agent.step(dt=dt)
        assert agent.current_phase == 0


# ======================================================================
# 3. is_green reflects current phase
# ======================================================================


class TestIsGreen:
    def test_initial_is_green(self, agent: IntersectionAgent):
        assert agent.is_green == [True, False]

    def test_after_phase_transition(self, agent: IntersectionAgent):
        dt = 0.1
        steps = int(agent.green_times[0] / dt)
        for _ in range(steps):
            agent.step(dt=dt)
        assert agent.is_green == [False, True]

    def test_three_phase_agent(self, model: StubModel):
        a = IntersectionAgent(
            model=model,
            position=np.array([0.0, 0.0]),
            node_id=1,
            n_phases=3,
            cycle_s=90.0,
        )
        assert a.is_green == [True, False, False]
        # Advance past phase 0 (30 s)
        for _ in range(300):
            a.step(dt=0.1)
        assert a.is_green == [False, True, False]


# ======================================================================
# 4. get_red_light_masses returns masses for non-green phases
# ======================================================================


class TestRedLightMasses:
    def test_initial_returns_masses_for_phase_1(self, agent: IntersectionAgent):
        """Phase 0 is green, so only phase 1 (EW) should produce masses."""
        masses = agent.get_red_light_masses()
        # Phase 1 is red -> 2 masses (+ and - offset)
        assert len(masses) == 2

    def test_after_transition_returns_masses_for_phase_0(self, agent: IntersectionAgent):
        dt = 0.1
        steps = int(agent.green_times[0] / dt)
        for _ in range(steps):
            agent.step(dt=dt)
        masses = agent.get_red_light_masses()
        # Phase 0 is now red -> 2 masses
        assert len(masses) == 2

    def test_mass_positions_are_float64(self, agent: IntersectionAgent):
        masses = agent.get_red_light_masses()
        for pos, _ in masses:
            assert pos.dtype == np.float64

    def test_mass_offsets_are_symmetric(self, agent: IntersectionAgent):
        masses = agent.get_red_light_masses()
        pos_a, _ = masses[0]
        pos_b, _ = masses[1]
        # The two masses should be symmetric around the intersection center
        midpoint = (pos_a + pos_b) / 2.0
        np.testing.assert_allclose(midpoint, agent.position, atol=1e-12)


# ======================================================================
# 5. Red-light mass is positive (congestion well)
# ======================================================================


class TestRedLightMassPositive:
    def test_all_masses_positive(self, agent: IntersectionAgent):
        masses = agent.get_red_light_masses()
        for _, mass_val in masses:
            assert mass_val > 0.0, f"Red-light mass must be positive, got {mass_val}"

    def test_mass_equals_configured_value(self, agent: IntersectionAgent):
        masses = agent.get_red_light_masses()
        for _, mass_val in masses:
            assert mass_val == pytest.approx(50.0)


# ======================================================================
# 6. try_optimize updates green_times when interval elapses
# ======================================================================


class TestTryOptimizeUpdates:
    def test_updates_after_interval(self, agent: IntersectionAgent):
        """After enough steps, try_optimize should call the optimizer."""
        # Force steps_since_optimize past the threshold
        agent.steps_since_optimize = agent.optimize_interval_steps

        # Create vehicle scenario: heavy NS traffic to skew timings
        rng = np.random.default_rng(99)
        n = 50
        # Vehicles aligned along y-axis (NS direction)
        positions = np.column_stack(
            [
                np.full(n, agent.position[0], dtype=np.float64),
                rng.uniform(agent.position[1] - 150, agent.position[1] + 150, n),
            ]
        ).astype(np.float64)
        masses = np.full(n, 5.0, dtype=np.float64)  # positive = slow/congested

        old_green = list(agent.green_times)
        agent.try_optimize(positions, masses, G_s=2.0)

        # green_times should have changed from the equal-split default
        assert agent.green_times != old_green
        # Counter should have been reset
        assert agent.steps_since_optimize == 0

    def test_cycle_s_updated(self, agent: IntersectionAgent):
        """Cycle length should be updated to match optimizer output."""
        agent.steps_since_optimize = agent.optimize_interval_steps

        positions = np.array([[100.0, 250.0]], dtype=np.float64)
        masses = np.array([5.0], dtype=np.float64)

        agent.try_optimize(positions, masses, G_s=2.0)
        # optimize_traffic_light returns horizon_s=120 by default
        assert agent.cycle_s == pytest.approx(120.0)

    def test_green_times_are_float(self, agent: IntersectionAgent):
        agent.steps_since_optimize = agent.optimize_interval_steps
        positions = np.array([[100.0, 250.0]], dtype=np.float64)
        masses = np.array([5.0], dtype=np.float64)
        agent.try_optimize(positions, masses)
        for gt in agent.green_times:
            assert isinstance(gt, float)


# ======================================================================
# 7. try_optimize does nothing before interval elapses
# ======================================================================


class TestTryOptimizeNoop:
    def test_noop_before_interval(self, agent: IntersectionAgent):
        """Green times must not change if interval has not elapsed."""
        original_green = list(agent.green_times)
        original_counter = agent.steps_since_optimize

        positions = np.array([[100.0, 250.0]], dtype=np.float64)
        masses = np.array([5.0], dtype=np.float64)

        agent.try_optimize(positions, masses, G_s=2.0)

        assert agent.green_times == original_green
        assert agent.steps_since_optimize == original_counter

    def test_noop_at_partial_interval(self, agent: IntersectionAgent):
        """Even at interval - 1, optimization should NOT trigger."""
        agent.steps_since_optimize = agent.optimize_interval_steps - 1
        original_green = list(agent.green_times)

        positions = np.array([[100.0, 250.0]], dtype=np.float64)
        masses = np.array([5.0], dtype=np.float64)

        agent.try_optimize(positions, masses, G_s=2.0)

        assert agent.green_times == original_green


# ======================================================================
# 8. to_dict returns all expected keys
# ======================================================================


class TestToDict:
    EXPECTED_KEYS = {"node_id", "x", "y", "current_phase", "green_times", "is_green"}

    def test_keys_present(self, agent: IntersectionAgent):
        d = agent.to_dict()
        assert set(d.keys()) == self.EXPECTED_KEYS

    def test_node_id_value(self, agent: IntersectionAgent):
        assert agent.to_dict()["node_id"] == 7

    def test_coordinates(self, agent: IntersectionAgent):
        d = agent.to_dict()
        assert d["x"] == pytest.approx(100.0)
        assert d["y"] == pytest.approx(200.0)

    def test_current_phase_type(self, agent: IntersectionAgent):
        assert isinstance(agent.to_dict()["current_phase"], int)

    def test_green_times_is_list(self, agent: IntersectionAgent):
        assert isinstance(agent.to_dict()["green_times"], list)

    def test_is_green_is_list_of_bool(self, agent: IntersectionAgent):
        ig = agent.to_dict()["is_green"]
        assert isinstance(ig, list)
        assert all(isinstance(v, bool) for v in ig)

    def test_values_are_json_serializable(self, agent: IntersectionAgent):
        """All values must be JSON-safe (no numpy scalars)."""
        import json

        d = agent.to_dict()
        # Should not raise
        json.dumps(d)
