"""Tests for VehicleAgent -- Mesa ABM agent for GravTraffic.

Covers:
    1. Agent creation with correct initial state
    2. speed property computes magnitude correctly
    3. speed_kmh conversion is correct
    4. update_from_simulation updates all fields
    5. mass_type classification: slow / fast / neutral
    6. to_dict returns all expected keys with correct types
    7. step() is callable (no-op, no error)
"""

from __future__ import annotations

import math

import mesa
import numpy as np
import pytest

from gravtraffic.agents.vehicle_agent import VehicleAgent

# ---------------------------------------------------------------------------
# Stub model (Mesa 3.x requires a Model instance for Agent.__init__)
# ---------------------------------------------------------------------------


class StubModel(mesa.Model):
    """Minimal Mesa model used exclusively for testing agents."""
    pass


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def stub_model() -> StubModel:
    """Fresh StubModel for each test."""
    return StubModel()


@pytest.fixture
def default_agent(stub_model: StubModel) -> VehicleAgent:
    """Agent at origin heading east at 25 m/s (90 km/h)."""
    return VehicleAgent(
        model=stub_model,
        position=np.array([0.0, 0.0]),
        velocity=np.array([25.0, 0.0]),
    )


# ---------------------------------------------------------------------------
# 1. Agent creation with correct initial state
# ---------------------------------------------------------------------------


class TestCreation:
    """Verify constructor sets all fields correctly."""

    def test_position_dtype(self, default_agent: VehicleAgent) -> None:
        assert default_agent.position.dtype == np.float64

    def test_velocity_dtype(self, default_agent: VehicleAgent) -> None:
        assert default_agent.velocity.dtype == np.float64

    def test_position_values(self, default_agent: VehicleAgent) -> None:
        np.testing.assert_array_equal(default_agent.position, [0.0, 0.0])

    def test_velocity_values(self, default_agent: VehicleAgent) -> None:
        np.testing.assert_array_equal(default_agent.velocity, [25.0, 0.0])

    def test_default_local_density(self, default_agent: VehicleAgent) -> None:
        assert default_agent.local_density == 30.0

    def test_default_v_max(self, default_agent: VehicleAgent) -> None:
        assert default_agent.v_max == 36.0

    def test_default_mass(self, default_agent: VehicleAgent) -> None:
        assert default_agent.mass == 0.0

    def test_default_mass_type(self, default_agent: VehicleAgent) -> None:
        assert default_agent.mass_type == "neutral"

    def test_unique_id_is_assigned(self, default_agent: VehicleAgent) -> None:
        """Mesa 3.x auto-assigns unique_id."""
        assert default_agent.unique_id is not None

    def test_custom_parameters(self, stub_model: StubModel) -> None:
        agent = VehicleAgent(
            model=stub_model,
            position=np.array([100.0, 2.5]),
            velocity=np.array([30.0, -1.0]),
            local_density=45.0,
            v_max=40.0,
        )
        assert agent.local_density == 45.0
        assert agent.v_max == 40.0
        np.testing.assert_array_equal(agent.position, [100.0, 2.5])

    def test_position_coerced_from_list(self, stub_model: StubModel) -> None:
        """Constructor must accept plain lists and coerce to float64."""
        agent = VehicleAgent(
            model=stub_model,
            position=[10, 20],
            velocity=[5, 0],
        )
        assert agent.position.dtype == np.float64
        assert agent.velocity.dtype == np.float64


# ---------------------------------------------------------------------------
# 2. speed property
# ---------------------------------------------------------------------------


class TestSpeed:
    """speed property must return the L2 norm of velocity."""

    def test_axis_aligned(self, default_agent: VehicleAgent) -> None:
        # velocity = [25, 0] -> speed = 25
        assert default_agent.speed == pytest.approx(25.0, abs=1e-12)

    def test_diagonal(self, stub_model: StubModel) -> None:
        agent = VehicleAgent(
            model=stub_model,
            position=[0.0, 0.0],
            velocity=[3.0, 4.0],
        )
        assert agent.speed == pytest.approx(5.0, abs=1e-12)

    def test_zero_velocity(self, stub_model: StubModel) -> None:
        agent = VehicleAgent(
            model=stub_model,
            position=[0.0, 0.0],
            velocity=[0.0, 0.0],
        )
        assert agent.speed == pytest.approx(0.0, abs=1e-15)

    def test_returns_float(self, default_agent: VehicleAgent) -> None:
        assert isinstance(default_agent.speed, float)


# ---------------------------------------------------------------------------
# 3. speed_kmh conversion
# ---------------------------------------------------------------------------


class TestSpeedKmh:
    """speed_kmh == speed * 3.6"""

    def test_conversion(self, default_agent: VehicleAgent) -> None:
        expected = 25.0 * 3.6  # 90 km/h
        assert default_agent.speed_kmh == pytest.approx(expected, abs=1e-12)

    def test_zero(self, stub_model: StubModel) -> None:
        agent = VehicleAgent(
            model=stub_model,
            position=[0.0, 0.0],
            velocity=[0.0, 0.0],
        )
        assert agent.speed_kmh == pytest.approx(0.0, abs=1e-15)

    @pytest.mark.parametrize(
        "vx, vy, expected_kmh",
        [
            (10.0, 0.0, 36.0),
            (0.0, 10.0, 36.0),
            (27.78, 0.0, 27.78 * 3.6),  # ~100 km/h
        ],
    )
    def test_parametrized(
        self, stub_model: StubModel, vx: float, vy: float, expected_kmh: float
    ) -> None:
        agent = VehicleAgent(
            model=stub_model,
            position=[0.0, 0.0],
            velocity=[vx, vy],
        )
        assert agent.speed_kmh == pytest.approx(expected_kmh, rel=1e-10)


# ---------------------------------------------------------------------------
# 4. update_from_simulation updates all fields
# ---------------------------------------------------------------------------


class TestUpdateFromSimulation:

    def test_position_updated(self, default_agent: VehicleAgent) -> None:
        default_agent.update_from_simulation(
            position=np.array([500.0, 1.5]),
            velocity=np.array([30.0, 0.0]),
            mass=0.5,
        )
        np.testing.assert_array_equal(default_agent.position, [500.0, 1.5])

    def test_velocity_updated(self, default_agent: VehicleAgent) -> None:
        default_agent.update_from_simulation(
            position=np.array([0.0, 0.0]),
            velocity=np.array([15.0, -2.0]),
            mass=-0.3,
        )
        np.testing.assert_array_equal(default_agent.velocity, [15.0, -2.0])

    def test_mass_updated(self, default_agent: VehicleAgent) -> None:
        default_agent.update_from_simulation(
            position=np.array([0.0, 0.0]),
            velocity=np.array([0.0, 0.0]),
            mass=1.234,
        )
        assert default_agent.mass == pytest.approx(1.234, abs=1e-15)

    def test_dtype_enforced(self, default_agent: VehicleAgent) -> None:
        """Even if caller passes float32, agent stores float64."""
        default_agent.update_from_simulation(
            position=np.array([1.0, 2.0], dtype=np.float32),
            velocity=np.array([3.0, 4.0], dtype=np.float32),
            mass=0.0,
        )
        assert default_agent.position.dtype == np.float64
        assert default_agent.velocity.dtype == np.float64

    def test_accepts_lists(self, default_agent: VehicleAgent) -> None:
        """update_from_simulation should accept plain lists."""
        default_agent.update_from_simulation(
            position=[100.0, 200.0],
            velocity=[10.0, 20.0],
            mass=0.05,
        )
        assert default_agent.position.dtype == np.float64


# ---------------------------------------------------------------------------
# 5. mass_type classification: slow / fast / neutral
# ---------------------------------------------------------------------------


class TestMassTypeClassification:
    """Thresholds: |mass| > 0.1 determines slow/fast; else neutral."""

    @pytest.mark.parametrize(
        "mass, expected_type",
        [
            (0.5, "slow"),       # positive, above threshold
            (0.11, "slow"),      # just above +threshold
            (0.1, "neutral"),    # exactly at threshold boundary
            (0.05, "neutral"),   # small positive
            (0.0, "neutral"),    # zero
            (-0.05, "neutral"),  # small negative
            (-0.1, "neutral"),   # exactly at -threshold boundary
            (-0.11, "fast"),     # just below -threshold
            (-0.5, "fast"),      # negative, below threshold
            (-2.0, "fast"),      # large negative
            (3.0, "slow"),       # large positive
        ],
    )
    def test_classification(
        self,
        default_agent: VehicleAgent,
        mass: float,
        expected_type: str,
    ) -> None:
        default_agent.update_from_simulation(
            position=[0.0, 0.0],
            velocity=[20.0, 0.0],
            mass=mass,
        )
        assert default_agent.mass_type == expected_type


# ---------------------------------------------------------------------------
# 6. to_dict serialisation
# ---------------------------------------------------------------------------


class TestToDict:

    def test_expected_keys(self, default_agent: VehicleAgent) -> None:
        d = default_agent.to_dict()
        expected_keys = {"id", "x", "y", "vx", "vy", "mass", "speed_kmh", "type"}
        assert set(d.keys()) == expected_keys

    def test_values_match_state(self, stub_model: StubModel) -> None:
        agent = VehicleAgent(
            model=stub_model,
            position=[100.0, 2.5],
            velocity=[30.0, -1.0],
        )
        agent.update_from_simulation(
            position=[200.0, 3.0],
            velocity=[28.0, 0.5],
            mass=0.75,
        )
        d = agent.to_dict()
        assert d["x"] == pytest.approx(200.0)
        assert d["y"] == pytest.approx(3.0)
        assert d["vx"] == pytest.approx(28.0)
        assert d["vy"] == pytest.approx(0.5)
        assert d["mass"] == pytest.approx(0.75)
        assert d["type"] == "slow"
        assert d["speed_kmh"] == pytest.approx(
            math.sqrt(28.0**2 + 0.5**2) * 3.6, rel=1e-10
        )

    def test_types_are_json_serialisable(self, default_agent: VehicleAgent) -> None:
        """All values must be plain Python types, not numpy."""
        d = default_agent.to_dict()
        assert isinstance(d["x"], float)
        assert isinstance(d["y"], float)
        assert isinstance(d["vx"], float)
        assert isinstance(d["vy"], float)
        assert isinstance(d["mass"], (int, float))
        assert isinstance(d["speed_kmh"], float)
        assert isinstance(d["type"], str)


# ---------------------------------------------------------------------------
# 7. step() is callable (no-op)
# ---------------------------------------------------------------------------


class TestStep:

    def test_step_runs_without_error(self, default_agent: VehicleAgent) -> None:
        """step() should be a no-op and not raise."""
        default_agent.step()

    def test_step_does_not_change_state(self, default_agent: VehicleAgent) -> None:
        pos_before = default_agent.position.copy()
        vel_before = default_agent.velocity.copy()
        mass_before = default_agent.mass
        default_agent.step()
        np.testing.assert_array_equal(default_agent.position, pos_before)
        np.testing.assert_array_equal(default_agent.velocity, vel_before)
        assert default_agent.mass == mass_before


# ---------------------------------------------------------------------------
# 8. __repr__ smoke test
# ---------------------------------------------------------------------------


class TestRepr:

    def test_repr_contains_id(self, default_agent: VehicleAgent) -> None:
        r = repr(default_agent)
        assert "VehicleAgent" in r
        assert str(default_agent.unique_id) in r
