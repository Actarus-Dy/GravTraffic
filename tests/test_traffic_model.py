"""Tests for TrafficModel -- the top-level Mesa Model for C-01 GravTraffic.

All tests use a small synthetic 3x3 grid network with 20-50 vehicles to keep
execution fast while exercising the full orchestration pipeline.

Author: Agent #01 Python Scientific Developer
Date: 2026-03-22
"""

from __future__ import annotations

import numpy as np
import pytest

from gravtraffic.agents.intersection_agent import IntersectionAgent
from gravtraffic.agents.traffic_model import TrafficModel
from gravtraffic.agents.vehicle_agent import VehicleAgent
from gravtraffic.network.road_network import RoadNetwork

# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def grid_network() -> RoadNetwork:
    """A small 3x3 grid network (9 nodes, 12 edges)."""
    return RoadNetwork.from_grid(3, 3)


@pytest.fixture
def model(grid_network: RoadNetwork) -> TrafficModel:
    """TrafficModel with 30 vehicles on a 3x3 grid."""
    return TrafficModel(
        network=grid_network,
        n_vehicles=30,
        G_s=2.0,
        beta=0.5,
        softening=10.0,
        theta=0.5,
        dt=0.1,
        v_max=36.0,
        signal_intersections=True,
        seed=42,
    )


@pytest.fixture
def model_no_signals(grid_network: RoadNetwork) -> TrafficModel:
    """TrafficModel with signals disabled."""
    return TrafficModel(
        network=grid_network,
        n_vehicles=20,
        signal_intersections=False,
        seed=99,
    )


# ======================================================================
# 1. Construction
# ======================================================================


class TestConstruction:
    """TrafficModel creates correctly with from_grid network and n vehicles."""

    def test_vehicle_count(self, model: TrafficModel) -> None:
        assert len(model.vehicle_agents) == 30

    def test_vehicle_agent_type(self, model: TrafficModel) -> None:
        for a in model.vehicle_agents:
            assert isinstance(a, VehicleAgent)

    def test_intersection_agents_created(self, model: TrafficModel) -> None:
        """3x3 grid: the single interior node (1,1) = node 4 has degree 4.
        Edge nodes with degree >= 3 are also signalized."""
        assert len(model.intersection_agents) > 0
        for a in model.intersection_agents:
            assert isinstance(a, IntersectionAgent)

    def test_no_signals_mode(self, model_no_signals: TrafficModel) -> None:
        assert len(model_no_signals.intersection_agents) == 0
        assert len(model_no_signals.vehicle_agents) == 20

    def test_simulation_initialized(self, model: TrafficModel) -> None:
        assert model.simulation.positions.shape == (30, 2)
        assert model.simulation.velocities.shape == (30, 2)
        assert model.simulation.masses.shape == (30,)

    def test_step_count_starts_zero(self, model: TrafficModel) -> None:
        assert model.step_count == 0

    def test_all_float64(self, model: TrafficModel) -> None:
        assert model.simulation.positions.dtype == np.float64
        assert model.simulation.velocities.dtype == np.float64
        assert model.simulation.masses.dtype == np.float64


# ======================================================================
# 2. step() returns correct dict
# ======================================================================


class TestStepReturn:
    """step() populates last_step_result with positions/velocities/masses.

    Note: Mesa >= 3.5 wraps Model.step() so it returns None.
    The physics result is available via model.last_step_result.
    """

    def test_last_step_result_is_dict(self, model: TrafficModel) -> None:
        model.step()
        result = model.last_step_result
        assert isinstance(result, dict)

    def test_required_keys(self, model: TrafficModel) -> None:
        model.step()
        result = model.last_step_result
        for key in ("positions", "velocities", "masses", "mean_speed", "dt_used", "step_count"):
            assert key in result, f"Missing key: {key}"

    def test_shapes(self, model: TrafficModel) -> None:
        model.step()
        result = model.last_step_result
        assert result["positions"].shape == (30, 2)
        assert result["velocities"].shape == (30, 2)
        assert result["masses"].shape == (30,)

    def test_step_count_increments(self, model: TrafficModel) -> None:
        model.step()
        assert model.step_count == 1
        model.step()
        result = model.last_step_result
        assert model.step_count == 2
        assert result["step_count"] == 2

    def test_last_step_result_none_before_step(self, model: TrafficModel) -> None:
        assert model.last_step_result is None


# ======================================================================
# 3. Run 10 steps without error
# ======================================================================


class TestMultiStep:
    """Run 10 steps without error."""

    def test_ten_steps(self, model: TrafficModel) -> None:
        for _ in range(10):
            model.step()
        assert model.step_count == 10
        assert model.last_step_result["step_count"] == 10

    def test_ten_steps_no_signals(self, model_no_signals: TrafficModel) -> None:
        for _ in range(10):
            model_no_signals.step()
        assert model_no_signals.step_count == 10


# ======================================================================
# 4. VehicleAgents are updated after each step (positions change)
# ======================================================================


class TestAgentUpdate:
    """VehicleAgents are updated after each step -- positions change."""

    def test_positions_change(self, model: TrafficModel) -> None:
        initial_positions = np.array([a.position.copy() for a in model.vehicle_agents])
        model.step()
        updated_positions = np.array([a.position for a in model.vehicle_agents])
        # At least some vehicles must have moved
        assert not np.allclose(initial_positions, updated_positions, atol=1e-15), (
            "No vehicle moved after one step"
        )

    def test_agents_match_simulation(self, model: TrafficModel) -> None:
        model.step()
        result = model.last_step_result
        for i, agent in enumerate(model.vehicle_agents):
            np.testing.assert_array_equal(agent.position, result["positions"][i])
            np.testing.assert_array_equal(agent.velocity, result["velocities"][i])
            assert agent.mass == pytest.approx(result["masses"][i])

    def test_mass_types_assigned(self, model: TrafficModel) -> None:
        model.step()
        types = {a.mass_type for a in model.vehicle_agents}
        # With 30 vehicles and random speeds, we expect at least 2 types
        assert len(types) >= 2, f"Only mass types found: {types}"


# ======================================================================
# 5. IntersectionAgents cycle phases after enough steps
# ======================================================================


class TestIntersectionPhases:
    """IntersectionAgents cycle phases after enough time."""

    def test_phase_cycles(self, model: TrafficModel) -> None:
        if not model.intersection_agents:
            pytest.skip("No intersection agents in model")
        ia = model.intersection_agents[0]
        initial_phase = ia.current_phase

        # Run enough steps to exceed one green phase duration.
        # Default green time = 120/2 = 60 s.  At dt ~0.1 s, need ~600 steps.
        # Use a shorter dt to trigger faster, or just run enough steps.
        green_duration = ia.green_times[initial_phase]
        dt = model.dt
        # Over-shoot by 20% to be safe
        n_steps = int(green_duration / dt * 1.2) + 10
        for _ in range(n_steps):
            model.step()

        # Phase should have changed at least once
        # (We check time_in_phase was reset, indicating a transition occurred)
        assert model.step_count == n_steps
        # The phase must have advanced: either current_phase differs, or
        # it wrapped back around.  We check the total elapsed time exceeds
        # one full green phase.
        sum(r.get("dt_used", model.dt) for r in [model.simulation.step() for _ in range(0)])
        # Simpler check: just ensure time_in_phase < green_duration
        # (which means a reset happened at some point)
        assert ia.time_in_phase < green_duration or ia.current_phase != initial_phase


# ======================================================================
# 6. DataCollector records KPIs
# ======================================================================


class TestDataCollector:
    """DataCollector records KPIs (mean_speed_kmh exists)."""

    def test_kpi_collected_after_step(self, model: TrafficModel) -> None:
        model.step()
        df = model.datacollector.get_model_vars_dataframe()
        assert len(df) == 1

    def test_mean_speed_kmh_exists(self, model: TrafficModel) -> None:
        model.step()
        df = model.datacollector.get_model_vars_dataframe()
        assert "mean_speed_kmh" in df.columns

    def test_congestion_index_in_range(self, model: TrafficModel) -> None:
        model.step()
        df = model.datacollector.get_model_vars_dataframe()
        ci = df["congestion_index"].iloc[0]
        assert 0.0 <= ci <= 1.0, f"congestion_index out of range: {ci}"

    def test_mass_type_counts_sum(self, model: TrafficModel) -> None:
        model.step()
        df = model.datacollector.get_model_vars_dataframe()
        row = df.iloc[0]
        total = row["n_slow"] + row["n_fast"] + row["n_neutral"]
        assert total == len(model.vehicle_agents)

    def test_multiple_steps_accumulate(self, model: TrafficModel) -> None:
        for _ in range(5):
            model.step()
        df = model.datacollector.get_model_vars_dataframe()
        assert len(df) == 5

    def test_mean_speed_positive(self, model: TrafficModel) -> None:
        model.step()
        df = model.datacollector.get_model_vars_dataframe()
        assert df["mean_speed_kmh"].iloc[0] > 0.0


# ======================================================================
# 7. get_potential_field returns valid grid
# ======================================================================


class TestPotentialField:
    """get_potential_field returns valid grid data."""

    def test_returns_dict(self, model: TrafficModel) -> None:
        model.step()
        pf = model.get_potential_field(resolution=50.0)
        assert isinstance(pf, dict)

    def test_required_keys(self, model: TrafficModel) -> None:
        model.step()
        pf = model.get_potential_field(resolution=50.0)
        for key in (
            "potential",
            "grid_centers",
            "grid_width",
            "grid_height",
            "x_min",
            "y_min",
            "x_max",
            "y_max",
        ):
            assert key in pf, f"Missing key: {key}"

    def test_potential_shape(self, model: TrafficModel) -> None:
        model.step()
        pf = model.get_potential_field(resolution=50.0)
        assert pf["potential"].ndim == 1
        assert len(pf["potential"]) == len(pf["grid_centers"])

    def test_grid_centers_shape(self, model: TrafficModel) -> None:
        model.step()
        pf = model.get_potential_field(resolution=50.0)
        assert pf["grid_centers"].ndim == 2
        assert pf["grid_centers"].shape[1] == 2

    def test_potential_is_float64(self, model: TrafficModel) -> None:
        model.step()
        pf = model.get_potential_field(resolution=50.0)
        assert pf["potential"].dtype == np.float64

    def test_potential_not_all_zero(self, model: TrafficModel) -> None:
        """With 30 vehicles, the potential field should be non-trivial."""
        model.step()
        pf = model.get_potential_field(resolution=50.0)
        assert not np.allclose(pf["potential"], 0.0), (
            "Potential field is all zeros -- physics not propagated"
        )


# ======================================================================
# 8. get_state returns vehicles and intersections lists
# ======================================================================


class TestGetState:
    """get_state returns vehicles and intersections lists."""

    def test_returns_dict(self, model: TrafficModel) -> None:
        state = model.get_state()
        assert isinstance(state, dict)

    def test_required_keys(self, model: TrafficModel) -> None:
        state = model.get_state()
        for key in ("step", "vehicles", "intersections", "kpi"):
            assert key in state, f"Missing key: {key}"

    def test_vehicles_list(self, model: TrafficModel) -> None:
        state = model.get_state()
        assert isinstance(state["vehicles"], list)
        assert len(state["vehicles"]) == 30

    def test_vehicle_dict_keys(self, model: TrafficModel) -> None:
        model.step()
        state = model.get_state()
        v = state["vehicles"][0]
        for key in ("id", "x", "y", "vx", "vy", "mass", "speed_kmh", "type"):
            assert key in v, f"Missing vehicle key: {key}"

    def test_intersections_list(self, model: TrafficModel) -> None:
        state = model.get_state()
        assert isinstance(state["intersections"], list)
        assert len(state["intersections"]) == len(model.intersection_agents)

    def test_intersection_dict_keys(self, model: TrafficModel) -> None:
        if not model.intersection_agents:
            pytest.skip("No intersection agents")
        state = model.get_state()
        ia = state["intersections"][0]
        for key in ("node_id", "x", "y", "current_phase", "green_times", "is_green"):
            assert key in ia, f"Missing intersection key: {key}"

    def test_kpi_empty_before_step(self, model: TrafficModel) -> None:
        state = model.get_state()
        assert state["kpi"] == {}

    def test_kpi_populated_after_step(self, model: TrafficModel) -> None:
        model.step()
        state = model.get_state()
        assert "mean_speed_kmh" in state["kpi"]
        assert "congestion_index" in state["kpi"]


# ======================================================================
# 9. No vehicle exceeds v_max
# ======================================================================


class TestVMaxEnforcement:
    """No vehicle exceeds v_max after simulation steps."""

    def test_initial_speeds_below_vmax(self, model: TrafficModel) -> None:
        """Initial speeds are sampled in [10, 30] m/s, below v_max=36."""
        for a in model.vehicle_agents:
            assert a.speed <= model.v_max + 1e-10, (
                f"Vehicle {a.unique_id} initial speed {a.speed:.2f} > v_max {model.v_max}"
            )

    def test_speeds_below_vmax_after_steps(self, model: TrafficModel) -> None:
        """After 20 steps, no vehicle should exceed v_max.

        The leapfrog integrator in GravSimulation clamps speeds to v_max.
        """
        for _ in range(20):
            model.step()
        for a in model.vehicle_agents:
            assert a.speed <= model.v_max + 1e-10, (
                f"Vehicle {a.unique_id} speed {a.speed:.2f} m/s exceeds "
                f"v_max {model.v_max} m/s at step {model.step_count}"
            )

    def test_speed_kmh_consistent(self, model: TrafficModel) -> None:
        """speed_kmh == speed * 3.6 for all agents after stepping."""
        model.step()
        for a in model.vehicle_agents:
            assert a.speed_kmh == pytest.approx(a.speed * 3.6, rel=1e-12)


# ======================================================================
# 10. Red-light masses are wired into GravSimulation
# ======================================================================


class TestRedLightObstacles:
    """Red-light masses from IntersectionAgents are injected as obstacles."""

    def test_obstacles_set_when_red_lights_exist(self, model: TrafficModel) -> None:
        """After a step, the simulation should have obstacles if any
        intersection has a red phase."""
        if not model.intersection_agents:
            pytest.skip("No intersection agents in model")

        model.step()

        # Count expected red-light obstacles
        expected_count = 0
        for ia in model.intersection_agents:
            expected_count += len(ia.get_red_light_masses())

        obs_count = len(model.simulation._obstacle_masses)

        if expected_count > 0:
            assert obs_count == expected_count, (
                f"Expected {expected_count} obstacle masses, got {obs_count}"
            )
        else:
            assert obs_count == 0

    def test_no_obstacles_without_signals(self, model_no_signals: TrafficModel) -> None:
        """Model with signals disabled should have zero obstacles."""
        model_no_signals.step()
        assert len(model_no_signals.simulation._obstacle_masses) == 0

    def test_red_light_affects_nearby_vehicles(self, grid_network: RoadNetwork) -> None:
        """Vehicles near a red light should behave differently than in a
        signal-free model.

        We compare positions after several steps between a signalized and
        non-signalized model with the same initial conditions.
        """
        seed = 12345
        kwargs = dict(
            network=grid_network,
            n_vehicles=30,
            G_s=2.0,
            beta=0.5,
            softening=10.0,
            theta=0.5,
            dt=0.1,
            v_max=36.0,
            seed=seed,
        )

        model_sig = TrafficModel(signal_intersections=True, **kwargs)
        model_nosig = TrafficModel(signal_intersections=False, **kwargs)

        n_steps = 10
        for _ in range(n_steps):
            model_sig.step()
            model_nosig.step()

        pos_sig = model_sig.simulation.positions
        pos_nosig = model_nosig.simulation.positions

        # The signalized model should produce different positions
        # because red-light masses alter the force field
        assert not np.allclose(pos_sig, pos_nosig, atol=1e-10), (
            "Signalized and non-signalized models produced identical "
            "positions -- red-light masses are not affecting physics"
        )

    def test_obstacle_masses_are_positive(self, model: TrafficModel) -> None:
        """Red-light obstacle masses must all be positive."""
        if not model.intersection_agents:
            pytest.skip("No intersection agents in model")

        model.step()
        if len(model.simulation._obstacle_masses) > 0:
            assert np.all(model.simulation._obstacle_masses > 0.0), (
                "Red-light masses should be positive"
            )
