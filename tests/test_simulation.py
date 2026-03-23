"""Integration tests for GravSimulation pipeline.

Tests
-----
1. GravSimulation with 50 vehicles runs 10 steps without error.
2. Masses are assigned correctly (slow -> positive, fast -> negative).
3. Step returns all expected keys.
4. Potential field computation works at current state.
5. run(n) returns list of n dicts.
6. Vehicles do not exceed v_max after stepping.

Author: Agent #01 Python Scientific Developer
Date: 2026-03-22
"""

from __future__ import annotations

import numpy as np
import pytest

from gravtraffic.core.simulation import GravSimulation


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture()
def sim_50() -> GravSimulation:
    """Create a GravSimulation with 50 vehicles on a 500 m road."""
    rng = np.random.default_rng(123)
    n = 50
    positions = np.column_stack([
        rng.uniform(0, 500, n),
        rng.uniform(-5, 5, n),
    ])
    velocities = np.column_stack([
        rng.uniform(10, 30, n),
        np.zeros(n),
    ])
    densities = rng.uniform(10, 80, n)

    sim = GravSimulation(G_s=2.0, beta=0.5, v_max=36.0, adaptive_dt=True)
    sim.init_vehicles(positions, velocities, densities)
    return sim


# ---------------------------------------------------------------------------
# Test 1: 50 vehicles, 10 steps, no error
# ---------------------------------------------------------------------------
class TestSimulationRunsWithoutError:
    """GravSimulation with 50 vehicles runs 10 steps without error."""

    def test_10_steps_no_exception(self, sim_50: GravSimulation) -> None:
        for _ in range(10):
            result = sim_50.step()
        assert sim_50.step_count == 10

    def test_positions_remain_finite(self, sim_50: GravSimulation) -> None:
        for _ in range(10):
            sim_50.step()
        assert np.all(np.isfinite(sim_50.positions))

    def test_velocities_remain_finite(self, sim_50: GravSimulation) -> None:
        for _ in range(10):
            sim_50.step()
        assert np.all(np.isfinite(sim_50.velocities))


# ---------------------------------------------------------------------------
# Test 2: Mass sign conventions
# ---------------------------------------------------------------------------
class TestMassAssignment:
    """Slow vehicles get positive mass, fast vehicles get negative mass."""

    def test_slow_vehicles_positive_mass(self) -> None:
        """A vehicle much slower than the mean should have positive mass."""
        n = 20
        # All vehicles at 25 m/s except one at 5 m/s (very slow)
        positions = np.column_stack([
            np.linspace(0, 500, n),
            np.zeros(n),
        ])
        velocities = np.full((n, 2), 0.0, dtype=np.float64)
        velocities[:, 0] = 25.0
        velocities[0, 0] = 5.0  # slow outlier
        densities = np.full(n, 40.0, dtype=np.float64)

        sim = GravSimulation(G_s=2.0, beta=0.5)
        sim.init_vehicles(positions, velocities, densities)

        # After init, masses are computed
        assert sim.masses[0] > 0.0, "Slow vehicle should have positive mass"

    def test_fast_vehicles_negative_mass(self) -> None:
        """A vehicle much faster than the mean should have negative mass."""
        n = 20
        positions = np.column_stack([
            np.linspace(0, 500, n),
            np.zeros(n),
        ])
        velocities = np.full((n, 2), 0.0, dtype=np.float64)
        velocities[:, 0] = 15.0
        velocities[0, 0] = 35.0  # fast outlier
        densities = np.full(n, 40.0, dtype=np.float64)

        sim = GravSimulation(G_s=2.0, beta=0.5)
        sim.init_vehicles(positions, velocities, densities)

        assert sim.masses[0] < 0.0, "Fast vehicle should have negative mass"


# ---------------------------------------------------------------------------
# Test 3: Step returns all expected keys
# ---------------------------------------------------------------------------
class TestStepReturnDict:
    """step() returns a dict with all required keys."""

    _EXPECTED_KEYS = {
        "positions",
        "velocities",
        "masses",
        "mean_speed",
        "dt_used",
        "step_count",
    }

    def test_keys_present(self, sim_50: GravSimulation) -> None:
        result = sim_50.step()
        assert set(result.keys()) == self._EXPECTED_KEYS

    def test_positions_shape(self, sim_50: GravSimulation) -> None:
        result = sim_50.step()
        assert result["positions"].shape == (50, 2)

    def test_velocities_shape(self, sim_50: GravSimulation) -> None:
        result = sim_50.step()
        assert result["velocities"].shape == (50, 2)

    def test_masses_shape(self, sim_50: GravSimulation) -> None:
        result = sim_50.step()
        assert result["masses"].shape == (50,)

    def test_mean_speed_is_float(self, sim_50: GravSimulation) -> None:
        result = sim_50.step()
        assert isinstance(result["mean_speed"], float)

    def test_dt_used_positive(self, sim_50: GravSimulation) -> None:
        result = sim_50.step()
        assert result["dt_used"] > 0.0

    def test_step_count_increments(self, sim_50: GravSimulation) -> None:
        r1 = sim_50.step()
        r2 = sim_50.step()
        assert r1["step_count"] == 1
        assert r2["step_count"] == 2

    def test_return_arrays_are_copies(self, sim_50: GravSimulation) -> None:
        """Returned arrays should be copies, not views of internal state."""
        result = sim_50.step()
        # Mutate the returned array; internal state should not change.
        old_pos = sim_50.positions.copy()
        result["positions"][:] = 0.0
        np.testing.assert_array_equal(sim_50.positions, old_pos)


# ---------------------------------------------------------------------------
# Test 4: Potential field computation
# ---------------------------------------------------------------------------
class TestPotentialField:
    """get_potential_field returns a sensible potential array."""

    def test_potential_field_shape(self, sim_50: GravSimulation) -> None:
        grid = np.array([[100.0, 0.0], [200.0, 0.0], [300.0, 0.0]])
        phi = sim_50.get_potential_field(grid)
        assert phi.shape == (3,)

    def test_potential_field_finite(self, sim_50: GravSimulation) -> None:
        grid = np.array([[100.0, 0.0], [200.0, 0.0], [300.0, 0.0]])
        phi = sim_50.get_potential_field(grid)
        assert np.all(np.isfinite(phi))

    def test_potential_field_after_stepping(self, sim_50: GravSimulation) -> None:
        """Potential field should still work after several steps."""
        for _ in range(5):
            sim_50.step()
        grid = np.array([[50.0, 0.0], [250.0, 0.0]])
        phi = sim_50.get_potential_field(grid)
        assert phi.shape == (2,)
        assert np.all(np.isfinite(phi))


# ---------------------------------------------------------------------------
# Test 5: run(n) returns list of n dicts
# ---------------------------------------------------------------------------
class TestRunMethod:
    """run(n) returns a list of exactly n step result dicts."""

    def test_run_length(self, sim_50: GravSimulation) -> None:
        results = sim_50.run(7)
        assert len(results) == 7

    def test_run_step_counts_sequential(self, sim_50: GravSimulation) -> None:
        results = sim_50.run(5)
        for i, r in enumerate(results, start=1):
            assert r["step_count"] == i

    def test_run_zero_steps(self, sim_50: GravSimulation) -> None:
        results = sim_50.run(0)
        assert results == []


# ---------------------------------------------------------------------------
# Test 6: Speed limiter enforcement
# ---------------------------------------------------------------------------
class TestSpeedLimiter:
    """Vehicles do not exceed v_max after stepping."""

    def test_speed_within_vmax(self) -> None:
        """After 20 steps, no vehicle should exceed v_max."""
        rng = np.random.default_rng(999)
        n = 30
        positions = np.column_stack([
            rng.uniform(0, 300, n),
            rng.uniform(-5, 5, n),
        ])
        # Give some vehicles speeds near v_max to provoke clipping
        velocities = np.column_stack([
            rng.uniform(30, 36, n),
            rng.uniform(-2, 2, n),
        ])
        densities = rng.uniform(20, 80, n)

        v_max = 36.0
        sim = GravSimulation(G_s=2.0, beta=0.5, v_max=v_max, adaptive_dt=True)
        sim.init_vehicles(positions, velocities, densities)

        for _ in range(20):
            sim.step()

        speeds = np.linalg.norm(sim.velocities, axis=1)
        # Allow a small floating-point tolerance
        assert np.all(speeds <= v_max + 1e-12), (
            f"Max speed {speeds.max():.6f} exceeds v_max {v_max}"
        )

    def test_speed_limiter_with_low_vmax(self) -> None:
        """With a very low v_max, all speeds should be clamped."""
        rng = np.random.default_rng(42)
        n = 20
        positions = np.column_stack([
            np.linspace(0, 400, n),
            np.zeros(n),
        ])
        velocities = np.column_stack([
            rng.uniform(10, 30, n),
            np.zeros(n),
        ])
        densities = np.full(n, 40.0, dtype=np.float64)

        v_max = 10.0
        sim = GravSimulation(G_s=2.0, beta=0.5, v_max=v_max, adaptive_dt=False, dt=0.05)
        sim.init_vehicles(positions, velocities, densities)

        for _ in range(10):
            sim.step()

        speeds = np.linalg.norm(sim.velocities, axis=1)
        assert np.all(speeds <= v_max + 1e-12), (
            f"Max speed {speeds.max():.6f} exceeds v_max {v_max}"
        )


# ---------------------------------------------------------------------------
# Test 7: dtype enforcement
# ---------------------------------------------------------------------------
class TestDtypeEnforcement:
    """All state arrays must be float64."""

    def test_positions_dtype(self, sim_50: GravSimulation) -> None:
        sim_50.step()
        assert sim_50.positions.dtype == np.float64

    def test_velocities_dtype(self, sim_50: GravSimulation) -> None:
        sim_50.step()
        assert sim_50.velocities.dtype == np.float64

    def test_masses_dtype(self, sim_50: GravSimulation) -> None:
        sim_50.step()
        assert sim_50.masses.dtype == np.float64


# ---------------------------------------------------------------------------
# Test 8: Input validation
# ---------------------------------------------------------------------------
class TestInputValidation:
    """init_vehicles rejects mismatched shapes."""

    def test_positions_shape_mismatch(self) -> None:
        sim = GravSimulation()
        with pytest.raises(ValueError, match="positions shape"):
            sim.init_vehicles(
                np.zeros((5, 2)),
                np.zeros((5, 2)),
                np.zeros(3),  # wrong length
            )

    def test_velocities_shape_mismatch(self) -> None:
        sim = GravSimulation()
        with pytest.raises(ValueError, match="velocities shape"):
            sim.init_vehicles(
                np.zeros((5, 2)),
                np.zeros((3, 2)),  # wrong length
                np.zeros(5),
            )


# ---------------------------------------------------------------------------
# Test 9: Local densities are updated during simulation
# ---------------------------------------------------------------------------
class TestLocalDensitiesUpdate:
    """local_densities must be recomputed from positions at every step."""

    def test_local_densities_update(self, sim_50: GravSimulation) -> None:
        """After 10 steps, local_densities should differ from the initial."""
        densities_step0 = sim_50.local_densities.copy()
        for _ in range(10):
            sim_50.step()
        densities_step10 = sim_50.local_densities.copy()
        # Densities must have changed (vehicles moved, so neighborhoods change)
        assert not np.array_equal(densities_step0, densities_step10), (
            "local_densities unchanged after 10 steps -- density is stale"
        )

    def test_local_densities_clustering(self) -> None:
        """Vehicles in a tight cluster should have higher density than spread-out ones."""
        n = 100
        # 50 vehicles clustered in [0, 20] x [-1, 1]
        rng = np.random.default_rng(777)
        cluster_pos = np.column_stack([
            rng.uniform(0, 20, 50),
            rng.uniform(-1, 1, 50),
        ])
        # 50 vehicles spread over [1000, 5000] x [-1, 1]
        spread_pos = np.column_stack([
            rng.uniform(1000, 5000, 50),
            rng.uniform(-1, 1, 50),
        ])
        positions = np.vstack([cluster_pos, spread_pos])
        velocities = np.column_stack([
            np.full(n, 20.0),
            np.zeros(n),
        ])
        densities_init = np.full(n, 30.0, dtype=np.float64)

        sim = GravSimulation(G_s=2.0, beta=0.5, adaptive_dt=False, dt=0.1)
        sim.init_vehicles(positions, velocities, densities_init)
        # Run one step so densities are recomputed from positions
        sim.step()

        cluster_density = sim.local_densities[:50].mean()
        spread_density = sim.local_densities[50:].mean()
        assert cluster_density > spread_density, (
            f"Cluster density {cluster_density:.2f} should exceed "
            f"spread density {spread_density:.2f}"
        )

    def test_local_densities_shape(self, sim_50: GravSimulation) -> None:
        """local_densities must be shape (N,) and dtype float64."""
        sim_50.step()
        assert sim_50.local_densities.shape == (50,)
        assert sim_50.local_densities.dtype == np.float64


# ---------------------------------------------------------------------------
# Test 10: Obstacle support (set_obstacles / clear_obstacles)
# ---------------------------------------------------------------------------
class TestObstacles:
    """Static obstacles (e.g. red-light masses) affect vehicle forces."""

    @staticmethod
    def _make_sim_with_fast_vehicle() -> GravSimulation:
        """Helper: create a simulation with one fast vehicle (negative mass)
        and several slow vehicles to establish a low mean speed.

        The fast vehicle is at index 0, positioned at x=0, moving right
        at 35 m/s.  Five slow vehicles are far away at x=-500 moving at
        5 m/s so the mean speed is low, giving the fast vehicle a strong
        negative mass.
        """
        n = 6
        positions = np.zeros((n, 2), dtype=np.float64)
        positions[0] = [0.0, 0.0]        # fast vehicle (test subject)
        positions[1:, 0] = -500.0         # slow vehicles far away

        velocities = np.zeros((n, 2), dtype=np.float64)
        velocities[0] = [35.0, 0.0]      # fast
        velocities[1:, 0] = 5.0           # slow

        densities = np.full(n, 30.0, dtype=np.float64)

        sim = GravSimulation(
            G_s=2.0, beta=0.5, softening=10.0, v_max=36.0,
            adaptive_dt=False, dt=0.1,
        )
        sim.init_vehicles(positions, velocities, densities)
        return sim

    def test_set_obstacles_stores_arrays(self) -> None:
        """set_obstacles stores positions and masses as float64 arrays."""
        sim = GravSimulation()
        sim.init_vehicles(
            np.zeros((2, 2)), np.ones((2, 2)) * 20.0, np.full(2, 30.0)
        )
        obs_pos = np.array([[100.0, 0.0], [200.0, 0.0]])
        obs_mass = np.array([50.0, 50.0])
        sim.set_obstacles(obs_pos, obs_mass)

        assert sim._obstacle_positions.shape == (2, 2)
        assert sim._obstacle_masses.shape == (2,)
        assert sim._obstacle_positions.dtype == np.float64
        assert sim._obstacle_masses.dtype == np.float64
        np.testing.assert_array_equal(sim._obstacle_positions, obs_pos)
        np.testing.assert_array_equal(sim._obstacle_masses, obs_mass)

    def test_clear_obstacles_removes_all(self) -> None:
        """clear_obstacles resets to empty arrays."""
        sim = GravSimulation()
        sim.init_vehicles(
            np.zeros((2, 2)), np.ones((2, 2)) * 20.0, np.full(2, 30.0)
        )
        sim.set_obstacles(
            np.array([[100.0, 0.0]]), np.array([50.0])
        )
        sim.clear_obstacles()

        assert sim._obstacle_positions.shape == (0, 2)
        assert sim._obstacle_masses.shape == (0,)

    def test_obstacles_affect_forces(self) -> None:
        """Forces differ when obstacles are present vs absent.

        Uses a population with clear fast/slow split so vehicles have
        significant non-zero masses that interact with the obstacle.
        """
        n = 10
        positions = np.column_stack([
            np.linspace(0, 200, n), np.zeros(n)
        ])
        velocities = np.zeros((n, 2), dtype=np.float64)
        # Half slow (5 m/s), half fast (35 m/s) -> mean ~20 m/s
        velocities[:5, 0] = 5.0
        velocities[5:, 0] = 35.0
        densities = np.full(n, 30.0, dtype=np.float64)

        # Run one step WITHOUT obstacles
        sim_a = GravSimulation(
            G_s=2.0, beta=0.5, softening=10.0, adaptive_dt=False, dt=0.1,
        )
        sim_a.init_vehicles(
            positions.copy(), velocities.copy(), densities.copy()
        )
        result_a = sim_a.step()

        # Run one step WITH a large positive obstacle near the vehicles
        sim_b = GravSimulation(
            G_s=2.0, beta=0.5, softening=10.0, adaptive_dt=False, dt=0.1,
        )
        sim_b.init_vehicles(
            positions.copy(), velocities.copy(), densities.copy()
        )
        sim_b.set_obstacles(
            np.array([[100.0, 0.0]], dtype=np.float64),
            np.array([200.0], dtype=np.float64),
        )
        result_b = sim_b.step()

        # After one step, the obstacle affects the second half-kick so
        # velocities diverge immediately.  Positions diverge after 2+ steps.
        assert not np.allclose(
            result_a["velocities"], result_b["velocities"]
        ), "Obstacle had no effect on vehicle velocities"

    def test_clear_obstacles_restores_baseline(self) -> None:
        """After clearing obstacles, forces match the no-obstacle baseline."""
        rng = np.random.default_rng(555)
        n = 5
        positions = np.column_stack([
            rng.uniform(0, 100, n), np.zeros(n)
        ])
        velocities = np.column_stack([
            rng.uniform(15, 25, n), np.zeros(n)
        ])
        densities = np.full(n, 30.0, dtype=np.float64)

        sim = GravSimulation(
            G_s=2.0, beta=0.5, softening=10.0, adaptive_dt=False, dt=0.1,
        )
        sim.init_vehicles(
            positions.copy(), velocities.copy(), densities.copy()
        )

        # Set then clear obstacles before stepping
        sim.set_obstacles(
            np.array([[50.0, 0.0]]), np.array([100.0])
        )
        sim.clear_obstacles()
        result_cleared = sim.step()

        # Baseline: fresh sim, never had obstacles
        sim2 = GravSimulation(
            G_s=2.0, beta=0.5, softening=10.0, adaptive_dt=False, dt=0.1,
        )
        sim2.init_vehicles(
            positions.copy(), velocities.copy(), densities.copy()
        )
        result_baseline = sim2.step()

        np.testing.assert_allclose(
            result_cleared["positions"],
            result_baseline["positions"],
            atol=1e-14,
            err_msg="Cleared obstacles still affecting forces",
        )

    def test_obstacle_decelerates_approaching_vehicle(self) -> None:
        """A fast vehicle (negative mass) approaching a positive-mass obstacle
        should be repelled, i.e. decelerated.

        Setup: vehicle 0 is fast (negative mass) at x=0 moving right toward
        an obstacle at x=100.  After several steps, vehicle 0's x-velocity
        should be lower (or position behind) vs the obstacle-free case.
        """
        # Without obstacle
        sim_free = self._make_sim_with_fast_vehicle()
        assert sim_free.masses[0] < 0.0, "Vehicle 0 should have negative mass"
        for _ in range(20):
            sim_free.step()
        x_free = sim_free.positions[0, 0]
        vx_free = sim_free.velocities[0, 0]

        # With positive-mass obstacle ahead of the fast vehicle
        sim_obs = self._make_sim_with_fast_vehicle()
        sim_obs.set_obstacles(
            np.array([[100.0, 0.0]], dtype=np.float64),
            np.array([200.0], dtype=np.float64),  # large positive mass
        )
        for _ in range(20):
            sim_obs.step()
        x_obs = sim_obs.positions[0, 0]
        vx_obs = sim_obs.velocities[0, 0]

        # The obstacle should have slowed the fast vehicle:
        # either position lags or velocity is lower.
        decelerated = (x_obs < x_free) or (vx_obs < vx_free)
        assert decelerated, (
            f"Obstacle did not decelerate vehicle: "
            f"x_free={x_free:.4f}, x_obs={x_obs:.4f}, "
            f"vx_free={vx_free:.4f}, vx_obs={vx_obs:.4f}"
        )
