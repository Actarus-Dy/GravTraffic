"""Tests for GravSimulation prediction (clone + predict).

Validates:
    1. clone() creates an independent deep copy
    2. predict() returns valid state after forward integration
    3. Prediction diverges from live simulation (independence)
    4. API endpoint POST /api/v1/predict works

Author: Agent #01 Python Scientific Developer
Date: 2026-03-23
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from gravtraffic.core.simulation import GravSimulation
from gravtraffic.api.app import app, state


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def sim() -> GravSimulation:
    """A small simulation with 20 vehicles for fast tests."""
    s = GravSimulation(G_s=5.0, beta=0.5, adaptive_dt=False, dt=0.1)
    rng = np.random.default_rng(42)
    n = 20
    positions = np.column_stack([
        rng.uniform(0, 500, n),
        np.zeros(n),
    ]).astype(np.float64)
    velocities = np.column_stack([
        rng.uniform(10, 30, n),
        np.zeros(n),
    ]).astype(np.float64)
    densities = np.full(n, 40.0, dtype=np.float64)
    s.init_vehicles(positions, velocities, densities)
    # Run a few steps to get non-trivial state
    s.run(10)
    return s


@pytest.fixture(autouse=True)
def _reset_api_state():
    state.reset()
    yield
    state.reset()


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


# ---------------------------------------------------------------------------
# Clone tests
# ---------------------------------------------------------------------------

class TestClone:
    def test_clone_has_same_state(self, sim: GravSimulation) -> None:
        """Clone should have identical positions, velocities, masses."""
        clone = sim.clone()
        np.testing.assert_array_equal(clone.positions, sim.positions)
        np.testing.assert_array_equal(clone.velocities, sim.velocities)
        np.testing.assert_array_equal(clone.masses, sim.masses)
        assert clone.step_count == sim.step_count
        assert clone.dt == sim.dt
        assert clone.G_s == sim.G_s

    def test_clone_is_independent(self, sim: GravSimulation) -> None:
        """Modifying the clone must not affect the original."""
        clone = sim.clone()
        original_pos = sim.positions.copy()

        # Run the clone forward
        clone.run(50)

        # Original must be unchanged
        np.testing.assert_array_equal(sim.positions, original_pos)
        assert sim.step_count == 10  # unchanged

        # Clone must have advanced
        assert clone.step_count == 60
        assert not np.array_equal(clone.positions, original_pos)

    def test_clone_preserves_obstacles(self, sim: GravSimulation) -> None:
        """Clone should copy obstacle state."""
        obs_pos = np.array([[100.0, 0.0], [200.0, 0.0]], dtype=np.float64)
        obs_mass = np.array([50.0, 50.0], dtype=np.float64)
        sim.set_obstacles(obs_pos, obs_mass)

        clone = sim.clone()
        np.testing.assert_array_equal(clone._obstacle_positions, obs_pos)
        np.testing.assert_array_equal(clone._obstacle_masses, obs_mass)

        # Modifying clone obstacles doesn't affect original
        clone.clear_obstacles()
        assert len(sim._obstacle_masses) == 2


# ---------------------------------------------------------------------------
# Predict tests
# ---------------------------------------------------------------------------

class TestPredict:
    def test_predict_returns_valid_state(self, sim: GravSimulation) -> None:
        """predict() should return a dict with all expected keys."""
        result = sim.predict(horizon_s=10.0)

        assert "positions" in result
        assert "velocities" in result
        assert "masses" in result
        assert "mean_speed" in result
        assert "step_count" in result
        assert "horizon_s" in result
        assert "n_steps_run" in result

        assert result["positions"].shape == (sim.n_vehicles, 2)
        assert result["n_steps_run"] > 0
        assert result["horizon_s"] >= 10.0

    def test_predict_does_not_mutate_original(self, sim: GravSimulation) -> None:
        """predict() must not change the live simulation state."""
        pos_before = sim.positions.copy()
        step_before = sim.step_count

        sim.predict(horizon_s=5.0)

        np.testing.assert_array_equal(sim.positions, pos_before)
        assert sim.step_count == step_before

    def test_predict_diverges_from_current(self, sim: GravSimulation) -> None:
        """Predicted state should differ from current state."""
        result = sim.predict(horizon_s=30.0)
        assert not np.allclose(result["positions"], sim.positions, atol=0.01)

    def test_predict_short_horizon(self, sim: GravSimulation) -> None:
        """Very short horizon should run at least 1 step."""
        result = sim.predict(horizon_s=0.01)
        assert result["n_steps_run"] >= 1

    def test_predict_zero_vehicles(self) -> None:
        """predict() with 0 vehicles should return immediately, not hang."""
        s = GravSimulation(G_s=5.0, adaptive_dt=False, dt=0.1)
        s.init_vehicles(
            np.empty((0, 2), dtype=np.float64),
            np.empty((0, 2), dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )
        result = s.predict(horizon_s=10.0)
        assert result["n_steps_run"] == 0
        assert result["positions"].shape == (0, 2)
        assert result["mean_speed"] == 0.0


# ---------------------------------------------------------------------------
# API endpoint tests
# ---------------------------------------------------------------------------

API_CONFIG = {
    "grid_rows": 3,
    "grid_cols": 3,
    "block_size": 100.0,
    "n_vehicles": 15,
    "G_s": 5.0,
    "beta": 0.5,
    "dt": 0.1,
    "seed": 42,
}


class TestPredictEndpoint:
    def test_predict_no_simulation(self, client: TestClient) -> None:
        resp = client.post("/api/v1/predict", params={"horizon_s": 10.0})
        assert resp.status_code == 409

    def test_predict_returns_vehicles(self, client: TestClient) -> None:
        client.post("/api/v1/simulate", json=API_CONFIG)
        client.post("/api/v1/step")  # need at least one step

        resp = client.post("/api/v1/predict", params={"horizon_s": 10.0})
        assert resp.status_code == 200
        body = resp.json()

        assert "vehicles" in body
        assert "horizon_s" in body
        assert "n_steps_run" in body
        assert "mean_speed_kmh" in body
        assert body["n_vehicles"] == 15
        assert body["horizon_s"] >= 10.0

        # Each vehicle has expected keys
        v = body["vehicles"][0]
        for key in ("x", "y", "vx", "vy", "mass", "speed_kmh"):
            assert key in v

    def test_predict_does_not_advance_live_sim(self, client: TestClient) -> None:
        """The predict endpoint must not change the live step counter."""
        client.post("/api/v1/simulate", json=API_CONFIG)
        client.post("/api/v1/step")
        step_before = client.get("/api/v1/status").json()["step"]

        client.post("/api/v1/predict", params={"horizon_s": 10.0})

        step_after = client.get("/api/v1/status").json()["step"]
        assert step_after == step_before

    def test_predict_invalid_horizon(self, client: TestClient) -> None:
        """horizon_s=0 or negative should be rejected."""
        client.post("/api/v1/simulate", json=API_CONFIG)
        resp = client.post("/api/v1/predict", params={"horizon_s": 0.0})
        assert resp.status_code == 422
