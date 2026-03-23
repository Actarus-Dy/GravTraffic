"""Tests for Numba JIT-compiled force engines.

Validates that Numba naive and Barnes-Hut engines produce forces
matching the CPU reference implementation within tight tolerances.

Author: Agent #01 Python Scientific Developer
Date: 2026-03-23
"""

from __future__ import annotations

import numpy as np
import pytest

from gravtraffic.core.force_engine import ForceEngine
from gravtraffic.core.force_engine_numba import (
    NUMBA_AVAILABLE,
    ForceEngineNumba,
    ForceEngineBHNumba,
)

requires_numba = pytest.mark.skipif(
    not NUMBA_AVAILABLE, reason="Numba not installed"
)


@pytest.fixture()
def cpu_engine():
    return ForceEngine(G_s=5.0, softening=10.0)


# ======================================================================
# ForceEngineNumba (naive JIT)
# ======================================================================

@requires_numba
class TestForceEngineNumba:
    def test_empty_input(self) -> None:
        engine = ForceEngineNumba(G_s=5.0, softening=10.0)
        result = engine.compute_all(
            np.empty((0, 2), dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )
        assert result.shape == (0, 2)

    def test_single_particle(self) -> None:
        engine = ForceEngineNumba(G_s=5.0, softening=10.0)
        result = engine.compute_all(
            np.array([[100.0, 50.0]], dtype=np.float64),
            np.array([3.0], dtype=np.float64),
        )
        np.testing.assert_allclose(result, 0.0, atol=1e-15)

    def test_matches_cpu_naive_two_particles(self, cpu_engine) -> None:
        engine = ForceEngineNumba(G_s=5.0, softening=10.0)
        pos = np.array([[0.0, 0.0], [100.0, 0.0]], dtype=np.float64)
        mass = np.array([2.0, -3.0], dtype=np.float64)

        cpu = cpu_engine.compute_all_naive(pos, mass)
        numba = engine.compute_all(pos, mass)

        np.testing.assert_allclose(numba, cpu, rtol=1e-12)

    def test_matches_cpu_naive_many_particles(self, cpu_engine) -> None:
        engine = ForceEngineNumba(G_s=5.0, softening=10.0)
        rng = np.random.default_rng(42)
        n = 200
        pos = rng.uniform(0, 1000, (n, 2)).astype(np.float64)
        mass = rng.uniform(-5, 5, n).astype(np.float64)

        cpu = cpu_engine.compute_all_naive(pos, mass)
        numba = engine.compute_all(pos, mass)

        np.testing.assert_allclose(numba, cpu, rtol=1e-10)

    def test_newton_third_law(self) -> None:
        engine = ForceEngineNumba(G_s=5.0, softening=10.0)
        pos = np.array([[0.0, 0.0], [50.0, 30.0]], dtype=np.float64)
        mass = np.array([3.0, -2.0], dtype=np.float64)
        forces = engine.compute_all(pos, mass)
        np.testing.assert_allclose(forces[0] + forces[1], [0.0, 0.0], atol=1e-12)

    def test_sign_convention(self) -> None:
        engine = ForceEngineNumba(G_s=5.0, softening=10.0)
        pos = np.array([[0.0, 0.0], [100.0, 0.0]], dtype=np.float64)
        mass = np.array([5.0, 5.0], dtype=np.float64)
        forces = engine.compute_all(pos, mass)
        assert forces[0, 0] > 0, "Same-sign should attract"
        assert forces[1, 0] < 0, "Same-sign should attract"


# ======================================================================
# ForceEngineBHNumba (Barnes-Hut + JIT traversal)
# ======================================================================

@requires_numba
class TestForceEngineBHNumba:
    def test_empty_input(self) -> None:
        engine = ForceEngineBHNumba(G_s=5.0, softening=10.0)
        result = engine.compute_all(
            np.empty((0, 2), dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )
        assert result.shape == (0, 2)

    def test_matches_cpu_naive_small(self, cpu_engine) -> None:
        engine = ForceEngineBHNumba(G_s=5.0, softening=10.0)
        rng = np.random.default_rng(99)
        n = 50
        pos = rng.uniform(0, 500, (n, 2)).astype(np.float64)
        mass = rng.uniform(-3, 3, n).astype(np.float64)

        cpu = cpu_engine.compute_all_naive(pos, mass)
        bh = engine.compute_all(pos, mass, theta=0.0)  # theta=0 = exact

        np.testing.assert_allclose(bh, cpu, rtol=1e-8)

    def test_theta_05_within_2_percent(self, cpu_engine) -> None:
        engine = ForceEngineBHNumba(G_s=5.0, softening=10.0)
        rng = np.random.default_rng(123)
        n = 100
        pos = rng.uniform(0, 1000, (n, 2)).astype(np.float64)
        mass = rng.uniform(-5, 5, n).astype(np.float64)

        cpu = cpu_engine.compute_all_naive(pos, mass)
        bh = engine.compute_all(pos, mass, theta=0.5)

        # Relative error per particle force magnitude
        cpu_mag = np.linalg.norm(cpu, axis=1)
        diff_mag = np.linalg.norm(bh - cpu, axis=1)
        # Only check particles with non-trivial forces
        mask = cpu_mag > 1e-6
        rel_err = diff_mag[mask] / cpu_mag[mask]
        assert np.mean(rel_err) < 0.02, f"Mean relative error {np.mean(rel_err):.4f} > 2%"


# ======================================================================
# Integration with GravSimulation
# ======================================================================

@requires_numba
class TestSimulationNumbaIntegration:
    def test_simulation_auto_selects_numba(self) -> None:
        from gravtraffic.core.simulation import GravSimulation

        sim = GravSimulation(G_s=5.0, use_gpu=False)
        assert isinstance(sim._force_engine, ForceEngineNumba)

    def test_simulation_runs_with_numba(self) -> None:
        from gravtraffic.core.simulation import GravSimulation

        sim = GravSimulation(G_s=5.0, use_gpu=False, adaptive_dt=False, dt=0.1)
        rng = np.random.default_rng(42)
        n = 20
        pos = np.column_stack([rng.uniform(0, 500, n), np.zeros(n)]).astype(np.float64)
        vel = np.column_stack([rng.uniform(10, 30, n), np.zeros(n)]).astype(np.float64)
        sim.init_vehicles(pos, vel, np.full(n, 50.0, dtype=np.float64))
        result = sim.step()
        assert result["step_count"] == 1
        assert result["positions"].shape == (n, 2)


@requires_numba
class TestBHNumbaEdgeCases:
    def test_single_particle(self) -> None:
        engine = ForceEngineBHNumba(G_s=5.0, softening=10.0)
        result = engine.compute_all(
            np.array([[100.0, 50.0]], dtype=np.float64),
            np.array([3.0], dtype=np.float64),
        )
        np.testing.assert_allclose(result, 0.0, atol=1e-15)

    def test_momentum_conservation_theta0(self, cpu_engine) -> None:
        """Sum of forces should be near zero (Newton's 3rd law)."""
        engine = ForceEngineBHNumba(G_s=5.0, softening=10.0)
        rng = np.random.default_rng(77)
        n = 50
        pos = rng.uniform(0, 500, (n, 2)).astype(np.float64)
        mass = rng.uniform(-5, 5, n).astype(np.float64)
        forces = engine.compute_all(pos, mass, theta=0.0)
        total = forces.sum(axis=0)
        np.testing.assert_allclose(total, [0.0, 0.0], atol=1e-6)

    def test_all_positive_masses(self, cpu_engine) -> None:
        """All positive masses — one tree empty, should still work."""
        engine = ForceEngineBHNumba(G_s=5.0, softening=10.0)
        pos = np.array([[0.0, 0.0], [100.0, 0.0], [200.0, 0.0]], dtype=np.float64)
        mass = np.array([3.0, 5.0, 2.0], dtype=np.float64)

        cpu = cpu_engine.compute_all_naive(pos, mass)
        bh = engine.compute_all(pos, mass, theta=0.0)
        np.testing.assert_allclose(bh, cpu, rtol=1e-8)

    def test_all_negative_masses(self, cpu_engine) -> None:
        """All negative masses — the other tree empty."""
        engine = ForceEngineBHNumba(G_s=5.0, softening=10.0)
        pos = np.array([[0.0, 0.0], [100.0, 0.0]], dtype=np.float64)
        mass = np.array([-3.0, -5.0], dtype=np.float64)

        cpu = cpu_engine.compute_all_naive(pos, mass)
        bh = engine.compute_all(pos, mass, theta=0.0)
        np.testing.assert_allclose(bh, cpu, rtol=1e-8)


class TestNumbaNotAvailable:
    @pytest.mark.skipif(NUMBA_AVAILABLE, reason="Numba IS installed")
    def test_raises_without_numba(self) -> None:
        with pytest.raises(RuntimeError, match="Numba is required"):
            ForceEngineNumba()
