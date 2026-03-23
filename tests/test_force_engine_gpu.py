"""Tests for GPU-accelerated force engine.

Tests are skipped if CuPy is not installed. When CuPy IS available,
validates that GPU forces match the CPU naive reference implementation
within a tight tolerance.

Author: Agent #01 Python Scientific Developer
Date: 2026-03-23
"""

from __future__ import annotations

import numpy as np
import pytest

from gravtraffic.core.force_engine import ForceEngine
from gravtraffic.core.force_engine_gpu import GPU_AVAILABLE, ForceEngineGPU

requires_gpu = pytest.mark.skipif(
    not GPU_AVAILABLE, reason="CuPy not installed — GPU tests skipped"
)


@pytest.fixture()
def engines():
    """CPU and GPU engines with identical parameters."""
    cpu = ForceEngine(G_s=5.0, softening=10.0)
    gpu = ForceEngineGPU(G_s=5.0, softening=10.0)
    return cpu, gpu


@requires_gpu
class TestGPUForceEngine:
    def test_empty_input(self, engines) -> None:
        _, gpu = engines
        result = gpu.compute_all(
            np.empty((0, 2), dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )
        assert result.shape == (0, 2)

    def test_single_particle(self, engines) -> None:
        _, gpu = engines
        pos = np.array([[100.0, 50.0]], dtype=np.float64)
        mass = np.array([3.0], dtype=np.float64)
        result = gpu.compute_all(pos, mass)
        assert result.shape == (1, 2)
        np.testing.assert_allclose(result, 0.0, atol=1e-15)

    def test_matches_cpu_naive_two_particles(self, engines) -> None:
        cpu, gpu = engines
        pos = np.array([[0.0, 0.0], [100.0, 0.0]], dtype=np.float64)
        mass = np.array([2.0, -3.0], dtype=np.float64)

        cpu_forces = cpu.compute_all_naive(pos, mass)
        gpu_forces = gpu.compute_all(pos, mass)

        np.testing.assert_allclose(gpu_forces, cpu_forces, rtol=1e-10)

    def test_matches_cpu_naive_many_particles(self, engines) -> None:
        cpu, gpu = engines
        rng = np.random.default_rng(42)
        n = 200
        pos = rng.uniform(0, 1000, (n, 2)).astype(np.float64)
        mass = rng.uniform(-5, 5, n).astype(np.float64)

        cpu_forces = cpu.compute_all_naive(pos, mass)
        gpu_forces = gpu.compute_all(pos, mass)

        np.testing.assert_allclose(gpu_forces, cpu_forces, rtol=1e-8, atol=1e-10)

    def test_sign_convention_same_sign_attraction(self, engines) -> None:
        """Same-sign masses: force should attract (toward each other)."""
        _, gpu = engines
        pos = np.array([[0.0, 0.0], [100.0, 0.0]], dtype=np.float64)
        mass = np.array([5.0, 5.0], dtype=np.float64)
        forces = gpu.compute_all(pos, mass)

        # Particle 0 should be pulled toward particle 1 (positive fx)
        assert forces[0, 0] > 0, f"Expected attraction, got fx={forces[0, 0]}"
        # Particle 1 should be pulled toward particle 0 (negative fx)
        assert forces[1, 0] < 0, f"Expected attraction, got fx={forces[1, 0]}"

    def test_sign_convention_opposite_sign_repulsion(self, engines) -> None:
        """Opposite-sign masses: force should repel."""
        _, gpu = engines
        pos = np.array([[0.0, 0.0], [100.0, 0.0]], dtype=np.float64)
        mass = np.array([5.0, -5.0], dtype=np.float64)
        forces = gpu.compute_all(pos, mass)

        # Particle 0 should be pushed away from particle 1 (negative fx)
        assert forces[0, 0] < 0, f"Expected repulsion, got fx={forces[0, 0]}"

    def test_newton_third_law(self, engines) -> None:
        """F_ij + F_ji = 0 (Newton's third law)."""
        _, gpu = engines
        pos = np.array([[0.0, 0.0], [50.0, 30.0]], dtype=np.float64)
        mass = np.array([3.0, -2.0], dtype=np.float64)
        forces = gpu.compute_all(pos, mass)

        np.testing.assert_allclose(
            forces[0] + forces[1], [0.0, 0.0], atol=1e-12
        )


class TestGPUNotAvailable:
    """Test graceful fallback when CuPy is not installed."""

    @pytest.mark.skipif(GPU_AVAILABLE, reason="CuPy IS installed — cannot test absence")
    def test_raises_on_init_without_cupy(self) -> None:
        with pytest.raises(RuntimeError, match="CuPy is required"):
            ForceEngineGPU()


class TestSimulationGPUIntegration:
    """Test that GravSimulation correctly uses GPU when available."""

    def test_simulation_with_gpu_flag(self) -> None:
        from gravtraffic.core.simulation import GravSimulation

        sim = GravSimulation(G_s=5.0, use_gpu=True)
        if GPU_AVAILABLE:
            assert isinstance(sim._force_engine, ForceEngineGPU)
        else:
            # Falls back to CPU (Numba or pure Python)
            assert not isinstance(sim._force_engine, ForceEngineGPU)

    def test_simulation_with_gpu_false(self) -> None:
        from gravtraffic.core.simulation import GravSimulation

        sim = GravSimulation(G_s=5.0, use_gpu=False)
        assert not isinstance(sim._force_engine, ForceEngineGPU)

    @requires_gpu
    def test_gpu_fallback_above_max_n(self) -> None:
        """When N > max_n, GPU engine should fallback to CPU and match."""
        cpu = ForceEngine(G_s=5.0, softening=10.0)
        gpu = ForceEngineGPU(G_s=5.0, softening=10.0, max_n=5)

        rng = np.random.default_rng(99)
        pos = rng.uniform(0, 500, (10, 2)).astype(np.float64)
        mass = rng.uniform(-3, 3, 10).astype(np.float64)

        cpu_forces = cpu.compute_all_naive(pos, mass)
        # N=10 > max_n=5, so GPU should fallback to CPU Barnes-Hut
        gpu_forces = gpu.compute_all(pos, mass)

        np.testing.assert_allclose(gpu_forces, cpu_forces, rtol=0.02)

    def test_gpu_fallback_without_cupy(self) -> None:
        """Without CuPy, ForceEngineGPU max_n fallback path is unreachable
        since __init__ raises. Just verify use_gpu=False works."""
        from gravtraffic.core.simulation import GravSimulation

        sim = GravSimulation(G_s=5.0, use_gpu=False, adaptive_dt=False, dt=0.1)
        assert not isinstance(sim._force_engine, ForceEngineGPU)

    def test_simulation_gpu_false_runs_step(self) -> None:
        """Verify simulation with use_gpu=False completes a full step."""
        from gravtraffic.core.simulation import GravSimulation

        sim = GravSimulation(G_s=5.0, use_gpu=False, adaptive_dt=False, dt=0.1)
        rng = np.random.default_rng(42)
        n = 10
        pos = np.column_stack([rng.uniform(0, 500, n), np.zeros(n)]).astype(np.float64)
        vel = np.column_stack([rng.uniform(10, 30, n), np.zeros(n)]).astype(np.float64)
        sim.init_vehicles(pos, vel, np.full(n, 50.0, dtype=np.float64))
        result = sim.step()
        assert result["step_count"] == 1
        assert result["positions"].shape == (n, 2)
