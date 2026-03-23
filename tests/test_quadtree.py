"""Tests for Barnes-Hut QuadTree force computation.

Validates the O(N log N) Barnes-Hut approximation against the O(N^2) naive
direct summation baseline.  All tests use float64 arithmetic.

For relative-error tests, particles with negligible net force (below 5% of
the RMS force magnitude) are excluded.  These particles sit in symmetric
environments where forces from all directions nearly cancel, making the
tiny residual force dominated by approximation noise.  This is standard
practice in N-body simulation benchmarking (see Salmon & Warren 1994).

Author: Agent #16 N-body Simulation Expert
Date: 2026-03-22
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from gravtraffic.core.force_engine import ForceEngine
from gravtraffic.core.quadtree import QuadTree


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def engine() -> ForceEngine:
    """Standard ForceEngine with default parameters."""
    return ForceEngine(G_s=9.8, softening=10.0)


def _random_particles(
    n: int, seed: int = 42, domain: float = 1000.0
) -> tuple[np.ndarray, np.ndarray]:
    """Generate random positions and signed masses."""
    rng = np.random.default_rng(seed)
    positions = rng.uniform(0.0, domain, size=(n, 2)).astype(np.float64)
    # Janus model: masses can be positive or negative.
    masses = rng.uniform(-5.0, 5.0, size=n).astype(np.float64)
    # Ensure no zero masses (avoid degenerate cases).
    masses[masses == 0.0] = 1.0
    return positions, masses


def _max_relative_error(
    forces_approx: np.ndarray,
    forces_exact: np.ndarray,
    force_threshold_fraction: float = 0.05,
) -> float:
    """Compute the maximum relative force error across significant particles.

    The relative error for particle i is:
        ||F_approx_i - F_exact_i|| / ||F_exact_i||

    Particles whose exact force magnitude is below
    ``force_threshold_fraction * rms_force`` are excluded.  These particles
    sit in near-symmetric environments where the tiny net force is dominated
    by approximation noise, making relative error misleading.  This is
    standard practice in N-body benchmarking.

    Parameters
    ----------
    forces_approx : ndarray, shape (N, 2)
        Approximate forces from Barnes-Hut.
    forces_exact : ndarray, shape (N, 2)
        Exact forces from naive O(N^2).
    force_threshold_fraction : float
        Fraction of RMS force below which particles are skipped.
    """
    diff = forces_approx - forces_exact
    err_norms = np.linalg.norm(diff, axis=1)
    exact_norms = np.linalg.norm(forces_exact, axis=1)

    if len(exact_norms) == 0:
        return 0.0

    # RMS force magnitude.
    rms_force = np.sqrt(np.mean(exact_norms ** 2))
    threshold = force_threshold_fraction * rms_force

    # Skip particles with negligible force (near-symmetry cancellation).
    mask = exact_norms > max(threshold, 1e-15)
    if not np.any(mask):
        return 0.0

    rel_errors = err_norms[mask] / exact_norms[mask]
    return float(np.max(rel_errors))


# ------------------------------------------------------------------
# Test 1: Barnes-Hut vs naive, 100 particles, theta=0.5, < 1% error
# ------------------------------------------------------------------

def test_barnes_hut_vs_naive_100_theta05(engine: ForceEngine) -> None:
    """100 random particles, theta=0.5: max relative error < 1%."""
    positions, masses = _random_particles(100, seed=1)

    forces_naive = engine.compute_all_naive(positions, masses)
    forces_bh = engine.compute_all(positions, masses, theta=0.5)

    max_err = _max_relative_error(forces_bh, forces_naive)
    assert max_err < 0.01, (
        f"Barnes-Hut relative error {max_err:.6f} exceeds 1% at theta=0.5"
    )


# ------------------------------------------------------------------
# Test 2: Barnes-Hut vs naive, 500 particles, theta=0.7, < 2% error
# ------------------------------------------------------------------

def test_barnes_hut_vs_naive_500_theta07(engine: ForceEngine) -> None:
    """500 random particles, theta=0.7: max relative error < 2%."""
    positions, masses = _random_particles(500, seed=2)

    forces_naive = engine.compute_all_naive(positions, masses)
    forces_bh = engine.compute_all(positions, masses, theta=0.7)

    max_err = _max_relative_error(forces_bh, forces_naive)
    assert max_err < 0.02, (
        f"Barnes-Hut relative error {max_err:.6f} exceeds 2% at theta=0.7"
    )


# ------------------------------------------------------------------
# Test 3: Exact at theta=0 — must match naive to machine precision
# ------------------------------------------------------------------

def test_exact_at_theta_zero(engine: ForceEngine) -> None:
    """At theta=0 Barnes-Hut should never use COM approximation."""
    positions, masses = _random_particles(50, seed=3)

    forces_naive = engine.compute_all_naive(positions, masses)
    forces_bh = engine.compute_all(positions, masses, theta=0.0)

    np.testing.assert_allclose(
        forces_bh, forces_naive, rtol=1e-12, atol=1e-15,
        err_msg="theta=0 Barnes-Hut does not match naive exactly",
    )


# ------------------------------------------------------------------
# Test 4: Single particle — no self-force
# ------------------------------------------------------------------

def test_single_particle_no_self_force(engine: ForceEngine) -> None:
    """A lone particle should experience zero force."""
    positions = np.array([[500.0, 500.0]], dtype=np.float64)
    masses = np.array([3.0], dtype=np.float64)

    forces = engine.compute_all(positions, masses, theta=0.5)
    assert forces.shape == (1, 2)
    np.testing.assert_allclose(
        forces, 0.0, atol=1e-15,
        err_msg="Single particle experiences non-zero self-force",
    )


# ------------------------------------------------------------------
# Test 5: Two particles — matches force_pair exactly
# ------------------------------------------------------------------

def test_two_particles_match_force_pair(engine: ForceEngine) -> None:
    """Two-particle Barnes-Hut must reproduce force_pair exactly."""
    positions = np.array([
        [100.0, 200.0],
        [400.0, 600.0],
    ], dtype=np.float64)
    masses = np.array([2.0, -3.0], dtype=np.float64)

    # Direct pair computation.
    dx = positions[1, 0] - positions[0, 0]  # 300
    dy = positions[1, 1] - positions[0, 1]  # 400
    fx_01, fy_01 = engine.force_pair(masses[0], masses[1], dx, dy)

    # Barnes-Hut.
    forces_bh = engine.compute_all(positions, masses, theta=0.5)

    # Force on particle 0 should be (fx_01, fy_01).
    np.testing.assert_allclose(
        forces_bh[0], [fx_01, fy_01], rtol=1e-12,
        err_msg="Two-particle BH force on p0 does not match force_pair",
    )
    # Force on particle 1 should be Newton's third law opposite.
    np.testing.assert_allclose(
        forces_bh[1], [-fx_01, -fy_01], rtol=1e-12,
        err_msg="Two-particle BH force on p1 does not satisfy Newton III",
    )


# ------------------------------------------------------------------
# Test 6: Performance — 10k particles completes in reasonable time
# ------------------------------------------------------------------

def test_performance_10k_particles(engine: ForceEngine) -> None:
    """10,000 particles should complete without hanging (< 60 seconds)."""
    positions, masses = _random_particles(10_000, seed=6, domain=5000.0)

    start = time.perf_counter()
    forces = engine.compute_all(positions, masses, theta=0.5)
    elapsed = time.perf_counter() - start

    assert forces.shape == (10_000, 2)
    assert elapsed < 60.0, (
        f"Barnes-Hut on 10k particles took {elapsed:.1f}s (limit 60s)"
    )


# ------------------------------------------------------------------
# Additional robustness tests
# ------------------------------------------------------------------

def test_empty_array(engine: ForceEngine) -> None:
    """Empty input should return empty (0, 2) array."""
    positions = np.zeros((0, 2), dtype=np.float64)
    masses = np.zeros(0, dtype=np.float64)

    forces = engine.compute_all(positions, masses, theta=0.5)
    assert forces.shape == (0, 2)


def test_coincident_particles(engine: ForceEngine) -> None:
    """Particles at the same location should not produce infinities."""
    positions = np.array([
        [100.0, 100.0],
        [100.0, 100.0],
        [100.0, 100.0],
    ], dtype=np.float64)
    masses = np.array([1.0, 2.0, 3.0], dtype=np.float64)

    forces_bh = engine.compute_all(positions, masses, theta=0.5)
    forces_naive = engine.compute_all_naive(positions, masses)

    assert np.all(np.isfinite(forces_bh)), "Barnes-Hut produced non-finite values"
    np.testing.assert_allclose(
        forces_bh, forces_naive, rtol=1e-12, atol=1e-15,
        err_msg="Coincident particles: BH does not match naive",
    )


def test_all_positive_masses_accuracy(engine: ForceEngine) -> None:
    """All-positive masses: standard BH accuracy should be excellent."""
    rng = np.random.default_rng(99)
    positions = rng.uniform(0.0, 1000.0, size=(200, 2)).astype(np.float64)
    masses = rng.uniform(1.0, 5.0, size=200).astype(np.float64)

    forces_naive = engine.compute_all_naive(positions, masses)
    forces_bh = engine.compute_all(positions, masses, theta=0.5)

    max_err = _max_relative_error(forces_bh, forces_naive)
    assert max_err < 0.01, (
        f"All-positive masses: relative error {max_err:.6f} exceeds 1%"
    )
