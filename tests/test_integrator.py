"""Tests for the leapfrog symplectic integrator and adaptive timestep.

Test suite covers:
    1. Energy conservation (2nd order) for near-circular orbit
    2. Straight-line motion under zero force
    3. Speed clipping under high-force scenarios
    4. Adaptive dt response to particle spacing
    5. Symmetry preservation for identical particles
    6. force_fn is called with updated positions (mock verification)

All tests use float64 arithmetic exclusively.

Author: Agent #01 Python Scientific Developer
Date: 2026-03-22
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from gravtraffic.core.integrator import _clip_speed, adaptive_dt, leapfrog_step

# ===================================================================
# Helpers
# ===================================================================


def _gravitational_force_fn(G: float, m: float, softening: float = 0.0):
    """Return a force function for two equal masses with Newtonian gravity.

    The returned callable computes gravitational acceleration on each
    particle due to all others using direct O(N^2) summation. This is
    used for the energy conservation test where we need a proper
    conservative force.

    Parameters
    ----------
    G : float
        Gravitational constant.
    m : float
        Mass of each particle (all equal).
    softening : float
        Plummer softening length.
    """

    def force_fn(positions: np.ndarray, velocities: np.ndarray = None) -> np.ndarray:
        n = positions.shape[0]
        forces = np.zeros_like(positions, dtype=np.float64)
        eps2 = softening * softening
        for i in range(n):
            for j in range(i + 1, n):
                dx = positions[j, 0] - positions[i, 0]
                dy = positions[j, 1] - positions[i, 1]
                d2 = dx * dx + dy * dy + eps2
                d = np.sqrt(d2)
                d3 = d2 * d
                coeff = G * m * m / d3  # attraction: force toward the other
                fx = coeff * dx
                fy = coeff * dy
                forces[i, 0] += fx
                forces[i, 1] += fy
                forces[j, 0] -= fx
                forces[j, 1] -= fy
        # Return acceleration = force / mass
        return forces / m

    return force_fn


def _total_energy(
    positions: np.ndarray, velocities: np.ndarray, G: float, m: float, softening: float = 0.0
) -> float:
    """Compute total energy (kinetic + potential) for equal-mass system.

    E = sum_i 0.5 * m * |v_i|^2 + sum_{i<j} -G * m^2 / d_ij

    Parameters
    ----------
    positions : ndarray, shape (N, 2)
    velocities : ndarray, shape (N, 2)
    G : float
    m : float
    softening : float

    Returns
    -------
    float
        Total energy.
    """
    n = positions.shape[0]
    # Kinetic energy
    KE = 0.5 * m * np.sum(velocities**2)

    # Potential energy
    PE = 0.0
    eps2 = softening * softening
    for i in range(n):
        for j in range(i + 1, n):
            dx = positions[j, 0] - positions[i, 0]
            dy = positions[j, 1] - positions[i, 1]
            d = np.sqrt(dx * dx + dy * dy + eps2)
            PE -= G * m * m / d

    return KE + PE


# ===================================================================
# Test 1: Energy conservation for near-circular orbit
# ===================================================================


class TestEnergyConservation:
    """Two equal positive masses in near-circular orbit.

    The leapfrog integrator is symplectic, so total energy should
    oscillate but not drift systematically. Over 100 steps with a
    well-chosen dt, the relative energy drift should be below 1%.
    """

    def test_two_body_energy_drift_below_one_percent(self):
        # Setup: two equal masses in circular orbit
        G = 1.0
        m = 1.0
        softening = 0.0

        # Place masses on x-axis, separated by distance r
        r = 10.0
        positions = np.array(
            [
                [-r / 2.0, 0.0],
                [r / 2.0, 0.0],
            ],
            dtype=np.float64,
        )

        # Circular orbit velocity: v = sqrt(G * m / (4 * r)) for two
        # equal masses separated by r orbiting their center of mass.
        # Each is at distance r/2 from center. Centripetal: m*v^2/(r/2) = G*m^2/r^2
        # => v = sqrt(G * m / (2 * r))
        # But with softened gravity off, for two bodies:
        # F = G*m^2/r^2, each feels F, acceleration = G*m/r^2
        # Circular at radius r/2: a = v^2/(r/2) => v = sqrt(a * r/2) = sqrt(G*m/(2*r))
        v_circ = np.sqrt(G * m / (2.0 * r))
        velocities = np.array(
            [
                [0.0, -v_circ],
                [0.0, v_circ],
            ],
            dtype=np.float64,
        )

        force_fn = _gravitational_force_fn(G, m, softening)

        # Initial forces
        forces = force_fn(positions)

        # Initial energy
        E0 = _total_energy(positions, velocities, G, m, softening)

        # Integrate for 100 steps with a small dt
        # Orbital period T = 2*pi*r / (2*v_circ) -- approximate
        # We pick dt so that 100 steps cover a fraction of the orbit
        dt = 0.5

        # Very high v_max so clipping does not interfere
        v_max = 1000.0

        for _ in range(100):
            positions, velocities, forces = leapfrog_step(
                positions, velocities, forces, dt, force_fn, v_max
            )

        E_final = _total_energy(positions, velocities, G, m, softening)

        relative_drift = abs(E_final - E0) / abs(E0)
        assert relative_drift < 0.01, (
            f"Energy drift {relative_drift:.6e} exceeds 1% threshold. "
            f"E0={E0:.10f}, E_final={E_final:.10f}"
        )

    def test_energy_oscillates_not_drifts(self):
        """Check that energy oscillates (symplectic property) rather than
        monotonically increasing or decreasing.

        We use a tight orbit (small r, large G) so the orbital period
        T = 2*pi*(r/2)/v_circ is short enough for 500 steps at dt=0.05
        to cover multiple orbits and reveal the oscillatory pattern.
        """
        G = 100.0
        m = 1.0
        r = 2.0
        # v_circ = sqrt(G * m / (2 * r))
        v_circ = np.sqrt(G * m / (2.0 * r))
        # Orbital period T = 2*pi*(r/2)/v_circ ~ 1.26 s
        # 500 steps * 0.002 = 1.0 s, covering most of an orbit

        positions = np.array([[-r / 2, 0.0], [r / 2, 0.0]], dtype=np.float64)
        velocities = np.array([[0.0, -v_circ], [0.0, v_circ]], dtype=np.float64)
        force_fn = _gravitational_force_fn(G, m)
        forces = force_fn(positions)
        dt = 0.002
        v_max = 1000.0

        energies = []
        for _ in range(500):
            positions, velocities, forces = leapfrog_step(
                positions, velocities, forces, dt, force_fn, v_max
            )
            E = _total_energy(positions, velocities, G, m)
            energies.append(E)

        # Check that energy changes sign of derivative at least once
        diffs = np.diff(energies)
        sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
        assert sign_changes >= 1, (
            f"Energy appears monotonic ({sign_changes} sign changes). "
            "Expected oscillatory behavior from symplectic integrator."
        )


# ===================================================================
# Test 2: Straight-line motion under zero force
# ===================================================================


class TestStraightLineMotion:
    """Zero force should produce constant velocity and exact linear motion."""

    def test_constant_velocity_exact(self):
        n = 5
        rng = np.random.default_rng(seed=123)
        positions = rng.uniform(-100, 100, (n, 2)).astype(np.float64)
        velocities = rng.uniform(-10, 10, (n, 2)).astype(np.float64)
        forces = np.zeros((n, 2), dtype=np.float64)

        def zero_force(pos, vel=None):
            return np.zeros_like(pos, dtype=np.float64)

        dt = 0.1
        pos0 = positions.copy()
        vel0 = velocities.copy()

        for step in range(10):
            positions, velocities, forces = leapfrog_step(
                positions, velocities, forces, dt, zero_force, v_max=1000.0
            )

        expected_pos = pos0 + vel0 * (10 * dt)

        np.testing.assert_allclose(
            positions,
            expected_pos,
            rtol=0,
            atol=1e-12,
            err_msg="Position diverged from exact straight-line solution",
        )
        np.testing.assert_allclose(
            velocities, vel0, rtol=0, atol=1e-12, err_msg="Velocity changed under zero force"
        )

    def test_single_particle_no_interaction(self):
        """Single particle with no force function interaction."""
        positions = np.array([[0.0, 0.0]], dtype=np.float64)
        velocities = np.array([[5.0, 3.0]], dtype=np.float64)
        forces = np.zeros((1, 2), dtype=np.float64)

        def zero_force(pos, vel=None):
            return np.zeros_like(pos, dtype=np.float64)

        dt = 0.05
        for _ in range(20):
            positions, velocities, forces = leapfrog_step(
                positions, velocities, forces, dt, zero_force, v_max=100.0
            )

        expected_pos = np.array([[5.0 * 1.0, 3.0 * 1.0]], dtype=np.float64)
        np.testing.assert_allclose(positions, expected_pos, atol=1e-12)


# ===================================================================
# Test 3: Speed clipping
# ===================================================================


class TestSpeedClipping:
    """Verify that speed never exceeds v_max under high forces."""

    def test_speed_capped_scalar_vmax(self):
        n = 10
        positions = np.zeros((n, 2), dtype=np.float64)
        positions[:, 0] = np.arange(n, dtype=np.float64) * 100.0
        velocities = np.zeros((n, 2), dtype=np.float64)
        v_max = 15.0

        # Force function that applies a huge constant acceleration
        def huge_force(pos, vel=None):
            f = np.full_like(pos, 1000.0, dtype=np.float64)
            return f

        forces = huge_force(positions)
        dt = 0.1

        for _ in range(50):
            positions, velocities, forces = leapfrog_step(
                positions, velocities, forces, dt, huge_force, v_max=v_max
            )
            speeds = np.linalg.norm(velocities, axis=1)
            assert np.all(speeds <= v_max + 1e-12), (
                f"Speed {np.max(speeds):.4f} exceeds v_max={v_max}"
            )

    def test_speed_capped_per_vehicle_vmax(self):
        n = 4
        positions = np.zeros((n, 2), dtype=np.float64)
        positions[:, 0] = np.arange(n, dtype=np.float64) * 100.0
        velocities = np.zeros((n, 2), dtype=np.float64)
        v_max_per = np.array([5.0, 10.0, 20.0, 30.0], dtype=np.float64)

        def huge_force(pos, vel=None):
            return np.full_like(pos, 500.0, dtype=np.float64)

        forces = huge_force(positions)
        dt = 0.1

        for _ in range(50):
            positions, velocities, forces = leapfrog_step(
                positions, velocities, forces, dt, huge_force, v_max=v_max_per
            )
            speeds = np.linalg.norm(velocities, axis=1)
            for i in range(n):
                assert speeds[i] <= v_max_per[i] + 1e-12, (
                    f"Vehicle {i}: speed {speeds[i]:.4f} exceeds v_max={v_max_per[i]}"
                )

    def test_clip_preserves_direction(self):
        """Clipping should reduce magnitude but preserve direction."""
        velocities = np.array(
            [
                [30.0, 40.0],  # speed = 50, should be clipped to 10
                [3.0, 4.0],  # speed = 5, below limit
            ],
            dtype=np.float64,
        )

        result = _clip_speed(velocities, v_max=10.0)

        # First particle: direction preserved, magnitude = 10
        speed_0 = np.linalg.norm(result[0])
        np.testing.assert_allclose(speed_0, 10.0, atol=1e-12)
        # Direction: (30, 40) / 50 = (0.6, 0.8)
        np.testing.assert_allclose(
            result[0] / speed_0,
            np.array([0.6, 0.8]),
            atol=1e-12,
        )

        # Second particle: unchanged
        np.testing.assert_allclose(result[1], velocities[1], atol=1e-15)

    def test_clip_zero_velocity_unchanged(self):
        """Zero velocity should remain zero after clipping."""
        velocities = np.array([[0.0, 0.0]], dtype=np.float64)
        result = _clip_speed(velocities, v_max=10.0)
        np.testing.assert_allclose(result, velocities, atol=1e-15)


# ===================================================================
# Test 4: Adaptive timestep
# ===================================================================


class TestAdaptiveDt:
    """Test that adaptive_dt responds correctly to particle spacing."""

    def test_close_particles_smaller_dt(self):
        """Closely spaced particles should produce a smaller dt."""
        # Closely spaced
        positions_close = np.array(
            [
                [0.0, 0.0],
                [0.1, 0.0],
            ],
            dtype=np.float64,
        )
        velocities = np.array(
            [
                [10.0, 0.0],
                [-10.0, 0.0],
            ],
            dtype=np.float64,
        )

        dt_close = adaptive_dt(positions_close, velocities, dt_max=1.0, dt_min=0.001)

        # Widely spaced
        positions_wide = np.array(
            [
                [0.0, 0.0],
                [1000.0, 0.0],
            ],
            dtype=np.float64,
        )

        dt_wide = adaptive_dt(positions_wide, velocities, dt_max=1.0, dt_min=0.001)

        assert dt_close < dt_wide, f"Close dt ({dt_close}) should be less than wide dt ({dt_wide})"

    def test_widely_spaced_returns_dt_max(self):
        """Very widely spaced, slow particles should return dt_max."""
        positions = np.array(
            [
                [0.0, 0.0],
                [1e6, 0.0],
            ],
            dtype=np.float64,
        )
        velocities = np.array(
            [
                [0.001, 0.0],
                [0.001, 0.0],
            ],
            dtype=np.float64,
        )

        dt = adaptive_dt(positions, velocities, dt_max=0.5, dt_min=0.01)
        assert dt == 0.5, f"Expected dt_max=0.5, got {dt}"

    def test_overlapping_particles_returns_dt_min(self):
        """Overlapping particles (d_min ~ 0) should return dt_min."""
        positions = np.array(
            [
                [5.0, 5.0],
                [5.0, 5.0],
            ],
            dtype=np.float64,
        )
        velocities = np.array(
            [
                [10.0, 0.0],
                [-10.0, 0.0],
            ],
            dtype=np.float64,
        )

        dt = adaptive_dt(positions, velocities, dt_max=0.2, dt_min=0.005)
        assert dt == 0.005, f"Expected dt_min=0.005, got {dt}"

    def test_single_particle_returns_dt_max(self):
        """Single particle means no pairwise constraint -> dt_max."""
        positions = np.array([[0.0, 0.0]], dtype=np.float64)
        velocities = np.array([[100.0, 0.0]], dtype=np.float64)

        dt = adaptive_dt(positions, velocities, dt_max=0.3, dt_min=0.01)
        assert dt == 0.3

    def test_stationary_particles_returns_dt_max(self):
        """All particles stationary -> dt_max."""
        positions = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
            ],
            dtype=np.float64,
        )
        velocities = np.zeros((2, 2), dtype=np.float64)

        dt = adaptive_dt(positions, velocities, dt_max=0.2, dt_min=0.01)
        assert dt == 0.2

    def test_cfl_formula_exact(self):
        """Verify the CFL formula: dt = d_min / (2 * v_max)."""
        positions = np.array(
            [
                [0.0, 0.0],
                [20.0, 0.0],
            ],
            dtype=np.float64,
        )
        velocities = np.array(
            [
                [5.0, 0.0],
                [0.0, 0.0],
            ],
            dtype=np.float64,
        )

        # d_min = 20.0, v_max = 5.0
        # dt_cfl = 20.0 / (2 * 5.0) = 2.0
        # Clamped by dt_max=1.0 -> 1.0
        dt = adaptive_dt(positions, velocities, dt_max=1.0, dt_min=0.001)
        assert dt == 1.0

        # Now with a higher dt_max
        dt = adaptive_dt(positions, velocities, dt_max=5.0, dt_min=0.001)
        np.testing.assert_allclose(dt, 2.0, atol=1e-12)

    def test_many_particles_sorted_approximation(self):
        """With multiple particles, the sort-by-x approximation should
        find a reasonable d_min."""
        positions = np.array(
            [
                [0.0, 0.0],
                [100.0, 0.0],
                [100.5, 0.0],  # closest pair: 0.5 apart
                [200.0, 0.0],
            ],
            dtype=np.float64,
        )
        velocities = np.full((4, 2), 10.0, dtype=np.float64)

        # d_min ~ 0.5, v_max ~ sqrt(200) ~ 14.14
        # dt_cfl ~ 0.5 / (2 * 14.14) ~ 0.0177
        dt = adaptive_dt(positions, velocities, dt_max=1.0, dt_min=0.001)
        assert dt < 0.1, f"Expected small dt due to close pair, got {dt}"
        assert dt >= 0.001, "dt should not go below dt_min"


# ===================================================================
# Test 5: Symmetry preservation
# ===================================================================


class TestSymmetry:
    """Two identical particles at symmetric positions should stay
    symmetric after integration."""

    def test_symmetric_particles_stay_symmetric(self):
        positions = np.array(
            [
                [-50.0, 0.0],
                [50.0, 0.0],
            ],
            dtype=np.float64,
        )
        velocities = np.array(
            [
                [0.0, 5.0],
                [0.0, -5.0],
            ],
            dtype=np.float64,
        )

        def symmetric_force(pos, vel=None):
            """Repulsive force along separation axis."""
            n = pos.shape[0]
            forces = np.zeros_like(pos, dtype=np.float64)
            for i in range(n):
                for j in range(i + 1, n):
                    diff = pos[j] - pos[i]
                    d = np.linalg.norm(diff)
                    if d > 1e-15:
                        # Repulsive: push apart
                        f = -100.0 / (d * d) * diff / d
                        forces[i] += f
                        forces[j] -= f
            return forces

        forces = symmetric_force(positions)
        dt = 0.01
        v_max = 1000.0

        for _ in range(100):
            positions, velocities, forces = leapfrog_step(
                positions, velocities, forces, dt, symmetric_force, v_max
            )

        # Check symmetry: center of mass should be at origin
        com = np.mean(positions, axis=0)
        np.testing.assert_allclose(
            com, [0.0, 0.0], atol=1e-12, err_msg="Center of mass drifted from origin"
        )

        # Positions should be mirror-symmetric about origin
        np.testing.assert_allclose(
            positions[0], -positions[1], atol=1e-12, err_msg="Positions lost mirror symmetry"
        )

        # Velocities should be antisymmetric
        np.testing.assert_allclose(
            velocities[0], -velocities[1], atol=1e-12, err_msg="Velocities lost antisymmetry"
        )


# ===================================================================
# Test 6: force_fn called with updated positions
# ===================================================================


class TestForceCallback:
    """Verify that force_fn receives the drifted (updated) positions,
    not the original positions."""

    def test_force_fn_receives_new_positions(self):
        positions = np.array([[0.0, 0.0], [10.0, 0.0]], dtype=np.float64)
        velocities = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float64)
        forces = np.zeros((2, 2), dtype=np.float64)
        dt = 0.5

        mock_force_fn = MagicMock(return_value=np.zeros((2, 2), dtype=np.float64))

        leapfrog_step(positions, velocities, forces, dt, mock_force_fn, v_max=100.0)

        # force_fn should have been called exactly once
        mock_force_fn.assert_called_once()

        # The first argument should be the drifted positions, not original
        called_positions = mock_force_fn.call_args[0][0]

        # After half-kick: v_half = [1,0] + 0.5*[0,0]*0.5 = [1,0]
        # After drift: x_new = [0,0] + [1,0]*0.5 = [0.5, 0]
        expected_new = np.array([[0.5, 0.0], [10.0, 0.0]], dtype=np.float64)

        np.testing.assert_allclose(
            called_positions,
            expected_new,
            atol=1e-12,
            err_msg="force_fn was not called with the drifted positions",
        )

    def test_force_fn_called_exactly_once_per_step(self):
        """force_fn should be called exactly once per leapfrog step."""
        n = 5
        positions = np.zeros((n, 2), dtype=np.float64)
        velocities = np.zeros((n, 2), dtype=np.float64)
        forces = np.zeros((n, 2), dtype=np.float64)

        mock_force_fn = MagicMock(return_value=np.zeros((n, 2), dtype=np.float64))

        num_steps = 7
        for _ in range(num_steps):
            positions, velocities, forces = leapfrog_step(
                positions, velocities, forces, 0.1, mock_force_fn, v_max=100.0
            )

        assert mock_force_fn.call_count == num_steps, (
            f"force_fn called {mock_force_fn.call_count} times, expected {num_steps}"
        )


# ===================================================================
# Test: dtype enforcement
# ===================================================================


class TestDtypeEnforcement:
    """Ensure all outputs are float64."""

    def test_output_dtypes_are_float64(self):
        positions = np.array([[0.0, 0.0]], dtype=np.float64)
        velocities = np.array([[1.0, 1.0]], dtype=np.float64)
        forces = np.zeros((1, 2), dtype=np.float64)

        def zero_force(pos, vel=None):
            return np.zeros_like(pos)

        p, v, f = leapfrog_step(positions, velocities, forces, 0.1, zero_force)
        assert p.dtype == np.float64, f"positions dtype: {p.dtype}"
        assert v.dtype == np.float64, f"velocities dtype: {v.dtype}"
        assert f.dtype == np.float64, f"forces dtype: {f.dtype}"

    def test_adaptive_dt_returns_python_float(self):
        positions = np.array([[0.0, 0.0], [10.0, 0.0]], dtype=np.float64)
        velocities = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float64)
        result = adaptive_dt(positions, velocities)
        assert isinstance(result, float), f"Expected float, got {type(result)}"
