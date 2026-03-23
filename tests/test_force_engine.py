"""Tests for ForceEngine — naive O(N^2) gravitational social force.

Covers all seven required validation cases:
1. Two positive masses attract (force pulls them together)
2. Two negative masses attract (force pulls them together)
3. Positive + negative mass repel (force pushes them apart)
4. Softening prevents infinite force at d=0
5. Force magnitude decreases with distance (inverse square law)
6. Symmetry: F_ij = -F_ji (Newton's third law)
7. Analytical 2-body case with exact force value

Author: Agent #01 Python Scientific Developer
Date: 2026-03-22
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from gravtraffic.core.force_engine import ForceEngine

# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def engine() -> ForceEngine:
    """Default ForceEngine with G_s=9.8, epsilon=10.0."""
    return ForceEngine(G_s=9.8, softening=10.0)


@pytest.fixture
def engine_tiny_softening() -> ForceEngine:
    """ForceEngine with very small softening for near-Newtonian tests."""
    return ForceEngine(G_s=1.0, softening=1e-6)


# ======================================================================
# Helper: compute expected force analytically
# ======================================================================


def analytical_force_pair(
    G_s: float, epsilon: float, m_i: float, m_j: float, dx: float, dy: float
) -> tuple[float, float]:
    """Reference calculation — direct from the corrected formula, no optimizations."""
    d = math.sqrt(dx * dx + dy * dy + epsilon * epsilon)
    coeff = G_s * m_i * m_j / (d * d * d)
    return (coeff * dx, coeff * dy)


# ======================================================================
# Test 1: Two positive masses -> attraction
# ======================================================================


class TestSameSignAttraction:
    """Same-sign masses must attract: force pulls i toward j."""

    def test_two_positive_masses_force_direction(self, engine: ForceEngine) -> None:
        """Two positive masses along the x-axis.

        With the corrected formula F = +G_s * m_i * m_j / d^3 * (dx, dy):
        - m_i=1, m_j=1, dx=100, dy=0
        - coeff = +9.8 * 1 * 1 / d^3 > 0
        - fx = coeff * 100 > 0  (points in +x direction, from i toward j)

        Same-sign masses attract: the force on i points toward j.
        """
        m_i, m_j = 1.0, 1.0
        dx, dy = 100.0, 0.0
        fx, fy = engine.force_pair(m_i, m_j, dx, dy)

        # Force: coeff = +G_s * m_i * m_j / d^3 > 0
        # Vector: fx has same sign as coeff * dx = (+) * (+) = positive (toward j)
        assert fx > 0.0, f"Expected fx > 0 for same-sign attraction, got {fx}"
        assert abs(fy) < 1e-15, f"Expected fy ~ 0 for x-axis alignment, got {fy}"

    def test_two_positive_masses_compute_all(self, engine: ForceEngine) -> None:
        """Two positive masses via compute_all_naive must attract."""
        positions = np.array([[0.0, 0.0], [100.0, 0.0]], dtype=np.float64)
        masses = np.array([1.0, 1.0], dtype=np.float64)
        forces = engine.compute_all_naive(positions, masses)

        # Particle 0: dx = x_j - x_i = 100, coeff = +G_s*1*1/d^3 > 0
        # fx = coeff * 100 > 0 (force on particle 0 points toward particle 1 = attraction)
        #
        # For compute_all_naive, we verify Newton's third law and
        # consistency with force_pair.
        fx_pair, fy_pair = engine.force_pair(1.0, 1.0, 100.0, 0.0)
        np.testing.assert_allclose(forces[0, 0], fx_pair, rtol=1e-14)
        np.testing.assert_allclose(forces[0, 1], fy_pair, atol=1e-15)
        np.testing.assert_allclose(forces[1, 0], -fx_pair, rtol=1e-14)
        np.testing.assert_allclose(forces[1, 1], -fy_pair, atol=1e-15)


# ======================================================================
# Test 2: Two negative masses -> attraction
# ======================================================================


class TestTwoNegativeMasses:
    """Two negative masses: m_i*m_j > 0, so they attract like positive pairs."""

    def test_two_negative_masses_same_behavior_as_positive(self, engine: ForceEngine) -> None:
        """Force between (-2, -3) should have same sign structure as (2, 3)."""
        dx, dy = 50.0, 30.0

        fx_pos, fy_pos = engine.force_pair(2.0, 3.0, dx, dy)
        fx_neg, fy_neg = engine.force_pair(-2.0, -3.0, dx, dy)

        # m_i*m_j is the same sign (+6) in both cases, so forces are identical
        np.testing.assert_allclose(fx_neg, fx_pos, rtol=1e-14)
        np.testing.assert_allclose(fy_neg, fy_pos, rtol=1e-14)

    def test_two_negative_masses_scalar_sign(self, engine: ForceEngine) -> None:
        """Scalar force F = +G_s * m_i * m_j / d^2 should be > 0 (attraction)."""
        m_i, m_j = -5.0, -3.0
        dx, dy = 80.0, 0.0
        fx, fy = engine.force_pair(m_i, m_j, dx, dy)

        # coeff = +G_s * (-5)*(-3) / d^3 = +G_s * 15 / d^3 > 0
        # fx = coeff * 80 > 0
        assert fx > 0.0


# ======================================================================
# Test 3: Positive + negative mass -> repulsion
# ======================================================================


class TestOppositeSignRepulsion:
    """Opposite-sign masses must repel: force pushes i away from j."""

    def test_positive_negative_force_direction(self, engine: ForceEngine) -> None:
        """m_i=+1, m_j=-1 along x-axis.

        coeff = +G_s * (+1)*(-1) / d^3 = -G_s / d^3 < 0
        fx = coeff * dx < 0 (opposite direction to dx, which points from i to j)

        Opposite-sign masses repel: force on i points away from j.
        """
        m_i, m_j = 1.0, -1.0
        dx, dy = 100.0, 0.0
        fx, fy = engine.force_pair(m_i, m_j, dx, dy)

        # coeff = +9.8 * 1 * (-1) / d^3 = -9.8 / d^3 < 0
        # fx = coeff * 100 < 0
        assert fx < 0.0, f"Expected fx < 0 for opposite-sign repulsion, got {fx}"
        assert abs(fy) < 1e-15

    def test_negative_positive_force_direction(self, engine: ForceEngine) -> None:
        """m_i=-1, m_j=+1: repulsion should also hold."""
        m_i, m_j = -1.0, 1.0
        dx, dy = 100.0, 0.0
        fx, fy = engine.force_pair(m_i, m_j, dx, dy)

        # coeff = +9.8 * (-1)*(+1) / d^3 = -9.8 / d^3 < 0
        assert fx < 0.0

    def test_opposite_vs_same_sign_direction_flip(self, engine: ForceEngine) -> None:
        """Opposite-sign force should have opposite direction to same-sign."""
        dx, dy = 60.0, 40.0

        fx_same, fy_same = engine.force_pair(2.0, 3.0, dx, dy)
        fx_opp, fy_opp = engine.force_pair(2.0, -3.0, dx, dy)

        # Same magnitude, opposite direction
        np.testing.assert_allclose(fx_opp, -fx_same, rtol=1e-14)
        np.testing.assert_allclose(fy_opp, -fy_same, rtol=1e-14)


# ======================================================================
# Test 4: Softening prevents infinite force at d=0
# ======================================================================


class TestSofteningRegularization:
    """Softening epsilon must prevent singularity when particles overlap."""

    def test_zero_distance_finite_force(self, engine: ForceEngine) -> None:
        """Particles at the same position: d_ij = 0, force must be finite."""
        fx, fy = engine.force_pair(1.0, 1.0, 0.0, 0.0)
        assert math.isfinite(fx)
        assert math.isfinite(fy)
        # At d=0, dx=dy=0, so force vector is (0, 0) regardless of softening
        assert fx == 0.0
        assert fy == 0.0

    def test_very_small_distance_bounded(self, engine: ForceEngine) -> None:
        """Particles extremely close: force bounded by softening."""
        dx, dy = 1e-10, 0.0
        fx, fy = engine.force_pair(1.0, 1.0, dx, dy)
        assert math.isfinite(fx)
        assert math.isfinite(fy)

        # Maximum possible force magnitude at d~0 is G_s * m_i * m_j / epsilon^2
        # = 9.8 * 1 * 1 / 100 = 0.098
        max_force = engine.G_s * 1.0 * 1.0 / (engine.epsilon**2)
        assert abs(fx) <= max_force + 1e-10

    def test_softening_controls_max_force(self) -> None:
        """Larger softening -> smaller maximum force."""
        engine_small = ForceEngine(G_s=1.0, softening=1.0)
        engine_large = ForceEngine(G_s=1.0, softening=100.0)

        dx, dy = 0.01, 0.0
        fx_small, _ = engine_small.force_pair(1.0, 1.0, dx, dy)
        fx_large, _ = engine_large.force_pair(1.0, 1.0, dx, dy)

        assert abs(fx_small) > abs(fx_large)

    def test_compute_all_naive_zero_distance(self, engine: ForceEngine) -> None:
        """Two particles at the same position: no crash, finite forces."""
        positions = np.array([[50.0, 50.0], [50.0, 50.0]], dtype=np.float64)
        masses = np.array([1.0, -1.0], dtype=np.float64)
        forces = engine.compute_all_naive(positions, masses)
        assert np.all(np.isfinite(forces))


# ======================================================================
# Test 5: Force magnitude decreases with distance (inverse square)
# ======================================================================


class TestInverseSquareFalloff:
    """Force magnitude must decrease with distance following ~1/d^2."""

    def test_force_decreases_with_distance(self, engine: ForceEngine) -> None:
        """Force at d=200 should be weaker than at d=100."""
        m_i, m_j = 1.0, 1.0
        fx_near, _ = engine.force_pair(m_i, m_j, 100.0, 0.0)
        fx_far, _ = engine.force_pair(m_i, m_j, 200.0, 0.0)

        assert abs(fx_near) > abs(fx_far)

    def test_inverse_square_ratio_large_distance(self, engine_tiny_softening: ForceEngine) -> None:
        """At large d >> epsilon, ratio of forces at d and 2d should be ~4.

        F ~ 1/d^2 for d >> epsilon, so F(d) / F(2d) ~ 4.
        Use large distances and tiny softening to approach Newtonian limit.
        """
        engine = engine_tiny_softening
        d1 = 1000.0
        d2 = 2000.0

        fx1, _ = engine.force_pair(1.0, 1.0, d1, 0.0)
        fx2, _ = engine.force_pair(1.0, 1.0, d2, 0.0)

        # |F(d1)| / |F(d2)| should be close to (d2/d1)^2 = 4
        # But actually the force magnitude is |coeff * dx| where
        # coeff = -G_s / d^3, so |F| = G_s * |dx| / d^3.
        # For dx = d (along x-axis): |F| = G_s * d / d^3 = G_s / d^2.
        # So ratio = d2^2 / d1^2 = 4.
        ratio = abs(fx1) / abs(fx2)
        np.testing.assert_allclose(ratio, 4.0, rtol=1e-6)

    @pytest.mark.parametrize("distance", [500.0, 1000.0, 5000.0, 10000.0])
    def test_force_magnitude_parametric(
        self, engine_tiny_softening: ForceEngine, distance: float
    ) -> None:
        """Force magnitude matches 1/d^2 law at various distances."""
        engine = engine_tiny_softening
        fx, _ = engine.force_pair(1.0, 1.0, distance, 0.0)
        # Expected: |F| = G_s / d^2 (for unit masses, dx=d, epsilon~0)
        expected = engine.G_s / (distance * distance)
        np.testing.assert_allclose(abs(fx), expected, rtol=1e-6)


# ======================================================================
# Test 6: Symmetry — Newton's third law: F_ij = -F_ji
# ======================================================================


class TestNewtonThirdLaw:
    """Force on i due to j must equal minus force on j due to i."""

    def test_pair_symmetry_same_sign(self, engine: ForceEngine) -> None:
        """F_ij = -F_ji for same-sign masses."""
        dx, dy = 73.0, -42.0
        m_i, m_j = 2.5, 3.7

        fx_ij, fy_ij = engine.force_pair(m_i, m_j, dx, dy)
        fx_ji, fy_ji = engine.force_pair(m_j, m_i, -dx, -dy)

        np.testing.assert_allclose(fx_ij, -fx_ji, rtol=1e-14)
        np.testing.assert_allclose(fy_ij, -fy_ji, rtol=1e-14)

    def test_pair_symmetry_opposite_sign(self, engine: ForceEngine) -> None:
        """F_ij = -F_ji for opposite-sign masses."""
        dx, dy = 55.0, 88.0
        m_i, m_j = 4.0, -2.0

        fx_ij, fy_ij = engine.force_pair(m_i, m_j, dx, dy)
        fx_ji, fy_ji = engine.force_pair(m_j, m_i, -dx, -dy)

        np.testing.assert_allclose(fx_ij, -fx_ji, rtol=1e-14)
        np.testing.assert_allclose(fy_ij, -fy_ji, rtol=1e-14)

    def test_compute_all_naive_third_law(self, engine: ForceEngine) -> None:
        """Total force sums to zero for the entire system (momentum conservation)."""
        rng = np.random.default_rng(seed=42)
        n = 20
        positions = rng.uniform(-500.0, 500.0, size=(n, 2)).astype(np.float64)
        masses = rng.uniform(-5.0, 5.0, size=n).astype(np.float64)

        forces = engine.compute_all_naive(positions, masses)

        # Sum of all forces must be zero (Newton's third law)
        total_force = forces.sum(axis=0)
        np.testing.assert_allclose(total_force, 0.0, atol=1e-10)

    def test_compute_all_naive_pairwise_symmetry(self, engine: ForceEngine) -> None:
        """Verify F_ij = -F_ji through compute_all_naive on a 2-body system."""
        positions = np.array([[10.0, 20.0], [80.0, -15.0]], dtype=np.float64)
        masses = np.array([3.0, -7.0], dtype=np.float64)

        forces = engine.compute_all_naive(positions, masses)

        np.testing.assert_allclose(forces[0], -forces[1], rtol=1e-14)


# ======================================================================
# Test 7: Analytical 2-body case with exact force value
# ======================================================================


class TestAnalytical2Body:
    """Verify exact numerical values against hand-calculated references."""

    def test_known_configuration_x_axis(self) -> None:
        """Two particles along x-axis, hand-computed reference value.

        Setup: G_s=9.8, epsilon=10.0, m_i=2.0, m_j=3.0
        Particle i at (0, 0), particle j at (100, 0).
        dx=100, dy=0.

        d = sqrt(100^2 + 0^2 + 10^2) = sqrt(10000 + 100) = sqrt(10100)
        d^3 = 10100 * sqrt(10100) = 10100 * 100.498756...

        coeff = -9.8 * 2.0 * 3.0 / d^3 = -58.8 / d^3
        fx = coeff * 100
        fy = coeff * 0 = 0
        """
        engine = ForceEngine(G_s=9.8, softening=10.0)
        m_i, m_j = 2.0, 3.0
        dx, dy = 100.0, 0.0

        d = math.sqrt(dx**2 + dy**2 + engine.epsilon**2)
        d3 = d**3
        expected_fx = 9.8 * m_i * m_j / d3 * dx
        expected_fy = 0.0

        fx, fy = engine.force_pair(m_i, m_j, dx, dy)

        np.testing.assert_allclose(fx, expected_fx, rtol=1e-14)
        np.testing.assert_allclose(fy, expected_fy, atol=1e-15)

        # Sanity: verify the actual numerical value
        # d = sqrt(10100) = 100.49875621...
        # d^3 = 1015075.125...
        # coeff = +58.8 / 1015075.125 = +5.7927...e-5
        # fx = +5.7927e-5 * 100 = +5.7927e-3
        assert abs(fx - (5.7927e-3)) < 1e-5  # rough sanity check

    def test_known_configuration_diagonal(self) -> None:
        """Two particles along the diagonal (45 degrees).

        Setup: G_s=1.0, epsilon=0.0 (set softening to near-zero),
        m_i=1.0, m_j=1.0, positions (0,0) and (1,1).
        dx=1, dy=1.
        d = sqrt(1 + 1 + eps^2) ~ sqrt(2)
        """
        engine = ForceEngine(G_s=1.0, softening=1e-10)
        m_i, m_j = 1.0, 1.0
        dx, dy = 1.0, 1.0

        d = math.sqrt(2.0 + 1e-20)  # effectively sqrt(2)
        d3 = d**3

        expected_fx = 1.0 * 1.0 * 1.0 / d3 * 1.0
        expected_fy = 1.0 * 1.0 * 1.0 / d3 * 1.0

        fx, fy = engine.force_pair(m_i, m_j, dx, dy)

        np.testing.assert_allclose(fx, expected_fx, rtol=1e-9)
        np.testing.assert_allclose(fy, expected_fy, rtol=1e-9)

        # fx and fy should be equal by symmetry of the diagonal
        np.testing.assert_allclose(fx, fy, rtol=1e-14)

    def test_compute_all_matches_force_pair(self, engine: ForceEngine) -> None:
        """compute_all_naive must produce identical results to manual force_pair."""
        positions = np.array([[0.0, 0.0], [100.0, 0.0], [50.0, 86.6]], dtype=np.float64)
        masses = np.array([2.0, -3.0, 1.5], dtype=np.float64)

        forces = engine.compute_all_naive(positions, masses)

        # Recompute by hand using force_pair
        expected = np.zeros((3, 2), dtype=np.float64)
        for i in range(3):
            for j in range(3):
                if i == j:
                    continue
                dx = positions[j, 0] - positions[i, 0]
                dy = positions[j, 1] - positions[i, 1]
                fx, fy = engine.force_pair(masses[i], masses[j], dx, dy)
                expected[i, 0] += fx
                expected[i, 1] += fy

        np.testing.assert_allclose(forces, expected, rtol=1e-14)


# ======================================================================
# Additional edge cases and dtype checks
# ======================================================================


class TestDtypeAndValidation:
    """Verify float64 enforcement and input validation."""

    def test_output_dtype_is_float64(self, engine: ForceEngine) -> None:
        """compute_all_naive must always return float64."""
        positions = np.array([[0.0, 0.0], [10.0, 10.0]], dtype=np.float64)
        masses = np.array([1.0, 1.0], dtype=np.float64)
        forces = engine.compute_all_naive(positions, masses)
        assert forces.dtype == np.float64

    def test_float32_input_upcast_to_float64(self, engine: ForceEngine) -> None:
        """float32 input must be silently upcast to float64."""
        positions = np.array([[0.0, 0.0], [10.0, 10.0]], dtype=np.float32)
        masses = np.array([1.0, 1.0], dtype=np.float32)
        forces = engine.compute_all_naive(positions, masses)
        assert forces.dtype == np.float64

    def test_shape_mismatch_raises_valueerror(self, engine: ForceEngine) -> None:
        """Incompatible positions/masses shapes must raise ValueError."""
        positions = np.array([[0.0, 0.0], [10.0, 10.0]], dtype=np.float64)
        masses = np.array([1.0, 1.0, 1.0], dtype=np.float64)  # 3 masses, 2 positions
        with pytest.raises(ValueError, match="incompatible"):
            engine.compute_all_naive(positions, masses)

    def test_single_particle_zero_force(self, engine: ForceEngine) -> None:
        """A single particle has zero net force."""
        positions = np.array([[42.0, -17.0]], dtype=np.float64)
        masses = np.array([5.0], dtype=np.float64)
        forces = engine.compute_all_naive(positions, masses)
        np.testing.assert_array_equal(forces, np.zeros((1, 2)))

    def test_empty_system(self, engine: ForceEngine) -> None:
        """Zero particles must return an empty (0, 2) array."""
        positions = np.empty((0, 2), dtype=np.float64)
        masses = np.empty((0,), dtype=np.float64)
        forces = engine.compute_all_naive(positions, masses)
        assert forces.shape == (0, 2)
        assert forces.dtype == np.float64
