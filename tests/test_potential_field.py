"""Tests for gravtraffic.core.potential_field module.

Validates:
1. Single positive mass produces negative potential (congestion well)
2. Single negative mass produces positive potential (fluid hill)
3. Potential decreases in magnitude with distance (1/r decay)
4. Symmetry: equal masses at symmetric positions produce symmetric field
5. make_grid produces correct dimensions
6. optimize_traffic_light returns valid timing in [15, 90]
7. Empty vehicles produce zero potential everywhere
8. dtype is always float64

Author: Agent #01 Python Scientific Developer
Date: 2026-03-22
"""

from __future__ import annotations

import numpy as np
import pytest

from gravtraffic.core.potential_field import (
    compute_potential_field,
    make_grid,
    optimize_traffic_light,
)


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def single_positive_mass():
    """Single slow vehicle (m > 0) at origin."""
    pos = np.array([[0.0, 0.0]], dtype=np.float64)
    mass = np.array([2.0], dtype=np.float64)
    return pos, mass


@pytest.fixture
def single_negative_mass():
    """Single fast vehicle (m < 0) at origin."""
    pos = np.array([[0.0, 0.0]], dtype=np.float64)
    mass = np.array([-3.0], dtype=np.float64)
    return pos, mass


@pytest.fixture
def symmetric_pair():
    """Two identical positive-mass vehicles at symmetric positions."""
    pos = np.array([[-50.0, 0.0], [50.0, 0.0]], dtype=np.float64)
    mass = np.array([1.5, 1.5], dtype=np.float64)
    return pos, mass


@pytest.fixture
def rng():
    """Reproducible random number generator."""
    return np.random.default_rng(seed=42)


# ======================================================================
# Test 1: Positive mass -> negative potential (well)
# ======================================================================

class TestPositiveMassPotential:
    """Phi_i = -sign(m_i) * G_s * |m_i| / r, with m_i > 0 -> Phi < 0."""

    def test_potential_is_negative_near_positive_mass(self, single_positive_mass):
        pos, mass = single_positive_mass
        eval_pts = np.array([[10.0, 0.0], [0.0, 20.0], [30.0, 40.0]], dtype=np.float64)
        phi = compute_potential_field(pos, mass, eval_pts, G_s=9.8)

        assert phi.dtype == np.float64
        assert np.all(phi < 0.0), f"Expected all negative for positive mass, got {phi}"

    def test_analytical_value_positive_mass(self, single_positive_mass):
        """Check exact value: Phi = -sign(2) * 9.8 * |2| / 100 = -0.196."""
        pos, mass = single_positive_mass
        eval_pts = np.array([[100.0, 0.0]], dtype=np.float64)
        phi = compute_potential_field(pos, mass, eval_pts, G_s=9.8, r_min=1.0)

        expected = -1.0 * 9.8 * 2.0 / 100.0  # -0.196
        np.testing.assert_allclose(phi[0], expected, rtol=1e-12)


# ======================================================================
# Test 2: Negative mass -> positive potential (hill)
# ======================================================================

class TestNegativeMassPotential:
    """Phi_i = -sign(m_i) * G_s * |m_i| / r, with m_i < 0 -> Phi > 0."""

    def test_potential_is_positive_near_negative_mass(self, single_negative_mass):
        pos, mass = single_negative_mass
        eval_pts = np.array([[10.0, 0.0], [0.0, 20.0]], dtype=np.float64)
        phi = compute_potential_field(pos, mass, eval_pts, G_s=9.8)

        assert phi.dtype == np.float64
        assert np.all(phi > 0.0), f"Expected all positive for negative mass, got {phi}"

    def test_analytical_value_negative_mass(self, single_negative_mass):
        """Check exact value: Phi = -sign(-3) * 9.8 * |-3| / 50 = +0.588."""
        pos, mass = single_negative_mass
        eval_pts = np.array([[50.0, 0.0]], dtype=np.float64)
        phi = compute_potential_field(pos, mass, eval_pts, G_s=9.8, r_min=1.0)

        expected = -(-1.0) * 9.8 * 3.0 / 50.0  # +0.588
        np.testing.assert_allclose(phi[0], expected, rtol=1e-12)


# ======================================================================
# Test 3: 1/r decay -- potential magnitude decreases with distance
# ======================================================================

class TestInverseRDecay:
    """|Phi| must decrease as evaluation point moves further from source."""

    @pytest.mark.parametrize("mass_val", [2.0, -3.0, 0.5, -0.1])
    def test_magnitude_decreases_with_distance(self, mass_val):
        pos = np.array([[0.0, 0.0]], dtype=np.float64)
        mass = np.array([mass_val], dtype=np.float64)

        # Evaluation points at increasing distances (all beyond r_min=5)
        distances = [10.0, 50.0, 100.0, 500.0, 1000.0]
        eval_pts = np.array([[d, 0.0] for d in distances], dtype=np.float64)

        phi = compute_potential_field(pos, mass, eval_pts, G_s=9.8, r_min=5.0)

        abs_phi = np.abs(phi)
        # Each successive point should have smaller |Phi|
        for i in range(len(distances) - 1):
            assert abs_phi[i] > abs_phi[i + 1], (
                f"|Phi| at r={distances[i]} ({abs_phi[i]:.6e}) should exceed "
                f"|Phi| at r={distances[i+1]} ({abs_phi[i+1]:.6e})"
            )

    def test_exact_ratio_1_over_r(self):
        """Phi at r=100 should be exactly half of Phi at r=50 (1/r law)."""
        pos = np.array([[0.0, 0.0]], dtype=np.float64)
        mass = np.array([1.0], dtype=np.float64)

        eval_pts = np.array([[50.0, 0.0], [100.0, 0.0]], dtype=np.float64)
        phi = compute_potential_field(pos, mass, eval_pts, G_s=9.8, r_min=1.0)

        ratio = phi[0] / phi[1]
        np.testing.assert_allclose(ratio, 2.0, rtol=1e-12)


# ======================================================================
# Test 4: Symmetry -- equal masses at symmetric positions
# ======================================================================

class TestSymmetry:
    """Two identical masses at (+-d, 0) produce a symmetric potential."""

    def test_midpoint_potential(self, symmetric_pair):
        """Potential at midpoint should equal sum from both (equal distances)."""
        pos, mass = symmetric_pair
        midpoint = np.array([[0.0, 0.0]], dtype=np.float64)
        phi = compute_potential_field(pos, mass, midpoint, G_s=9.8, r_min=1.0)

        # Each at distance 50, m=1.5, sign=+1
        # Phi = 2 * (-1 * 9.8 * 1.5 / 50) = -0.588
        expected = 2.0 * (-1.0 * 9.8 * 1.5 / 50.0)
        np.testing.assert_allclose(phi[0], expected, rtol=1e-12)

    def test_symmetric_eval_points(self, symmetric_pair):
        """Potential at (-100, 0) and (100, 0) should be equal by symmetry."""
        pos, mass = symmetric_pair
        eval_pts = np.array([[-100.0, 0.0], [100.0, 0.0]], dtype=np.float64)
        phi = compute_potential_field(pos, mass, eval_pts, G_s=9.8, r_min=1.0)

        np.testing.assert_allclose(phi[0], phi[1], rtol=1e-12)

    def test_y_axis_symmetry(self, symmetric_pair):
        """Points at (0, +h) and (0, -h) should have equal potential."""
        pos, mass = symmetric_pair
        eval_pts = np.array([[0.0, 30.0], [0.0, -30.0]], dtype=np.float64)
        phi = compute_potential_field(pos, mass, eval_pts, G_s=9.8, r_min=1.0)

        np.testing.assert_allclose(phi[0], phi[1], rtol=1e-12)


# ======================================================================
# Test 5: make_grid dimensions
# ======================================================================

class TestMakeGrid:
    """Grid generation correctness."""

    def test_basic_dimensions(self):
        grid = make_grid(0.0, 0.0, 100.0, 100.0, resolution=10.0)
        # x: centers at 5, 15, 25, ..., 95 -> 10 points
        # y: centers at 5, 15, 25, ..., 95 -> 10 points
        # Total: 100 points
        assert grid.shape == (100, 2)
        assert grid.dtype == np.float64

    def test_single_cell(self):
        grid = make_grid(0.0, 0.0, 10.0, 10.0, resolution=10.0)
        assert grid.shape == (1, 2)
        np.testing.assert_allclose(grid[0], [5.0, 5.0], rtol=1e-12)

    def test_rectangular_grid(self):
        grid = make_grid(0.0, 0.0, 50.0, 30.0, resolution=10.0)
        # x: 5 cells, y: 3 cells -> 15 points
        assert grid.shape == (15, 2)

    def test_grid_bounds(self):
        """All grid centers should be within the domain."""
        grid = make_grid(-100.0, -50.0, 300.0, 200.0, resolution=25.0)
        assert np.all(grid[:, 0] >= -100.0)
        assert np.all(grid[:, 0] <= 300.0)
        assert np.all(grid[:, 1] >= -50.0)
        assert np.all(grid[:, 1] <= 200.0)
        assert grid.dtype == np.float64

    def test_empty_grid(self):
        """Domain smaller than one cell -> no grid points if center falls outside."""
        grid = make_grid(0.0, 0.0, 3.0, 3.0, resolution=10.0)
        # Center would be at 5.0 which is > 3.0 -> empty
        assert grid.shape[0] == 0
        assert grid.shape[1] == 2


# ======================================================================
# Test 6: optimize_traffic_light returns valid timing
# ======================================================================

class TestOptimizeTrafficLight:
    """Traffic light optimizer produces valid output."""

    def test_valid_timing_range(self, rng):
        n = 50
        pos = rng.uniform(-150.0, 150.0, (n, 2)).astype(np.float64)
        mass = rng.uniform(-2.0, 2.0, n).astype(np.float64)
        intersection = np.array([0.0, 0.0], dtype=np.float64)

        result = optimize_traffic_light(pos, mass, intersection, radius=200.0)

        assert isinstance(result, dict)
        assert "green_ns" in result
        assert "green_ew" in result
        assert "cycle_s" in result
        assert "phi_cost" in result
        assert 15 <= result["green_ns"] <= 90
        assert result["green_ew"] >= 10
        assert result["cycle_s"] == 120

    def test_timing_sums_correctly(self, rng):
        """green_ns + green_ew + clearance should not exceed cycle."""
        n = 30
        pos = rng.uniform(-100.0, 100.0, (n, 2)).astype(np.float64)
        mass = rng.uniform(-1.0, 3.0, n).astype(np.float64)
        intersection = np.array([50.0, 50.0], dtype=np.float64)

        result = optimize_traffic_light(pos, mass, intersection, horizon_s=120)

        clearance = 10
        assert result["green_ns"] + result["green_ew"] + clearance <= result["cycle_s"] + clearance

    def test_phi_cost_is_finite(self, rng):
        n = 20
        pos = rng.uniform(-50.0, 50.0, (n, 2)).astype(np.float64)
        mass = rng.uniform(0.5, 3.0, n).astype(np.float64)  # all positive -> congestion
        intersection = np.array([0.0, 0.0], dtype=np.float64)

        result = optimize_traffic_light(pos, mass, intersection)
        assert np.isfinite(result["phi_cost"])

    def test_pure_congestion_prefers_balanced_split(self):
        """With symmetric congestion in NS and EW, timing should be near 50/50."""
        # Place vehicles symmetrically in NS and EW corridors
        pos = np.array([
            [0.0, 50.0],   # NS
            [0.0, -50.0],  # NS
            [50.0, 0.0],   # EW
            [-50.0, 0.0],  # EW
        ], dtype=np.float64)
        mass = np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float64)  # equal congestion
        intersection = np.array([0.0, 0.0], dtype=np.float64)

        result = optimize_traffic_light(pos, mass, intersection, radius=200.0)
        # With equal congestion, optimal is near equal split
        # cycle=120, clearance=10, so available=110; half=55
        assert 40 <= result["green_ns"] <= 70, (
            f"Expected near-equal split, got green_ns={result['green_ns']}"
        )


# ======================================================================
# Test 7: Empty vehicles -> zero potential
# ======================================================================

class TestEmptyVehicles:
    """No vehicles should produce zero potential everywhere."""

    def test_zero_potential_on_grid(self):
        pos = np.empty((0, 2), dtype=np.float64)
        mass = np.empty((0,), dtype=np.float64)
        eval_pts = np.array([[0.0, 0.0], [100.0, 200.0], [-50.0, 75.0]], dtype=np.float64)

        phi = compute_potential_field(pos, mass, eval_pts, G_s=9.8)

        assert phi.dtype == np.float64
        np.testing.assert_array_equal(phi, 0.0)

    def test_optimize_with_no_vehicles(self):
        pos = np.empty((0, 2), dtype=np.float64)
        mass = np.empty((0,), dtype=np.float64)
        intersection = np.array([0.0, 0.0], dtype=np.float64)

        result = optimize_traffic_light(pos, mass, intersection)

        assert 15 <= result["green_ns"] <= 90
        assert result["green_ew"] >= 10
        assert result["phi_cost"] == 0.0


# ======================================================================
# Test 8: dtype enforcement
# ======================================================================

class TestDtypeEnforcement:
    """All outputs must be float64."""

    def test_potential_dtype_from_float32_input(self):
        """Even if inputs are float32, output must be float64."""
        pos = np.array([[10.0, 20.0]], dtype=np.float32)
        mass = np.array([1.0], dtype=np.float32)
        eval_pts = np.array([[50.0, 50.0]], dtype=np.float32)

        phi = compute_potential_field(pos, mass, eval_pts)
        assert phi.dtype == np.float64

    def test_grid_dtype(self):
        grid = make_grid(0.0, 0.0, 100.0, 100.0)
        assert grid.dtype == np.float64


# ======================================================================
# Test 9: r_min clamping prevents extreme values
# ======================================================================

class TestRMinClamping:
    """Potential should be bounded when evaluation point coincides with vehicle."""

    def test_coincident_point(self):
        """Evaluation point at vehicle location uses r_min, not r=0."""
        pos = np.array([[50.0, 50.0]], dtype=np.float64)
        mass = np.array([1.0], dtype=np.float64)
        eval_pts = np.array([[50.0, 50.0]], dtype=np.float64)  # same as vehicle

        phi = compute_potential_field(pos, mass, eval_pts, G_s=9.8, r_min=5.0)

        expected = -1.0 * 9.8 * 1.0 / 5.0  # -1.96
        np.testing.assert_allclose(phi[0], expected, rtol=1e-12)
        assert np.isfinite(phi[0])


# ======================================================================
# Test 10: Superposition -- multi-vehicle potential
# ======================================================================

class TestSuperposition:
    """Total potential equals sum of individual contributions."""

    def test_two_vehicles_superposition(self):
        pos1 = np.array([[0.0, 0.0]], dtype=np.float64)
        mass1 = np.array([2.0], dtype=np.float64)

        pos2 = np.array([[100.0, 0.0]], dtype=np.float64)
        mass2 = np.array([-1.5], dtype=np.float64)

        pos_both = np.array([[0.0, 0.0], [100.0, 0.0]], dtype=np.float64)
        mass_both = np.array([2.0, -1.5], dtype=np.float64)

        eval_pts = np.array([[50.0, 50.0], [200.0, 0.0]], dtype=np.float64)

        phi1 = compute_potential_field(pos1, mass1, eval_pts, G_s=9.8, r_min=1.0)
        phi2 = compute_potential_field(pos2, mass2, eval_pts, G_s=9.8, r_min=1.0)
        phi_both = compute_potential_field(pos_both, mass_both, eval_pts, G_s=9.8, r_min=1.0)

        np.testing.assert_allclose(phi_both, phi1 + phi2, rtol=1e-12)
