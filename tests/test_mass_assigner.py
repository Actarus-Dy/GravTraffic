"""Tests for gravtraffic.core.mass_assigner.MassAssigner.

Covers:
    1. Vehicle slower than v_mean  ->  positive mass
    2. Vehicle faster than v_mean  ->  negative mass
    3. Vehicle at v_mean           ->  mass ~ 0 (neutral)
    4. beta parameter affects magnitude correctly
    5. rho_scale normalisation works
    6. Vectorization stress test (1 M vehicles)
    7. classify returns correct labels
    8. Constructor validation (edge cases)
"""

from __future__ import annotations

import numpy as np
import pytest

from gravtraffic.core.mass_assigner import MassAssigner

# =====================================================================
# Fixtures
# =====================================================================


@pytest.fixture
def default_assigner() -> MassAssigner:
    """MassAssigner with default parameters (beta=1, rho_scale=30)."""
    return MassAssigner(beta=1.0, rho_scale=30.0)


# =====================================================================
# 1. Slow vehicle -> positive mass
# =====================================================================


class TestSlowVehicle:
    """A vehicle slower than v_mean must receive positive mass."""

    def test_single_slow_vehicle(self, default_assigner: MassAssigner) -> None:
        speeds = np.array([20.0], dtype=np.float64)
        v_mean = 30.0
        densities = np.array([30.0], dtype=np.float64)

        masses = default_assigner.assign(speeds, v_mean, densities)

        # delta = 30 - 20 = 10;  m = 10 * 30/30 = 10
        assert masses.dtype == np.float64
        assert masses[0] == pytest.approx(10.0, rel=1e-12)
        assert masses[0] > 0.0

    def test_multiple_slow_vehicles(self, default_assigner: MassAssigner) -> None:
        speeds = np.array([10.0, 15.0, 25.0], dtype=np.float64)
        v_mean = 30.0
        densities = np.full(3, 30.0, dtype=np.float64)

        masses = default_assigner.assign(speeds, v_mean, densities)

        np.testing.assert_array_less(0.0, masses)  # all positive


# =====================================================================
# 2. Fast vehicle -> negative mass
# =====================================================================


class TestFastVehicle:
    """A vehicle faster than v_mean must receive negative mass."""

    def test_single_fast_vehicle(self, default_assigner: MassAssigner) -> None:
        speeds = np.array([40.0], dtype=np.float64)
        v_mean = 30.0
        densities = np.array([30.0], dtype=np.float64)

        masses = default_assigner.assign(speeds, v_mean, densities)

        # delta = 30 - 40 = -10;  m = -10 * 30/30 = -10
        assert masses[0] == pytest.approx(-10.0, rel=1e-12)
        assert masses[0] < 0.0

    def test_multiple_fast_vehicles(self, default_assigner: MassAssigner) -> None:
        speeds = np.array([35.0, 40.0, 50.0], dtype=np.float64)
        v_mean = 30.0
        densities = np.full(3, 30.0, dtype=np.float64)

        masses = default_assigner.assign(speeds, v_mean, densities)

        np.testing.assert_array_less(masses, 0.0)  # all negative


# =====================================================================
# 3. Vehicle at v_mean -> neutral (mass ~ 0)
# =====================================================================


class TestNeutralVehicle:
    """A vehicle exactly at v_mean should have mass = 0."""

    def test_exact_mean_speed(self, default_assigner: MassAssigner) -> None:
        speeds = np.array([30.0], dtype=np.float64)
        v_mean = 30.0
        densities = np.array([30.0], dtype=np.float64)

        masses = default_assigner.assign(speeds, v_mean, densities)

        assert masses[0] == pytest.approx(0.0, abs=1e-15)

    def test_near_mean_speed_classified_neutral(self, default_assigner: MassAssigner) -> None:
        speeds = np.array([29.95, 30.0, 30.05], dtype=np.float64)
        v_mean = 30.0
        densities = np.full(3, 30.0, dtype=np.float64)

        masses = default_assigner.assign(speeds, v_mean, densities)
        labels = default_assigner.classify(masses)

        np.testing.assert_array_equal(labels, ["neutral", "neutral", "neutral"])


# =====================================================================
# 4. Beta parameter affects magnitude
# =====================================================================


class TestBetaParameter:
    """The exponent beta must correctly scale the mass magnitude."""

    def test_beta_one_linear(self) -> None:
        assigner = MassAssigner(beta=1.0, rho_scale=1.0)
        speeds = np.array([20.0], dtype=np.float64)
        v_mean = 30.0
        densities = np.array([1.0], dtype=np.float64)

        masses = assigner.assign(speeds, v_mean, densities)

        # |delta|^1 = 10
        assert masses[0] == pytest.approx(10.0, rel=1e-12)

    def test_beta_two_quadratic(self) -> None:
        assigner = MassAssigner(beta=2.0, rho_scale=1.0)
        speeds = np.array([20.0], dtype=np.float64)
        v_mean = 30.0
        densities = np.array([1.0], dtype=np.float64)

        masses = assigner.assign(speeds, v_mean, densities)

        # |delta|^2 = 100, sign positive
        assert masses[0] == pytest.approx(100.0, rel=1e-12)

    def test_beta_half_sqrt(self) -> None:
        assigner = MassAssigner(beta=0.5, rho_scale=1.0)
        speeds = np.array([21.0], dtype=np.float64)
        v_mean = 30.0
        densities = np.array([1.0], dtype=np.float64)

        masses = assigner.assign(speeds, v_mean, densities)

        expected = np.sqrt(9.0)  # |30 - 21|^0.5 = 3.0
        assert masses[0] == pytest.approx(expected, rel=1e-12)

    def test_beta_zero_sign_only(self) -> None:
        assigner = MassAssigner(beta=0.0, rho_scale=1.0)
        speeds = np.array([20.0, 40.0], dtype=np.float64)
        v_mean = 30.0
        densities = np.array([1.0, 1.0], dtype=np.float64)

        masses = assigner.assign(speeds, v_mean, densities)

        # |delta|^0 = 1 for nonzero delta, so mass = +/-1
        assert masses[0] == pytest.approx(1.0, rel=1e-12)
        assert masses[1] == pytest.approx(-1.0, rel=1e-12)

    def test_negative_beta_raises(self) -> None:
        with pytest.raises(ValueError, match="beta must be non-negative"):
            MassAssigner(beta=-0.5)


# =====================================================================
# 5. rho_scale normalisation
# =====================================================================


class TestRhoScale:
    """Density normalisation by rho_scale must divide correctly."""

    def test_density_doubles_mass(self) -> None:
        assigner = MassAssigner(beta=1.0, rho_scale=30.0)
        speeds = np.array([20.0, 20.0], dtype=np.float64)
        v_mean = 30.0
        densities = np.array([30.0, 60.0], dtype=np.float64)

        masses = assigner.assign(speeds, v_mean, densities)

        # Both have delta=10.  First: 10*30/30=10, Second: 10*60/30=20
        assert masses[0] == pytest.approx(10.0, rel=1e-12)
        assert masses[1] == pytest.approx(20.0, rel=1e-12)

    def test_rho_scale_halves_mass(self) -> None:
        a1 = MassAssigner(beta=1.0, rho_scale=30.0)
        a2 = MassAssigner(beta=1.0, rho_scale=60.0)
        speeds = np.array([20.0], dtype=np.float64)
        v_mean = 30.0
        densities = np.array([30.0], dtype=np.float64)

        m1 = a1.assign(speeds, v_mean, densities)
        m2 = a2.assign(speeds, v_mean, densities)

        assert m1[0] / m2[0] == pytest.approx(2.0, rel=1e-12)

    def test_zero_density_gives_zero_mass(self, default_assigner: MassAssigner) -> None:
        speeds = np.array([20.0], dtype=np.float64)
        v_mean = 30.0
        densities = np.array([0.0], dtype=np.float64)

        masses = default_assigner.assign(speeds, v_mean, densities)

        assert masses[0] == pytest.approx(0.0, abs=1e-15)

    def test_non_positive_rho_scale_raises(self) -> None:
        with pytest.raises(ValueError, match="rho_scale must be positive"):
            MassAssigner(rho_scale=0.0)
        with pytest.raises(ValueError, match="rho_scale must be positive"):
            MassAssigner(rho_scale=-10.0)


# =====================================================================
# 6. Vectorization stress test (1 M vehicles)
# =====================================================================


class TestVectorization:
    """Ensure the computation is vectorized and can handle large arrays."""

    def test_one_million_vehicles(self, default_assigner: MassAssigner) -> None:
        rng = np.random.default_rng(seed=42)
        n = 1_000_000
        speeds = rng.uniform(10.0, 50.0, size=n).astype(np.float64)
        v_mean = 30.0
        densities = rng.uniform(5.0, 60.0, size=n).astype(np.float64)

        masses = default_assigner.assign(speeds, v_mean, densities)

        assert masses.shape == (n,)
        assert masses.dtype == np.float64
        # Sanity: mix of positive and negative masses expected
        assert np.any(masses > 0.0)
        assert np.any(masses < 0.0)

    def test_classify_one_million(self, default_assigner: MassAssigner) -> None:
        rng = np.random.default_rng(seed=99)
        n = 1_000_000
        masses = rng.uniform(-5.0, 5.0, size=n).astype(np.float64)

        labels = default_assigner.classify(masses)

        assert labels.shape == (n,)
        assert set(np.unique(labels)).issubset({"slow", "fast", "neutral"})


# =====================================================================
# 7. classify returns correct labels
# =====================================================================


class TestClassify:
    """Verify classification thresholds."""

    def test_positive_above_threshold(self, default_assigner: MassAssigner) -> None:
        masses = np.array([0.5, 1.0, 10.0], dtype=np.float64)
        labels = default_assigner.classify(masses)
        np.testing.assert_array_equal(labels, ["slow", "slow", "slow"])

    def test_negative_below_threshold(self, default_assigner: MassAssigner) -> None:
        masses = np.array([-0.5, -1.0, -10.0], dtype=np.float64)
        labels = default_assigner.classify(masses)
        np.testing.assert_array_equal(labels, ["fast", "fast", "fast"])

    def test_neutral_zone(self, default_assigner: MassAssigner) -> None:
        masses = np.array([0.0, 0.05, -0.05, 0.1, -0.1], dtype=np.float64)
        labels = default_assigner.classify(masses)
        np.testing.assert_array_equal(
            labels, ["neutral", "neutral", "neutral", "neutral", "neutral"]
        )

    def test_boundary_just_above(self, default_assigner: MassAssigner) -> None:
        eps = 1e-10
        masses = np.array([0.1 + eps, -(0.1 + eps)], dtype=np.float64)
        labels = default_assigner.classify(masses)
        np.testing.assert_array_equal(labels, ["slow", "fast"])

    def test_empty_array(self, default_assigner: MassAssigner) -> None:
        masses = np.array([], dtype=np.float64)
        labels = default_assigner.classify(masses)
        assert labels.shape == (0,)


# =====================================================================
# 8. End-to-end: assign + classify pipeline
# =====================================================================


class TestEndToEnd:
    """Full pipeline: speeds -> masses -> labels."""

    def test_mixed_population(self, default_assigner: MassAssigner) -> None:
        # Slow, fast, and at-mean vehicles
        speeds = np.array([20.0, 40.0, 30.0], dtype=np.float64)
        v_mean = 30.0
        densities = np.full(3, 30.0, dtype=np.float64)

        masses = default_assigner.assign(speeds, v_mean, densities)
        labels = default_assigner.classify(masses)

        assert labels[0] == "slow"
        assert labels[1] == "fast"
        assert labels[2] == "neutral"

    def test_analytical_values(self) -> None:
        """Check exact analytical result with beta=2, rho_scale=10."""
        assigner = MassAssigner(beta=2.0, rho_scale=10.0)
        speeds = np.array([25.0, 35.0], dtype=np.float64)
        v_mean = 30.0
        densities = np.array([20.0, 40.0], dtype=np.float64)

        masses = assigner.assign(speeds, v_mean, densities)

        # Vehicle 0: delta=5, |5|^2=25, sign=+1, rho/rho0=2.0 -> +50.0
        assert masses[0] == pytest.approx(50.0, rel=1e-12)
        # Vehicle 1: delta=-5, |5|^2=25, sign=-1, rho/rho0=4.0 -> -100.0
        assert masses[1] == pytest.approx(-100.0, rel=1e-12)
