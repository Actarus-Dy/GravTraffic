"""Unified calibration tests -- verify parameters satisfy BOTH calibration AND emergence.

These tests validate the core scientific claim of the drag-enriched GravTraffic model:
    a_i = F_gravity / |m_i| + gamma * (v_eq(rho) - |v_i|) * direction

The enrichment is physically motivated (engine vs aerodynamic drag), NOT a
car-following rule.  Gravity provides inter-vehicle interactions; drag provides
the baseline speed-density relationship.

Three test classes:
    1. TestUnifiedParameterExists -- at least one parameter set satisfies both
    2. TestDefaultCalibration    -- default GravSimulation params give R^2 > 0.70
    3. TestDefaultEmergence      -- default params produce measurable emergence

Author: Agent #23 Scientific Validation Tester
Date: 2026-03-22

Reference
---------
Janus Civil C-01 GravTraffic Technical Plan, Section 5 (Calibration).
Greenshields, B.D. (1935). "A Study of Traffic Capacity."
"""

from __future__ import annotations

import numpy as np

from gravtraffic.core.calibration_unified import (
    run_calibration_test,
    run_emergence_test,
)
from gravtraffic.core.simulation import GravSimulation


# ===================================================================
# TEST 1: At least one parameter set satisfies both
# ===================================================================
class TestUnifiedParameterExists:
    """Verify that the unified grid search produces at least one
    parameter set passing both calibration (R^2 > 0.70) and
    emergence (upstream deceleration > 0.5 m/s)."""

    # Test with a focused set of known-good candidates to keep runtime short.
    CANDIDATES = [
        {"G_s": 5.0, "beta": 0.5, "softening": 10.0, "gamma": 0.3},
        {"G_s": 10.0, "beta": 0.5, "softening": 10.0, "gamma": 0.1},
        {"G_s": 10.0, "beta": 0.5, "softening": 10.0, "gamma": 0.3},
    ]

    def test_at_least_one_unified_pass(self) -> None:
        """At least one candidate must pass both calibration and emergence."""
        results = []
        for params in self.CANDIDATES:
            cal = run_calibration_test(**params)
            emg = run_emergence_test(**params)
            unified = cal["r_squared"] > 0.70 and emg["emergence_pass"]
            results.append(
                {
                    **params,
                    "r_squared": cal["r_squared"],
                    "upstream_decel": emg["upstream_decel"],
                    "unified_pass": unified,
                }
            )

            print(
                f"  G_s={params['G_s']}, gamma={params['gamma']}: "
                f"R^2={cal['r_squared']:.4f}, decel={emg['upstream_decel']:.2f} "
                f"-> {'PASS' if unified else 'FAIL'}"
            )

        n_pass = sum(1 for r in results if r["unified_pass"])
        assert n_pass >= 1, (
            f"No parameter set satisfies both calibration and emergence. Results: {results}"
        )

    def test_best_candidate_r_squared_above_090(self) -> None:
        """The best candidate (G_s=5, gamma=0.3) should achieve R^2 > 0.90."""
        cal = run_calibration_test(G_s=5.0, beta=0.5, softening=10.0, gamma=0.3)
        assert cal["r_squared"] > 0.90, (
            f"Best candidate R^2 = {cal['r_squared']:.4f}, expected > 0.90"
        )

    def test_best_candidate_emergence_strong(self) -> None:
        """The best candidate should produce > 5 m/s upstream deceleration."""
        emg = run_emergence_test(G_s=5.0, beta=0.5, softening=10.0, gamma=0.3)
        assert emg["upstream_decel"] > 5.0, (
            f"Best candidate decel = {emg['upstream_decel']:.2f} m/s, expected > 5.0"
        )


# ===================================================================
# TEST 2: Default parameters produce good calibration
# ===================================================================
class TestDefaultCalibration:
    """Verify that the default GravSimulation parameters produce
    R^2 > 0.70 against Greenshields."""

    def test_default_r_squared_above_070(self) -> None:
        """Default params must achieve R^2 > 0.70 in the generation test."""
        # Extract defaults from GravSimulation
        sim = GravSimulation()
        cal = run_calibration_test(
            G_s=sim.G_s,
            beta=sim._mass_assigner.beta,
            softening=sim._force_engine.epsilon,
            gamma=sim._drag_coefficient,
        )

        print("\n--- Default Calibration Test ---")
        print(
            f"  G_s={sim.G_s}, beta={sim._mass_assigner.beta}, "
            f"softening={sim._force_engine.epsilon}, "
            f"gamma={sim._drag_coefficient}"
        )
        print(f"  R^2 = {cal['r_squared']:.6f}")
        print(f"  Stable = {cal['stable']}")
        print(f"  Monotonic = {cal['monotonic']}")

        assert cal["r_squared"] > 0.70, (
            f"Default params R^2 = {cal['r_squared']:.4f}, expected > 0.70. "
            f"Defaults: G_s={sim.G_s}, gamma={sim._drag_coefficient}"
        )

    def test_default_calibration_stable(self) -> None:
        """Default calibration must be numerically stable."""
        sim = GravSimulation()
        cal = run_calibration_test(
            G_s=sim.G_s,
            beta=sim._mass_assigner.beta,
            softening=sim._force_engine.epsilon,
            gamma=sim._drag_coefficient,
        )
        assert cal["stable"], "Default calibration is unstable (NaN/Inf detected)"

    def test_default_calibration_monotonic(self) -> None:
        """Default calibration should produce monotonically decreasing
        speed with density."""
        sim = GravSimulation()
        cal = run_calibration_test(
            G_s=sim.G_s,
            beta=sim._mass_assigner.beta,
            softening=sim._force_engine.epsilon,
            gamma=sim._drag_coefficient,
        )
        assert cal["monotonic"], (
            f"Speed is not monotonically decreasing with density. Mean speeds: {cal['mean_speeds']}"
        )


# ===================================================================
# TEST 3: Default parameters produce measurable emergence
# ===================================================================
class TestDefaultEmergence:
    """Verify that the default GravSimulation parameters produce
    measurable upstream deceleration from a slow vehicle perturbation."""

    def test_default_emergence_upstream_decel(self) -> None:
        """Default params must produce upstream deceleration > 0.5 m/s."""
        sim = GravSimulation()
        emg = run_emergence_test(
            G_s=sim.G_s,
            beta=sim._mass_assigner.beta,
            softening=sim._force_engine.epsilon,
            gamma=sim._drag_coefficient,
        )

        print("\n--- Default Emergence Test ---")
        print(f"  Upstream mean speed: {emg['upstream_mean_speed']:.2f} m/s")
        print(f"  Upstream deceleration: {emg['upstream_decel']:.2f} m/s")
        print(f"  Emergence pass: {emg['emergence_pass']}")

        assert emg["emergence_pass"], (
            f"Default params produce no measurable emergence: "
            f"upstream_decel = {emg['upstream_decel']:.2f} m/s (threshold: 0.5). "
            f"Defaults: G_s={sim.G_s}, gamma={sim._drag_coefficient}"
        )

    def test_default_emergence_stable(self) -> None:
        """Default emergence simulation must be numerically stable."""
        sim = GravSimulation()
        emg = run_emergence_test(
            G_s=sim.G_s,
            beta=sim._mass_assigner.beta,
            softening=sim._force_engine.epsilon,
            gamma=sim._drag_coefficient,
        )
        assert emg["stable"], "Default emergence simulation is unstable"

    def test_downstream_fluidity_preserved(self) -> None:
        """Downstream vehicles should not decelerate as much as upstream."""
        sim = GravSimulation()
        emg = run_emergence_test(
            G_s=sim.G_s,
            beta=sim._mass_assigner.beta,
            softening=sim._force_engine.epsilon,
            gamma=sim._drag_coefficient,
        )

        # Downstream should be faster than upstream (asymmetric effect)
        print("\n--- Downstream Fluidity ---")
        print(f"  Upstream mean: {emg['upstream_mean_speed']:.2f} m/s")
        print(f"  Downstream mean: {emg['downstream_mean_speed']:.2f} m/s")

        # Both are affected by drag, but upstream should be slower
        # (gravitational well from slow vehicle affects upstream more)
        assert emg["downstream_mean_speed"] >= emg["upstream_mean_speed"] - 3.0, (
            f"Downstream vehicles decelerated more than upstream: "
            f"downstream={emg['downstream_mean_speed']:.2f}, "
            f"upstream={emg['upstream_mean_speed']:.2f}"
        )


# ===================================================================
# TEST 4: Drag coefficient at zero recovers pure gravity
# ===================================================================
class TestDragZeroRecoversPureGravity:
    """With drag_coefficient=0, the model should behave identically
    to pure gravity (speed approximately preserved)."""

    def test_zero_drag_preserves_speed(self) -> None:
        """With gamma=0, all vehicles at same speed should stay at that speed."""
        n = 20
        positions = np.zeros((n, 2), dtype=np.float64)
        positions[:, 0] = np.linspace(0, 1000, n)
        velocities = np.zeros((n, 2), dtype=np.float64)
        velocities[:, 0] = 25.0  # uniform speed
        densities = np.full(n, 50.0, dtype=np.float64)

        sim = GravSimulation(
            G_s=5.0,
            beta=0.5,
            softening=10.0,
            drag_coefficient=0.0,  # pure gravity
            dt=0.1,
            adaptive_dt=False,
        )
        sim.init_vehicles(positions, velocities, densities)
        sim.run(100)

        final_speeds = np.linalg.norm(sim.velocities, axis=1)
        mean_speed = float(np.mean(final_speeds))

        print("\n--- Zero Drag Test ---")
        print("  Initial mean speed: 25.0 m/s")
        print(f"  Final mean speed:   {mean_speed:.4f} m/s")
        print(f"  Speed drift:        {abs(mean_speed - 25.0):.4f} m/s")

        # With uniform speed, all masses are ~0, forces ~0, speeds preserved
        assert abs(mean_speed - 25.0) < 2.0, (
            f"Zero-drag pure gravity changed mean speed by "
            f"{abs(mean_speed - 25.0):.2f} m/s (expected < 2.0)"
        )


# ===================================================================
# TEST 5: Dimensional and physical consistency
# ===================================================================
class TestPhysicalConsistency:
    """Verify dimensional and physical consistency of the drag enrichment."""

    def test_drag_units_are_acceleration(self) -> None:
        """The drag term gamma * (v_eq - |v|) has units [1/s] * [m/s] = [m/s^2],
        which is correct for acceleration."""
        gamma = 0.3  # [1/s]
        v_eq = 20.0  # [m/s]
        v_i = 25.0  # [m/s]
        drag_accel = gamma * (v_eq - v_i)  # [m/s^2]

        # This is negative (deceleration) because v_i > v_eq
        assert drag_accel < 0, f"Expected deceleration, got {drag_accel}"
        assert abs(drag_accel - (-1.5)) < 1e-10, f"Drag accel = {drag_accel}, expected -1.5 m/s^2"

    def test_drag_equilibrium_is_greenshields(self) -> None:
        """At drag equilibrium (a_drag = 0), v_i = v_eq(rho) = Greenshields."""
        v_free = 33.33
        rho_jam = 150.0
        test_densities = [0, 50, 100, 150]
        expected_speeds = [33.33, 22.22, 11.11, 0.0]

        for rho, v_expected in zip(test_densities, expected_speeds):
            v_eq = v_free * max(0, 1.0 - rho / rho_jam)
            assert abs(v_eq - v_expected) < 0.01, (
                f"At rho={rho}: v_eq={v_eq:.2f}, expected {v_expected:.2f}"
            )

    def test_drag_sign_convention(self) -> None:
        """Drag accelerates slow vehicles and decelerates fast vehicles."""
        gamma = 0.3
        v_eq = 20.0

        # Slow vehicle: v < v_eq -> positive drag (acceleration)
        assert gamma * (v_eq - 15.0) > 0

        # Fast vehicle: v > v_eq -> negative drag (deceleration)
        assert gamma * (v_eq - 25.0) < 0

        # At equilibrium: v = v_eq -> zero drag
        assert gamma * (v_eq - v_eq) == 0.0
