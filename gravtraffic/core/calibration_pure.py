"""Pure-gravity calibration test -- can gravity alone reproduce a fundamental diagram?

This module tests whether the gravitational traffic model WITHOUT any relaxation,
damping, or desired-speed term can reproduce a plausible speed-density relationship
(fundamental diagram).

The old calibration module (calibration.py) used:
    - A relaxation term: damping * (v_desired - v_i)
    - A force_scale factor of 0.01
    - A desired speed from Greenshields
These terms dominated the dynamics, making it impossible to tell whether the
gravitational model itself contributed anything.

Here we test PURE GRAVITY:
    acceleration_i = F_gravity_i / |m_i|
No relaxation. No damping. No desired speed. Only MassAssigner -> ForceEngine -> Leapfrog.

Key physics insight
-------------------
The mass formula is: m_i = sgn(v_mean - v_i) * |v_mean - v_i|^beta * rho/rho_0

At equilibrium (all vehicles at v_mean), ALL masses are zero and ALL forces vanish.
This is a fixed point. The gravitational force acts as a RESTORING force that pushes
deviating vehicles back toward the mean speed. But it does NOT determine what the mean
speed should be -- the mean speed is conserved (approximately) by the gravitational
dynamics alone.

Therefore, pure gravity can PRESERVE a speed-density relationship that is established
by initial conditions, but it cannot CREATE one from arbitrary initial conditions.
This is an important distinction.

Test protocol
-------------
1. For each density rho in [10, 20, ..., 140] veh/km:
   - Place N = rho * L vehicles uniformly on a 2 km segment
   - Set initial speed = v_free * (1 - rho/rho_jam) + small noise
   - Run pure gravitational simulation for 200 steps (20s)
   - Record final mean speed
2. Compare to Greenshields: v_expected = v_free * (1 - rho/rho_jam)
3. Grid search over (G_s, beta, softening) to find best parameters

Reference
---------
Janus Civil C-01 GravTraffic Technical Plan, Section 5 (Calibration).
Greenshields, B.D. (1935). "A Study of Traffic Capacity."
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from gravtraffic.core.simulation import GravSimulation

__all__ = [
    "run_pure_gravity_test",
    "run_generation_test",
    "pure_gravity_grid_search",
]

# ---------------------------------------------------------------------------
# Greenshields constants
# ---------------------------------------------------------------------------
V_FREE_MS: float = 33.33  # m/s (120 km/h)
RHO_JAM: float = 150.0  # veh/km
SEGMENT_LENGTH_M: float = 2000.0  # 2 km highway segment
NOISE_SIGMA: float = 1.0  # m/s -- small perturbation around equilibrium


def _greenshields_speed(rho: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Greenshields speed: v(rho) = v_free * (1 - rho / rho_jam)."""
    return np.float64(V_FREE_MS) * (1.0 - np.asarray(rho, dtype=np.float64) / RHO_JAM)


def _r_squared(
    y_true: npt.NDArray[np.float64],
    y_pred: npt.NDArray[np.float64],
) -> float:
    """Coefficient of determination. Can be negative."""
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return 1.0 - ss_res / ss_tot


def _rmse(
    y_true: npt.NDArray[np.float64],
    y_pred: npt.NDArray[np.float64],
) -> float:
    """Root mean square error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _setup_scenario(
    rho_veh_per_km: float,
    v_free: float,
    rho_jam: float,
    rng: np.random.Generator,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Create initial positions, velocities, and local densities for one density.

    Parameters
    ----------
    rho_veh_per_km : float
        Target density in vehicles per km.
    v_free : float
        Free-flow speed in m/s.
    rho_jam : float
        Jam density in veh/km.
    rng : numpy.random.Generator
        Random number generator for noise.

    Returns
    -------
    positions : ndarray, shape (N, 2)
    velocities : ndarray, shape (N, 2)
    local_densities : ndarray, shape (N,)
    """
    # Number of vehicles on our 2 km segment
    n_vehicles = max(2, int(round(rho_veh_per_km * SEGMENT_LENGTH_M / 1000.0)))

    # Uniform spacing
    spacing = SEGMENT_LENGTH_M / n_vehicles
    positions_x = np.arange(n_vehicles, dtype=np.float64) * spacing + spacing * 0.5

    # 2D positions (y = 0 for 1D highway)
    positions = np.zeros((n_vehicles, 2), dtype=np.float64)
    positions[:, 0] = positions_x

    # Initial speed near Greenshields equilibrium + small noise
    v_eq = max(0.0, v_free * (1.0 - rho_veh_per_km / rho_jam))
    noise = rng.normal(0.0, NOISE_SIGMA, size=n_vehicles)
    speeds = np.clip(v_eq + noise, 0.0, v_free).astype(np.float64)

    # 2D velocities (all motion in x-direction)
    velocities = np.zeros((n_vehicles, 2), dtype=np.float64)
    velocities[:, 0] = speeds

    # Local density: all vehicles see the same uniform density
    local_densities = np.full(n_vehicles, rho_veh_per_km, dtype=np.float64)

    return positions, velocities, local_densities


def run_pure_gravity_test(
    G_s: float,
    beta: float,
    softening: float,
    densities: list[float] | None = None,
    n_steps: int = 200,
    dt: float = 0.1,
    v_free: float = V_FREE_MS,
    rho_jam: float = RHO_JAM,
    seed: int = 42,
) -> dict:
    """Run a pure-gravity fundamental diagram test.

    For each density, creates a uniform vehicle distribution at the Greenshields
    equilibrium speed (plus noise), runs the PURE gravitational simulation
    (no relaxation, no damping, no desired speed), and records the final mean speed.

    Parameters
    ----------
    G_s : float
        Social gravitational constant.
    beta : float
        Mass-assignment exponent.
    softening : float
        Force softening length in meters.
    densities : list of float, optional
        Densities to test in veh/km. Default: [10, 20, ..., 140].
    n_steps : int
        Number of simulation steps per density. Default: 200.
    dt : float
        Fixed timestep in seconds. Default: 0.1.
    v_free : float
        Free-flow speed in m/s. Default: 33.33.
    rho_jam : float
        Jam density in veh/km. Default: 150.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        Keys:
        - 'G_s', 'beta', 'softening': parameter values
        - 'densities': ndarray of tested densities
        - 'mean_speeds': ndarray of final mean speeds (m/s)
        - 'greenshields_speeds': ndarray of analytical speeds (m/s)
        - 'r_squared': float, R^2 against Greenshields
        - 'rmse_ms': float, RMSE in m/s
        - 'monotonic': bool, True if speed decreases with density
        - 'stable': bool, True if no NaN/Inf and no negative speeds
        - 'speed_drift_pct': float, mean absolute drift from initial as %
        - 'notes': str, human-readable interpretation
    """
    if densities is None:
        densities = list(range(10, 141, 10))

    rng = np.random.default_rng(seed)
    density_arr = np.array(densities, dtype=np.float64)
    greenshields = _greenshields_speed(density_arr)

    mean_speeds = np.zeros(len(densities), dtype=np.float64)
    initial_mean_speeds = np.zeros(len(densities), dtype=np.float64)
    all_stable = True

    for idx, rho in enumerate(densities):
        # Create fresh scenario for this density
        positions, velocities, local_dens = _setup_scenario(rho, v_free, rho_jam, rng)
        initial_mean_speeds[idx] = float(np.mean(np.linalg.norm(velocities, axis=1)))

        # Create simulation with PURE GRAVITY -- no relaxation, no damping
        sim = GravSimulation(
            G_s=G_s,
            beta=beta,
            softening=softening,
            rho_scale=30.0,
            theta=0.5,
            dt=dt,
            v_max=v_free + 5.0,  # allow slight overshoot for stability detection
            adaptive_dt=False,
        )
        sim.init_vehicles(positions, velocities, local_dens)

        # Run simulation
        had_nan = False
        for _ in range(n_steps):
            result = sim.step()
            if np.any(np.isnan(result["velocities"])) or np.any(np.isinf(result["velocities"])):
                had_nan = True
                break

        if had_nan:
            mean_speeds[idx] = np.nan
            all_stable = False
        else:
            final_speeds = np.linalg.norm(sim.velocities, axis=1)
            final_mean = float(np.mean(final_speeds))
            mean_speeds[idx] = final_mean

            # Check for negative speeds (in 1D, check vx)
            if np.any(sim.velocities[:, 0] < -1.0):
                all_stable = False

    # Stability check
    stable = all_stable and not np.any(np.isnan(mean_speeds))

    # Monotonicity check: is mean speed (weakly) decreasing with density?
    if stable:
        # Allow small non-monotonicity (up to 1 m/s) due to noise
        diffs = np.diff(mean_speeds)
        monotonic = bool(np.all(diffs <= 1.0))  # allow small increases
    else:
        monotonic = False

    # R^2 and RMSE (only on finite values)
    finite_mask = np.isfinite(mean_speeds)
    if np.sum(finite_mask) >= 2:
        r2 = _r_squared(greenshields[finite_mask], mean_speeds[finite_mask])
        rmse_val = _rmse(greenshields[finite_mask], mean_speeds[finite_mask])
    else:
        r2 = float("-inf")
        rmse_val = float("inf")

    # Speed drift: how much did the mean speed change from initial conditions?
    if stable:
        drift_pct = float(
            np.mean(np.abs(mean_speeds - initial_mean_speeds))
            / max(np.mean(np.abs(initial_mean_speeds)), 1e-10)
            * 100.0
        )
    else:
        drift_pct = float("inf")

    # Interpretation notes
    notes = _interpret_result(r2, stable, monotonic, drift_pct)

    return {
        "G_s": float(G_s),
        "beta": float(beta),
        "softening": float(softening),
        "densities": density_arr,
        "mean_speeds": mean_speeds,
        "greenshields_speeds": greenshields,
        "r_squared": float(r2),
        "rmse_ms": float(rmse_val),
        "monotonic": monotonic,
        "stable": stable,
        "speed_drift_pct": drift_pct,
        "notes": notes,
    }


def _interpret_result(r2: float, stable: bool, monotonic: bool, drift_pct: float) -> str:
    """Generate human-readable interpretation of test results."""
    parts: list[str] = []

    if not stable:
        parts.append("UNSTABLE: simulation produced NaN/Inf or negative speeds.")
        return " ".join(parts)

    if r2 >= 0.90:
        parts.append(f"EXCELLENT: R^2={r2:.4f} against Greenshields.")
    elif r2 >= 0.70:
        parts.append(f"GOOD: R^2={r2:.4f} against Greenshields.")
    elif r2 >= 0.30:
        parts.append(f"WEAK: R^2={r2:.4f} -- gravity partially reproduces the diagram.")
    else:
        parts.append(f"POOR: R^2={r2:.4f} -- gravity alone does not reproduce Greenshields.")

    if monotonic:
        parts.append("Speed is monotonically decreasing with density (physically correct).")
    else:
        parts.append("WARNING: Speed is NOT monotonically decreasing with density.")

    if drift_pct < 5.0:
        parts.append(f"Speed drift {drift_pct:.1f}% -- gravity preserves initial equilibrium well.")
    elif drift_pct < 20.0:
        parts.append(f"Speed drift {drift_pct:.1f}% -- moderate perturbation from initial speeds.")
    else:
        parts.append(f"Speed drift {drift_pct:.1f}% -- large deviation from initial conditions.")

    return " ".join(parts)


def run_generation_test(
    G_s: float,
    beta: float,
    softening: float,
    densities: list[float] | None = None,
    n_steps: int = 200,
    dt: float = 0.1,
    v_free: float = V_FREE_MS,
    rho_jam: float = RHO_JAM,
    seed: int = 42,
) -> dict:
    """Test whether pure gravity can GENERATE a speed-density relationship.

    Unlike run_pure_gravity_test (which starts near Greenshields equilibrium),
    this test starts ALL vehicles at v_free regardless of density. If the
    gravitational model can generate a fundamental diagram, vehicles at high
    density should slow down and vehicles at low density should stay fast.

    This is the critical test that distinguishes PRESERVATION from GENERATION.

    Parameters
    ----------
    Same as run_pure_gravity_test.

    Returns
    -------
    dict
        Same keys as run_pure_gravity_test, plus:
        - 'initial_speeds': ndarray of initial mean speeds (all ~v_free)
        - 'speed_reduction': ndarray of (v_free - final_mean_speed) per density
        - 'generates_fd': bool, True if speed decreases with density
    """
    if densities is None:
        densities = list(range(10, 141, 10))

    rng = np.random.default_rng(seed)
    density_arr = np.array(densities, dtype=np.float64)
    greenshields = _greenshields_speed(density_arr)

    mean_speeds = np.zeros(len(densities), dtype=np.float64)
    initial_mean_speeds = np.zeros(len(densities), dtype=np.float64)
    all_stable = True

    for idx, rho in enumerate(densities):
        n_vehicles = max(2, int(round(rho * SEGMENT_LENGTH_M / 1000.0)))
        spacing = SEGMENT_LENGTH_M / n_vehicles
        positions_x = np.arange(n_vehicles, dtype=np.float64) * spacing + spacing * 0.5

        positions = np.zeros((n_vehicles, 2), dtype=np.float64)
        positions[:, 0] = positions_x

        # KEY DIFFERENCE: all vehicles start at v_free + small noise
        noise = rng.normal(0.0, NOISE_SIGMA, size=n_vehicles)
        speeds = np.clip(v_free + noise, 0.0, v_free + 3.0).astype(np.float64)

        velocities = np.zeros((n_vehicles, 2), dtype=np.float64)
        velocities[:, 0] = speeds

        local_dens = np.full(n_vehicles, rho, dtype=np.float64)
        initial_mean_speeds[idx] = float(np.mean(speeds))

        sim = GravSimulation(
            G_s=G_s,
            beta=beta,
            softening=softening,
            rho_scale=30.0,
            theta=0.5,
            dt=dt,
            v_max=v_free + 5.0,
            adaptive_dt=False,
        )
        sim.init_vehicles(positions, velocities, local_dens)

        had_nan = False
        for _ in range(n_steps):
            result = sim.step()
            if np.any(np.isnan(result["velocities"])) or np.any(np.isinf(result["velocities"])):
                had_nan = True
                break

        if had_nan:
            mean_speeds[idx] = np.nan
            all_stable = False
        else:
            final_speeds = np.linalg.norm(sim.velocities, axis=1)
            mean_speeds[idx] = float(np.mean(final_speeds))
            if np.any(sim.velocities[:, 0] < -1.0):
                all_stable = False

    stable = all_stable and not np.any(np.isnan(mean_speeds))

    # Speed reduction from v_free -- does gravity slow vehicles at high density?
    speed_reduction = v_free - mean_speeds

    # Does gravity generate a fundamental diagram?
    # Criterion: speed reduction should be LARGER at high density
    if stable:
        diffs = np.diff(mean_speeds)
        monotonic = bool(np.all(diffs <= 1.0))
        # More stringent: speed must actually decrease significantly
        generates_fd = (
            monotonic and (mean_speeds[0] - mean_speeds[-1]) > 5.0  # at least 5 m/s range
        )
    else:
        monotonic = False
        generates_fd = False

    finite_mask = np.isfinite(mean_speeds)
    if np.sum(finite_mask) >= 2:
        r2 = _r_squared(greenshields[finite_mask], mean_speeds[finite_mask])
        rmse_val = _rmse(greenshields[finite_mask], mean_speeds[finite_mask])
    else:
        r2 = float("-inf")
        rmse_val = float("inf")

    drift_pct = (
        float(
            np.mean(np.abs(mean_speeds - initial_mean_speeds))
            / max(np.mean(np.abs(initial_mean_speeds)), 1e-10)
            * 100.0
        )
        if stable
        else float("inf")
    )

    # Interpretation
    parts: list[str] = []
    if not stable:
        parts.append("UNSTABLE in generation test.")
    elif generates_fd:
        parts.append(
            f"GENERATION TEST PASSED: R^2={r2:.4f}. "
            "Gravity creates density-dependent speed reduction."
        )
    else:
        parts.append(
            f"GENERATION TEST FAILED: R^2={r2:.4f}. "
            "Starting at v_free, gravity does NOT produce different speeds "
            "for different densities. This confirms that pure gravity PRESERVES "
            "but does not GENERATE a fundamental diagram."
        )

    return {
        "G_s": float(G_s),
        "beta": float(beta),
        "softening": float(softening),
        "densities": density_arr,
        "mean_speeds": mean_speeds,
        "greenshields_speeds": greenshields,
        "initial_speeds": initial_mean_speeds,
        "speed_reduction": speed_reduction,
        "r_squared": float(r2),
        "rmse_ms": float(rmse_val),
        "monotonic": monotonic,
        "stable": stable,
        "generates_fd": bool(generates_fd),
        "speed_drift_pct": drift_pct,
        "notes": " ".join(parts),
    }


def pure_gravity_grid_search(seed: int = 42) -> list[dict]:
    """Run grid search over G_s, beta, softening parameters.

    Tests a wide range of parameter combinations and returns results sorted
    by R^2 descending.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    list of dict
        Each dict is the output of run_pure_gravity_test for one parameter
        combination. Sorted by R^2 descending.
    """
    G_s_values = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    beta_values = [0.3, 0.5, 0.7, 1.0, 1.5]
    softening_values = [2.0, 5.0, 10.0, 20.0]

    results: list[dict] = []

    len(G_s_values) * len(beta_values) * len(softening_values)
    count = 0

    for G_s in G_s_values:
        for beta in beta_values:
            for softening in softening_values:
                count += 1
                try:
                    result = run_pure_gravity_test(
                        G_s=G_s,
                        beta=beta,
                        softening=softening,
                        seed=seed,
                    )
                    results.append(result)
                except Exception as exc:
                    # Record the failure but continue the search
                    results.append(
                        {
                            "G_s": float(G_s),
                            "beta": float(beta),
                            "softening": float(softening),
                            "densities": np.array([]),
                            "mean_speeds": np.array([]),
                            "greenshields_speeds": np.array([]),
                            "r_squared": float("-inf"),
                            "rmse_ms": float("inf"),
                            "monotonic": False,
                            "stable": False,
                            "speed_drift_pct": float("inf"),
                            "notes": f"EXCEPTION: {exc!r}",
                        }
                    )

    # Sort by R^2 descending
    results.sort(key=lambda r: r["r_squared"], reverse=True)

    return results


def print_grid_search_report(results: list[dict], top_n: int = 10) -> None:
    """Print a human-readable report of grid search results.

    Parameters
    ----------
    results : list of dict
        Output of pure_gravity_grid_search.
    top_n : int
        Number of top results to show in detail.
    """
    print("=" * 80)
    print("PURE GRAVITY CALIBRATION -- GRID SEARCH REPORT")
    print("=" * 80)
    print(f"Total configurations tested: {len(results)}")

    stable_count = sum(1 for r in results if r["stable"])
    monotonic_count = sum(1 for r in results if r["monotonic"])
    print(f"Stable (no NaN/negative): {stable_count}/{len(results)}")
    print(f"Monotonic speed-density: {monotonic_count}/{len(results)}")

    # Best R^2
    if results:
        best = results[0]
        print(f"\nBest R^2: {best['r_squared']:.6f}")
        print(f"  G_s={best['G_s']}, beta={best['beta']}, softening={best['softening']}")
        print(f"  RMSE={best['rmse_ms']:.4f} m/s ({best['rmse_ms'] * 3.6:.2f} km/h)")
        print(f"  Monotonic: {best['monotonic']}, Stable: {best['stable']}")
        print(f"  Drift: {best['speed_drift_pct']:.1f}%")

    print(
        f"\n{'Rank':<6} {'G_s':>6} {'beta':>6} {'soft':>6} "
        f"{'R^2':>10} {'RMSE(m/s)':>10} {'Mon':>5} {'Stab':>5} {'Drift%':>8}"
    )
    print("-" * 80)

    for i, r in enumerate(results[:top_n]):
        print(
            f"{i + 1:<6} {r['G_s']:>6.1f} {r['beta']:>6.2f} {r['softening']:>6.1f} "
            f"{r['r_squared']:>10.6f} {r['rmse_ms']:>10.4f} "
            f"{'Y' if r['monotonic'] else 'N':>5} "
            f"{'Y' if r['stable'] else 'N':>5} "
            f"{r['speed_drift_pct']:>8.1f}"
        )
    print("-" * 80)

    # Scientific assessment
    print("\n" + "=" * 80)
    print("SCIENTIFIC ASSESSMENT")
    print("=" * 80)

    if results and results[0]["r_squared"] >= 0.90:
        print("RESULT: Pure gravity CAN reproduce a Greenshields-like fundamental diagram.")
        print("The gravitational model alone is sufficient for speed-density relationships.")
    elif results and results[0]["r_squared"] >= 0.70:
        print("RESULT: Pure gravity produces a REASONABLE fundamental diagram (R^2 >= 0.70).")
        print("The model captures the qualitative shape but may need enrichment for precision.")
    elif results and results[0]["r_squared"] >= 0.30:
        print("RESULT: Pure gravity produces a WEAK fundamental diagram (0.30 <= R^2 < 0.70).")
        print("The gravitational model captures some trend but is insufficient alone.")
        _print_enrichment_suggestions()
    else:
        print("RESULT: Pure gravity CANNOT reproduce a plausible fundamental diagram.")
        print("This is expected and physically justified. Here is why:")
        print()
        _print_physics_explanation()
        _print_enrichment_suggestions()

    print("=" * 80)


def _print_physics_explanation() -> None:
    """Print the physics explanation for why pure gravity may not suffice."""
    print("PHYSICS EXPLANATION:")
    print("  The mass formula m_i = sgn(v_mean - v_i) * |v_mean - v_i|^beta * rho/rho_0")
    print("  has an important property: at equilibrium (all v_i = v_mean), all masses")
    print("  are ZERO and all gravitational forces vanish. This means:")
    print()
    print("  1. The equilibrium is a FIXED POINT of the gravitational dynamics.")
    print("  2. Gravity acts as a RESTORING force that pushes vehicles back toward")
    print("     the mean speed, but does not determine WHAT the mean speed should be.")
    print("  3. The mean speed is approximately CONSERVED by pure gravity -- it depends")
    print("     entirely on initial conditions.")
    print("  4. Therefore, if we START near Greenshields equilibrium, we STAY near it.")
    print("     This is preservation of initial conditions, not emergence from gravity.")
    print()
    print("  The high R^2 we observe (if any) reflects the system's stability, not")
    print("  the gravitational model's ability to generate a fundamental diagram.")
    print()


def _print_enrichment_suggestions() -> None:
    """Print minimal enrichment suggestions."""
    print("MINIMAL ENRICHMENT NEEDED:")
    print("  The simplest physically justified addition is a DRAG/FRICTION term:")
    print()
    print("    a_drag = -gamma * v_i")
    print()
    print("  where gamma > 0 is a friction coefficient. This is NOT a car-following")
    print("  rule -- it represents aerodynamic drag and rolling resistance, which are")
    print("  real physical forces. Combined with gravity:")
    print()
    print("    a_i = F_gravity_i / |m_i| - gamma * v_i")
    print()
    print("  At equilibrium (F_gravity = 0), this gives v_i -> 0, which is wrong.")
    print("  A better enrichment is a CONSTANT DRIVE FORCE (engine) plus drag:")
    print()
    print("    a_i = F_gravity_i / |m_i| + F_drive - gamma * v_i")
    print()
    print("  At equilibrium: v_eq = F_drive / gamma")
    print("  Making F_drive density-dependent: F_drive(rho) = gamma * v_free * (1 - rho/rho_jam)")
    print("  This recovers Greenshields exactly at equilibrium, with gravity providing")
    print("  the inter-vehicle interaction dynamics (clustering, platoon formation, etc.)")
    print()
    print("  This enrichment is physically motivated (engine + drag) and minimal --")
    print("  it does not introduce any car-following logic or desired-speed tracking.")
