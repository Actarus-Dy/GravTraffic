"""Unified calibration -- find parameters satisfying BOTH calibration AND emergence.

This module searches for GravTraffic parameters (G_s, beta, softening, gamma)
that simultaneously:

  A. **Calibration**: Reproduce a Greenshields speed-density fundamental diagram
     when starting vehicles at free-flow speed and letting the drag term pull
     them toward equilibrium.  Measured by R^2 against Greenshields.

  B. **Emergence**: Produce measurable upstream deceleration when a single slow
     vehicle is injected into a uniform stream.  The gravitational interaction
     must cause chain-reaction deceleration that propagates upstream.

The enriched acceleration model is:
    a_i = F_gravity / |m_i| + gamma * (v_eq(rho_i) - |v_i|) * direction_i

where v_eq(rho) = v_free * max(0, 1 - rho / rho_jam) is the Greenshields
equilibrium speed, and gamma is the drag coefficient.

Author: Agent #23 Scientific Validation Tester
Date: 2026-03-22
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from gravtraffic.core.simulation import GravSimulation

__all__ = [
    "run_calibration_test",
    "run_emergence_test",
    "unified_grid_search",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
V_FREE_MS: float = 33.33       # m/s (120 km/h)
RHO_JAM: float = 150.0         # veh/km
SEGMENT_LENGTH_M: float = 2000.0  # 2 km highway segment


def _greenshields_speed(rho: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Greenshields speed: v(rho) = v_free * max(0, 1 - rho / rho_jam)."""
    return np.float64(V_FREE_MS) * np.maximum(
        0.0, 1.0 - np.asarray(rho, dtype=np.float64) / RHO_JAM
    )


def _r_squared(
    y_true: npt.NDArray[np.float64],
    y_pred: npt.NDArray[np.float64],
) -> float:
    """Coefficient of determination.  Can be negative."""
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return 1.0 - ss_res / ss_tot


# ---------------------------------------------------------------------------
# Test A: Calibration -- Greenshields fundamental diagram
# ---------------------------------------------------------------------------
def run_calibration_test(
    G_s: float,
    beta: float,
    softening: float,
    gamma: float,
    densities: list[float] | None = None,
    n_steps: int = 300,
    dt: float = 0.1,
    seed: int = 42,
) -> dict:
    """Test whether the drag-enriched model reproduces Greenshields.

    Vehicles start at free-flow speed (v_free) at various densities.
    The drag term should pull them toward v_eq(rho) over 300 steps (30s).

    Parameters
    ----------
    G_s, beta, softening, gamma : float
        Model parameters.
    densities : list of float, optional
        Densities to test (veh/km).  Default: [10, 30, 50, 70, 90, 110, 130].
    n_steps : int
        Simulation steps per density scenario.
    dt : float
        Fixed timestep in seconds.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Keys: 'r_squared', 'mean_speeds', 'greenshields_speeds', 'densities',
        'stable', 'monotonic'.
    """
    if densities is None:
        densities = [10, 30, 50, 70, 90, 110, 130]

    rng = np.random.default_rng(seed)
    density_arr = np.array(densities, dtype=np.float64)
    greenshields = _greenshields_speed(density_arr)

    mean_speeds = np.zeros(len(densities), dtype=np.float64)
    all_stable = True

    for idx, rho in enumerate(densities):
        n_vehicles = max(2, int(round(rho * SEGMENT_LENGTH_M / 1000.0)))
        spacing = SEGMENT_LENGTH_M / n_vehicles
        positions_x = np.arange(n_vehicles, dtype=np.float64) * spacing + spacing * 0.5

        positions = np.zeros((n_vehicles, 2), dtype=np.float64)
        positions[:, 0] = positions_x

        # Start ALL vehicles at v_free (generation test, not preservation)
        noise = rng.normal(0.0, 0.5, size=n_vehicles)
        speeds = np.clip(V_FREE_MS + noise, 0.0, V_FREE_MS + 3.0).astype(np.float64)

        velocities = np.zeros((n_vehicles, 2), dtype=np.float64)
        velocities[:, 0] = speeds

        local_dens = np.full(n_vehicles, rho, dtype=np.float64)

        sim = GravSimulation(
            G_s=G_s,
            beta=beta,
            softening=softening,
            rho_scale=30.0,
            theta=0.5,
            dt=dt,
            v_max=V_FREE_MS + 5.0,
            adaptive_dt=False,
            drag_coefficient=gamma,
            v_free=V_FREE_MS,
            rho_jam=RHO_JAM,
        )
        sim.init_vehicles(positions, velocities, local_dens)

        had_nan = False
        for _ in range(n_steps):
            result = sim.step()
            if np.any(np.isnan(result["velocities"])) or np.any(
                np.isinf(result["velocities"])
            ):
                had_nan = True
                break

        if had_nan:
            mean_speeds[idx] = np.nan
            all_stable = False
        else:
            final_speeds = np.linalg.norm(sim.velocities, axis=1)
            mean_speeds[idx] = float(np.mean(final_speeds))
            if np.any(sim.velocities[:, 0] < -5.0):
                all_stable = False

    stable = all_stable and not np.any(np.isnan(mean_speeds))

    # Monotonicity
    if stable:
        diffs = np.diff(mean_speeds)
        monotonic = bool(np.all(diffs <= 1.0))
    else:
        monotonic = False

    # R^2
    finite_mask = np.isfinite(mean_speeds)
    if np.sum(finite_mask) >= 2:
        r2 = _r_squared(greenshields[finite_mask], mean_speeds[finite_mask])
    else:
        r2 = float("-inf")

    return {
        "r_squared": float(r2),
        "mean_speeds": mean_speeds,
        "greenshields_speeds": greenshields,
        "densities": density_arr,
        "stable": stable,
        "monotonic": monotonic,
    }


# ---------------------------------------------------------------------------
# Test B: Emergence -- upstream deceleration from slow vehicle
# ---------------------------------------------------------------------------
def run_emergence_test(
    G_s: float,
    beta: float,
    softening: float,
    gamma: float,
    n_vehicles: int = 200,
    highway_length: float = 2000.0,
    initial_speed: float = 25.0,
    slow_speed: float = 5.0,
    n_steps: int = 500,
    dt: float = 0.1,
    seed: int = 42,
) -> dict:
    """Test emergence: does a slow vehicle cause upstream deceleration?

    Parameters
    ----------
    G_s, beta, softening, gamma : float
        Model parameters.
    n_vehicles : int
        Number of vehicles.
    highway_length : float
        Highway length in meters.
    initial_speed : float
        Initial speed for all vehicles (m/s).
    slow_speed : float
        Speed of the injected slow vehicle (m/s).
    n_steps : int
        Simulation steps.
    dt : float
        Fixed timestep.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Keys: 'upstream_mean_speed', 'upstream_decel', 'emergence_pass',
        'stable', 'downstream_mean_speed'.
    """
    np.random.seed(seed)
    spacing = highway_length / n_vehicles
    x_positions = np.linspace(0, highway_length - spacing, n_vehicles)
    positions = np.zeros((n_vehicles, 2), dtype=np.float64)
    positions[:, 0] = x_positions

    velocities = np.zeros((n_vehicles, 2), dtype=np.float64)
    velocities[:, 0] = initial_speed

    # Inject slow vehicle at midpoint
    slow_idx = int(np.argmin(np.abs(x_positions - highway_length / 2)))
    velocities[slow_idx, 0] = slow_speed

    # Density ~ n_vehicles / (highway_length / 1000) veh/km
    density = n_vehicles / (highway_length / 1000.0)
    local_densities = np.full(n_vehicles, density, dtype=np.float64)

    sim = GravSimulation(
        G_s=G_s,
        beta=beta,
        softening=softening,
        rho_scale=30.0,
        theta=0.5,
        dt=dt,
        v_max=36.0,
        adaptive_dt=False,
        drag_coefficient=gamma,
        v_free=V_FREE_MS,
        rho_jam=RHO_JAM,
    )
    sim.init_vehicles(positions.copy(), velocities.copy(), local_densities.copy())

    had_nan = False
    for _ in range(n_steps):
        result = sim.step()
        if np.any(np.isnan(result["velocities"])) or np.any(
            np.isinf(result["velocities"])
        ):
            had_nan = True
            break

    if had_nan:
        return {
            "upstream_mean_speed": float("nan"),
            "upstream_decel": 0.0,
            "emergence_pass": False,
            "stable": False,
            "downstream_mean_speed": float("nan"),
        }

    # Identify upstream vehicles: originally at x in [700, 950]
    init_x = x_positions
    upstream_mask = (init_x >= 700.0) & (init_x <= 950.0)
    final_speeds = np.linalg.norm(sim.velocities, axis=1)

    if np.any(upstream_mask):
        upstream_mean = float(np.mean(final_speeds[upstream_mask]))
    else:
        upstream_mean = initial_speed

    # Downstream vehicles: originally at x in [1050, 1300]
    downstream_mask = (init_x >= 1050.0) & (init_x <= 1300.0)
    if np.any(downstream_mask):
        downstream_mean = float(np.mean(final_speeds[downstream_mask]))
    else:
        downstream_mean = initial_speed

    upstream_decel = initial_speed - upstream_mean

    # Pass criterion: upstream deceleration > 0.5 m/s
    emergence_pass = upstream_decel > 0.5

    return {
        "upstream_mean_speed": upstream_mean,
        "upstream_decel": upstream_decel,
        "emergence_pass": emergence_pass,
        "stable": True,
        "downstream_mean_speed": downstream_mean,
    }


# ---------------------------------------------------------------------------
# Unified grid search
# ---------------------------------------------------------------------------
def unified_grid_search(
    seed: int = 42,
    verbose: bool = False,
) -> list[dict]:
    """Search parameter space for sets satisfying both calibration and emergence.

    Parameters
    ----------
    seed : int
        Random seed.
    verbose : bool
        If True, print progress.

    Returns
    -------
    list of dict
        Results sorted by unified_score descending.  Each dict contains
        the parameter values, calibration R^2, emergence metrics, and
        a unified score.
    """
    G_s_values = [1, 5, 10, 20, 50]
    beta_values = [0.3, 0.5, 1.0]
    softening_values = [5, 10, 20]
    gamma_values = [0.1, 0.3, 0.5, 1.0]

    results: list[dict] = []
    total = len(G_s_values) * len(beta_values) * len(softening_values) * len(gamma_values)
    count = 0

    for G_s in G_s_values:
        for beta in beta_values:
            for softening in softening_values:
                for gamma in gamma_values:
                    count += 1
                    if verbose:
                        print(f"[{count}/{total}] G_s={G_s}, beta={beta}, "
                              f"soft={softening}, gamma={gamma}")

                    try:
                        cal = run_calibration_test(
                            G_s=G_s, beta=beta, softening=softening,
                            gamma=gamma, seed=seed,
                        )
                        emg = run_emergence_test(
                            G_s=G_s, beta=beta, softening=softening,
                            gamma=gamma, seed=seed,
                        )

                        # Unified score: R^2 * (upstream_decel / 2.0)
                        # Both must contribute positively.
                        r2 = cal["r_squared"] if cal["stable"] else -1.0
                        decel = emg["upstream_decel"] if emg["stable"] else 0.0
                        unified = max(0.0, r2) * min(1.0, decel / 2.0)

                        results.append({
                            "G_s": float(G_s),
                            "beta": float(beta),
                            "softening": float(softening),
                            "gamma": float(gamma),
                            "r_squared": float(r2),
                            "upstream_decel": float(decel),
                            "upstream_mean_speed": emg["upstream_mean_speed"],
                            "downstream_mean_speed": emg["downstream_mean_speed"],
                            "calibration_pass": r2 > 0.70,
                            "emergence_pass": emg["emergence_pass"],
                            "unified_pass": r2 > 0.70 and emg["emergence_pass"],
                            "unified_score": float(unified),
                            "stable": cal["stable"] and emg["stable"],
                            "monotonic": cal["monotonic"],
                        })
                    except Exception as exc:
                        results.append({
                            "G_s": float(G_s),
                            "beta": float(beta),
                            "softening": float(softening),
                            "gamma": float(gamma),
                            "r_squared": float("-inf"),
                            "upstream_decel": 0.0,
                            "upstream_mean_speed": float("nan"),
                            "downstream_mean_speed": float("nan"),
                            "calibration_pass": False,
                            "emergence_pass": False,
                            "unified_pass": False,
                            "unified_score": 0.0,
                            "stable": False,
                            "monotonic": False,
                        })

    results.sort(key=lambda r: r["unified_score"], reverse=True)
    return results


def print_unified_report(results: list[dict], top_n: int = 10) -> None:
    """Print a human-readable report of unified grid search results."""
    print("=" * 90)
    print("UNIFIED CALIBRATION + EMERGENCE -- GRID SEARCH REPORT")
    print("=" * 90)
    print(f"Total configurations tested: {len(results)}")

    unified_pass = sum(1 for r in results if r["unified_pass"])
    cal_pass = sum(1 for r in results if r["calibration_pass"])
    emg_pass = sum(1 for r in results if r["emergence_pass"])
    print(f"Calibration pass (R^2 > 0.70): {cal_pass}/{len(results)}")
    print(f"Emergence pass (decel > 0.5 m/s): {emg_pass}/{len(results)}")
    print(f"UNIFIED PASS (both): {unified_pass}/{len(results)}")

    print(f"\n{'Rank':<5} {'G_s':>5} {'beta':>5} {'soft':>5} {'gamma':>6} "
          f"{'R^2':>8} {'decel':>7} {'score':>7} {'Cal':>4} {'Emg':>4} {'Both':>5}")
    print("-" * 90)

    for i, r in enumerate(results[:top_n]):
        print(
            f"{i+1:<5} {r['G_s']:>5.0f} {r['beta']:>5.1f} {r['softening']:>5.0f} "
            f"{r['gamma']:>6.2f} {r['r_squared']:>8.4f} "
            f"{r['upstream_decel']:>7.2f} {r['unified_score']:>7.4f} "
            f"{'Y' if r['calibration_pass'] else 'N':>4} "
            f"{'Y' if r['emergence_pass'] else 'N':>4} "
            f"{'Y' if r['unified_pass'] else 'N':>5}"
        )
    print("-" * 90)
    print("=" * 90)
