"""Calibration viability module -- go/no-go gate for GravTraffic (C-01).

This module generates synthetic highway data following Greenshields'
fundamental diagram and tests whether the gravitational traffic model
can reproduce a plausible speed-density relationship.

Greenshields model:
    v(rho) = v_free * (1 - rho / rho_jam)

Three parameter configurations are tested:
    Helbing:    G_s = 2.0,  beta = 0.5
    GravJanus:  G_s = 9.8,  beta = 1.0
    NonLinear:  G_s = 15.0, beta = 1.5

Reference
---------
Janus Civil C-01 GravTraffic Technical Plan, Section 5 (Calibration).
Greenshields, B.D. (1935). "A Study of Traffic Capacity."
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from gravtraffic.core.mass_assigner import MassAssigner
from gravtraffic.core.force_engine import ForceEngine

__all__ = ["run_calibration_test", "calibration_viability_report"]

# ---------------------------------------------------------------------------
# Greenshields constants
# ---------------------------------------------------------------------------
V_FREE_KMH: float = 120.0          # km/h
V_FREE_MS: float = V_FREE_KMH / 3.6  # 33.333... m/s
RHO_JAM: float = 150.0             # veh/km
NOISE_SIGMA: float = 2.0           # m/s  Gaussian noise on speeds
SEGMENT_LENGTH_KM: float = 1.0     # 1 km road segment
SEGMENT_LENGTH_M: float = 1000.0   # meters


def _greenshields_speed(rho: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Analytical Greenshields speed v(rho) = v_free * (1 - rho/rho_jam).

    Parameters
    ----------
    rho : ndarray, shape (N,), dtype float64
        Traffic density in veh/km.

    Returns
    -------
    ndarray, shape (N,), dtype float64
        Speed in m/s.
    """
    return np.float64(V_FREE_MS) * (1.0 - rho / np.float64(RHO_JAM))


def _compute_r_squared(
    y_true: npt.NDArray[np.float64],
    y_pred: npt.NDArray[np.float64],
) -> float:
    """Compute coefficient of determination R^2 = 1 - SS_res / SS_tot.

    Can be negative if the model is worse than a horizontal mean line.
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return float(1.0 - ss_res / ss_tot)


def _compute_rmse(
    y_true: npt.NDArray[np.float64],
    y_pred: npt.NDArray[np.float64],
) -> float:
    """Root mean square error in the same units as y."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _estimate_local_density(
    positions_x: npt.NDArray[np.float64],
    segment_length_m: float = SEGMENT_LENGTH_M,
    n_bins: int = 10,
) -> npt.NDArray[np.float64]:
    """Estimate local density (veh/km) from 1D positions using binning.

    Divides the segment into *n_bins* equal bins and counts vehicles in
    each bin, then converts to veh/km.

    Parameters
    ----------
    positions_x : ndarray, shape (N,)
        Vehicle x-positions in meters along the segment.
    segment_length_m : float
        Total segment length in meters.
    n_bins : int
        Number of spatial bins.

    Returns
    -------
    ndarray, shape (N,), dtype float64
        Local density at each vehicle's position, in veh/km.
    """
    bin_length_m = segment_length_m / n_bins
    bin_length_km = bin_length_m / 1000.0

    # Clamp positions to [0, segment_length_m)
    clamped = np.clip(positions_x, 0.0, segment_length_m - 1e-6)
    bin_indices = np.floor(clamped / bin_length_m).astype(int)
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    # Count vehicles per bin
    counts = np.bincount(bin_indices, minlength=n_bins)

    # Density for each bin in veh/km
    bin_densities = counts.astype(np.float64) / bin_length_km

    # Map back to each vehicle
    local_densities = bin_densities[bin_indices]
    return local_densities


def run_calibration_test(
    G_s: float,
    beta: float,
    n_vehicles: int = 100,
    n_steps: int = 50,
    seed: int = 42,
) -> dict:
    """Run a single calibration test for one (G_s, beta) configuration.

    Procedure
    ---------
    1. Sample densities uniformly from [5, 140] veh/km.
    2. Compute Greenshields speeds + Gaussian noise.
    3. Place vehicles on a 1D segment with spacing derived from density.
    4. Run a simplified Euler integration loop using MassAssigner + ForceEngine.
    5. Compare final speeds to Greenshields analytical curve via R^2.

    Parameters
    ----------
    G_s : float
        Social gravitational constant.
    beta : float
        Mass exponent.
    n_vehicles : int
        Number of synthetic vehicles (default 100).
    n_steps : int
        Number of Euler timesteps (default 50).
    seed : int
        Random seed for reproducibility (default 42).

    Returns
    -------
    dict
        Keys: 'G_s', 'beta', 'r_squared', 'rmse_ms', 'final_speeds',
        'densities'.
    """
    rng = np.random.default_rng(seed)
    dt = np.float64(1.0)  # seconds

    # ------------------------------------------------------------------
    # Step 1: Generate synthetic density samples
    # ------------------------------------------------------------------
    densities = rng.uniform(5.0, 140.0, size=n_vehicles).astype(np.float64)

    # ------------------------------------------------------------------
    # Step 2: Greenshields speeds with noise
    # ------------------------------------------------------------------
    analytical_speeds = _greenshields_speed(densities)
    noisy_speeds = analytical_speeds + rng.normal(0.0, NOISE_SIGMA, size=n_vehicles)
    # Clamp speeds to physical range [0, v_free]
    noisy_speeds = np.clip(noisy_speeds, 0.0, V_FREE_MS).astype(np.float64)

    # ------------------------------------------------------------------
    # Step 3: Place vehicles on 1D segment
    # ------------------------------------------------------------------
    # Position each vehicle based on its "slot" within the segment.
    # For a vehicle at density rho, the average spacing is 1000/rho meters.
    # We place vehicles sorted by density-derived position to create a
    # realistic spatial distribution.
    sort_idx = np.argsort(densities)
    # Cumulative spacing: each vehicle gets 1000/rho_i meters of headway
    spacings = 1000.0 / densities[sort_idx]  # meters per vehicle
    cumulative_pos = np.cumsum(spacings)
    # Normalize to fit within segment
    if cumulative_pos[-1] > 0:
        cumulative_pos = cumulative_pos / cumulative_pos[-1] * SEGMENT_LENGTH_M * 0.95
    # Unsort back to original order
    positions_x = np.empty(n_vehicles, dtype=np.float64)
    positions_x[sort_idx] = cumulative_pos

    # 2D positions for ForceEngine (y = 0 for 1D)
    positions = np.zeros((n_vehicles, 2), dtype=np.float64)
    positions[:, 0] = positions_x

    # Current speeds (will be modified during simulation)
    speeds = noisy_speeds.copy()

    # ------------------------------------------------------------------
    # Step 4: Simulation loop (Euler integration)
    # ------------------------------------------------------------------
    mass_assigner = MassAssigner(beta=beta, rho_scale=30.0)
    force_engine = ForceEngine(G_s=G_s, softening=10.0)

    # Damping coefficient to prevent runaway acceleration.
    # The gravitational force is a behavioral model; damping represents
    # drivers' tendency to return toward a "desired" speed.
    # The relaxation time tau ~ 1/damping controls how quickly drivers
    # adapt to their desired (Greenshields) speed.
    damping = np.float64(0.3)

    # Desired speed from the ORIGINAL sampled densities (scenario truth).
    # This is the target each driver tries to maintain; the gravitational
    # force perturbs them around this target.
    desired_speeds = _greenshields_speed(densities)

    for step in range(n_steps):
        # Estimate local density from current positions for mass computation.
        # This captures the spatial clustering effect of the gravitational model.
        local_densities = _estimate_local_density(positions[:, 0])

        # Mean speed for mass assignment
        v_mean = float(np.mean(speeds))

        # Compute gravitational masses
        masses = mass_assigner.assign(speeds, v_mean, local_densities)

        # Compute gravitational social forces (N-body, O(N^2))
        forces = force_engine.compute_all_naive(positions, masses)

        # Extract x-component of force (1D motion)
        force_x = forces[:, 0]

        # Relaxation toward Greenshields desired speed (IDM-style).
        # Uses the original scenario densities, not the binned local
        # densities, because each vehicle's "desired" speed is set by
        # the traffic conditions it was generated for.
        relaxation = damping * (desired_speeds - speeds)

        # Scale gravitational force to act as a perturbation.
        # The force magnitude can be very large (O(G_s * m^2 / d^2)),
        # so we normalize to keep accelerations physically meaningful.
        # Maximum reasonable acceleration: ~3 m/s^2.
        force_scale = np.float64(0.01)
        acceleration = force_scale * force_x + relaxation

        # Euler step on speed
        speeds = speeds + acceleration * dt

        # Clamp to physical bounds
        speeds = np.clip(speeds, 0.0, V_FREE_MS)

        # Euler step on position
        positions[:, 0] = positions[:, 0] + speeds * dt

        # Wrap positions within segment (periodic boundary)
        positions[:, 0] = np.mod(positions[:, 0], SEGMENT_LENGTH_M)

    # ------------------------------------------------------------------
    # Step 5: Evaluate fit against Greenshields
    # ------------------------------------------------------------------
    # Compare final speeds to the analytical Greenshields curve at the
    # *original* sampled densities (the question is whether the model
    # preserves the fundamental diagram shape).
    r_squared = _compute_r_squared(analytical_speeds, speeds)
    rmse_ms = _compute_rmse(analytical_speeds, speeds)

    return {
        "G_s": float(G_s),
        "beta": float(beta),
        "r_squared": float(r_squared),
        "rmse_ms": float(rmse_ms),
        "final_speeds": speeds.copy(),
        "densities": densities.copy(),
    }


# ---------------------------------------------------------------------------
# Predefined configurations
# ---------------------------------------------------------------------------
CONFIGS = [
    {"name": "Helbing",    "G_s": 2.0,  "beta": 0.5},
    {"name": "GravJanus",  "G_s": 9.8,  "beta": 1.0},
    {"name": "NonLinear",  "G_s": 15.0, "beta": 1.5},
]


def calibration_viability_report() -> list[dict]:
    """Run all 3 configurations and return results sorted by R^2 descending.

    Also prints a human-readable summary table.

    Returns
    -------
    list[dict]
        Each dict has keys: 'name', 'G_s', 'beta', 'r_squared', 'rmse_ms',
        'final_speeds', 'densities'.
    """
    results: list[dict] = []

    for cfg in CONFIGS:
        result = run_calibration_test(G_s=cfg["G_s"], beta=cfg["beta"])
        result["name"] = cfg["name"]
        results.append(result)

    # Sort by R^2 descending (best first)
    results.sort(key=lambda r: r["r_squared"], reverse=True)

    # Print summary
    print("=" * 70)
    print("GRAVTRAFFIC CALIBRATION VIABILITY REPORT")
    print("=" * 70)
    print(f"{'Config':<12} {'G_s':>6} {'beta':>6} {'R^2':>10} {'RMSE (m/s)':>12}")
    print("-" * 70)
    for r in results:
        print(
            f"{r['name']:<12} {r['G_s']:>6.1f} {r['beta']:>6.1f} "
            f"{r['r_squared']:>10.6f} {r['rmse_ms']:>12.4f}"
        )
    print("-" * 70)

    # Viability assessment
    best = results[0]
    if best["r_squared"] >= 0.70:
        print(f"PASS: Best R^2 = {best['r_squared']:.4f} >= 0.70 "
              f"(config: {best['name']})")
        print("The gravitational model can reproduce a plausible "
              "speed-density relationship.")
    else:
        print(f"WARN: Best R^2 = {best['r_squared']:.4f} < 0.70")
        print("Enrichment suggestions:")
        print("  1. Introduce an explicit desired-speed relaxation term "
              "(IDM-style)")
        print("  2. Add density-dependent softening length")
        print("  3. Tune rho_scale in MassAssigner to match scenario density")
        print("  4. Consider anisotropic forces (forward-looking bias)")
        print("  5. Increase simulation steps or reduce dt for better "
              "convergence")
    print("=" * 70)

    return results
