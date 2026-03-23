"""Fundamental Diagram validation -- Greenshields speed-density fit.

Runs a systematic density sweep on a 1-D highway and compares the
emergent mean speed at each density against the Greenshields model:

    v(rho) = v_free * (1 - rho / rho_jam)

Reports R², RMSE, and per-density data points for plotting.

Author: Agent #23 Scientific Validation Tester
Date: 2026-03-23
"""

from __future__ import annotations

import numpy as np

from gravtraffic.core.simulation import GravSimulation


def greenshields_speed(
    rho: float, v_free: float = 33.33, rho_jam: float = 150.0
) -> float:
    """Theoretical Greenshields equilibrium speed."""
    return v_free * max(0.0, 1.0 - rho / rho_jam)


def run_fd_sweep(
    densities: list[float] | None = None,
    G_s: float = 5.0,
    beta: float = 0.5,
    gamma: float = 0.3,
    v_free: float = 33.33,
    rho_jam: float = 150.0,
    highway_length: float = 2000.0,
    n_steps: int = 800,
    warmup_steps: int = 500,
    seed: int = 42,
) -> dict:
    """Run a density sweep and measure emergent speed-density relationship.

    Parameters
    ----------
    densities : list[float], optional
        Densities in veh/km to test. Default: 10 to 140 in steps of 10.
    n_steps : int
        Total simulation steps per density point.
    warmup_steps : int
        Steps to discard before measuring (let transients settle).

    Returns
    -------
    dict with keys:
        - densities: list[float] — tested densities [veh/km]
        - measured_speeds: list[float] — mean speed at each density [m/s]
        - theoretical_speeds: list[float] — Greenshields prediction [m/s]
        - r_squared: float — R² goodness of fit
        - rmse: float — root mean squared error [m/s]
        - data_points: list[dict] — per-density detail
    """
    if densities is None:
        densities = list(range(10, 141, 10))

    rng = np.random.default_rng(seed)
    results = []

    for rho in densities:
        # Number of vehicles from density and highway length
        n_veh = max(2, int(rho * highway_length / 1000.0))

        # Initial conditions: uniform spacing on 1-D highway
        positions = np.zeros((n_veh, 2), dtype=np.float64)
        positions[:, 0] = np.linspace(0, highway_length, n_veh, endpoint=False)

        # Initial speed: v_free for ALL densities (NOT v_eq!) to avoid
        # circular validation. The model must converge to the correct
        # equilibrium from arbitrary initial conditions.
        speeds = np.full(n_veh, v_free, dtype=np.float64)
        speeds += rng.normal(0, 1.0, n_veh)  # small noise for symmetry breaking
        speeds = np.clip(speeds, 0.5, v_free)
        velocities = np.zeros((n_veh, 2), dtype=np.float64)
        velocities[:, 0] = speeds

        local_densities = np.full(n_veh, float(rho), dtype=np.float64)

        sim = GravSimulation(
            G_s=G_s, beta=beta, softening=10.0,
            dt=0.1, v_max=v_free + 3.0, adaptive_dt=False,
            drag_coefficient=gamma, v_free=v_free, rho_jam=rho_jam,
            use_gpu=False,
        )
        sim.init_vehicles(positions, velocities, local_densities)

        # Run warmup
        sim.run(warmup_steps)

        # Measure over remaining steps
        speed_samples = []
        for _ in range(n_steps - warmup_steps):
            sim.step()
            mean_spd = float(np.mean(np.linalg.norm(sim.velocities, axis=1)))
            speed_samples.append(mean_spd)

        measured = float(np.mean(speed_samples))
        theoretical = greenshields_speed(rho, v_free, rho_jam)

        results.append({
            "density": rho,
            "n_vehicles": n_veh,
            "measured_speed": measured,
            "theoretical_speed": theoretical,
            "speed_std": float(np.std(speed_samples)),
        })

    # Compute fit metrics
    measured_arr = np.array([r["measured_speed"] for r in results])
    theoretical_arr = np.array([r["theoretical_speed"] for r in results])

    ss_res = np.sum((measured_arr - theoretical_arr) ** 2)
    ss_tot = np.sum((measured_arr - np.mean(measured_arr)) ** 2)
    r_squared = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    rmse = float(np.sqrt(np.mean((measured_arr - theoretical_arr) ** 2)))

    return {
        "densities": [r["density"] for r in results],
        "measured_speeds": [r["measured_speed"] for r in results],
        "theoretical_speeds": [r["theoretical_speed"] for r in results],
        "r_squared": r_squared,
        "rmse": rmse,
        "data_points": results,
        "parameters": {
            "G_s": G_s, "beta": beta, "gamma": gamma,
            "v_free": v_free, "rho_jam": rho_jam,
        },
    }
