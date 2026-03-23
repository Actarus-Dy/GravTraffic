"""Emergence quantification -- shock wave and jam metrics.

Measures the key emergent phenomena from the Janus gravitational model:
1. Upstream deceleration wave speed
2. Gini coefficient of vehicle spacings (clustering)
3. Jam formation and dissolution times
4. Speed variance amplification

All metrics compare gravity-on vs gravity-off (baseline) to isolate
the gravitational contribution to traffic dynamics.

Author: Agent #23 Scientific Validation Tester
Date: 2026-03-23
"""

from __future__ import annotations

import numpy as np

from gravtraffic.core.simulation import GravSimulation


def _build_scenario(
    n_vehicles: int = 100,
    highway_length: float = 2000.0,
    initial_speed: float = 25.0,
    slow_speed: float = 5.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Build a standard emergence scenario: uniform flow + one slow vehicle."""
    positions = np.zeros((n_vehicles, 2), dtype=np.float64)
    positions[:, 0] = np.linspace(0, highway_length, n_vehicles, endpoint=False)

    velocities = np.zeros((n_vehicles, 2), dtype=np.float64)
    velocities[:, 0] = initial_speed

    # Inject slow vehicle at midpoint
    slow_idx = n_vehicles // 2
    velocities[slow_idx, 0] = slow_speed

    density = n_vehicles / (highway_length / 1000.0)
    local_densities = np.full(n_vehicles, density, dtype=np.float64)

    return positions, velocities, local_densities, slow_idx


def gini_coefficient(values: np.ndarray) -> float:
    """Compute the Gini coefficient of a 1-D array (0=equal, 1=unequal)."""
    values = np.sort(values)
    if np.any(values < 0):
        raise ValueError("Gini coefficient requires non-negative values")
    n = len(values)
    if n == 0 or np.sum(values) == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2.0 * np.sum(index * values) - (n + 1) * np.sum(values)) / (n * np.sum(values)))


def run_emergence_analysis(
    G_s: float = 5.0,
    beta: float = 0.5,
    gamma: float = 0.3,
    n_vehicles: int = 100,
    n_steps: int = 500,
    seed: int = 42,
) -> dict:
    """Run emergence analysis comparing gravity-on vs gravity-off.

    Returns
    -------
    dict with keys:
        - gravity_on: dict of metrics with gravity
        - gravity_off: dict of metrics without gravity (G_s=0 equivalent)
        - emergence_score: float — composite score (higher = more emergence)
    """
    pos, vel, dens, slow_idx = _build_scenario(n_vehicles=n_vehicles, seed=seed)

    results = {}
    for label, g_s in [("gravity_on", G_s), ("gravity_off", 0.0)]:
        sim = GravSimulation(
            G_s=g_s,
            beta=beta,
            softening=10.0,
            dt=0.1,
            v_max=36.0,
            adaptive_dt=False,
            drag_coefficient=gamma,
            use_gpu=False,
        )
        sim.init_vehicles(pos.copy(), vel.copy(), dens.copy())

        # Track history
        speed_history = []
        spacing_gini_history = []

        for step in range(n_steps):
            sim.step()
            speeds = np.linalg.norm(sim.velocities, axis=1)
            speed_history.append(speeds.copy())

            # Vehicle spacings (sorted by x)
            x_sorted = np.sort(sim.positions[:, 0])
            spacings = np.diff(x_sorted)
            if len(spacings) > 0:
                spacing_gini_history.append(gini_coefficient(spacings))

        init_speeds = np.linalg.norm(vel, axis=1)
        final_speeds = speed_history[-1]

        # Upstream deceleration: vehicles approaching from behind (x < slow_x)
        # in a flow moving toward +x. These are the vehicles that should
        # decelerate as the gravitational shock wave propagates backward.
        init_x = pos[:, 0]
        slow_x = init_x[slow_idx]
        upstream_mask = (init_x < slow_x - 50) & (init_x > slow_x - 500)
        upstream_decel = float(
            np.mean(init_speeds[upstream_mask]) - np.mean(final_speeds[upstream_mask])
        )

        # Speed variance amplification
        init_std = float(np.std(init_speeds))
        final_std = float(np.std(final_speeds))
        variance_ratio = final_std / max(init_std, 0.01)

        # Gini evolution
        gini_initial = gini_coefficient(np.diff(np.sort(pos[:, 0])))
        gini_final = float(np.mean(spacing_gini_history[-50:])) if spacing_gini_history else 0.0

        # Wave speed estimation: find the furthest upstream vehicle that decelerated
        decel_threshold = 0.5  # m/s
        decel_mask = (init_speeds - final_speeds) > decel_threshold
        if decel_mask.any():
            furthest_decel_x = float(np.max(init_x[decel_mask]))
            wave_distance = furthest_decel_x - slow_x
            wave_speed = wave_distance / (n_steps * 0.1) if wave_distance > 0 else 0.0
        else:
            wave_distance = 0.0
            wave_speed = 0.0

        results[label] = {
            "upstream_deceleration_ms": upstream_decel,
            "speed_std_initial": init_std,
            "speed_std_final": final_std,
            "variance_ratio": variance_ratio,
            "gini_initial": gini_initial,
            "gini_final": gini_final,
            "gini_increase": gini_final - gini_initial,
            "wave_distance_m": wave_distance,
            "wave_speed_ms": wave_speed,
            "mean_speed_final_ms": float(np.mean(final_speeds)),
        }

    # Composite emergence score (qualitative indicator, not a physical metric).
    # Measures how much gravity amplifies perturbation spreading compared to
    # the drag-only baseline. Each component is normalized to [0, ~1]:
    #   - decel_delta: extra upstream deceleration from gravity [m/s], capped at 5
    #   - variance_delta: extra speed variance amplification, capped at 2
    #   - gini_delta: extra clustering (Gini of spacings), capped at 0.1
    g_on = results["gravity_on"]
    g_off = results["gravity_off"]
    decel_delta = max(0, g_on["upstream_deceleration_ms"] - g_off["upstream_deceleration_ms"])
    variance_delta = max(0, g_on["variance_ratio"] - g_off["variance_ratio"])
    gini_delta = max(0, g_on["gini_increase"] - g_off["gini_increase"])

    emergence_score = (
        min(decel_delta / 5.0, 1.0) + min(variance_delta / 2.0, 1.0) + min(gini_delta / 0.1, 1.0)
    ) / 3.0  # average of 3 normalized components -> [0, 1]

    return {
        "gravity_on": g_on,
        "gravity_off": g_off,
        "emergence_score": float(emergence_score),
        "parameters": {"G_s": G_s, "beta": beta, "gamma": gamma},
    }
