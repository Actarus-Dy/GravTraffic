"""Traffic metrics and KPI calculator for GravTraffic (C-01).

This module provides fully vectorized NumPy functions that compute traffic
performance indicators from vehicle state arrays.  No Python-level loops
iterate over vehicles -- all hot paths use broadcasting and array operations.

Metrics implemented
-------------------
- **Throughput** -- vehicles crossing a counting gate per hour.
- **Mean speed** -- population-average scalar speed [m/s].
- **Delay** -- excess travel time relative to free-flow speed [s/km].
- **Stops** -- count of vehicles below a speed threshold.
- **Congestion index** -- fraction of vehicles with positive gravitational mass
  (slow / congested as defined by the MassAssigner classification threshold).
- **Level of Service** -- HCM-style A-F grading based on speed ratio.
- **Travel time** -- mean time for vehicles to traverse a corridor segment.
- **Snapshot KPIs** -- convenience dict aggregating all instantaneous metrics.

Reference
---------
Janus Civil C-01 GravTraffic Technical Plan, Section 6 -- Performance Metrics.

Author: Agent #01 Python Scientific Developer
Date: 2026-03-22
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

__all__ = [
    "compute_throughput",
    "compute_mean_speed",
    "compute_delay",
    "compute_stops",
    "compute_congestion_index",
    "compute_level_of_service",
    "compute_travel_time",
    "compute_snapshot_kpis",
]


def compute_throughput(
    positions: npt.NDArray[np.float64],
    prev_positions: npt.NDArray[np.float64],
    gate_x: float,
    dt: float,
) -> float:
    """Count vehicles crossing *gate_x* between two timesteps.

    A crossing is detected when a vehicle's x-coordinate moves from one
    side of ``gate_x`` to the other (or exactly onto it) between the
    previous and current positions.  Both left-to-right and right-to-left
    crossings are counted.

    Parameters
    ----------
    positions : ndarray, shape (N, 2), dtype float64
        Current (x, y) coordinates.
    prev_positions : ndarray, shape (N, 2), dtype float64
        Previous (x, y) coordinates.
    gate_x : float
        x-coordinate of the virtual counting gate.
    dt : float
        Time elapsed between the two snapshots [seconds].  Must be > 0.

    Returns
    -------
    float
        Throughput in vehicles per hour.

    Notes
    -----
    Fully vectorized -- one boolean mask operation on the x-columns.
    """
    positions = np.asarray(positions, dtype=np.float64)
    prev_positions = np.asarray(prev_positions, dtype=np.float64)

    if positions.size == 0:
        return 0.0

    prev_x = prev_positions[:, 0]
    curr_x = positions[:, 0]

    crossed = (
        ((prev_x < gate_x) & (curr_x >= gate_x))
        | ((prev_x > gate_x) & (curr_x <= gate_x))
    )

    return float(np.sum(crossed)) / dt * 3600.0


def compute_mean_speed(velocities: npt.NDArray[np.float64]) -> float:
    """Return the population-average scalar speed [m/s].

    Parameters
    ----------
    velocities : ndarray, shape (N, 2), dtype float64
        Velocity vectors (vx, vy) of each vehicle.

    Returns
    -------
    float
        Mean speed, or 0.0 if the array is empty.
    """
    velocities = np.asarray(velocities, dtype=np.float64)
    if velocities.size == 0:
        return 0.0
    speeds = np.linalg.norm(velocities, axis=1)
    return float(np.mean(speeds))


def compute_delay(
    velocities: npt.NDArray[np.float64],
    v_free: float = 33.33,
) -> float:
    """Mean delay per vehicle [seconds per kilometre].

    Delay is the excess travel time compared to free-flow conditions:

        delay_i = 1000 / v_i  -  1000 / v_free

    Vehicle speeds are clamped to a minimum of 0.1 m/s to avoid division
    by zero.

    Parameters
    ----------
    velocities : ndarray, shape (N, 2), dtype float64
        Velocity vectors.
    v_free : float, default 33.33
        Free-flow speed [m/s] (~120 km/h).

    Returns
    -------
    float
        Mean delay [s/km], or 0.0 for empty input.
    """
    velocities = np.asarray(velocities, dtype=np.float64)
    if velocities.size == 0:
        return 0.0

    speeds = np.linalg.norm(velocities, axis=1)
    speeds = np.maximum(speeds, 0.1)  # clamp to avoid division by zero
    time_actual = 1000.0 / speeds     # seconds to travel 1 km at actual speed
    time_free = 1000.0 / v_free       # seconds to travel 1 km at free-flow
    return float(np.mean(time_actual - time_free))


def compute_stops(
    velocities: npt.NDArray[np.float64],
    threshold: float = 2.0,
) -> int:
    """Count vehicles whose scalar speed is below *threshold*.

    Parameters
    ----------
    velocities : ndarray, shape (N, 2), dtype float64
        Velocity vectors.
    threshold : float, default 2.0
        Speed below which a vehicle is considered stopped [m/s].

    Returns
    -------
    int
        Number of stopped vehicles.
    """
    velocities = np.asarray(velocities, dtype=np.float64)
    if velocities.size == 0:
        return 0
    speeds = np.linalg.norm(velocities, axis=1)
    return int(np.sum(speeds < threshold))


def compute_congestion_index(masses: npt.NDArray[np.float64]) -> float:
    """Fraction of vehicles with positive gravitational mass (slow/congested).

    A vehicle is considered congested when its mass exceeds 0.1, consistent
    with the neutral-threshold used by ``MassAssigner.classify``.

    Parameters
    ----------
    masses : ndarray, shape (N,), dtype float64
        Signed gravitational masses.

    Returns
    -------
    float
        Congestion index in [0, 1].
    """
    masses = np.asarray(masses, dtype=np.float64)
    if masses.size == 0:
        return 0.0
    return float(np.sum(masses > 0.1) / len(masses))


def compute_level_of_service(
    velocities: npt.NDArray[np.float64],
    v_free: float = 33.33,
) -> str:
    """HCM-style Level of Service grading (A through F).

    The grade is based on the ratio of mean speed to free-flow speed:

    =====  ===============
    Grade  Speed ratio
    =====  ===============
    A      > 0.90
    B      > 0.70
    C      > 0.50
    D      > 0.40
    E      > 0.25
    F      <= 0.25
    =====  ===============

    Parameters
    ----------
    velocities : ndarray, shape (N, 2), dtype float64
        Velocity vectors.
    v_free : float, default 33.33
        Free-flow speed [m/s].

    Returns
    -------
    str
        Single-character grade ``'A'`` to ``'F'``.
    """
    ratio = compute_mean_speed(velocities) / v_free if v_free > 0 else 0.0
    if ratio > 0.90:
        return "A"
    if ratio > 0.70:
        return "B"
    if ratio > 0.50:
        return "C"
    if ratio > 0.40:
        return "D"
    if ratio > 0.25:
        return "E"
    return "F"


def compute_travel_time(
    positions_history: list[npt.NDArray[np.float64]],
    start_x: float,
    end_x: float,
    dt: float,
) -> float | None:
    """Mean travel time for vehicles traversing from *start_x* to *end_x*.

    The function scans the time series of position snapshots to determine
    when each vehicle first reaches ``start_x`` (entry) and ``end_x``
    (exit).  Only vehicles that complete the full trip contribute to the
    average.

    Per-timestep logic is vectorized over vehicles (no inner Python loop
    on the vehicle dimension).  The outer loop over timesteps is
    unavoidable because each step depends on the previous state.

    Parameters
    ----------
    positions_history : list of ndarray, each shape (N, 2), dtype float64
        Chronological position snapshots.
    start_x : float
        x-coordinate marking the corridor entrance.
    end_x : float
        x-coordinate marking the corridor exit.
    dt : float
        Timestep duration [seconds].

    Returns
    -------
    float or None
        Mean travel time [seconds], or ``None`` if no vehicle completed
        the trip.
    """
    if len(positions_history) == 0:
        return None

    n_vehicles = positions_history[0].shape[0]
    if n_vehicles == 0:
        return None

    entry_step = np.full(n_vehicles, -1, dtype=np.int64)
    exit_step = np.full(n_vehicles, -1, dtype=np.int64)

    for t, pos in enumerate(positions_history):
        pos = np.asarray(pos, dtype=np.float64)
        x = pos[:, 0]

        # Mark entry: first time x >= start_x
        entering = (entry_step == -1) & (x >= start_x)
        entry_step[entering] = t

        # Mark exit: first time x >= end_x after entry
        exiting = (entry_step >= 0) & (exit_step == -1) & (x >= end_x)
        exit_step[exiting] = t

    completed = (entry_step >= 0) & (exit_step >= 0)
    if not np.any(completed):
        return None

    travel_steps = exit_step[completed] - entry_step[completed]
    return float(np.mean(travel_steps) * dt)


def compute_snapshot_kpis(
    positions: npt.NDArray[np.float64],
    velocities: npt.NDArray[np.float64],
    masses: npt.NDArray[np.float64],
    v_free: float = 33.33,
) -> dict:
    """Compute all instantaneous KPIs and return them as a dictionary.

    This is a convenience aggregator that calls the individual metric
    functions.  Intended for logging, dashboard display, or API
    serialisation.

    Parameters
    ----------
    positions : ndarray, shape (N, 2), dtype float64
        Current positions.
    velocities : ndarray, shape (N, 2), dtype float64
        Current velocities.
    masses : ndarray, shape (N,), dtype float64
        Current gravitational masses.
    v_free : float, default 33.33
        Free-flow speed [m/s].

    Returns
    -------
    dict
        Keys: ``mean_speed_ms``, ``mean_speed_kmh``, ``n_stops``,
        ``congestion_index``, ``delay_s_per_km``, ``level_of_service``,
        ``n_vehicles``.
    """
    mean_spd = compute_mean_speed(velocities)
    return {
        "mean_speed_ms": mean_spd,
        "mean_speed_kmh": mean_spd * 3.6,
        "n_stops": compute_stops(velocities),
        "congestion_index": compute_congestion_index(masses),
        "delay_s_per_km": compute_delay(velocities, v_free),
        "level_of_service": compute_level_of_service(velocities, v_free),
        "n_vehicles": len(positions),
    }
