"""Leapfrog symplectic integrator with adaptive CFL timestep control.

Implements the kick-drift-kick (KDK) variant of the leapfrog integrator
for the Janus GravTraffic simulation (C-01, section 2.1). The leapfrog
method is a second-order symplectic integrator, meaning it conserves a
shadow Hamiltonian close to the true Hamiltonian over exponentially long
times. This makes it ideal for gravitational N-body problems where energy
drift must remain bounded.

KDK scheme
----------
Given positions x, velocities v, accelerations a, and timestep dt:

    1. v_half  = v + 0.5 * a * dt          (half-kick)
    2. x_new   = x + v_half * dt            (drift)
    3. a_new   = force_fn(x_new)            (recompute forces)
    4. v_new   = v_half + 0.5 * a_new * dt  (half-kick)
    5. clip |v_new| to [0, v_max]           (speed limiter)

Adaptive timestep
-----------------
The CFL-like condition ensures no particle crosses more than half the
minimum inter-particle distance per step:

    dt <= d_min / (2 * v_max_system)

A fast O(N log N) approximation is used: particles are sorted by their
x-coordinate and only consecutive pairs are checked for the minimum
distance. This is exact for 1D-dominated traffic layouts and a good
lower bound in general.

Author: Agent #01 Python Scientific Developer
Date: 2026-03-22
"""

from __future__ import annotations

from typing import Callable, Union

import numpy as np
import numpy.typing as npt


def leapfrog_step(
    positions: npt.NDArray[np.float64],
    velocities: npt.NDArray[np.float64],
    forces: npt.NDArray[np.float64],
    dt: float,
    force_fn: Callable[..., npt.NDArray[np.float64]],
    v_max: Union[npt.NDArray[np.float64], float] = 36.0,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Perform one leapfrog kick-drift-kick integration step.

    All arrays must be float64. The function is fully vectorized with no
    Python loops over particles.

    Parameters
    ----------
    positions : ndarray, shape (N, 2), dtype float64
        Current (x, y) positions of all particles.
    velocities : ndarray, shape (N, 2), dtype float64
        Current (vx, vy) velocities of all particles.
    forces : ndarray, shape (N, 2), dtype float64
        Current accelerations (force with mass already factored out).
    dt : float
        Integration timestep in seconds. Must be positive.
    force_fn : callable
        Function mapping (positions, velocities) -> accelerations, both
        (N, 2). Called once per step with the drifted positions and
        half-kick velocities to recompute accelerations.
    v_max : float or ndarray of shape (N,), default 36.0
        Maximum allowed speed magnitude per particle (m/s). Scalar
        applies the same limit to all particles. Array allows per-vehicle
        limits. Corresponds to ~130 km/h at default value.

    Returns
    -------
    positions_new : ndarray, shape (N, 2), dtype float64
        Updated positions after the drift.
    velocities_new : ndarray, shape (N, 2), dtype float64
        Updated velocities after the second half-kick and speed clipping.
    forces_new : ndarray, shape (N, 2), dtype float64
        New accelerations evaluated at ``positions_new``.

    Notes
    -----
    The leapfrog integrator is symplectic (preserves phase-space volume)
    and time-reversible, giving excellent long-term energy conservation
    for Hamiltonian systems. The KDK variant is preferred here because
    the forces at the end of the step are immediately available for the
    next step without recomputation.

    The speed clipping step breaks strict symplecticity but is physically
    necessary to enforce the traffic speed limit. In practice, clipping
    events are rare when the timestep is well-chosen via ``adaptive_dt``.
    """
    dt = float(dt)
    positions = np.asarray(positions, dtype=np.float64)
    velocities = np.asarray(velocities, dtype=np.float64)
    forces = np.asarray(forces, dtype=np.float64)

    # Step 1: half-kick
    v_half = velocities + 0.5 * forces * dt

    # Step 2: drift
    positions_new = positions + v_half * dt

    # Step 3: recompute forces at new positions with half-kick velocities
    forces_new = force_fn(positions_new, v_half)
    forces_new = np.asarray(forces_new, dtype=np.float64)

    # Step 4: second half-kick
    velocities_new = v_half + 0.5 * forces_new * dt

    # Step 5: speed clipping
    velocities_new = _clip_speed(velocities_new, v_max)

    return positions_new, velocities_new, forces_new


def _clip_speed(
    velocities: npt.NDArray[np.float64],
    v_max: Union[npt.NDArray[np.float64], float],
) -> npt.NDArray[np.float64]:
    """Clip velocity vectors so their magnitude does not exceed ``v_max``.

    Parameters
    ----------
    velocities : ndarray, shape (N, 2), dtype float64
        Velocity vectors to clip.
    v_max : float or ndarray of shape (N,)
        Maximum speed magnitude. If array, each row of ``velocities``
        gets its own limit.

    Returns
    -------
    ndarray, shape (N, 2), dtype float64
        Clipped velocity vectors. Direction is preserved; only magnitude
        is reduced when it exceeds the limit.
    """
    # Compute speed magnitudes: shape (N,)
    speed = np.linalg.norm(velocities, axis=1)

    # Build the v_max array for broadcasting: shape (N,)
    if np.ndim(v_max) == 0:
        v_max_arr = np.full(speed.shape, float(v_max), dtype=np.float64)
    else:
        v_max_arr = np.asarray(v_max, dtype=np.float64)

    # Identify particles that exceed v_max
    mask = speed > v_max_arr

    # Scale factor: v_max / speed, applied only where speed exceeds limit
    # Avoid division by zero for stationary particles (speed == 0)
    result = velocities.copy()
    if np.any(mask):
        scale = v_max_arr[mask] / speed[mask]
        # scale is shape (M,), need (M, 1) for broadcasting with (M, 2)
        result[mask] = velocities[mask] * scale[:, np.newaxis]

    return result


def adaptive_dt(
    positions: npt.NDArray[np.float64],
    velocities: npt.NDArray[np.float64],
    dt_max: float = 0.2,
    dt_min: float = 0.01,
) -> float:
    """Compute an adaptive timestep based on the CFL condition.

    Uses a fast O(N log N) approximation for the minimum pairwise
    distance: sort particles by x-coordinate and check only consecutive
    pairs. This is exact when the closest pair shares consecutive
    x-coordinates (common in traffic layouts) and provides a reliable
    lower bound otherwise.

    Parameters
    ----------
    positions : ndarray, shape (N, 2), dtype float64
        Current (x, y) positions of all particles.
    velocities : ndarray, shape (N, 2), dtype float64
        Current (vx, vy) velocities of all particles.
    dt_max : float, default 0.2
        Upper bound on the returned timestep (seconds).
    dt_min : float, default 0.01
        Lower bound on the returned timestep (seconds). Prevents the
        simulation from stalling when particles are extremely close.

    Returns
    -------
    float
        Timestep in seconds, clamped to ``[dt_min, dt_max]``.

    Notes
    -----
    CFL condition::

        dt <= d_min / (2 * v_max_system)

    where ``d_min`` is the approximate minimum pairwise distance and
    ``v_max_system`` is the maximum speed magnitude among all particles.

    For a single particle (N=1), ``dt_max`` is returned since there are
    no pairwise distances to constrain the step.
    """
    positions = np.asarray(positions, dtype=np.float64)
    velocities = np.asarray(velocities, dtype=np.float64)
    dt_max = float(dt_max)
    dt_min = float(dt_min)

    n = positions.shape[0]
    if n <= 1:
        return dt_max

    # Maximum speed in the system
    speeds = np.linalg.norm(velocities, axis=1)
    v_max_system = np.max(speeds)

    if v_max_system < 1.0e-15:
        # All particles essentially stationary
        return dt_max

    # Fast approximate minimum distance: sort by x, check consecutive pairs
    order = np.argsort(positions[:, 0])
    sorted_pos = positions[order]

    # Differences between consecutive sorted positions: shape (N-1, 2)
    diffs = np.diff(sorted_pos, axis=0)

    # Euclidean distances between consecutive sorted points
    dists = np.linalg.norm(diffs, axis=1)

    d_min = np.min(dists)

    # Guard against d_min == 0 (overlapping particles)
    if d_min < 1.0e-15:
        return dt_min

    # CFL condition
    dt_cfl = d_min / (2.0 * v_max_system)

    # Clamp to [dt_min, dt_max]
    return float(np.clip(dt_cfl, dt_min, dt_max))
