"""GreenWaveCoordinator -- synchronize traffic lights along a corridor.

A green wave offsets signal phases so that a platoon driving at a target
speed encounters a continuous sequence of green lights.  The phase offset
for intersection *i* is:

    offset_i = d_i / v_wave

where ``d_i`` is the projected distance from the first intersection along
the corridor direction and ``v_wave`` is the target progression speed.

The ``optimize_wave_speed`` method performs a brute-force bandwidth
maximization: it sweeps candidate speeds and selects the one that
maximizes the fraction of the cycle during which a vehicle can traverse
the entire corridor on green (the *bandwidth*).

Physics analogy (Janus C-01, section 3.2):
    The green wave propagates like a coherent wavefront in the
    gravitational field -- intersections fire in phase so that the
    potential wells (red lights) are never encountered by the platoon.

Mesa compatibility: Mesa >= 3.0 (no unique_id in Agent constructor).

Author: Agent #01 Python Scientific Developer
Date: 2026-03-22
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from gravtraffic.agents.intersection_agent import IntersectionAgent

__all__ = ["GreenWaveCoordinator"]


class GreenWaveCoordinator:
    """Synchronize traffic light phases along a corridor for green wave progression.

    The offset for each intersection is::

        offset_i = distance_i / v_wave

    where ``distance_i`` is the projection of the intersection position onto
    the corridor direction, measured relative to the first (closest)
    intersection.

    Parameters
    ----------
    wave_speed : float
        Target progression speed in m/s.  Default 13.888... m/s = 50 km/h.

    Attributes
    ----------
    wave_speed : float
        Current wave speed in m/s.

    References
    ----------
    .. [1] Webster, F. V. (1958). *Traffic Signal Settings*.
       Road Research Technical Paper No. 39.
    """

    def __init__(self, wave_speed: float = 50.0 / 3.6) -> None:
        if wave_speed <= 0.0:
            raise ValueError(f"wave_speed must be positive, got {wave_speed}")
        self.wave_speed: float = float(wave_speed)

    # ------------------------------------------------------------------
    # Offset computation
    # ------------------------------------------------------------------

    def compute_offsets(
        self,
        intersection_positions: npt.NDArray[np.float64],
        corridor_direction: npt.NDArray[np.float64] | None = None,
    ) -> npt.NDArray[np.float64]:
        """Compute phase offsets for each intersection along the corridor.

        Projects intersection positions onto the corridor direction vector,
        then computes ``offset_i = projected_distance_i / wave_speed``.

        Parameters
        ----------
        intersection_positions : ndarray, shape (K, 2), dtype float64
            (x, y) coordinates of each intersection center in meters.
        corridor_direction : ndarray, shape (2,), dtype float64, optional
            Unit vector giving the corridor direction.  Defaults to the
            positive x-axis ``[1, 0]``.

        Returns
        -------
        ndarray, shape (K,), dtype float64
            Phase offsets in seconds.  The intersection closest to the
            corridor origin has offset 0.
        """
        intersection_positions = np.asarray(intersection_positions, dtype=np.float64)
        if intersection_positions.ndim != 2 or intersection_positions.shape[1] != 2:
            raise ValueError(
                f"intersection_positions must have shape (K, 2), got {intersection_positions.shape}"
            )

        if corridor_direction is None:
            corridor_direction = np.array([1.0, 0.0], dtype=np.float64)
        else:
            corridor_direction = np.asarray(corridor_direction, dtype=np.float64)

        # Normalize to unit vector
        norm = np.linalg.norm(corridor_direction)
        if norm < 1e-15:
            raise ValueError("corridor_direction must be non-zero")
        corridor_direction = corridor_direction / norm

        # Project positions onto corridor axis
        projections: npt.NDArray[np.float64] = intersection_positions @ corridor_direction

        # Offset relative to the first (minimum-projection) intersection
        offsets: npt.NDArray[np.float64] = (projections - projections.min()) / self.wave_speed

        return offsets

    # ------------------------------------------------------------------
    # Apply offsets to intersection agents
    # ------------------------------------------------------------------

    def apply_offsets(
        self,
        intersections: list[IntersectionAgent],
        offsets: npt.NDArray[np.float64],
    ) -> None:
        """Apply computed offsets to intersection agents.

        Sets each agent's ``current_phase`` and ``time_in_phase`` so that
        the agent behaves as though it started its cycle ``offset_i``
        seconds ago.

        Parameters
        ----------
        intersections : list of IntersectionAgent
            Intersection agents along the corridor, in the same order as
            the positions used to compute *offsets*.
        offsets : ndarray, shape (K,), dtype float64
            Phase offsets in seconds, as returned by :meth:`compute_offsets`.

        Raises
        ------
        ValueError
            If *intersections* and *offsets* have different lengths.
        """
        offsets = np.asarray(offsets, dtype=np.float64)
        if len(intersections) != len(offsets):
            raise ValueError(
                f"Length mismatch: {len(intersections)} intersections vs {len(offsets)} offsets"
            )

        for agent, offset in zip(intersections, offsets):
            cycle = sum(agent.green_times)
            offset_mod = float(offset) % cycle

            # Walk through phases to find where offset_mod falls
            cumulative = 0.0
            for phase_idx, gt in enumerate(agent.green_times):
                if cumulative + gt > offset_mod:
                    agent.current_phase = phase_idx
                    agent.time_in_phase = offset_mod - cumulative
                    break
                cumulative += gt

    # ------------------------------------------------------------------
    # Bandwidth-optimal wave speed
    # ------------------------------------------------------------------

    def optimize_wave_speed(
        self,
        intersection_positions: npt.NDArray[np.float64],
        green_times: list[float],
        speed_range: tuple[float, float] = (8.0, 20.0),
        n_candidates: int = 25,
        corridor_direction: npt.NDArray[np.float64] | None = None,
    ) -> float:
        """Find the wave speed that maximizes the green-wave bandwidth.

        The *bandwidth* is the fraction of the cycle during which a vehicle
        can traverse all intersections on green without stopping.  It is
        computed as::

            bandwidth = max(0, main_green - spread) / cycle

        where ``spread`` is the range of fractional offsets (mod cycle) across
        intersections, and ``main_green`` is the green time for the corridor
        direction (first element of *green_times*).

        A brute-force sweep over *n_candidates* speeds in *speed_range* is
        performed.  The speed yielding the largest bandwidth is returned.

        Parameters
        ----------
        intersection_positions : ndarray, shape (K, 2), dtype float64
            Intersection coordinates in meters.
        green_times : list of float
            Green durations for each phase.  ``green_times[0]`` is assumed
            to be the corridor (main) direction.
        speed_range : (float, float), default (8.0, 20.0)
            Min and max candidate speeds in m/s.
        n_candidates : int, default 25
            Number of speeds to evaluate in the sweep.
        corridor_direction : ndarray, shape (2,), optional
            Corridor direction vector.  Defaults to ``[1, 0]``.

        Returns
        -------
        float
            Optimal wave speed in m/s.
        """
        intersection_positions = np.asarray(intersection_positions, dtype=np.float64)

        if corridor_direction is None:
            corridor_dir = np.array([1.0, 0.0], dtype=np.float64)
        else:
            corridor_dir = np.asarray(corridor_direction, dtype=np.float64)
            norm = np.linalg.norm(corridor_dir)
            if norm < 1e-15:
                raise ValueError("corridor_direction must be non-zero")
            corridor_dir = corridor_dir / norm

        projections = intersection_positions @ corridor_dir
        distances = projections - projections.min()

        cycle = float(sum(green_times))
        main_green = float(green_times[0])

        best_speed = self.wave_speed
        best_bandwidth = -1.0

        for v in np.linspace(speed_range[0], speed_range[1], n_candidates, dtype=np.float64):
            offsets = distances / v
            fractional_offsets = offsets % cycle
            spread = float(np.max(fractional_offsets) - np.min(fractional_offsets))
            bandwidth = max(0.0, main_green - spread) / cycle

            if bandwidth > best_bandwidth:
                best_bandwidth = bandwidth
                best_speed = float(v)

        return best_speed
