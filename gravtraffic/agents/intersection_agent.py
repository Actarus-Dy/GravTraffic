"""IntersectionAgent -- signalized intersection for the GravTraffic ABM.

Controls traffic-light phases and periodically re-optimizes green timings
using the local gravitational potential field as a congestion proxy.

Physics analogy (Janus C-01, section 2.4):
    A red light acts as a temporary large positive mass in the gravitational
    field, creating a potential well that decelerates (repels) approaching
    vehicles.  When the light turns green the mass is removed and vehicles
    coast through the intersection under the ambient field only.

Mesa compatibility: Mesa >= 3.0 (no unique_id in Agent constructor).

Author: Agent #01 Python Scientific Developer
Date: 2026-03-22
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import mesa

from gravtraffic.core.signal_optimizer import optimize_signal_timing

__all__ = ["IntersectionAgent"]


class IntersectionAgent(mesa.Agent):
    """A signalized intersection in the GravTraffic simulation.

    Controls traffic light phases.  Periodically optimizes green timings
    using the local gravitational potential field.

    Parameters
    ----------
    model : mesa.Model
        Parent Mesa model instance.
    position : array-like, shape (2,)
        (x, y) coordinates of the intersection center in meters.
    node_id : int
        Unique identifier for this intersection node in the road network.
    n_phases : int, default 2
        Number of distinct signal phases (e.g. 2 for NS/EW).
    cycle_s : float, default 120.0
        Total signal cycle length in seconds.
    red_light_mass : float, default 50.0
        Gravitational mass injected into the potential field for each
        red-light direction.  Must be positive (creates a congestion well).
    optimize_interval_steps : int, default 300
        Number of simulation steps between optimization calls.
        At dt = 0.1 s this corresponds to 30 s of simulated time.

    Attributes
    ----------
    current_phase : int
        Index of the currently active (green) phase.
    green_times : list[float]
        Duration (seconds) of the green interval for each phase.
    time_in_phase : float
        Elapsed time (seconds) within the current green interval.
    steps_since_optimize : int
        Counter used to trigger periodic re-optimization.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        model: mesa.Model,
        position: npt.ArrayLike,
        node_id: int,
        n_phases: int = 2,
        cycle_s: float = 120.0,
        red_light_mass: float = 50.0,
        optimize_interval_steps: int = 300,
    ) -> None:
        super().__init__(model)

        # Immutable configuration
        self.position: npt.NDArray[np.float64] = np.asarray(
            position, dtype=np.float64
        )
        self.node_id: int = int(node_id)
        self.n_phases: int = int(n_phases)
        self.cycle_s: float = float(cycle_s)
        self.red_light_mass: float = float(red_light_mass)
        self.optimize_interval_steps: int = int(optimize_interval_steps)

        # Validate
        if self.position.shape != (2,):
            raise ValueError(
                f"position must have shape (2,), got {self.position.shape}"
            )
        if self.n_phases < 1:
            raise ValueError(f"n_phases must be >= 1, got {self.n_phases}")
        if self.red_light_mass <= 0.0:
            raise ValueError(
                f"red_light_mass must be positive, got {self.red_light_mass}"
            )

        # Phase state -- equal split initially
        self.current_phase: int = 0
        self.green_times: list[float] = [
            self.cycle_s / self.n_phases
        ] * self.n_phases
        self.time_in_phase: float = 0.0
        self.steps_since_optimize: int = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_green(self) -> list[bool]:
        """Per-phase green status.  ``is_green[i]`` is True iff phase *i*
        is the currently active (green) phase."""
        return [i == self.current_phase for i in range(self.n_phases)]

    # ------------------------------------------------------------------
    # Gravitational field interface
    # ------------------------------------------------------------------

    def get_red_light_masses(self) -> list[tuple[npt.NDArray[np.float64], float]]:
        """Return ``(position, mass)`` pairs for each red-light direction.

        These are injected into the gravitational field as temporary
        positive masses that create congestion wells, decelerating
        approaching vehicles.

        For a 2-phase intersection the convention is:
            - Phase 0 = NS direction (offset along y-axis)
            - Phase 1 = EW direction (offset along x-axis)

        Each red phase contributes two masses, one on each side of the
        intersection center, offset by 15 m (approximate stop-line distance).

        Returns
        -------
        list of (ndarray shape (2,), float)
            Position and mass for every red-light obstacle.
        """
        masses: list[tuple[npt.NDArray[np.float64], float]] = []
        for i in range(self.n_phases):
            if i != self.current_phase:  # red phase
                # Phase 0 = NS -> offset along y; Phase 1 = EW -> offset along x
                if i == 0:
                    offset = np.array([0.0, 15.0], dtype=np.float64)
                else:
                    offset = np.array([15.0, 0.0], dtype=np.float64)
                masses.append((self.position + offset, self.red_light_mass))
                masses.append((self.position - offset, self.red_light_mass))
        return masses

    # ------------------------------------------------------------------
    # Stepping
    # ------------------------------------------------------------------

    def step(self, dt: float = 0.1) -> None:
        """Advance the traffic light by *dt* seconds.

        Increments ``time_in_phase`` and triggers a phase transition when
        the current green interval expires.  The phase index wraps modulo
        ``n_phases``.
        """
        self.time_in_phase += dt
        self.steps_since_optimize += 1

        # Phase transition
        if self.time_in_phase >= self.green_times[self.current_phase]:
            self.current_phase = (self.current_phase + 1) % self.n_phases
            self.time_in_phase = 0.0

    # ------------------------------------------------------------------
    # Optimization
    # ------------------------------------------------------------------

    def try_optimize(
        self,
        vehicle_positions: npt.NDArray[np.float64],
        vehicle_masses: npt.NDArray[np.float64],
        G_s: float = 5.0,
        vehicle_velocities: npt.NDArray[np.float64] | None = None,
    ) -> dict | None:
        """Re-optimize green timings if the optimization interval has elapsed.

        Calls :func:`gravtraffic.core.signal_optimizer.optimize_signal_timing`
        with the current vehicle state to determine new green-phase durations
        using time-integrated potential over a 2-minute horizon.

        Parameters
        ----------
        vehicle_positions : ndarray, shape (N, 2), dtype float64
            Current (x, y) positions of all vehicles.
        vehicle_masses : ndarray, shape (N,), dtype float64
            Signed gravitational masses of all vehicles.
        G_s : float, default 5.0
            Social gravitational constant passed to the optimizer
            (unified calibration).
        vehicle_velocities : ndarray, shape (N, 2), dtype float64, optional
            Current (vx, vy) velocities of all vehicles.  If *None*, zero
            velocities are assumed (falls back to instantaneous evaluation).

        Returns
        -------
        dict or None
            The full optimization result dict if optimization was triggered,
            or *None* if the interval has not yet elapsed.
        """
        if self.steps_since_optimize >= self.optimize_interval_steps:
            self.steps_since_optimize = 0

            n = len(vehicle_masses)
            if vehicle_velocities is None:
                vehicle_velocities = np.zeros((n, 2), dtype=np.float64)

            result = optimize_signal_timing(
                vehicle_positions,
                vehicle_velocities,
                vehicle_masses,
                self.position,
                radius=200.0,
                G_s=G_s,
                red_light_mass=self.red_light_mass,
            )
            if self.n_phases == 2:
                self.green_times = [
                    float(result["green_ns"]),
                    float(result["green_ew"]),
                ]
                self.cycle_s = float(result["cycle_s"])
            return result
        return None

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Return a JSON-serializable dictionary of the agent state.

        Useful for data collection, API export, and snapshot logging.

        Returns
        -------
        dict
            Keys: ``node_id``, ``x``, ``y``, ``current_phase``,
            ``green_times``, ``is_green``.
        """
        return {
            "node_id": self.node_id,
            "x": float(self.position[0]),
            "y": float(self.position[1]),
            "current_phase": self.current_phase,
            "green_times": list(self.green_times),
            "is_green": self.is_green,
        }
