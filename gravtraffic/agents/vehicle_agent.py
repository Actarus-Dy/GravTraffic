"""VehicleAgent -- Mesa ABM agent for GravTraffic simulation.

Each vehicle holds its kinematic state (position, velocity) and its
gravitational mass as computed by the central :class:`GravSimulation` engine.
The agent does **not** compute forces or make behavioural decisions; all
physics is resolved centrally, and results are pushed to the agent via
:meth:`update_from_simulation`.

Mass classification follows the same thresholds as
:mod:`gravtraffic.core.mass_assigner`:

    * ``mass >  +0.1``  -->  ``"slow"``   (positive mass, attractor)
    * ``mass <  -0.1``  -->  ``"fast"``   (negative mass, repulsor)
    * ``|mass| <= 0.1`` -->  ``"neutral"``

Reference
---------
Janus Civil C-01 GravTraffic Technical Plan, Section 5.1.
"""

from __future__ import annotations

import numpy as np
import mesa

__all__ = ["VehicleAgent"]

# Must match gravtraffic.core.mass_assigner._NEUTRAL_THRESHOLD
_NEUTRAL_THRESHOLD: float = 0.1


class VehicleAgent(mesa.Agent):
    """A vehicle in the GravTraffic simulation.

    This agent does **not** compute its own forces or make behavioural
    decisions.  All physics is computed centrally by ``GravSimulation``,
    and results are pushed to the agent via :meth:`update_from_simulation`.

    The agent's role is to:

    - Hold its current state (position, velocity, mass).
    - Provide its state to the model for physics computation.
    - Receive updated state after physics computation.
    - Track its history for data collection.

    Parameters
    ----------
    model : mesa.Model
        The parent Mesa model (Mesa 3.x API).
    position : array_like, shape (2,)
        Initial position ``[x, y]`` in metres.
    velocity : array_like, shape (2,)
        Initial velocity ``[vx, vy]`` in m/s.
    local_density : float, default 30.0
        Initial local traffic density at vehicle position [veh/km].
    v_max : float, default 36.0
        Maximum allowed speed for this vehicle [m/s] (approx. 130 km/h).
    """

    def __init__(
        self,
        model: mesa.Model,
        position: np.ndarray,
        velocity: np.ndarray,
        local_density: float = 30.0,
        v_max: float = 36.0,
    ) -> None:
        # Mesa 3.x: Agent.__init__(self, model) -- unique_id auto-assigned
        super().__init__(model)

        # Kinematic state -- always float64 for numerical accuracy
        self.position: np.ndarray = np.asarray(position, dtype=np.float64)
        self.velocity: np.ndarray = np.asarray(velocity, dtype=np.float64)

        # Traffic parameters
        self.local_density: float = float(local_density)
        self.v_max: float = float(v_max)

        # Gravitational mass (set by MassAssigner via update_from_simulation)
        self.mass: float = 0.0
        self.mass_type: str = "neutral"  # 'slow', 'fast', 'neutral'

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def speed(self) -> float:
        """Scalar speed magnitude in m/s.

        Returns
        -------
        float
            ``||velocity||_2``
        """
        return float(np.linalg.norm(self.velocity))

    @property
    def speed_kmh(self) -> float:
        """Speed converted to km/h.

        Returns
        -------
        float
            ``speed * 3.6``
        """
        return self.speed * 3.6

    # ------------------------------------------------------------------
    # State update (called by the central simulation engine)
    # ------------------------------------------------------------------
    def update_from_simulation(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        mass: float,
    ) -> None:
        """Receive updated state from the central GravSimulation engine.

        Parameters
        ----------
        position : array_like, shape (2,)
            New position ``[x, y]`` in metres.
        velocity : array_like, shape (2,)
            New velocity ``[vx, vy]`` in m/s.
        mass : float
            New gravitational mass (signed).
        """
        self.position = np.asarray(position, dtype=np.float64)
        self.velocity = np.asarray(velocity, dtype=np.float64)
        self.mass = float(mass)

        # Classify using the same threshold as MassAssigner
        if mass > _NEUTRAL_THRESHOLD:
            self.mass_type = "slow"
        elif mass < -_NEUTRAL_THRESHOLD:
            self.mass_type = "fast"
        else:
            self.mass_type = "neutral"

    # ------------------------------------------------------------------
    # Mesa step (no-op)
    # ------------------------------------------------------------------
    def step(self) -> None:
        """No-op: physics is computed centrally, not per-agent."""
        pass

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        """Serialize agent state for API / visualisation.

        Returns
        -------
        dict
            Keys: ``id``, ``x``, ``y``, ``vx``, ``vy``, ``mass``,
            ``speed_kmh``, ``type``.
        """
        return {
            "id": self.unique_id,
            "x": float(self.position[0]),
            "y": float(self.position[1]),
            "vx": float(self.velocity[0]),
            "vy": float(self.velocity[1]),
            "mass": self.mass,
            "speed_kmh": self.speed_kmh,
            "type": self.mass_type,
        }

    def __repr__(self) -> str:
        return (
            f"VehicleAgent(id={self.unique_id}, "
            f"pos=[{self.position[0]:.1f}, {self.position[1]:.1f}], "
            f"speed={self.speed_kmh:.1f} km/h, "
            f"mass={self.mass:.3f}, type={self.mass_type!r})"
        )
