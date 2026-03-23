"""GravSimulation -- full GravTraffic pipeline integrating all core modules.

Connects :class:`MassAssigner`, :class:`ForceEngine`, the leapfrog integrator,
and the potential-field evaluator into a single coherent simulation loop.

Pipeline per step
-----------------
1. Compute segment mean speed from current velocities.
2. Assign signed gravitational masses via :class:`MassAssigner`.
3. Compute forces via :meth:`ForceEngine.compute_all` (Barnes-Hut O(N log N)).
4. Integrate positions and velocities via :func:`leapfrog_step` (KDK).
5. Optionally recompute an adaptive timestep for the next step.

All arithmetic is float64.  No Python loops on the hot path -- the only loops
live inside the Barnes-Hut tree traversal (which is inherently recursive).

Author: Agent #01 Python Scientific Developer
Date: 2026-03-22
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import numpy.typing as npt

from gravtraffic.core.force_engine import ForceEngine
from gravtraffic.core.force_engine_gpu import GPU_AVAILABLE, ForceEngineGPU
from gravtraffic.core.force_engine_numba import (
    NUMBA_AVAILABLE,
    ForceEngineNumba,
    ForceEngineBHNumba,
)
from gravtraffic.core.integrator import adaptive_dt, leapfrog_step
from gravtraffic.core.mass_assigner import MassAssigner
from gravtraffic.core.potential_field import compute_potential_field

__all__ = ["GravSimulation"]

# Floor for |m_i| when converting force -> acceleration, to avoid instability
# for near-zero mass particles.
_MASS_FLOOR: float = 0.01


class GravSimulation:
    """Full GravTraffic simulation pipeline.

    Parameters
    ----------
    G_s : float, default 5.0
        Social gravitational constant (calibrated for unified parameters).
    beta : float, default 0.5
        Mass-assignment exponent (calibrated value from viability test).
    softening : float, default 10.0
        Force softening length in meters.
    rho_scale : float, default 30.0
        Reference density for mass normalisation [veh/km].
    theta : float, default 0.5
        Barnes-Hut opening-angle parameter.
    dt : float, default 0.1
        Base integration timestep in seconds.
    v_max : float, default 36.0
        Maximum allowed vehicle speed in m/s (~130 km/h).
    adaptive_dt : bool, default True
        If True, recompute the timestep after each step using the CFL
        condition.  Otherwise use the fixed ``dt``.
    drag_coefficient : float, default 0.3
        Greenshields drag coefficient (gamma). When > 0, adds a drag
        enrichment term: ``a_drag = gamma * (v_eq(rho) - |v|) * direction``.
        Set to 0.0 for pure gravity (no drag).
    v_free : float, default 33.33
        Free-flow speed in m/s (120 km/h) for the Greenshields model.
    rho_jam : float, default 150.0
        Jam density in vehicles/km for the Greenshields model.
    use_gpu : bool or None, default None
        If True, use CuPy GPU-accelerated force engine. If False, use CPU.
        If None (default), auto-detect: use GPU if CuPy is available.

    Attributes
    ----------
    positions : ndarray, shape (N, 2), dtype float64
        Current vehicle positions.
    velocities : ndarray, shape (N, 2), dtype float64
        Current vehicle velocities.
    masses : ndarray, shape (N,), dtype float64
        Most recently assigned gravitational masses.
    local_densities : ndarray, shape (N,), dtype float64
        Local traffic density at each vehicle [veh/km].
    step_count : int
        Number of completed simulation steps.
    """

    def __init__(
        self,
        G_s: float = 5.0,
        beta: float = 0.5,
        softening: float = 10.0,
        rho_scale: float = 30.0,
        theta: float = 0.5,
        dt: float = 0.1,
        v_max: float = 36.0,
        adaptive_dt: bool = True,
        drag_coefficient: float = 0.3,
        v_free: float = 33.33,
        rho_jam: float = 150.0,
        use_gpu: bool | None = None,
    ) -> None:
        self.G_s: float = float(G_s)
        self.theta: float = float(theta)
        self.dt: float = float(dt)
        self.v_max: float = float(v_max)
        self.use_adaptive_dt: bool = adaptive_dt

        # Drag enrichment parameters (Greenshields equilibrium speed model).
        # When drag_coefficient > 0, an additional acceleration term is applied:
        #   a_drag_i = gamma * (v_eq(rho_i) - |v_i|) * direction_i
        # where v_eq(rho) = v_free * max(0, 1 - rho / rho_jam).
        # This represents engine thrust vs aerodynamic drag, NOT a car-following
        # rule.  Gravity still provides inter-vehicle interactions.
        self._drag_coefficient: float = float(drag_coefficient)
        self._v_free: float = float(v_free)
        self._rho_jam: float = float(rho_jam)

        # GPU auto-detection
        if use_gpu is None:
            use_gpu = GPU_AVAILABLE
        self.use_gpu: bool = use_gpu and GPU_AVAILABLE

        # Sub-modules — engine auto-selection: GPU > Numba > Python
        # Numba naive O(N²) is fastest for N < ~2000 due to JIT + N3L.
        # Numba BH O(N log N) wins for larger N. GPU wins for N < max_n.
        self._mass_assigner = MassAssigner(beta=beta, rho_scale=rho_scale)
        if self.use_gpu:
            self._force_engine = ForceEngineGPU(G_s=G_s, softening=softening)
        elif NUMBA_AVAILABLE:
            self._force_engine = ForceEngineNumba(G_s=G_s, softening=softening)
            self._force_engine_bh = ForceEngineBHNumba(G_s=G_s, softening=softening)
        else:
            self._force_engine = ForceEngine(G_s=G_s, softening=softening)

        # State arrays -- set by init_vehicles
        self.positions: npt.NDArray[np.float64] = np.empty((0, 2), dtype=np.float64)
        self.velocities: npt.NDArray[np.float64] = np.empty((0, 2), dtype=np.float64)
        self.local_densities: npt.NDArray[np.float64] = np.empty(0, dtype=np.float64)
        self.masses: npt.NDArray[np.float64] = np.empty(0, dtype=np.float64)

        # Internal force cache (accelerations) for leapfrog continuity
        self._forces: npt.NDArray[np.float64] = np.empty((0, 2), dtype=np.float64)

        # Static obstacles (e.g. red-light masses) -- participate in force
        # computation but are NOT integrated.
        self._obstacle_positions: npt.NDArray[np.float64] = np.empty(
            (0, 2), dtype=np.float64
        )
        self._obstacle_masses: npt.NDArray[np.float64] = np.empty(
            0, dtype=np.float64
        )

        # Bookkeeping
        self.step_count: int = 0
        self._mean_speed: float = 0.0

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    def init_vehicles(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        local_densities: np.ndarray,
    ) -> None:
        """Set initial conditions for the vehicle population.

        Parameters
        ----------
        positions : ndarray, shape (N, 2), dtype float64
            Initial (x, y) positions of all vehicles.
        velocities : ndarray, shape (N, 2), dtype float64
            Initial (vx, vy) velocities of all vehicles.
        local_densities : ndarray, shape (N,), dtype float64
            Local traffic density at each vehicle position [veh/km].
        """
        self.positions = np.asarray(positions, dtype=np.float64)
        self.velocities = np.asarray(velocities, dtype=np.float64)
        self.local_densities = np.asarray(local_densities, dtype=np.float64)

        n = len(self.local_densities)
        if self.positions.shape != (n, 2):
            raise ValueError(
                f"positions shape {self.positions.shape} incompatible with "
                f"{n} densities; expected ({n}, 2)"
            )
        if self.velocities.shape != (n, 2):
            raise ValueError(
                f"velocities shape {self.velocities.shape} incompatible with "
                f"{n} densities; expected ({n}, 2)"
            )

        # Compute initial masses and forces so that the first leapfrog step
        # has a valid force cache.
        self._mean_speed = self._compute_mean_speed()
        self.masses = self._mass_assigner.assign(
            self._speeds(), self._mean_speed, self.local_densities
        )
        self._forces = self._compute_accelerations(self.positions)
        self.step_count = 0

    # ------------------------------------------------------------------
    # State cloning (for prediction without mutating the live sim)
    # ------------------------------------------------------------------
    def clone(self) -> "GravSimulation":
        """Create an independent deep copy of this simulation.

        The clone shares no mutable state with the original -- modifying
        one does not affect the other.  This is used for prediction:
        clone the live simulation, run the clone forward T seconds, and
        read off the predicted state.

        Returns
        -------
        GravSimulation
            A new simulation with identical configuration and state.
        """
        c = GravSimulation(
            G_s=self.G_s,
            beta=self._mass_assigner.beta,
            softening=self._force_engine.epsilon,
            rho_scale=self._mass_assigner.rho_scale,
            theta=self.theta,
            dt=self.dt,
            v_max=self.v_max,
            adaptive_dt=self.use_adaptive_dt,
            drag_coefficient=self._drag_coefficient,
            v_free=self._v_free,
            rho_jam=self._rho_jam,
            use_gpu=self.use_gpu,
        )
        # Deep-copy all state arrays
        c.positions = self.positions.copy()
        c.velocities = self.velocities.copy()
        c.local_densities = self.local_densities.copy()
        c.masses = self.masses.copy()
        c._forces = self._forces.copy()
        c._obstacle_positions = self._obstacle_positions.copy()
        c._obstacle_masses = self._obstacle_masses.copy()
        c.step_count = self.step_count
        c._mean_speed = self._mean_speed
        return c

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(self, horizon_s: float) -> dict:
        """Run a cloned simulation forward and return the predicted state.

        Parameters
        ----------
        horizon_s : float
            Prediction horizon in seconds (e.g. 900 for T+15min).

        Returns
        -------
        dict
            Keys: ``'positions'``, ``'velocities'``, ``'masses'``,
            ``'mean_speed'``, ``'step_count'``, ``'horizon_s'``,
            ``'n_steps_run'``.
        """
        if self.n_vehicles == 0:
            return {
                "positions": np.empty((0, 2), dtype=np.float64),
                "velocities": np.empty((0, 2), dtype=np.float64),
                "masses": np.empty(0, dtype=np.float64),
                "mean_speed": 0.0,
                "step_count": self.step_count,
                "horizon_s": 0.0,
                "n_steps_run": 0,
            }

        clone = self.clone()
        return clone.run_until(horizon_s)

    def run_until(self, horizon_s: float) -> dict:
        """Run THIS simulation forward for *horizon_s* seconds (in-place).

        Unlike :meth:`predict`, this does NOT clone -- it mutates ``self``.
        Use on a clone obtained via :meth:`clone` to avoid modifying the
        live simulation.

        Parameters
        ----------
        horizon_s : float
            Time horizon in seconds.

        Returns
        -------
        dict
            Same keys as :meth:`predict`.
        """
        elapsed = 0.0
        n_steps = 0
        max_steps = max(1, int(horizon_s / 0.005))
        while elapsed < horizon_s and n_steps < max_steps:
            dt_before = self.dt
            self.step()
            elapsed += dt_before
            n_steps += 1

        return {
            "positions": self.positions.copy(),
            "velocities": self.velocities.copy(),
            "masses": self.masses.copy(),
            "mean_speed": float(np.mean(np.linalg.norm(self.velocities, axis=1)))
            if self.n_vehicles > 0
            else 0.0,
            "step_count": self.step_count,
            "horizon_s": elapsed,
            "n_steps_run": n_steps,
        }

    # ------------------------------------------------------------------
    # Dynamic vehicle injection / removal
    # ------------------------------------------------------------------
    def add_vehicles(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        local_densities: np.ndarray,
    ) -> npt.NDArray[np.intp]:
        """Add K vehicles to the running simulation.

        Parameters
        ----------
        positions : array_like, shape (K, 2)
            Positions of the new vehicles.
        velocities : array_like, shape (K, 2)
            Velocities of the new vehicles.
        local_densities : array_like, shape (K,)
            Local traffic density at each new vehicle [veh/km].

        Returns
        -------
        ndarray, shape (K,), dtype intp
            Indices assigned to the new vehicles in the state arrays.
        """
        positions = np.asarray(positions, dtype=np.float64).reshape(-1, 2)
        velocities = np.asarray(velocities, dtype=np.float64).reshape(-1, 2)
        local_densities = np.asarray(local_densities, dtype=np.float64).ravel()

        k = len(positions)
        old_n = len(self.positions)

        self.positions = np.vstack([self.positions, positions])
        self.velocities = np.vstack([self.velocities, velocities])
        self.local_densities = np.concatenate([self.local_densities, local_densities])

        # Compute masses for new vehicles using current mean speed
        new_masses = self._mass_assigner.assign(
            np.linalg.norm(velocities, axis=1),
            self._mean_speed,
            local_densities,
        )
        self.masses = np.concatenate([self.masses, new_masses])

        # Extend forces array (zero initial force for new vehicles)
        self._forces = np.vstack([self._forces, np.zeros((k, 2), dtype=np.float64)])

        return np.arange(old_n, old_n + k)

    def remove_vehicles(self, indices: np.ndarray) -> None:
        """Remove vehicles at given indices from the simulation.

        Parameters
        ----------
        indices : array_like, dtype intp
            Indices of vehicles to remove.  Must be valid indices into
            the current state arrays.
        """
        indices = np.asarray(indices, dtype=np.intp)
        mask = np.ones(len(self.positions), dtype=bool)
        mask[indices] = False

        self.positions = self.positions[mask]
        self.velocities = self.velocities[mask]
        self.local_densities = self.local_densities[mask]
        self.masses = self.masses[mask]
        self._forces = self._forces[mask]

    @property
    def n_vehicles(self) -> int:
        """Current number of vehicles in the simulation."""
        return len(self.positions)

    # ------------------------------------------------------------------
    # Obstacle management (red-light masses, barriers, etc.)
    # ------------------------------------------------------------------
    def set_obstacles(
        self,
        positions: np.ndarray,
        masses: np.ndarray,
    ) -> None:
        """Set static obstacle positions and masses for force computation.

        Obstacles participate in force calculation (they exert forces on
        vehicles) but are NOT integrated -- their positions do not change.

        Parameters
        ----------
        positions : array-like, shape (K, 2), dtype float64
            (x, y) positions of the K obstacles.
        masses : array-like, shape (K,), dtype float64
            Signed gravitational mass for each obstacle.

        Raises
        ------
        ValueError
            If *positions* and *masses* have incompatible shapes.
        """
        self._obstacle_positions = np.asarray(
            positions, dtype=np.float64
        ).reshape(-1, 2)
        self._obstacle_masses = np.asarray(masses, dtype=np.float64).ravel()

        k = len(self._obstacle_masses)
        if self._obstacle_positions.shape != (k, 2):
            raise ValueError(
                f"obstacle positions shape {self._obstacle_positions.shape} "
                f"incompatible with {k} masses; expected ({k}, 2)"
            )

    def clear_obstacles(self) -> None:
        """Remove all obstacles from the simulation."""
        self._obstacle_positions = np.empty((0, 2), dtype=np.float64)
        self._obstacle_masses = np.empty(0, dtype=np.float64)

    # ------------------------------------------------------------------
    # Simulation step
    # ------------------------------------------------------------------
    def step(self) -> dict:
        """Execute one simulation step.

        Returns
        -------
        dict
            Step results with keys:

            - ``'positions'``: (N, 2) ndarray
            - ``'velocities'``: (N, 2) ndarray
            - ``'masses'``: (N,) ndarray
            - ``'mean_speed'``: float
            - ``'dt_used'``: float
            - ``'step_count'``: int
        """
        dt = self.dt

        # Leapfrog KDK integration.  The force_fn callback recomputes
        # masses at the new positions (using the current mean speed) and
        # converts the resulting forces to accelerations.
        pos_new, vel_new, forces_new = leapfrog_step(
            self.positions,
            self.velocities,
            self._forces,
            dt,
            force_fn=self._compute_accelerations,
            v_max=self.v_max,
        )

        # Commit new state
        self.positions = pos_new
        self.velocities = vel_new
        self._forces = forces_new

        # Update local densities from current positions (once per step)
        self.local_densities = self._compute_local_densities(self.positions)

        # Recompute mean speed and masses at the committed state
        self._mean_speed = self._compute_mean_speed()
        self.masses = self._mass_assigner.assign(
            self._speeds(), self._mean_speed, self.local_densities
        )

        self.step_count += 1

        # Adaptive timestep for the next step
        if self.use_adaptive_dt:
            self.dt = adaptive_dt(self.positions, self.velocities)

        return {
            "positions": self.positions.copy(),
            "velocities": self.velocities.copy(),
            "masses": self.masses.copy(),
            "mean_speed": self._mean_speed,
            "dt_used": dt,
            "step_count": self.step_count,
        }

    # ------------------------------------------------------------------
    # Multi-step runner
    # ------------------------------------------------------------------
    def run(self, n_steps: int) -> list[dict]:
        """Run *n_steps* simulation steps.

        Parameters
        ----------
        n_steps : int
            Number of steps to execute.

        Returns
        -------
        list[dict]
            List of length *n_steps*, each element being the dict returned
            by :meth:`step`.
        """
        return [self.step() for _ in range(n_steps)]

    # ------------------------------------------------------------------
    # Potential field
    # ------------------------------------------------------------------
    def get_potential_field(
        self, grid_centers: np.ndarray
    ) -> npt.NDArray[np.float64]:
        """Compute the gravitational potential field at the current state.

        Parameters
        ----------
        grid_centers : ndarray, shape (M, 2), dtype float64
            Evaluation points where the potential is computed.

        Returns
        -------
        ndarray, shape (M,), dtype float64
            Scalar potential at each grid point.
        """
        return compute_potential_field(
            self.positions, self.masses, grid_centers, G_s=self.G_s
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _compute_local_densities(
        self, positions: np.ndarray, radius: float = 100.0
    ) -> npt.NDArray[np.float64]:
        """Compute local traffic density for each vehicle.

        Counts vehicles within a Euclidean radius and converts to
        vehicles/km.  Uses ``scipy.spatial.cKDTree`` for O(N log N)
        vectorized neighbor counting.

        Parameters
        ----------
        positions : ndarray, shape (N, 2), dtype float64
            Current vehicle positions.
        radius : float, default 100.0
            Neighborhood radius in meters for density calculation.

        Returns
        -------
        densities : ndarray, shape (N,), dtype float64
            Local traffic density at each vehicle [veh/km].
        """
        from scipy.spatial import cKDTree

        n = len(positions)
        if n <= 1:
            return np.full(n, 1.0 / (2 * radius / 1000), dtype=np.float64)

        tree = cKDTree(positions)
        # count_neighbors returns the number of points within radius
        # (including self)
        counts = tree.query_ball_point(positions, r=radius, return_length=True)
        counts = np.asarray(counts, dtype=np.float64)

        # Convert count in radius to vehicles/km
        # Window length = 2*radius (vehicles can be on either side)
        window_km = 2 * radius / 1000.0
        densities = counts / window_km

        return densities

    def _speeds(self) -> npt.NDArray[np.float64]:
        """Return speed magnitudes for all vehicles.  Shape (N,)."""
        return np.linalg.norm(self.velocities, axis=1)

    def _compute_mean_speed(self) -> float:
        """Return the population mean speed."""
        if self.velocities.shape[0] == 0:
            return 0.0
        return float(np.mean(self._speeds()))

    def _compute_accelerations(
        self,
        positions: npt.NDArray[np.float64],
        velocities: npt.NDArray[np.float64] | None = None,
    ) -> npt.NDArray[np.float64]:
        """Force callback for the leapfrog integrator.

        1. Recompute masses at *positions* using the current mean speed.
        2. Compute forces via Barnes-Hut.
        3. Convert forces to accelerations: a_i = F_i / max(|m_i|, 0.01).

        Parameters
        ----------
        positions : ndarray, shape (N, 2), dtype float64
            Particle positions (may differ from ``self.positions`` during
            the leapfrog drift sub-step).
        velocities : ndarray, shape (N, 2), dtype float64, optional
            Particle velocities to use for mass assignment and drag.
            When called from the leapfrog integrator this is ``v_half``
            (the half-kick velocities), ensuring symplecticity.
            If *None*, falls back to ``self.velocities`` (used during
            init_vehicles bootstrap).

        Returns
        -------
        ndarray, shape (N, 2), dtype float64
            Acceleration vectors for all vehicles.
        """
        # Use provided velocities (v_half from leapfrog) when available,
        # otherwise fall back to self.velocities (bootstrap / init).
        if velocities is None:
            velocities = self.velocities

        # Recompute masses at the (possibly drifted) positions.
        # We use the current mean speed -- it does not change within a step.
        speeds = np.linalg.norm(velocities, axis=1)
        masses = self._mass_assigner.assign(
            speeds, self._mean_speed, self.local_densities
        )

        n_vehicles = len(masses)

        # Concatenate static obstacles (red-light masses, etc.) so they
        # participate in force computation.  Obstacles exert forces on
        # vehicles but are NOT integrated -- we slice them off afterwards.
        if len(self._obstacle_masses) > 0:
            all_positions = np.vstack([positions, self._obstacle_positions])
            all_masses = np.concatenate([masses, self._obstacle_masses])
        else:
            all_positions = positions
            all_masses = masses

        # Force computation — auto-select engine by N:
        # Numba naive for N < 2000, Numba BH for N >= 2000 (if available)
        engine = self._force_engine
        if (hasattr(self, '_force_engine_bh')
                and len(all_masses) >= 2000):
            engine = self._force_engine_bh
        all_forces = engine.compute_all(
            all_positions, all_masses, theta=self.theta
        )

        # Keep only the vehicle forces; discard forces on obstacles.
        forces = all_forces[:n_vehicles]

        # Convert force -> acceleration: a = F / |m|, with floor to avoid
        # division by zero for near-zero mass particles.
        abs_masses = np.maximum(np.abs(masses), _MASS_FLOOR)  # (N,)
        accelerations = forces / abs_masses[:, np.newaxis]    # (N, 2)

        # --- Drag enrichment (Greenshields equilibrium speed model) ---
        # Physically motivated: engine thrust vs aerodynamic drag.
        # a_drag_i = gamma * (v_eq(rho_i) - |v_i|) * direction_i
        # When |v_i| > v_eq: deceleration.  When |v_i| < v_eq: acceleration.
        if self._drag_coefficient > 0:
            speed = np.linalg.norm(velocities, axis=1, keepdims=True)  # (N, 1)
            speed_scalar = speed.ravel()                                # (N,)

            # Greenshields equilibrium speed from local density
            v_eq = self._v_free * np.maximum(
                0.0, 1.0 - self.local_densities / self._rho_jam
            )  # (N,)

            # Unit direction vector (along current velocity); fallback to +x
            # for stationary vehicles to avoid zero-division.
            safe_speed = np.maximum(speed, 1e-6)
            direction = np.where(
                speed > 1e-6,
                velocities / safe_speed,
                np.array([[1.0, 0.0]], dtype=np.float64),
            )  # (N, 2)

            # Drag acceleration: scalar (v_eq - |v|) applied along direction
            drag_scalar = self._drag_coefficient * (v_eq - speed_scalar)  # (N,)
            accelerations += drag_scalar[:, np.newaxis] * direction       # (N, 2)

        return accelerations
