"""TrafficModel -- Mesa Model orchestrating GravSimulation and agents.

This is the top-level simulation object for C-01 GravTraffic.  It coordinates
data flow between three subsystems:

1. **GravSimulation** (physics engine):  mass assignment, Barnes-Hut force
   computation, leapfrog integration.  All numerics live here.
2. **VehicleAgent** (per-vehicle state):  holds position, velocity, mass for
   visualization and data collection.  Does *not* compute its own physics.
3. **IntersectionAgent** (traffic lights):  phase cycling and potential-field
   based green-time optimization.

Data flow per step
------------------
1. Collect red-light masses from IntersectionAgents (tracked for future use).
2. Run one physics step via ``GravSimulation.step()``.
3. Push updated positions, velocities, and masses to VehicleAgents.
4. Advance IntersectionAgents (phase cycling) and attempt re-optimization.
5. Collect KPI data via ``mesa.DataCollector``.

Mesa compatibility: Mesa >= 3.0.  Uses ``rng=`` (not deprecated ``seed=``).

Author: Agent #01 Python Scientific Developer
Date: 2026-03-22
"""

from __future__ import annotations

import mesa
import numpy as np
import numpy.typing as npt

from gravtraffic.agents.intersection_agent import IntersectionAgent
from gravtraffic.agents.vehicle_agent import VehicleAgent
from gravtraffic.core.potential_field import compute_potential_field, make_grid
from gravtraffic.core.simulation import GravSimulation
from gravtraffic.network.road_network import RoadNetwork

__all__ = ["TrafficModel"]


class TrafficModel(mesa.Model):
    """GravTraffic Mesa Model -- orchestrates physics simulation and agents.

    Parameters
    ----------
    network : RoadNetwork
        Road network providing topology, positions, and speed limits.
    n_vehicles : int, default 100
        Number of vehicle agents to spawn on the network.
    G_s : float, default 5.0
        Social gravitational constant (calibrated for unified parameters).
    beta : float, default 0.5
        Mass-assignment exponent (calibrated).
    softening : float, default 10.0
        Force softening length in metres.
    theta : float, default 0.5
        Barnes-Hut opening-angle parameter.
    dt : float, default 0.1
        Base integration timestep in seconds.
    v_max : float, default 36.0
        Maximum allowed vehicle speed in m/s (~130 km/h).
    signal_intersections : bool, default True
        If True, create IntersectionAgents at nodes with degree >= 3.
    seed : int, default 42
        Random seed for reproducibility.
    drag_coefficient : float, default 0.3
        Greenshields drag coefficient (gamma).
    v_free : float, default 33.33
        Free-flow speed in m/s (120 km/h).
    rho_jam : float, default 150.0
        Jam density in vehicles/km.

    Attributes
    ----------
    vehicle_agents : list[VehicleAgent]
        All vehicle agents, index-aligned with the physics engine arrays.
    intersection_agents : list[IntersectionAgent]
        All signalized intersection agents.
    simulation : GravSimulation
        The central physics engine.
    step_count : int
        Number of completed simulation steps.
    """

    def __init__(
        self,
        network: RoadNetwork,
        n_vehicles: int = 100,
        G_s: float = 5.0,
        beta: float = 0.5,
        softening: float = 10.0,
        theta: float = 0.5,
        dt: float = 0.1,
        v_max: float = 36.0,
        signal_intersections: bool = True,
        seed: int = 42,
        drag_coefficient: float = 0.3,
        v_free: float = 33.33,
        rho_jam: float = 150.0,
    ) -> None:
        # Mesa 3.5+: use rng= to avoid FutureWarning on deprecated seed=
        super().__init__(rng=np.random.default_rng(seed))

        self.network: RoadNetwork = network
        self.G_s: float = float(G_s)
        self.v_max: float = float(v_max)
        self.dt: float = float(dt)
        self.step_count: int = 0
        self._last_step_result: dict | None = None

        # ---- Physics engine ------------------------------------------------
        self.simulation = GravSimulation(
            G_s=G_s,
            beta=beta,
            softening=softening,
            theta=theta,
            dt=dt,
            v_max=v_max,
            adaptive_dt=True,
            drag_coefficient=drag_coefficient,
            v_free=v_free,
            rho_jam=rho_jam,
        )

        # ---- Spawn vehicles on the network ---------------------------------
        rng = np.random.default_rng(seed)
        positions = network.sample_positions(n_vehicles, rng=rng)

        # Initial velocities: random speed along a random unit direction
        # (uniformly distributed on the circle, scaled by speed magnitude)
        speeds = rng.uniform(10.0, 30.0, n_vehicles)  # m/s
        angles = rng.uniform(0.0, 2.0 * np.pi, n_vehicles)
        velocities = np.column_stack(
            [
                speeds * np.cos(angles),
                speeds * np.sin(angles),
            ]
        )

        # Local densities: uniform initial estimate
        densities = np.full(n_vehicles, 30.0, dtype=np.float64)

        # Initialize the physics engine with vehicle state
        self.simulation.init_vehicles(positions, velocities, densities)

        # ---- VehicleAgents --------------------------------------------------
        self.vehicle_agents: list[VehicleAgent] = []
        for i in range(n_vehicles):
            agent = VehicleAgent(
                self,
                position=positions[i],
                velocity=velocities[i],
                local_density=densities[i],
                v_max=v_max,
            )
            self.vehicle_agents.append(agent)

        # ---- IntersectionAgents at high-degree nodes ------------------------
        self.intersection_agents: list[IntersectionAgent] = []
        if signal_intersections:
            for node_info in network.intersections:
                if node_info["degree"] >= 3:
                    agent = IntersectionAgent(
                        self,
                        position=np.array([node_info["x"], node_info["y"]], dtype=np.float64),
                        node_id=node_info["node_id"],
                    )
                    self.intersection_agents.append(agent)

        # ---- Mesa DataCollector ---------------------------------------------
        self.datacollector = mesa.DataCollector(
            model_reporters={
                "mean_speed_kmh": _mean_speed_kmh,
                "congestion_index": _congestion_index,
                "n_slow": _n_slow,
                "n_fast": _n_fast,
                "n_neutral": _n_neutral,
            }
        )

    # ------------------------------------------------------------------
    # Dynamic vehicle injection / removal
    # ------------------------------------------------------------------

    def spawn_vehicle(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        local_density: float = 30.0,
    ) -> VehicleAgent:
        """Spawn a single vehicle and add it to both simulation and agent list.

        Parameters
        ----------
        position : array_like, shape (2,)
            Initial position ``[x, y]`` in metres.
        velocity : array_like, shape (2,)
            Initial velocity ``[vx, vy]`` in m/s.
        local_density : float, default 30.0
            Local traffic density at the spawn point [veh/km].

        Returns
        -------
        VehicleAgent
            The newly created agent.
        """
        position = np.asarray(position, dtype=np.float64)
        velocity = np.asarray(velocity, dtype=np.float64)

        self.simulation.add_vehicles(
            position.reshape(1, 2),
            velocity.reshape(1, 2),
            np.array([local_density], dtype=np.float64),
        )
        agent = VehicleAgent(
            self,
            position=position,
            velocity=velocity,
            local_density=local_density,
            v_max=self.v_max,
        )
        self.vehicle_agents.append(agent)
        return agent

    def despawn_vehicle(self, index: int) -> None:
        """Remove a vehicle by index from both simulation and agent list.

        Parameters
        ----------
        index : int
            Index of the vehicle to remove (into ``vehicle_agents`` and the
            physics engine arrays, which are kept index-aligned).
        """
        self.simulation.remove_vehicles(np.array([index], dtype=np.intp))
        self.vehicle_agents.pop(index)

    def despawn_out_of_bounds(
        self,
        x_min: float,
        y_min: float,
        x_max: float,
        y_max: float,
    ) -> int:
        """Remove all vehicles outside a bounding box.

        Parameters
        ----------
        x_min, y_min, x_max, y_max : float
            Bounding box limits in metres.

        Returns
        -------
        int
            Number of vehicles removed.
        """
        positions = self.simulation.positions
        out = (
            (positions[:, 0] < x_min)
            | (positions[:, 0] > x_max)
            | (positions[:, 1] < y_min)
            | (positions[:, 1] > y_max)
        )

        if not out.any():
            return 0

        indices = np.where(out)[0]

        # Remove from physics engine in one batch (boolean mask internally)
        self.simulation.remove_vehicles(indices)

        # Remove agents in reverse order to preserve earlier indices
        for i in sorted(indices, reverse=True):
            self.vehicle_agents.pop(i)

        return int(out.sum())

    # ------------------------------------------------------------------
    # Simulation step
    # ------------------------------------------------------------------

    def step(self) -> None:
        """Execute one simulation step.

        In Mesa >= 3.5, ``Model.step()`` is replaced by an internal wrapper
        that drives the event-based scheduler.  The user-defined ``step()``
        is called internally and its return value is discarded.  To access
        the physics result dict, use :attr:`last_step_result` after calling
        ``step()``.

        The result dict (also stored in ``self._last_step_result``) has keys:
        ``'positions'``, ``'velocities'``, ``'masses'``,
        ``'mean_speed'``, ``'dt_used'``, ``'step_count'``.
        """
        self.step_count += 1

        # 1. Collect red-light masses from IntersectionAgents and inject
        #    them into GravSimulation as static obstacles.  Positive masses
        #    repel fast vehicles (deceleration) and attract slow vehicles
        #    (jam formation) -- physically correct per Janus sign convention.
        obstacle_positions: list[npt.NDArray[np.float64]] = []
        obstacle_masses: list[float] = []
        for ia in self.intersection_agents:
            for pos, mass in ia.get_red_light_masses():
                obstacle_positions.append(pos)
                obstacle_masses.append(mass)

        if obstacle_positions:
            self.simulation.set_obstacles(
                np.array(obstacle_positions, dtype=np.float64),
                np.array(obstacle_masses, dtype=np.float64),
            )
        else:
            self.simulation.clear_obstacles()

        # 2. Run one physics step (mass assignment + forces + integration)
        result = self.simulation.step()

        # 3. Push updated state to VehicleAgents
        positions = result["positions"]
        velocities = result["velocities"]
        masses = result["masses"]

        for i, agent in enumerate(self.vehicle_agents):
            agent.update_from_simulation(positions[i], velocities[i], masses[i])

        # 4. Step IntersectionAgents (phase cycling + optimization attempt)
        dt_used = result.get("dt_used", self.dt)
        for ia in self.intersection_agents:
            ia.step(dt=dt_used)
            ia.try_optimize(positions, masses, G_s=self.G_s, vehicle_velocities=velocities)

        # 5. Collect KPI data
        self.datacollector.collect(self)

        # Store result for external access (Mesa 3.5 discards return values)
        self._last_step_result = result

    @property
    def last_step_result(self) -> dict | None:
        """Result dict from the most recent physics step.

        Returns ``None`` if :meth:`step` has not been called yet.

        Keys: ``'positions'``, ``'velocities'``, ``'masses'``,
        ``'mean_speed'``, ``'dt_used'``, ``'step_count'``.
        """
        return self._last_step_result

    # ------------------------------------------------------------------
    # Potential field
    # ------------------------------------------------------------------

    def get_potential_field(self, resolution: float = 10.0) -> dict:
        """Compute the current gravitational potential on a regular grid.

        Parameters
        ----------
        resolution : float, default 10.0
            Grid spacing in metres.

        Returns
        -------
        dict
            Keys: ``'potential'`` (1-D array), ``'grid_centers'`` (M x 2),
            ``'grid_width'``, ``'grid_height'``,
            ``'x_min'``, ``'y_min'``, ``'x_max'``, ``'y_max'``.
        """
        positions = np.array([a.position for a in self.vehicle_agents], dtype=np.float64)
        masses = np.array([a.mass for a in self.vehicle_agents], dtype=np.float64)

        # Compute bounds with padding
        margin = 50.0
        x_min = float(positions[:, 0].min()) - margin
        y_min = float(positions[:, 1].min()) - margin
        x_max = float(positions[:, 0].max()) + margin
        y_max = float(positions[:, 1].max()) + margin

        grid = make_grid(x_min, y_min, x_max, y_max, resolution)
        phi = compute_potential_field(positions, masses, grid, G_s=self.G_s)

        grid_w = int((x_max - x_min) / resolution) + 1
        grid_h = int((y_max - y_min) / resolution) + 1

        return {
            "potential": phi,
            "grid_centers": grid,
            "grid_width": grid_w,
            "grid_height": grid_h,
            "x_min": x_min,
            "y_min": y_min,
            "x_max": x_max,
            "y_max": y_max,
        }

    # ------------------------------------------------------------------
    # State export
    # ------------------------------------------------------------------

    def get_state(self) -> dict:
        """Return the full simulation state for API / visualization.

        Returns
        -------
        dict
            Keys: ``'step'``, ``'vehicles'`` (list of dicts),
            ``'intersections'`` (list of dicts), ``'kpi'`` (dict of latest
            KPI values or empty dict if no steps have been run).
        """
        if self.step_count > 0:
            df = self.datacollector.get_model_vars_dataframe()
            kpi = df.iloc[-1].to_dict()
        else:
            kpi = {}

        return {
            "step": self.step_count,
            "vehicles": [a.to_dict() for a in self.vehicle_agents],
            "intersections": [a.to_dict() for a in self.intersection_agents],
            "kpi": kpi,
        }


# ======================================================================
# Model reporter functions (top-level for pickle-ability)
# ======================================================================


def _mean_speed_kmh(model: TrafficModel) -> float:
    """Mean vehicle speed in km/h across all agents."""
    if not model.vehicle_agents:
        return 0.0
    return float(np.mean([a.speed_kmh for a in model.vehicle_agents]))


def _congestion_index(model: TrafficModel) -> float:
    """Fraction of vehicles classified as 'slow' (positive-mass attractors).

    Returns a value in [0, 1].  A value of 0 means no congestion seeds;
    a value of 1 means every vehicle is slow relative to the mean flow.
    """
    n = len(model.vehicle_agents)
    if n == 0:
        return 0.0
    n_slow = sum(1 for a in model.vehicle_agents if a.mass_type == "slow")
    return float(n_slow / n)


def _n_slow(model: TrafficModel) -> int:
    """Number of vehicles with mass_type == 'slow'."""
    return sum(1 for a in model.vehicle_agents if a.mass_type == "slow")


def _n_fast(model: TrafficModel) -> int:
    """Number of vehicles with mass_type == 'fast'."""
    return sum(1 for a in model.vehicle_agents if a.mass_type == "fast")


def _n_neutral(model: TrafficModel) -> int:
    """Number of vehicles with mass_type == 'neutral'."""
    return sum(1 for a in model.vehicle_agents if a.mass_type == "neutral")
