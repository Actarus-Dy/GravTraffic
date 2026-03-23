"""Rivoli corridor benchmark scenario -- 12 intersections, fixed vs optimized signals.

Milestone S10: "Gain debit >= +15% vs feux fixes sur corridor (12 carrefours)"

This scenario creates a linear corridor simulating a simplified Rue de Rivoli:
- 12 intersections spaced 150 m apart (total 1.65 km)
- 2 directions (eastbound and westbound)
- v_max = 13.9 m/s (50 km/h)
- Vehicles injected at both ends at a steady rate

The scenario uses GravSimulation directly with manual intersection management
(no TrafficModel / Mesa overhead).

Modes
-----
- **fixed**: All intersections cycle with identical, uncoordinated green times.
- **optimized**: Green-wave coordination + periodic signal timing optimization
  via the time-integrated potential optimizer.

Author: Agent #23 Scientific Validation Tester
Date: 2026-03-22
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from gravtraffic.core.simulation import GravSimulation
from gravtraffic.core.signal_optimizer import optimize_signal_timing
from gravtraffic.core.green_wave import GreenWaveCoordinator
from gravtraffic.core.metrics import (
    compute_throughput,
    compute_mean_speed,
    compute_stops,
    compute_congestion_index,
    compute_snapshot_kpis,
)

__all__ = ["RivoliCorridor"]


class RivoliCorridor:
    """Rivoli corridor benchmark: 12 intersections, fixed vs optimized signals.

    Parameters
    ----------
    n_intersections : int
        Number of signalized intersections along the corridor.
    spacing : float
        Distance in metres between adjacent intersections.
    v_max : float
        Speed limit in m/s (13.9 ~= 50 km/h).
    G_s : float
        Social gravitational constant.
    beta : float
        Mass-assignment exponent.
    drag_coefficient : float
        Greenshields drag coefficient.
    injection_rate : float
        Vehicle injection rate per direction [vehicles/second].
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_intersections: int = 12,
        spacing: float = 150.0,
        v_max: float = 13.9,
        G_s: float = 5.0,
        beta: float = 0.5,
        drag_coefficient: float = 0.3,
        injection_rate: float = 0.5,
        seed: int = 42,
    ):
        self.n_intersections = n_intersections
        self.spacing = spacing
        self.v_max = v_max
        self.G_s = G_s
        self.beta = beta
        self.drag_coefficient = drag_coefficient
        self.injection_rate = injection_rate
        self.seed = seed

        # Intersection positions along x-axis
        self.intersection_x = np.arange(n_intersections) * spacing
        self.corridor_length = (n_intersections - 1) * spacing

        # Simulation parameters (fixed dt, no adaptive for benchmark stability)
        self.sim_params = dict(
            G_s=G_s,
            beta=beta,
            softening=10.0,
            theta=0.5,
            dt=0.1,
            v_max=v_max,
            adaptive_dt=False,
            drag_coefficient=drag_coefficient,
            v_free=v_max,       # free-flow = speed limit for urban corridor
            rho_jam=150.0,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run_fixed_timing(
        self,
        duration_s: float = 300.0,
        green_ns: float = 60.0,
        green_ew: float = 50.0,
    ) -> dict:
        """Run with fixed, uncoordinated signal timing.

        All 12 intersections cycle identically with no phase offset
        (worst-case for a corridor).

        Returns
        -------
        dict
            KPI dictionary with mode, speeds, stops, vehicle counts.
        """
        return self._run(
            duration_s, mode="fixed", green_ns=green_ns, green_ew=green_ew
        )

    def run_optimized(
        self,
        duration_s: float = 300.0,
        optimize_interval: int = 300,
    ) -> dict:
        """Run with green-wave coordination + periodic optimization.

        Phase offsets are computed so that eastbound traffic encounters a
        continuous green wave at the corridor speed limit.  Signal timings
        are periodically re-optimized using the gravitational potential
        integral.

        Parameters
        ----------
        duration_s : float
            Simulation duration in seconds.
        optimize_interval : int
            Number of simulation steps between re-optimization calls.

        Returns
        -------
        dict
            KPI dictionary.
        """
        return self._run(
            duration_s,
            mode="optimized",
            optimize_interval=optimize_interval,
        )

    def compare(self, duration_s: float = 300.0) -> dict:
        """Run both modes and compare KPIs.

        Returns
        -------
        dict
            Keys: ``fixed``, ``optimized``, ``speed_gain_pct``,
            ``stops_reduction_pct``, ``throughput_gain_pct``.
        """
        fixed = self.run_fixed_timing(duration_s)
        optimized = self.run_optimized(duration_s)

        speed_gain_pct = (
            (optimized["mean_speed_kmh"] - fixed["mean_speed_kmh"])
            / max(fixed["mean_speed_kmh"], 0.1)
            * 100
        )
        stops_reduction_pct = (
            (fixed["mean_stops"] - optimized["mean_stops"])
            / max(fixed["mean_stops"], 0.1)
            * 100
        )
        throughput_gain_pct = (
            (optimized["total_throughput"] - fixed["total_throughput"])
            / max(fixed["total_throughput"], 0.1)
            * 100
        )

        return {
            "fixed": fixed,
            "optimized": optimized,
            "speed_gain_pct": speed_gain_pct,
            "stops_reduction_pct": stops_reduction_pct,
            "throughput_gain_pct": throughput_gain_pct,
        }

    # ------------------------------------------------------------------
    # Core simulation loop
    # ------------------------------------------------------------------

    def _run(self, duration_s: float, mode: str, **kwargs) -> dict:
        """Execute the corridor simulation.

        Parameters
        ----------
        duration_s : float
            Total simulated time in seconds.
        mode : str
            ``'fixed'`` or ``'optimized'``.
        **kwargs
            Mode-specific parameters (green_ns, green_ew, optimize_interval).
        """
        rng = np.random.default_rng(self.seed)
        dt = 0.1
        n_steps = int(duration_s / dt)

        # --- Initialize vehicles on the corridor ---
        # ~1 vehicle per 20 m, split roughly 50/50 eastbound/westbound
        n_initial = int(self.corridor_length / 20)
        x_positions = rng.uniform(0, self.corridor_length, n_initial)
        # Alternate lanes: y=2 (eastbound lane), y=-2 (westbound lane)
        directions = rng.choice([-1, 1], n_initial)
        y_positions = directions * 2.0

        positions = np.column_stack([x_positions, y_positions])
        speeds = rng.uniform(8.0, self.v_max, n_initial)
        velocities = np.column_stack([
            directions * speeds,
            np.zeros(n_initial),
        ])
        densities = np.full(n_initial, 50.0, dtype=np.float64)

        # --- Create simulation ---
        sim = GravSimulation(**self.sim_params)
        sim.init_vehicles(positions, velocities, densities)

        # --- Signal state ---
        # Phase 0 = NS green (EW red -- blocks corridor traffic)
        # Phase 1 = EW green (NS red -- corridor flows)
        intersection_phase = np.zeros(self.n_intersections, dtype=int)
        intersection_timer = np.zeros(self.n_intersections, dtype=np.float64)

        if mode == "fixed":
            green_ns = kwargs.get("green_ns", 60.0)
            green_ew = kwargs.get("green_ew", 50.0)
            green_times = [green_ns, green_ew]
        else:
            green_times = [60.0, 50.0]  # initial timing, will be optimized

            # Apply green wave: compute phase offsets so eastbound platoon
            # sees continuous green (phase 1 = EW green)
            gw = GreenWaveCoordinator(wave_speed=self.v_max)
            int_positions_2d = np.column_stack([
                self.intersection_x,
                np.zeros(self.n_intersections),
            ])
            offsets = gw.compute_offsets(int_positions_2d)

            # Shift timers by offset so that EW green phases are staggered
            # for the wave.  We add green_ns to the offset so the wave
            # aligns with phase 1 (EW green) rather than phase 0 (NS green).
            cycle = sum(green_times)
            intersection_timer = (offsets + green_times[0]) % cycle

            # Start all intersections in the phase corresponding to their
            # timer position within the cycle
            for i in range(self.n_intersections):
                if intersection_timer[i] < green_times[0]:
                    intersection_phase[i] = 0  # NS green
                else:
                    intersection_phase[i] = 1  # EW green
                    intersection_timer[i] -= green_times[0]

        # --- Metrics accumulators ---
        speed_history: list[float] = []
        stops_history: list[int] = []
        throughput_count = 0  # total gate crossings
        gate_x = self.corridor_length / 2.0

        for step in range(n_steps):
            # --- Inject vehicles at corridor ends ---
            if rng.random() < self.injection_rate * dt:
                new_pos = np.array([[0.0, 2.0]], dtype=np.float64)
                new_vel = np.array(
                    [[rng.uniform(8.0, self.v_max), 0.0]], dtype=np.float64
                )
                sim.add_vehicles(new_pos, new_vel, np.array([50.0]))

            if rng.random() < self.injection_rate * dt:
                new_pos = np.array(
                    [[self.corridor_length, -2.0]], dtype=np.float64
                )
                new_vel = np.array(
                    [[-rng.uniform(8.0, self.v_max), 0.0]], dtype=np.float64
                )
                sim.add_vehicles(new_pos, new_vel, np.array([50.0]))

            # --- Update signal phases and build obstacle list ---
            obstacle_positions: list[list[float]] = []
            obstacle_masses: list[float] = []

            for i in range(self.n_intersections):
                intersection_timer[i] += dt
                current_green = green_times[intersection_phase[i]]

                if intersection_timer[i] >= current_green:
                    intersection_timer[i] = 0.0
                    intersection_phase[i] = 1 - intersection_phase[i]

                # Place red-light obstacle masses.
                # When phase == 0 (NS green): EW is red -> block x-direction
                # When phase == 1 (EW green): NS is red -> block y-direction
                ix = float(self.intersection_x[i])
                if intersection_phase[i] == 0:
                    # NS green, EW red: obstacles block east-west traffic
                    obstacle_positions.append([ix + 15.0, 0.0])
                    obstacle_positions.append([ix - 15.0, 0.0])
                    obstacle_masses.extend([50.0, 50.0])
                else:
                    # EW green, NS red: obstacles block north-south traffic
                    obstacle_positions.append([ix, 15.0])
                    obstacle_positions.append([ix, -15.0])
                    obstacle_masses.extend([50.0, 50.0])

            if obstacle_positions:
                sim.set_obstacles(
                    np.array(obstacle_positions, dtype=np.float64),
                    np.array(obstacle_masses, dtype=np.float64),
                )
            else:
                sim.clear_obstacles()

            # --- Periodic signal optimization (optimized mode only) ---
            opt_interval = kwargs.get("optimize_interval", 300)
            if (
                mode == "optimized"
                and step > 0
                and step % opt_interval == 0
                and sim.n_vehicles > 0
            ):
                for i in range(self.n_intersections):
                    ix = float(self.intersection_x[i])
                    int_pos = np.array([ix, 0.0], dtype=np.float64)
                    result = optimize_signal_timing(
                        sim.positions,
                        sim.velocities,
                        sim.masses,
                        int_pos,
                        radius=200.0,
                        G_s=self.G_s,
                    )
                    # Update green times from optimizer result
                    green_times = [result["green_ns"], result["green_ew"]]

            # --- Record pre-step positions for throughput gate ---
            if sim.n_vehicles > 0:
                prev_positions = sim.positions.copy()
            else:
                prev_positions = np.empty((0, 2), dtype=np.float64)

            # --- Physics step ---
            sim.step()

            # --- Throughput: count gate crossings ---
            if sim.n_vehicles > 0 and prev_positions.shape[0] == sim.n_vehicles:
                tp = compute_throughput(
                    sim.positions, prev_positions, gate_x, dt
                )
                throughput_count += tp * dt / 3600.0  # convert back to count

            # --- Remove vehicles that left the corridor ---
            if sim.n_vehicles > 0:
                out_mask = (sim.positions[:, 0] < -50.0) | (
                    sim.positions[:, 0] > self.corridor_length + 50.0
                )
                if out_mask.any():
                    sim.remove_vehicles(np.where(out_mask)[0])

            # --- Collect metrics every 10 steps ---
            if step % 10 == 0 and sim.n_vehicles > 0:
                speed_history.append(compute_mean_speed(sim.velocities))
                stops_history.append(compute_stops(sim.velocities))

        # --- Final KPIs ---
        mean_speed = float(np.mean(speed_history)) if speed_history else 0.0
        mean_stops = float(np.mean(stops_history)) if stops_history else 0.0

        return {
            "mode": mode,
            "duration_s": duration_s,
            "mean_speed_ms": mean_speed,
            "mean_speed_kmh": mean_speed * 3.6,
            "mean_stops": mean_stops,
            "total_throughput": float(throughput_count),
            "final_n_vehicles": sim.n_vehicles,
            "total_steps": n_steps,
        }
