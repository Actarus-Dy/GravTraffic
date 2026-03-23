"""GravTraffic FastAPI application -- REST + WebSocket API.

Provides:
- REST endpoints for simulation lifecycle (create, step, query state)
- WebSocket endpoint for real-time potential-field streaming

All endpoints are versioned under ``/api/v1/``.  The WebSocket stream
lives at ``/ws/stream/potential``.

Error responses follow RFC 7807 Problem Details where applicable.

Author: Agent #06 API Architect
Date: 2026-03-22
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import os
import pathlib
import time

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

_STATIC_DIR = pathlib.Path(__file__).parent / "static"

from gravtraffic.agents.traffic_model import TrafficModel
from gravtraffic.network.road_network import RoadNetwork


# ======================================================================
# Pydantic request / response models
# ======================================================================


class SimulationConfig(BaseModel):
    """Configuration payload for creating a new simulation."""

    grid_rows: int = Field(default=5, ge=1, le=50, description="Grid intersection rows")
    grid_cols: int = Field(default=5, ge=1, le=50, description="Grid intersection columns")
    block_size: float = Field(default=200.0, gt=0, description="Block size in metres")
    n_vehicles: int = Field(default=100, ge=1, le=100_000, description="Number of vehicles")
    G_s: float = Field(default=5.0, gt=0, description="Social gravitational constant (unified calibration)")
    beta: float = Field(default=0.5, ge=0, description="Mass-assignment exponent")
    dt: float = Field(default=0.1, gt=0, le=1.0, description="Integration timestep (seconds)")
    seed: int = Field(default=42, description="Random seed for reproducibility")


class ProblemDetail(BaseModel):
    """RFC 7807 Problem Details for HTTP APIs."""

    type: str = "about:blank"
    title: str
    status: int
    detail: str


# ======================================================================
# Application state (module-level singleton)
# ======================================================================


class AppState:
    """Mutable container for the simulation state held by the server.

    Uses an asyncio lock to serialise access between REST handlers and
    the WebSocket streaming loop (both run on the same event loop in
    single-worker uvicorn).
    """

    def __init__(self) -> None:
        self.model: TrafficModel | None = None
        self.config: SimulationConfig | None = None
        self.running: bool = False
        self.lock: asyncio.Lock = asyncio.Lock()

    def reset(self) -> None:
        """Reset to pristine state (useful for testing)."""
        self.model = None
        self.config = None
        self.running = False
        self.lock = asyncio.Lock()


state = AppState()


# ======================================================================
# FastAPI application
# ======================================================================

app = FastAPI(
    title="GravTraffic API",
    version="0.2.0",
    description=(
        "REST and WebSocket API for real-time gravitational traffic simulation "
        "based on the Janus Cosmological Model (C-01).\n\n"
        "**Core physics:** vehicles have positive mass (slow, attract, jam) or "
        "negative mass (fast, repel, fluid). Traffic phenomena emerge from "
        "gravitational interactions.\n\n"
        "**Endpoints:** simulation lifecycle, real-time stepping, T+15min prediction, "
        "potential field visualization, Prometheus metrics."
    ),
    openapi_tags=[
        {"name": "infrastructure", "description": "Health checks and monitoring"},
        {"name": "simulation", "description": "Simulation lifecycle and stepping"},
        {"name": "analysis", "description": "Potential field and prediction"},
    ],
)

# Static files and dashboard route
if _STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


@app.get("/dashboard", tags=["infrastructure"], include_in_schema=False)
async def dashboard():
    """Serve the real-time visualization dashboard."""
    html = _STATIC_DIR / "dashboard.html"
    if not html.exists():
        raise HTTPException(status_code=404, detail="Dashboard not found")
    return FileResponse(str(html), media_type="text/html")


# ------------------------------------------------------------------
# CORS -- configurable via GRAVTRAFFIC_CORS_ORIGINS env var
# Default: localhost only. Set to "*" for open access.
# ------------------------------------------------------------------
_cors_origins = os.environ.get(
    "GRAVTRAFFIC_CORS_ORIGINS", "http://localhost:3000,http://localhost:8080"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _cors_origins],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------
# Rate limiting -- simple in-memory token bucket per endpoint
# ------------------------------------------------------------------
class _RateLimiter:
    """In-memory sliding-window rate limiter.

    Tracks request timestamps per key (typically ``endpoint:client_ip``)
    and rejects requests that exceed ``max_requests`` within ``window_s``
    seconds.
    """

    def __init__(self, max_requests: int = 60, window_s: float = 60.0) -> None:
        self.max_requests = max_requests
        self.window_s = window_s
        self._requests: dict[str, list[float]] = {}
        self._call_count: int = 0

    def check(self, key: str) -> bool:
        """Return True if the request is allowed, False if rate-limited."""
        self._call_count += 1
        if self._call_count % 1000 == 0:
            self.cleanup()
        now = time.monotonic()
        timestamps = self._requests.get(key, [])
        # Prune old timestamps
        cutoff = now - self.window_s
        timestamps = [t for t in timestamps if t > cutoff]
        if len(timestamps) >= self.max_requests:
            self._requests[key] = timestamps
            return False
        timestamps.append(now)
        self._requests[key] = timestamps
        return True

    def cleanup(self) -> None:
        """Remove keys with no recent requests (call periodically)."""
        now = time.monotonic()
        cutoff = now - self.window_s
        dead = [k for k, v in self._requests.items()
                if not any(t > cutoff for t in v)]
        for k in dead:
            del self._requests[k]


_rate_limiter = _RateLimiter(max_requests=120, window_s=60.0)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _build_model(config: SimulationConfig) -> TrafficModel:
    """Construct a TrafficModel from a SimulationConfig."""
    network = RoadNetwork.from_grid(config.grid_rows, config.grid_cols, config.block_size)
    return TrafficModel(
        network=network,
        n_vehicles=config.n_vehicles,
        G_s=config.G_s,
        beta=config.beta,
        dt=config.dt,
        seed=config.seed,
    )


def _require_simulation() -> None:
    """Raise 409 if no simulation exists."""
    if state.model is None:
        raise HTTPException(
            status_code=409,
            detail={
                "type": "about:blank",
                "title": "No simulation",
                "status": 409,
                "detail": "No simulation created. POST /api/v1/simulate first.",
            },
        )


def _check_rate_limit(key: str) -> None:
    """Raise 429 if rate limit exceeded for the given key."""
    if not _rate_limiter.check(key):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Try again later.",
        )


# ------------------------------------------------------------------
# Infrastructure endpoints (health, readiness, metrics)
# ------------------------------------------------------------------

_start_time = time.monotonic()


@app.get("/health", tags=["infrastructure"])
async def health() -> dict[str, str]:
    """Liveness probe. Always returns 200 if the process is running."""
    return {"status": "ok"}


@app.get("/ready", tags=["infrastructure"])
async def ready() -> dict[str, Any]:
    """Readiness probe. Returns 200 if a simulation is loaded, 503 otherwise."""
    async with state.lock:
        if state.model is None:
            raise HTTPException(status_code=503, detail="No simulation loaded")
        n = len(state.model.vehicle_agents)
    return {"status": "ready", "n_vehicles": n}


@app.get("/metrics", tags=["infrastructure"])
async def metrics():
    """Prometheus-compatible metrics in text exposition format.

    WARNING: This endpoint is unauthenticated. In production, expose it
    only on an internal network (not the public interface) or add an
    authentication middleware.
    """
    uptime = time.monotonic() - _start_time
    async with state.lock:
        if state.model is not None:
            step = state.model.step_count
            n_veh = len(state.model.vehicle_agents)
            mean_spd = state.model.simulation._mean_speed * 3.6
        else:
            step = 0
            n_veh = 0
            mean_spd = 0.0

    lines = [
        "# HELP gravtraffic_uptime_seconds Server uptime in seconds",
        "# TYPE gravtraffic_uptime_seconds gauge",
        f"gravtraffic_uptime_seconds {uptime:.1f}",
        "# HELP gravtraffic_step_count Total simulation steps completed",
        "# TYPE gravtraffic_step_count counter",
        f"gravtraffic_step_count {step}",
        "# HELP gravtraffic_n_vehicles Current number of vehicles",
        "# TYPE gravtraffic_n_vehicles gauge",
        f"gravtraffic_n_vehicles {n_veh}",
        "# HELP gravtraffic_mean_speed_kmh Mean vehicle speed in km/h",
        "# TYPE gravtraffic_mean_speed_kmh gauge",
        f"gravtraffic_mean_speed_kmh {mean_spd:.2f}",
        "# HELP gravtraffic_rate_limit_keys Active rate limiter keys",
        "# TYPE gravtraffic_rate_limit_keys gauge",
        f"gravtraffic_rate_limit_keys {len(_rate_limiter._requests)}",
        "",
    ]
    return PlainTextResponse(
        "\n".join(lines), media_type="text/plain; version=0.0.4; charset=utf-8"
    )


# ------------------------------------------------------------------
# REST endpoints -- /api/v1/
# ------------------------------------------------------------------


@app.get("/api/v1/status", tags=["simulation"])
async def get_status() -> dict[str, Any]:
    """Return current simulation status."""
    async with state.lock:
        if state.model is None:
            return {"running": False, "step": 0, "n_vehicles": 0}
        return {
            "running": state.running,
            "step": state.model.step_count,
            "n_vehicles": len(state.model.vehicle_agents),
        }


@app.post("/api/v1/simulate", status_code=201, tags=["simulation"])
async def create_simulation(config: SimulationConfig) -> dict[str, Any]:
    """Create a new simulation with the given configuration.

    Replaces any existing simulation.
    """
    async with state.lock:
        state.model = _build_model(config)
        state.config = config
        state.running = False
    return {"status": "created", "n_vehicles": config.n_vehicles}


@app.post("/api/v1/step", tags=["simulation"])
async def step_simulation(request: Request) -> dict[str, Any]:
    """Execute one simulation step and return the full state snapshot."""
    _check_rate_limit(f"step:{request.client.host}" if request.client else "step")
    async with state.lock:
        _require_simulation()
        state.model.step()
        return state.model.get_state()


@app.get("/api/v1/network/state", tags=["simulation"])
async def get_network_state() -> dict[str, Any]:
    """Return the current gravitational state of the simulation."""
    async with state.lock:
        _require_simulation()
        return state.model.get_state()


@app.get("/api/v1/potential", tags=["analysis"])
async def get_potential(
    request: Request,
    resolution: float = Query(default=20.0, ge=1.0, le=500.0),
) -> dict[str, Any]:
    """Compute and return the current gravitational potential field.

    Parameters
    ----------
    resolution : float
        Grid spacing in metres (query parameter, default 20.0, min 1.0).
    """
    _check_rate_limit(f"potential:{request.client.host}" if request.client else "potential")
    async with state.lock:
        _require_simulation()
        field = state.model.get_potential_field(resolution=resolution)
        return {
            "grid_width": field["grid_width"],
            "grid_height": field["grid_height"],
            "potential": field["potential"].tolist(),
            "x_min": field["x_min"],
            "y_min": field["y_min"],
            "x_max": field["x_max"],
            "y_max": field["y_max"],
        }


@app.post("/api/v1/predict", tags=["analysis"])
async def predict_simulation(
    request: Request,
    horizon_s: float = Query(default=900.0, ge=1.0, le=3600.0),
) -> dict[str, Any]:
    """Run a prediction T+horizon_s seconds into the future.

    Clones the current simulation, runs the clone forward, and returns
    the predicted state.  The live simulation is NOT modified.

    Parameters
    ----------
    horizon_s : float
        Prediction horizon in seconds (default 900 = 15 min, max 3600 = 1h).
    """
    _check_rate_limit(f"predict:{request.client.host}" if request.client else "predict")
    # Lock only for clone(); run prediction outside the lock since
    # the clone is independent from the live simulation.
    async with state.lock:
        if state.model is None:
            raise HTTPException(status_code=409, detail="No simulation created.")
        sim_clone = state.model.simulation.clone()
    # Prediction loop runs on the clone — does not hold the lock
    prediction = sim_clone.run_until(horizon_s)

    # Convert arrays to lists for JSON serialisation
    positions = prediction["positions"]
    velocities = prediction["velocities"]
    masses = prediction["masses"]

    vehicles = []
    for i in range(len(masses)):
        speed = float(np.linalg.norm(velocities[i]))
        vehicles.append({
            "x": float(positions[i, 0]),
            "y": float(positions[i, 1]),
            "vx": float(velocities[i, 0]),
            "vy": float(velocities[i, 1]),
            "mass": float(masses[i]),
            "speed_kmh": speed * 3.6,
        })

    return {
        "horizon_s": prediction["horizon_s"],
        "n_steps_run": prediction["n_steps_run"],
        "predicted_step": prediction["step_count"],
        "mean_speed_kmh": prediction["mean_speed"] * 3.6,
        "n_vehicles": len(vehicles),
        "vehicles": vehicles,
    }


# ------------------------------------------------------------------
# WebSocket endpoint -- /ws/stream/potential
# ------------------------------------------------------------------


@app.websocket("/ws/stream/potential")
async def stream_potential(websocket: WebSocket) -> None:
    """Stream real-time simulation state over WebSocket.

    On connect the server auto-creates a default simulation if none exists
    and begins pushing frames at approximately 1 Hz.

    Client commands (JSON text messages)
    -------------------------------------
    ``{"type": "start"}``  -- start / resume the simulation loop
    ``{"type": "stop"}``   -- pause the simulation loop
    ``{"type": "step"}``   -- execute a single simulation step (while paused)
    ``{"type": "config", ...}``  -- reconfigure (all SimulationConfig fields)

    Server frames (JSON)
    --------------------
    Each frame has ``type: "frame"`` plus ``step``, ``vehicles``,
    ``potential``, ``grid_width``, ``grid_height``, and ``kpi``.
    """
    await websocket.accept()

    # Auto-create a default simulation if the user has not POST-ed one.
    async with state.lock:
        if state.model is None:
            config = SimulationConfig()
            state.model = _build_model(config)
            state.config = config
        state.running = True

    try:
        while True:
            # --- Non-blocking check for client messages ---
            try:
                raw = await asyncio.wait_for(websocket.receive_text(), timeout=0.05)
                msg = json.loads(raw)
                msg_type = msg.get("type", "")

                if msg_type == "stop":
                    async with state.lock:
                        state.running = False
                elif msg_type == "start":
                    async with state.lock:
                        state.running = True
                elif msg_type == "step":
                    if state.model is not None:
                        async with state.lock:
                            state.model.step()
                elif msg_type == "config":
                    try:
                        async with state.lock:
                            cfg_data = msg.get("config", msg)
                            new_cfg = SimulationConfig(**{
                                k: v for k, v in cfg_data.items()
                                if k != "type"
                            })
                            state.model = _build_model(new_cfg)
                            state.config = new_cfg
                        await websocket.send_json(
                            {"type": "config_ok", "n_vehicles": new_cfg.n_vehicles}
                        )
                    except Exception as exc:
                        await websocket.send_json(
                            {"type": "error", "detail": str(exc)}
                        )
            except asyncio.TimeoutError:
                pass

            # --- Produce a frame if running ---
            if state.running and state.model is not None:
                async with state.lock:
                    # Advance simulation: 10 micro-steps per frame at dt=0.1 -> 1 s sim-time
                    for _ in range(10):
                        state.model.step()

                    vehicles = [a.to_dict() for a in state.model.vehicle_agents]
                    step_count = state.model.step_count

                    # Coarse potential grid for visualisation
                    field = state.model.get_potential_field(resolution=20.0)

                frame: dict[str, Any] = {
                    "type": "frame",
                    "step": step_count,
                    "grid_width": field["grid_width"],
                    "grid_height": field["grid_height"],
                    "potential": field["potential"].tolist(),
                    "vehicles": vehicles,
                    "kpi": {
                        "mean_speed_kmh": float(
                            np.mean([v["speed_kmh"] for v in vehicles]) if vehicles else 0.0
                        ),
                        "congestion_index": float(
                            sum(1 for v in vehicles if v["type"] == "slow")
                            / max(len(vehicles), 1)
                        ),
                        "n_vehicles": len(vehicles),
                    },
                }

                await websocket.send_json(frame)

            # Target ~1 Hz frame rate
            await asyncio.sleep(1.0)

    except WebSocketDisconnect:
        state.running = False


# ======================================================================
# CLI entry point
# ======================================================================


def main() -> None:
    """Run the development server on port 8000."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
