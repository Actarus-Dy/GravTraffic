# GravTraffic API Reference

> **Version:** 0.2.0
> **Date:** 2026-03-23
> **Project:** GravTraffic (C-01) -- Janus Civil Software Catalogue

---

## Table of Contents

1. [Overview](#1-overview)
2. [Authentication](#2-authentication)
3. [Rate Limiting](#3-rate-limiting)
4. [Infrastructure Endpoints](#4-infrastructure-endpoints)
5. [Simulation Lifecycle](#5-simulation-lifecycle)
6. [Simulation Control](#6-simulation-control)
7. [Analysis](#7-analysis)
8. [WebSocket -- Real-Time Streaming](#8-websocket----real-time-streaming)
9. [Error Handling](#9-error-handling)
10. [CORS Configuration](#10-cors-configuration)

---

## 1. Overview

GravTraffic exposes a REST API and a WebSocket endpoint for real-time gravitational traffic simulation based on the Janus Cosmological Model.

| Property | Value |
|---|---|
| **Base URL** | `http://<host>:8000` |
| **API prefix** | `/api/v1/` |
| **Content-Type** | `application/json` (REST), `text/plain` (`/metrics`) |
| **WebSocket** | `/ws/stream/potential` |
| **OpenAPI spec** | `GET /openapi.json` |
| **Interactive docs** | `GET /docs` (Swagger UI), `GET /redoc` (ReDoc) |

All REST endpoints accept and return JSON unless stated otherwise. The API follows semantic versioning; breaking changes will increment the version prefix (e.g., `/api/v2/`).

### Starting the Server

```bash
uvicorn gravtraffic.api.app:app --host 0.0.0.0 --port 8000
```

---

## 2. Authentication

**There is no authentication mechanism.** The API is intended for internal or development use. In production deployments, place the service behind a reverse proxy (e.g., nginx, Traefik) that enforces authentication and TLS termination.

> **Limitation:** The `/metrics` endpoint exposes operational data (uptime, vehicle count, mean speed). Restrict access to this endpoint to internal networks only.

---

## 3. Rate Limiting

All rate-limited endpoints enforce an in-memory sliding-window limit.

| Parameter | Value |
|---|---|
| **Max requests** | 120 per key |
| **Window** | 60 seconds |
| **Key format** | `<endpoint>:<client_ip>` |

When the limit is exceeded, the server returns HTTP `429 Too Many Requests`.

Rate-limited endpoints: `POST /api/v1/step`, `GET /api/v1/potential`, `POST /api/v1/predict`.

Infrastructure endpoints (`/health`, `/ready`, `/metrics`) and simulation creation (`POST /api/v1/simulate`) are **not** rate-limited.

---

## 4. Infrastructure Endpoints

### 4.1 GET /health

Liveness probe. Always returns `200 OK` if the process is running.

```bash
curl http://localhost:8000/health
```

**Response** `200 OK`:

```json
{
  "status": "ok"
}
```

---

### 4.2 GET /ready

Readiness probe. Returns `200 OK` if a simulation is loaded, `503 Service Unavailable` otherwise.

```bash
curl http://localhost:8000/ready
```

**Response** `200 OK` (simulation loaded):

```json
{
  "status": "ready",
  "n_vehicles": 100
}
```

**Response** `503 Service Unavailable` (no simulation):

```json
{
  "detail": "No simulation loaded"
}
```

---

### 4.3 GET /metrics

Prometheus-compatible metrics in text exposition format (`text/plain; version=0.0.4`).

```bash
curl http://localhost:8000/metrics
```

**Response** `200 OK`:

```text
# HELP gravtraffic_uptime_seconds Server uptime in seconds
# TYPE gravtraffic_uptime_seconds gauge
gravtraffic_uptime_seconds 342.7
# HELP gravtraffic_step_count Total simulation steps completed
# TYPE gravtraffic_step_count counter
gravtraffic_step_count 1500
# HELP gravtraffic_n_vehicles Current number of vehicles
# TYPE gravtraffic_n_vehicles gauge
gravtraffic_n_vehicles 100
# HELP gravtraffic_mean_speed_kmh Mean vehicle speed in km/h
# TYPE gravtraffic_mean_speed_kmh gauge
gravtraffic_mean_speed_kmh 72.45
# HELP gravtraffic_rate_limit_keys Active rate limiter keys
# TYPE gravtraffic_rate_limit_keys gauge
gravtraffic_rate_limit_keys 3
```

| Metric | Type | Description |
|---|---|---|
| `gravtraffic_uptime_seconds` | gauge | Server uptime in seconds |
| `gravtraffic_step_count` | counter | Total simulation steps completed |
| `gravtraffic_n_vehicles` | gauge | Current number of vehicles |
| `gravtraffic_mean_speed_kmh` | gauge | Mean vehicle speed in km/h |
| `gravtraffic_rate_limit_keys` | gauge | Number of active rate limiter tracking keys |

---

## 5. Simulation Lifecycle

### 5.1 POST /api/v1/simulate

Create a new simulation. Replaces any existing simulation.

```bash
curl -X POST http://localhost:8000/api/v1/simulate \
  -H "Content-Type: application/json" \
  -d '{
    "grid_rows": 5,
    "grid_cols": 5,
    "block_size": 200.0,
    "n_vehicles": 100,
    "G_s": 5.0,
    "beta": 0.5,
    "dt": 0.1,
    "seed": 42
  }'
```

**Request Body** (`SimulationConfig`):

| Field | Type | Default | Constraints | Description |
|---|---|---|---|---|
| `grid_rows` | int | `5` | 1--50 | Grid intersection rows |
| `grid_cols` | int | `5` | 1--50 | Grid intersection columns |
| `block_size` | float | `200.0` | > 0 | Block size in metres |
| `n_vehicles` | int | `100` | 1--100,000 | Number of vehicles |
| `G_s` | float | `5.0` | > 0 | Social gravitational constant |
| `beta` | float | `0.5` | >= 0 | Mass-assignment exponent |
| `dt` | float | `0.1` | 0 < dt <= 1.0 | Integration timestep (seconds) |
| `seed` | int | `42` | -- | Random seed for reproducibility |

All fields are optional; defaults produce the standard calibrated scenario.

**Response** `201 Created`:

```json
{
  "status": "created",
  "n_vehicles": 100
}
```

**Response** `422 Unprocessable Entity` (validation error):

```json
{
  "detail": [
    {
      "loc": ["body", "n_vehicles"],
      "msg": "ensure this value is greater than or equal to 1",
      "type": "value_error.number.not_ge"
    }
  ]
}
```

---

### 5.2 GET /api/v1/status

Return the current simulation status. Does not require an active simulation.

```bash
curl http://localhost:8000/api/v1/status
```

**Response** `200 OK` (simulation active):

```json
{
  "running": true,
  "step": 150,
  "n_vehicles": 100
}
```

**Response** `200 OK` (no simulation):

```json
{
  "running": false,
  "step": 0,
  "n_vehicles": 0
}
```

---

## 6. Simulation Control

### 6.1 POST /api/v1/step

Execute one simulation step and return the full state snapshot.

**Rate-limited:** 120 requests per minute per IP.

```bash
curl -X POST http://localhost:8000/api/v1/step
```

**Response** `200 OK`:

```json
{
  "step": 151,
  "vehicles": [
    {
      "id": 0,
      "x": 234.5,
      "y": 102.3,
      "vx": 12.1,
      "vy": 0.5,
      "mass": 0.83,
      "speed_kmh": 43.6,
      "type": "slow"
    }
  ],
  "mean_speed_kmh": 72.45,
  "n_vehicles": 100
}
```

**Error** `409 Conflict` (no simulation created):

```json
{
  "detail": {
    "type": "about:blank",
    "title": "No simulation",
    "status": 409,
    "detail": "No simulation created. POST /api/v1/simulate first."
  }
}
```

---

### 6.2 GET /api/v1/network/state

Return the current gravitational state of the simulation without advancing it.

```bash
curl http://localhost:8000/api/v1/network/state
```

**Response** `200 OK`: Same structure as `POST /api/v1/step` (state snapshot without stepping).

**Error** `409 Conflict`: Same as `POST /api/v1/step` when no simulation exists.

---

## 7. Analysis

### 7.1 GET /api/v1/potential

Compute and return the current gravitational potential field on a 2D grid.

**Rate-limited:** 120 requests per minute per IP.

| Query Parameter | Type | Default | Constraints | Description |
|---|---|---|---|---|
| `resolution` | float | `20.0` | 1.0--500.0 | Grid spacing in metres |

```bash
curl "http://localhost:8000/api/v1/potential?resolution=50.0"
```

**Response** `200 OK`:

```json
{
  "grid_width": 20,
  "grid_height": 20,
  "x_min": 0.0,
  "y_min": 0.0,
  "x_max": 1000.0,
  "y_max": 1000.0,
  "potential": [[0.12, -0.05, ...], ...]
}
```

| Field | Type | Description |
|---|---|---|
| `grid_width` | int | Number of columns in the potential grid |
| `grid_height` | int | Number of rows in the potential grid |
| `x_min`, `y_min` | float | Bottom-left corner of the grid (metres) |
| `x_max`, `y_max` | float | Top-right corner of the grid (metres) |
| `potential` | float[][] | 2D array of potential values (row-major) |

**Error** `409 Conflict`: No simulation created.

---

### 7.2 POST /api/v1/predict

Run a forward prediction on a cloned simulation. The live simulation is **not modified**.

**Rate-limited:** 120 requests per minute per IP.

| Query Parameter | Type | Default | Constraints | Description |
|---|---|---|---|---|
| `horizon_s` | float | `900.0` | 1.0--3600.0 | Prediction horizon in seconds |

```bash
curl -X POST "http://localhost:8000/api/v1/predict?horizon_s=900"
```

**Response** `200 OK`:

```json
{
  "horizon_s": 900.0,
  "n_steps_run": 9000,
  "predicted_step": 9150,
  "mean_speed_kmh": 68.2,
  "n_vehicles": 100,
  "vehicles": [
    {
      "x": 450.2,
      "y": 310.8,
      "vx": 10.5,
      "vy": -0.3,
      "mass": -0.42,
      "speed_kmh": 37.8
    }
  ]
}
```

| Field | Type | Description |
|---|---|---|
| `horizon_s` | float | Actual prediction horizon used (seconds) |
| `n_steps_run` | int | Number of integration steps executed |
| `predicted_step` | int | Step count in the cloned simulation |
| `mean_speed_kmh` | float | Mean speed of all vehicles at prediction time |
| `n_vehicles` | int | Number of vehicles |
| `vehicles` | object[] | Predicted state of each vehicle |

**Error** `409 Conflict`: No simulation created.

> **Note:** Prediction with `horizon_s=900` (15 minutes) at `dt=0.1` executes 9000 steps on the server. Expect response times of several seconds depending on `n_vehicles`.

---

## 8. WebSocket -- Real-Time Streaming

### Endpoint

```
ws://<host>:8000/ws/stream/potential
```

### Connection Behavior

1. On connect, the server auto-creates a default simulation (5x5 grid, 100 vehicles) if none exists via REST.
2. The simulation starts in **running** mode immediately.
3. The server pushes frames at approximately **1 Hz**.
4. Each frame advances the simulation by 10 micro-steps (1 second of simulation time at `dt=0.1`).

### Client Commands

Send JSON text messages to control the simulation.

#### start

Resume the simulation loop.

```json
{"type": "start"}
```

#### stop

Pause the simulation loop. Frames stop being sent.

```json
{"type": "stop"}
```

#### step

Execute a single simulation step (useful while paused).

```json
{"type": "step"}
```

#### config

Reconfigure and reset the simulation. Accepts all `SimulationConfig` fields alongside `"type": "config"`.

```json
{
  "type": "config",
  "grid_rows": 8,
  "grid_cols": 8,
  "n_vehicles": 500,
  "G_s": 5.0,
  "dt": 0.1
}
```

**Server acknowledgement:**

```json
{"type": "config_ok", "n_vehicles": 500}
```

**Server error (invalid config):**

```json
{"type": "error", "detail": "1 validation error for SimulationConfig ..."}
```

### Server Frame Format

Each frame is a JSON message with the following structure:

```json
{
  "type": "frame",
  "step": 1500,
  "grid_width": 50,
  "grid_height": 50,
  "potential": [[0.12, -0.05, ...], ...],
  "vehicles": [
    {
      "id": 0,
      "x": 234.5,
      "y": 102.3,
      "vx": 12.1,
      "vy": 0.5,
      "mass": 0.83,
      "speed_kmh": 43.6,
      "type": "slow"
    }
  ],
  "kpi": {
    "mean_speed_kmh": 72.45,
    "congestion_index": 0.35,
    "n_vehicles": 100
  }
}
```

| Field | Type | Description |
|---|---|---|
| `type` | string | Always `"frame"` |
| `step` | int | Current simulation step count |
| `grid_width` | int | Potential grid columns |
| `grid_height` | int | Potential grid rows |
| `potential` | float[][] | 2D potential field (resolution = 20 m) |
| `vehicles` | object[] | Current state of each vehicle |
| `kpi.mean_speed_kmh` | float | Mean speed across all vehicles |
| `kpi.congestion_index` | float | Fraction of vehicles classified as "slow" (positive mass) |
| `kpi.n_vehicles` | int | Total vehicle count |

### Example (websocat)

```bash
# Connect and receive frames
websocat ws://localhost:8000/ws/stream/potential

# Send a pause command
echo '{"type": "stop"}' | websocat ws://localhost:8000/ws/stream/potential
```

### Example (Python)

```python
import asyncio
import websockets
import json

async def stream():
    async with websockets.connect("ws://localhost:8000/ws/stream/potential") as ws:
        # Receive 5 frames
        for _ in range(5):
            frame = json.loads(await ws.recv())
            print(f"Step {frame['step']}, "
                  f"speed={frame['kpi']['mean_speed_kmh']:.1f} km/h, "
                  f"congestion={frame['kpi']['congestion_index']:.2f}")

        # Pause
        await ws.send(json.dumps({"type": "stop"}))

asyncio.run(stream())
```

---

## 9. Error Handling

The API uses standard HTTP status codes. Error responses for REST endpoints follow the RFC 7807 Problem Details structure where applicable.

### Status Codes

| Code | Meaning | When |
|---|---|---|
| `200` | OK | Successful GET or POST (step, state, potential, predict) |
| `201` | Created | Simulation created via `POST /api/v1/simulate` |
| `409` | Conflict | Operation requires a simulation but none exists |
| `422` | Unprocessable Entity | Request body fails Pydantic validation |
| `429` | Too Many Requests | Rate limit exceeded (120 req/min per IP per endpoint) |
| `503` | Service Unavailable | Readiness probe fails (no simulation loaded) |

### 409 Conflict -- No Simulation

Returned by `POST /api/v1/step`, `GET /api/v1/network/state`, `GET /api/v1/potential`, `POST /api/v1/predict` when no simulation has been created.

```json
{
  "detail": {
    "type": "about:blank",
    "title": "No simulation",
    "status": 409,
    "detail": "No simulation created. POST /api/v1/simulate first."
  }
}
```

### 422 Unprocessable Entity -- Validation Error

Returned by `POST /api/v1/simulate` when the request body fails schema validation.

```json
{
  "detail": [
    {
      "loc": ["body", "grid_rows"],
      "msg": "ensure this value is less than or equal to 50",
      "type": "value_error.number.not_le"
    }
  ]
}
```

### 429 Too Many Requests -- Rate Limited

Returned when a client exceeds 120 requests per minute on a rate-limited endpoint.

```json
{
  "detail": "Rate limit exceeded. Try again later."
}
```

### 503 Service Unavailable -- Not Ready

Returned by `GET /ready` when no simulation is loaded.

```json
{
  "detail": "No simulation loaded"
}
```

---

## 10. CORS Configuration

Cross-Origin Resource Sharing is configured via the `GRAVTRAFFIC_CORS_ORIGINS` environment variable.

| Setting | Value |
|---|---|
| **Environment variable** | `GRAVTRAFFIC_CORS_ORIGINS` |
| **Default** | `http://localhost:3000,http://localhost:8080` |
| **Format** | Comma-separated list of allowed origins |
| **Wildcard** | Set to `*` for open access (development only) |
| **Allowed methods** | `GET`, `POST` |
| **Allowed headers** | `*` (all) |
| **Credentials** | Enabled |

### Examples

```bash
# Default (localhost only)
uvicorn gravtraffic.api.app:app --host 0.0.0.0 --port 8000

# Allow a specific frontend origin
GRAVTRAFFIC_CORS_ORIGINS="https://dashboard.example.com" \
  uvicorn gravtraffic.api.app:app --host 0.0.0.0 --port 8000

# Open access (development only -- do NOT use in production)
GRAVTRAFFIC_CORS_ORIGINS="*" \
  uvicorn gravtraffic.api.app:app --host 0.0.0.0 --port 8000

# Multiple origins
GRAVTRAFFIC_CORS_ORIGINS="https://app.example.com,https://staging.example.com" \
  uvicorn gravtraffic.api.app:app --host 0.0.0.0 --port 8000
```

---

## Appendix: Quick Reference

| Method | Endpoint | Rate Limited | Requires Simulation |
|---|---|---|---|
| GET | `/health` | No | No |
| GET | `/ready` | No | Yes (503 if absent) |
| GET | `/metrics` | No | No |
| GET | `/api/v1/status` | No | No |
| POST | `/api/v1/simulate` | No | No (creates one) |
| POST | `/api/v1/step` | Yes | Yes (409 if absent) |
| GET | `/api/v1/network/state` | No | Yes (409 if absent) |
| GET | `/api/v1/potential` | Yes | Yes (409 if absent) |
| POST | `/api/v1/predict` | Yes | Yes (409 if absent) |
| WS | `/ws/stream/potential` | No | No (auto-creates) |
