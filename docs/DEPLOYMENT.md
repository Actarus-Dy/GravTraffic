# GravTraffic -- Deployment Guide

Deployment, configuration, and production operations for the GravTraffic (C-01) API server.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Local Development](#local-development)
3. [Docker](#docker)
4. [Docker Compose](#docker-compose)
5. [Environment Variables](#environment-variables)
6. [CI/CD Pipeline](#cicd-pipeline)
7. [Health Monitoring](#health-monitoring)
8. [Prometheus Integration](#prometheus-integration)
9. [Production Recommendations](#production-recommendations)
10. [GPU Acceleration](#gpu-acceleration)

---

## Prerequisites

| Requirement   | Version   | Notes                                      |
|---------------|-----------|--------------------------------------------|
| Python        | 3.12+     | CPython only (CuPy requires it)            |
| pip           | 23.0+     | Ships with Python 3.12                     |
| Docker        | 24+       | Optional -- for containerised deployments  |
| Docker Compose| 2.20+     | Optional -- `docker compose` v2 syntax     |
| NVIDIA Driver | 535+      | Optional -- only for GPU acceleration      |

---

## Local Development

### Install dependencies

```bash
pip install -e ".[dev,api]"
```

This installs the core engine, test tooling (pytest, ruff), and the FastAPI/uvicorn stack.

### Start the development server

```bash
uvicorn gravtraffic.api.app:app --host 0.0.0.0 --port 8000 --reload --reload-dir gravtraffic
```

### Verify

```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

### Dashboard

Once the server is running, open the real-time visualization dashboard at:

```
http://localhost:8000/dashboard
```

### Run tests

```bash
# Fast suite (~70 s) -- excludes slow calibration and corridor tests
python -m pytest tests/ \
  --ignore=tests/test_calibration_pure.py \
  --ignore=tests/test_calibration_unified.py \
  --ignore=tests/test_corridor_rivoli.py \
  -v --tb=short

# Full suite (~15 min)
python -m pytest tests/ -v --tb=short
```

---

## Docker

### Build -- CPU only

```bash
docker build -t gravtraffic .
```

### Build -- with GPU support

```bash
docker build --build-arg INSTALL_GPU=1 -t gravtraffic:gpu .
```

This installs the `cupy-cuda12x` package inside the image. The host must have the NVIDIA Container Toolkit (`nvidia-docker`) configured.

### Run

```bash
docker run -p 8000:8000 gravtraffic
```

With GPU:

```bash
docker run --gpus all -p 8000:8000 gravtraffic:gpu
```

### Override CORS origins

```bash
docker run -p 8000:8000 \
  -e GRAVTRAFFIC_CORS_ORIGINS="https://app.example.com" \
  gravtraffic
```

### Built-in health check

The Dockerfile includes a `HEALTHCHECK` directive that polls `/health` every 30 seconds with a 5-second timeout, 10-second start period, and 3 retries. Docker marks the container as `unhealthy` if these fail.

---

## Docker Compose

### Development mode (default)

```bash
docker compose up
```

This starts the API server with `--reload` and a read-only bind mount of the `gravtraffic/` source directory, so code changes take effect immediately without rebuilding the image.

Default CORS origins: `http://localhost:3000,http://localhost:8080`.

### Production mode

```bash
docker compose --profile prod up
```

The `gravtraffic-prod` service:

- Does **not** mount source code (uses the baked image).
- Applies a 2 GB memory limit via `deploy.resources.limits.memory`.
- Uses `restart: unless-stopped` for automatic recovery.
- Reads `GRAVTRAFFIC_CORS_ORIGINS` from the host environment (defaults to `http://localhost:3000`).

### Rebuild after code changes

```bash
docker compose build
docker compose up
```

---

## Environment Variables

| Variable                     | Default                                        | Description                                                     |
|------------------------------|------------------------------------------------|-----------------------------------------------------------------|
| `GRAVTRAFFIC_CORS_ORIGINS`   | `http://localhost:3000,http://localhost:8080`   | Comma-separated list of allowed CORS origins. Set to `*` for open access (not recommended in production). |

Set it on the command line:

```bash
export GRAVTRAFFIC_CORS_ORIGINS="https://dashboard.example.com,https://admin.example.com"
uvicorn gravtraffic.api.app:app --host 0.0.0.0 --port 8000
```

---

## CI/CD Pipeline

The GitHub Actions workflow (`.github/workflows/ci.yml`) runs on every push to `main` and on every pull request targeting `main`. It has three sequential jobs:

### 1. Lint (ruff)

- Runs `ruff check` and `ruff format --check` against `gravtraffic/` and `tests/`.
- Fails fast -- the test job does not start if linting fails.

### 2. Test (fast suite)

- Installs `.[dev,api]` with pip caching.
- Runs the fast test suite (excludes `test_calibration_pure.py`, `test_calibration_unified.py`, and `test_corridor_rivoli.py`).
- Produces a JUnit XML report uploaded as a build artifact.

### 3. Docker build + smoke test

- Builds the Docker image (`gravtraffic:ci`).
- Starts a container and polls `/health` for up to 30 seconds.
- Verifies that both `/health` and `/api/v1/status` return HTTP 200.
- Stops the container.

All three jobs must pass for the pipeline to be green.

---

## Health Monitoring

The API exposes three infrastructure endpoints, all unauthenticated:

### `GET /health` -- Liveness probe

Returns HTTP 200 with `{"status": "ok"}` as long as the process is running. Use this for Kubernetes `livenessProbe` or Docker `HEALTHCHECK`.

```bash
curl -sf http://localhost:8000/health
```

### `GET /ready` -- Readiness probe

Returns HTTP 200 with `{"status": "ready", "n_vehicles": <N>}` if a simulation is loaded. Returns HTTP 503 if no simulation exists. Use this for Kubernetes `readinessProbe` to avoid routing traffic before the simulation is initialized.

```bash
curl -sf http://localhost:8000/ready
```

### `GET /metrics` -- Prometheus metrics

Returns metrics in Prometheus text exposition format (`text/plain; version=0.0.4`). See the next section for details.

```bash
curl -s http://localhost:8000/metrics
```

---

## Prometheus Integration

The `/metrics` endpoint exports the following gauges and counters:

| Metric                            | Type    | Description                              |
|-----------------------------------|---------|------------------------------------------|
| `gravtraffic_uptime_seconds`      | gauge   | Server uptime in seconds                 |
| `gravtraffic_step_count`          | counter | Total simulation steps completed         |
| `gravtraffic_n_vehicles`          | gauge   | Current number of vehicles               |
| `gravtraffic_mean_speed_kmh`      | gauge   | Mean vehicle speed in km/h               |
| `gravtraffic_rate_limit_keys`     | gauge   | Active rate limiter keys (connection count proxy) |

### Prometheus scrape configuration

Add the following job to your `prometheus.yml`:

```yaml
scrape_configs:
  - job_name: "gravtraffic"
    scrape_interval: 15s
    metrics_path: "/metrics"
    static_configs:
      - targets: ["localhost:8000"]
        labels:
          instance: "gravtraffic-01"
          project: "janus-civil"
          app_id: "C-01"
```

> **Security note:** The `/metrics` endpoint is unauthenticated. In production, expose it only on an internal network or add an authentication middleware. Do not expose it on the public interface.

---

## Production Recommendations

### Single worker only

GravTraffic **must** run with a single uvicorn worker (`--workers 1`). The `GravSimulation` object holds mutable NumPy arrays and is not fork-safe. Multiple workers would create independent simulation copies with divergent state. The Dockerfile enforces this by default.

### Memory limits

- A 5x5 grid with 1,000 vehicles uses approximately 200 MB.
- The Docker Compose production profile sets a 2 GB memory limit.
- For large simulations (10,000+ vehicles), increase the limit accordingly:

```yaml
deploy:
  resources:
    limits:
      memory: 4G
```

### Reverse proxy

Place the API behind a reverse proxy (nginx, Caddy, Traefik) for:

- TLS termination
- Request buffering
- Rate limiting at the infrastructure level (the built-in rate limiter is in-memory and per-process only)

### Restart policy

Use `restart: unless-stopped` (Docker Compose) or a systemd unit with `Restart=on-failure` for bare-metal deployments.

### Logging

Uvicorn writes access logs to stdout by default. Pipe them to your log aggregation system. For structured JSON logging, add `--log-config` with a custom logging dictionary.

---

## GPU Acceleration

GravTraffic includes a GPU-accelerated O(N^2) force engine (`ForceEngineGPU`) that uses CuPy for CUDA computation. For the Janus signed-mass model, naive O(N^2) on GPU outperforms CPU Barnes-Hut O(N log N) for moderate N.

### Install CuPy

```bash
pip install "gravtraffic[gpu]"
# or directly:
pip install cupy-cuda12x>=13.0
```

Requires an NVIDIA GPU with CUDA 12.x drivers.

### Auto-detection

`GravSimulation` auto-detects GPU availability at startup:

- If CuPy is installed and a compatible GPU is present, the GPU engine is used automatically.
- If CuPy is not installed, the simulation falls back to the CPU engine (Barnes-Hut or naive) with no code changes required.
- You can force CPU mode by passing `use_gpu=False` to `GravSimulation`.

### `max_n` threshold

`ForceEngineGPU` has a `max_n` parameter (default: 10,000) that controls the upper bound for GPU computation. When the number of vehicles exceeds `max_n`, the engine falls back to CPU naive computation to avoid GPU out-of-memory errors.

- 10,000 vehicles requires approximately 1.5 GB of VRAM for the N x N distance and force matrices.
- Increase `max_n` if your GPU has sufficient VRAM (e.g., 20,000 for 6+ GB VRAM).

### Docker with GPU

```bash
docker build --build-arg INSTALL_GPU=1 -t gravtraffic:gpu .
docker run --gpus all -p 8000:8000 gravtraffic:gpu
```

The `--gpus all` flag requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) to be installed on the host.
