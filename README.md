# GravTraffic (C-01)

![Tests](https://img.shields.io/badge/tests-508%20passed-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.12%2B-blue)
![API](https://img.shields.io/badge/API-v0.2.0-009688)

**Gravitational traffic flow simulator based on the Janus Cosmological Model.**

GravTraffic models each vehicle as a gravitational body with signed mass.
Vehicles slower than the segment average receive **positive mass** (attract
neighbours, form jams), while faster vehicles receive **negative mass** (repel
neighbours, create fluid zones). Traffic phenomena -- congestion waves, convoy
formation, shock propagation, overtaking -- emerge naturally from gravitational
interactions combined with a physically motivated drag enrichment. No
car-following rules are coded; all collective behaviour is emergent.

Part of the **Actarus-Dy Software** catalogue (26 applications, 8 domains,
46 formulas).

---

## Quickstart

```bash
# 1. Install (Python 3.12+)
pip install -e ".[api]"

# 2. Start the API server
uvicorn gravtraffic.api.app:app --host 0.0.0.0 --port 8000

# 3. Create a simulation
curl -X POST http://localhost:8000/api/v1/simulate \
  -H "Content-Type: application/json" \
  -d '{"grid_rows": 5, "grid_cols": 5, "n_vehicles": 200}'

# 4. Open the real-time dashboard
#    http://localhost:8000/dashboard
```

Time to first result: under 2 minutes (install + launch + create simulation).

---

## Architecture

```
gravtraffic/
+-- core/                      Janus physics engine
|   +-- mass_assigner.py           Mass m_i from speed deviation (vectorized NumPy)
|   +-- force_engine.py            Naive O(N^2) + Barnes-Hut O(N log N)
|   +-- quadtree.py                Dual-tree Barnes-Hut for signed masses
|   +-- integrator.py              Leapfrog symplectic + adaptive CFL dt
|   +-- potential_field.py         Phi(x,y) gravitational potential field
|   +-- signal_optimizer.py        Minimize integral(Phi) dt over 2-min horizon
|   +-- green_wave.py              Green wave corridor synchronization
|   +-- simulation.py              GravSimulation pipeline (orchestrator)
|   +-- metrics.py                 Throughput, delay, stops, LOS, travel time
|   +-- calibration_unified.py     Unified parameter search (gravity + drag)
|
+-- agents/                    Mesa 3.x ABM layer
|   +-- vehicle_agent.py          Passive agent -- physics pushed from simulation
|   +-- intersection_agent.py     Traffic light with red-light mass injection
|   +-- traffic_model.py          Mesa Model wiring simulation + agents
|
+-- network/                   Road network
|   +-- road_network.py           OSMnx loader + synthetic grid generator
|
+-- api/                       FastAPI REST + WebSocket
|   +-- app.py                    REST endpoints + WS /stream/potential at 1 Hz
|   +-- static/                   Dashboard HTML/JS assets
|
+-- scenarios/                 Benchmark scenarios
    +-- rivoli.py                 12-intersection corridor (Rue de Rivoli)

tests/                         508 tests (24 test files, 1 xfail)
benchmarks/                    Performance benchmarking scripts
```

---

## Physics

### Gravitational Force

```
F_vec = +G_s * m_i * m_j / d^3 * (x_j - x_i, y_j - y_i)
```

- Same-sign masses (`m_i * m_j > 0`): attraction -- vehicles cluster, jams form
- Opposite-sign masses (`m_i * m_j < 0`): repulsion -- vehicles spread, flow improves
- Softened distance: `d = sqrt(dx^2 + dy^2 + epsilon^2)`, epsilon = 10 m

### Mass Assignment

```
m_i = sgn(v_mean - v_i) * |v_mean - v_i|^beta * rho(x_i) / rho_0
```

Slow vehicles (below segment average) get positive mass. Fast vehicles get
negative mass. The exponent `beta` controls nonlinearity.

### Drag Enrichment

```
a_drag = gamma * (v_eq(rho) - |v_i|) * direction_i
v_eq(rho) = v_free * max(0, 1 - rho / rho_jam)
```

The drag term provides baseline speed-density behaviour (Greenshields
fundamental diagram). Gravity provides inter-vehicle interactions and
emergent collective phenomena on top of this baseline.

### Calibrated Parameters

| Parameter   | Value   | Description                        |
|-------------|--------:|------------------------------------|
| `G_s`       |     5.0 | Social gravitational constant      |
| `beta`      |     0.5 | Mass-assignment exponent           |
| `softening` |    10.0 | Softening length (metres)          |
| `gamma`     |     0.3 | Drag coefficient                   |
| `v_free`    |   33.33 | Free-flow speed (m/s, 120 km/h)    |
| `rho_jam`   |   150.0 | Jam density (vehicles/km)          |

---

## API

All REST endpoints are versioned under `/api/v1/`. The WebSocket stream lives
at `/ws/stream/potential`.

| Method | Path                     | Description                                      |
|--------|--------------------------|--------------------------------------------------|
| GET    | `/health`                | Liveness probe (always 200)                      |
| GET    | `/ready`                 | Readiness probe (200 if simulation loaded)       |
| GET    | `/metrics`               | Prometheus-compatible metrics (text exposition)   |
| GET    | `/api/v1/status`         | Current simulation status (step, vehicle count)  |
| POST   | `/api/v1/simulate`       | Create a new simulation (replaces existing)      |
| POST   | `/api/v1/step`           | Execute one simulation step, return state         |
| GET    | `/api/v1/network/state`  | Current gravitational state snapshot              |
| GET    | `/api/v1/potential`      | Compute potential field (query: `resolution`)     |
| POST   | `/api/v1/predict`        | T+N prediction (query: `horizon_s`, default 900s) |
| WS     | `/ws/stream/potential`   | Real-time frames at 1 Hz (vehicles + potential)  |
| GET    | `/dashboard`             | Serve the real-time visualization dashboard       |

Rate limiting: 120 requests per 60 seconds per client IP on step/potential/predict
endpoints. CORS configurable via `GRAVTRAFFIC_CORS_ORIGINS` environment variable.

---

## Dashboard

The built-in dashboard at `/dashboard` provides real-time visualization of the
simulation:

- **Vehicle map** -- Colour-coded dots: red for positive-mass (slow/congested)
  vehicles, blue for negative-mass (fast/fluid) vehicles, positioned on the
  road network grid.
- **Potential field heatmap** -- 2D gravitational potential overlay showing
  attraction wells (congestion zones) and repulsion hills (fluid zones),
  updated at 1 Hz via WebSocket.
- **Live KPI panel** -- Mean speed (km/h), congestion index, vehicle count,
  and Level of Service (A-F) displayed in real time.
- **Playback controls** -- Start, stop, and single-step buttons. Configuration
  panel to adjust vehicle count, grid size, and physics parameters without
  restarting the server.

---

## Performance

Benchmarks measured on a 2 km highway segment, 10 simulation steps, single
thread (unless noted). Times are per step.

| Backend        | 1,000 vehicles | 5,000 vehicles | 10,000 vehicles |
|----------------|---------------:|---------------:|----------------:|
| Python (naive O(N^2)) | ~12 ms   | ~280 ms        | ~1,100 ms       |
| Barnes-Hut O(N log N) | ~8 ms    | ~55 ms         | ~130 ms         |
| Numba JIT              | ~2 ms    | ~45 ms         | ~170 ms         |
| GPU (CuPy CUDA)        | ~1 ms    | ~3 ms          | ~8 ms           |

Install optional accelerators:

```bash
pip install -e ".[accel]"   # Numba JIT
pip install -e ".[gpu]"     # CuPy CUDA 12.x
```

Run the benchmark suite:

```bash
python benchmarks/bench_gravcore.py
```

---

## Scientific Validation

GravTraffic is validated against the Greenshields fundamental diagram and
tested for emergent traffic phenomena.

| Metric                          | Value     | Target    |
|---------------------------------|-----------|-----------|
| Greenshields R^2                | >= 0.95   | >= 0.90   |
| Speed RMSE (vs Greenshields)    | < 3 m/s   | < 5 m/s   |
| Emergence score (upstream decel)| > 0.3     | > 0.1     |

**Emergence validation:** When a single slow vehicle is injected into a
uniform stream, gravitational attraction causes chain-reaction deceleration
that propagates upstream -- a shock wave emerges purely from N-body physics,
with no explicit car-following rules.

**Corridor validation:** The Rue de Rivoli scenario (12 intersections, 1.8 km)
demonstrates signal optimization reducing mean delay by over 20% compared to
fixed-timing signals, using potential-field minimization.

---

## Docker

### Build and run

```bash
# Standard build
docker build -t gravtraffic .

# GPU-enabled build
docker build --build-arg INSTALL_GPU=1 -t gravtraffic:gpu .

# Run
docker run -p 8000:8000 gravtraffic
```

### Docker Compose

```bash
# Development (hot reload)
docker compose up

# Production
docker compose --profile prod up
```

The production profile enforces a 2 GB memory limit and `unless-stopped`
restart policy. The simulation server runs a single uvicorn worker
(GravSimulation is not fork-safe).

---

## Testing

508 tests across 24 test files, including unit tests, integration tests,
scientific validation tests, and API endpoint tests.

```bash
# Run all tests
python -m pytest tests/ -v --tb=short

# Fast tests only (~70 seconds, skips calibration and corridor)
python -m pytest tests/ \
  --ignore=tests/test_calibration_pure.py \
  --ignore=tests/test_calibration_unified.py \
  --ignore=tests/test_corridor_rivoli.py

# Single test file
python -m pytest tests/test_force_engine.py -v

# With coverage
python -m pytest tests/ --cov=gravtraffic --cov-report=term-missing
```

Install dev dependencies:

```bash
pip install -e ".[dev,api]"
```

---

## Project Status

| Phase | Description                                        | Status   |
|-------|----------------------------------------------------|----------|
| 1     | GravCore engine (force, mass, integrator, QuadTree) | Complete |
| 2     | Mesa ABM, road network, API, emergence validation   | Complete |
| 3     | Signal optimization, green wave, Rivoli corridor    | Complete |
| DA    | Sign fix, obstacle injection, unified calibration   | Complete |
| 4     | Production API, T+15min prediction, GPU (CUDA/CuPy) | Planned  |

---

## License

GravTraffic is released under the [MIT License](LICENSE).

Copyright (c) 2026 Actarus-Dy Software.
