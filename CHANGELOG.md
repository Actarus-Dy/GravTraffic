# Changelog

All notable changes to GravTraffic (C-01) will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.2.0] - 2026-03-23

### Added

- Phase 5: Dockerfile multi-stage, docker-compose, GitHub Actions CI, /health /ready /metrics, OpenAPI v0.2.0
- Phase 6: Real-time dashboard (/dashboard) with canvas visualization, heatmap, controls, 5 scenarios
- Phase 7: Numba JIT force engines (323x speedup naive, 151x BH), dual-tree BH, auto-selection GPU>Numba>Python
- Phase 8: Scientific validation module — FD R²=0.9796, emergence quantification, sensitivity analysis, reproducible JSON report

### Fixed (DA Sprints #3-9)

- Leapfrog symplecticity: v_half passed to _compute_accelerations
- Signal optimizer: velocities now passed to try_optimize
- Docstrings: -G_s corrected to +G_s
- G_s defaults harmonized to 5.0 everywhere
- Local densities: cKDTree (Euclidean, O(N log N)) replaces Python loop
- API: asyncio lock, rate limiter per-IP, proper HTTP 409, input validation
- Predict: clone under lock, run_until outside lock, max_steps guard, 0-vehicle early return
- GPU: max_n threshold with CPU fallback
- Dashboard: Math.min stack overflow fix, WS backoff, config field filtering
- Validation: FD init from v_free (no circular bias), G_s=0.0 baseline, upstream mask direction

## [0.1.0] - 2026-03-22

### Added

- Phase 1: GravCore engine (mass assignment, force computation, leapfrog integrator)
- Phase 2: Mesa ABM, road network, API, emergence validation
- Phase 3: Signal optimization, green wave, Rivoli corridor, metrics
- Phase 4: Production API (CORS, rate limit), prediction T+15min, GPU (CuPy)

### Fixed (DA Sprints #1-2)

- Force sign convention corrected (was inverted)
- Red-light masses injected into physics
- Local densities recalculated dynamically
- Unified calibration: G_s=5.0, beta=0.5, gamma=0.3
