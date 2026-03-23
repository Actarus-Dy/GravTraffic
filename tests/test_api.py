"""Tests for the GravTraffic FastAPI application.

Covers REST endpoints and WebSocket streaming using FastAPI's TestClient.
Each test resets the module-level ``state`` object via a pytest fixture to
prevent cross-test contamination.

Author: Agent #06 API Architect
Date: 2026-03-22
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from gravtraffic.api.app import app, state


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture(autouse=True)
def _reset_state():
    """Reset the global AppState before every test."""
    state.reset()
    yield
    state.reset()


@pytest.fixture()
def client() -> TestClient:
    """Fresh TestClient bound to the FastAPI app."""
    return TestClient(app)


DEFAULT_CONFIG = {
    "grid_rows": 3,
    "grid_cols": 3,
    "block_size": 100.0,
    "n_vehicles": 20,
    "G_s": 5.0,
    "beta": 0.5,
    "dt": 0.1,
    "seed": 42,
}


# ======================================================================
# REST -- GET /api/v1/status
# ======================================================================


class TestGetStatus:
    def test_status_no_simulation(self, client: TestClient) -> None:
        resp = client.get("/api/v1/status")
        assert resp.status_code == 200
        body = resp.json()
        assert body["running"] is False
        assert body["step"] == 0
        assert body["n_vehicles"] == 0

    def test_status_after_create(self, client: TestClient) -> None:
        client.post("/api/v1/simulate", json=DEFAULT_CONFIG)
        resp = client.get("/api/v1/status")
        body = resp.json()
        assert body["running"] is False
        assert body["n_vehicles"] == 20


# ======================================================================
# REST -- POST /api/v1/simulate
# ======================================================================


class TestCreateSimulation:
    def test_create_returns_201(self, client: TestClient) -> None:
        resp = client.post("/api/v1/simulate", json=DEFAULT_CONFIG)
        assert resp.status_code == 201
        body = resp.json()
        assert body["status"] == "created"
        assert body["n_vehicles"] == 20

    def test_create_replaces_previous(self, client: TestClient) -> None:
        client.post("/api/v1/simulate", json=DEFAULT_CONFIG)
        new_cfg = {**DEFAULT_CONFIG, "n_vehicles": 10}
        resp = client.post("/api/v1/simulate", json=new_cfg)
        assert resp.status_code == 201
        assert resp.json()["n_vehicles"] == 10
        # Status reflects the new simulation
        assert client.get("/api/v1/status").json()["n_vehicles"] == 10


# ======================================================================
# REST -- POST /api/v1/step
# ======================================================================


class TestStepSimulation:
    def test_step_without_simulation(self, client: TestClient) -> None:
        resp = client.post("/api/v1/step")
        assert resp.status_code == 409
        body = resp.json()
        assert body["detail"]["title"] == "No simulation"

    def test_step_returns_state(self, client: TestClient) -> None:
        client.post("/api/v1/simulate", json=DEFAULT_CONFIG)
        resp = client.post("/api/v1/step")
        assert resp.status_code == 200
        body = resp.json()
        assert "step" in body
        assert body["step"] == 1
        assert "vehicles" in body
        assert isinstance(body["vehicles"], list)
        assert len(body["vehicles"]) == 20
        assert "intersections" in body

    def test_step_increments(self, client: TestClient) -> None:
        client.post("/api/v1/simulate", json=DEFAULT_CONFIG)
        client.post("/api/v1/step")
        client.post("/api/v1/step")
        resp = client.post("/api/v1/step")
        assert resp.json()["step"] == 3

    def test_vehicle_dict_structure(self, client: TestClient) -> None:
        client.post("/api/v1/simulate", json=DEFAULT_CONFIG)
        resp = client.post("/api/v1/step")
        v = resp.json()["vehicles"][0]
        for key in ("id", "x", "y", "vx", "vy", "mass", "speed_kmh", "type"):
            assert key in v, f"Missing key {key!r} in vehicle dict"


# ======================================================================
# REST -- GET /api/v1/network/state
# ======================================================================


class TestNetworkState:
    def test_network_state_no_sim(self, client: TestClient) -> None:
        resp = client.get("/api/v1/network/state")
        assert resp.status_code == 409

    def test_network_state_after_step(self, client: TestClient) -> None:
        client.post("/api/v1/simulate", json=DEFAULT_CONFIG)
        client.post("/api/v1/step")
        resp = client.get("/api/v1/network/state")
        assert resp.status_code == 200
        body = resp.json()
        assert "vehicles" in body
        assert "intersections" in body
        assert "kpi" in body


# ======================================================================
# REST -- GET /api/v1/potential
# ======================================================================


class TestPotentialField:
    def test_potential_no_sim(self, client: TestClient) -> None:
        resp = client.get("/api/v1/potential")
        assert resp.status_code == 409

    def test_potential_returns_grid(self, client: TestClient) -> None:
        client.post("/api/v1/simulate", json=DEFAULT_CONFIG)
        resp = client.get("/api/v1/potential", params={"resolution": 50.0})
        assert resp.status_code == 200
        body = resp.json()
        assert "grid_width" in body
        assert "grid_height" in body
        assert "potential" in body
        assert isinstance(body["potential"], list)
        assert len(body["potential"]) > 0
        # Bounds present
        for key in ("x_min", "y_min", "x_max", "y_max"):
            assert key in body


# ======================================================================
# WebSocket -- /ws/stream/potential
# ======================================================================


class TestWebSocketStream:
    def test_receive_frames(self, client: TestClient) -> None:
        """Connect, receive at least 2 frames, verify JSON structure."""
        # Pre-create a small simulation so the WS does not use the default
        client.post("/api/v1/simulate", json=DEFAULT_CONFIG)

        with client.websocket_connect("/ws/stream/potential") as ws:
            frames = []
            for _ in range(2):
                raw = ws.receive_json()
                frames.append(raw)

            assert len(frames) >= 2
            for frame in frames:
                assert frame["type"] == "frame"
                assert "step" in frame
                assert "vehicles" in frame
                assert "potential" in frame
                assert "grid_width" in frame
                assert "grid_height" in frame
                assert "kpi" in frame
                assert "mean_speed_kmh" in frame["kpi"]
                assert "congestion_index" in frame["kpi"]
                assert "n_vehicles" in frame["kpi"]

    def test_stop_command(self, client: TestClient) -> None:
        """Send stop command and verify no further frames arrive."""
        client.post("/api/v1/simulate", json=DEFAULT_CONFIG)

        with client.websocket_connect("/ws/stream/potential") as ws:
            # Receive the first frame to confirm streaming works
            first = ws.receive_json()
            assert first["type"] == "frame"
            step_at_stop = first["step"]

            # Send stop
            ws.send_json({"type": "stop"})

            # Wait and then send a probe -- after stop, step should not advance
            # because the server only advances when running=True.
            # We receive one more potential message (the server may have already
            # been computing the next frame), then it should stop.
            # Try to receive -- if the server stopped, this will time out or
            # return a frame computed before the stop took effect.
            import time

            time.sleep(1.5)

            # Send start again so we can read the step counter
            ws.send_json({"type": "start"})
            after = ws.receive_json()

            # The step counter should have advanced by roughly 10 (one frame)
            # from the start command, NOT from continuous running during the
            # 1.5 s pause.  Allow some tolerance.
            steps_during_pause = after["step"] - step_at_stop
            # During the pause (~1.5 s) at most one extra frame might sneak through
            # (race between stop arriving and next iteration).  Without the pause
            # we would expect ~15 frames worth of steps.  Assert we got at most 2
            # frames' worth.
            assert steps_during_pause <= 30, (
                f"Expected simulation to pause, but step jumped by {steps_during_pause}"
            )

    def test_auto_creates_simulation(self) -> None:
        """If no simulation exists, the WS endpoint creates a default one."""
        assert state.model is None
        test_client = TestClient(app)
        with test_client.websocket_connect("/ws/stream/potential") as ws:
            frame = ws.receive_json()
            assert frame["type"] == "frame"
            assert frame["kpi"]["n_vehicles"] == 100  # default config

    def test_step_command_while_stopped(self, client: TestClient) -> None:
        """Send step command while stopped; verify step count advances by 1.

        Because the WebSocket server loop processes commands asynchronously
        (with a 1 s sleep between iterations), we need to wait long enough
        for the server to pick up the step command and then resume so we
        can read the updated step count via the next frame.
        """
        client.post("/api/v1/simulate", json=DEFAULT_CONFIG)

        with client.websocket_connect("/ws/stream/potential") as ws:
            # Receive the first frame (running=True, 10 micro-steps)
            first_frame = ws.receive_json()
            assert first_frame["type"] == "frame"

            # Stop the simulation loop
            ws.send_json({"type": "stop"})

            # Wait for the server to process the stop (next loop iteration)
            import time
            time.sleep(1.5)

            # Record the step count from the REST endpoint.
            # At this point the loop is paused so the counter is stable.
            step_before = client.get("/api/v1/status").json()["step"]

            # Send exactly one manual step, then immediately resume so the
            # server produces a frame we can read to confirm the step landed.
            ws.send_json({"type": "step"})
            # Give the server time to process the step command
            time.sleep(1.5)

            # Check the step counter advanced by exactly 1
            step_after = client.get("/api/v1/status").json()["step"]
            assert step_after == step_before + 1, (
                f"Expected step to advance by 1 (from {step_before}), "
                f"got {step_after}"
            )


# ======================================================================
# Input validation
# ======================================================================


class TestInputValidation:
    def test_resolution_too_small(self, client: TestClient) -> None:
        """resolution < 1.0 should be rejected by FastAPI validation."""
        client.post("/api/v1/simulate", json=DEFAULT_CONFIG)
        resp = client.get("/api/v1/potential", params={"resolution": 0.1})
        assert resp.status_code == 422

    def test_resolution_too_large(self, client: TestClient) -> None:
        """resolution > 500 should be rejected."""
        client.post("/api/v1/simulate", json=DEFAULT_CONFIG)
        resp = client.get("/api/v1/potential", params={"resolution": 1000.0})
        assert resp.status_code == 422

    def test_invalid_n_vehicles_zero(self, client: TestClient) -> None:
        """n_vehicles=0 should be rejected."""
        bad = {**DEFAULT_CONFIG, "n_vehicles": 0}
        resp = client.post("/api/v1/simulate", json=bad)
        assert resp.status_code == 422

    def test_invalid_G_s_negative(self, client: TestClient) -> None:
        """G_s <= 0 should be rejected."""
        bad = {**DEFAULT_CONFIG, "G_s": -1.0}
        resp = client.post("/api/v1/simulate", json=bad)
        assert resp.status_code == 422

    def test_invalid_dt_too_large(self, client: TestClient) -> None:
        """dt > 1.0 should be rejected."""
        bad = {**DEFAULT_CONFIG, "dt": 5.0}
        resp = client.post("/api/v1/simulate", json=bad)
        assert resp.status_code == 422


# ======================================================================
# CORS headers
# ======================================================================


class TestCORS:
    def test_cors_headers_present(self, client: TestClient) -> None:
        """Preflight OPTIONS request should return CORS headers."""
        resp = client.options(
            "/api/v1/status",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert "access-control-allow-origin" in resp.headers

    def test_cors_disallowed_origin(self, client: TestClient) -> None:
        """An unknown origin should not get CORS allow-origin header."""
        resp = client.options(
            "/api/v1/status",
            headers={
                "Origin": "http://evil.example.com",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert resp.headers.get("access-control-allow-origin") != "http://evil.example.com"


# ======================================================================
# Proper HTTP status codes
# ======================================================================


# ======================================================================
# Infrastructure endpoints (health, ready, metrics)
# ======================================================================


class TestHealthEndpoints:
    def test_health_always_200(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_ready_503_without_simulation(self, client: TestClient) -> None:
        resp = client.get("/ready")
        assert resp.status_code == 503

    def test_ready_200_with_simulation(self, client: TestClient) -> None:
        client.post("/api/v1/simulate", json=DEFAULT_CONFIG)
        resp = client.get("/ready")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ready"
        assert resp.json()["n_vehicles"] == 20

    def test_metrics_returns_prometheus_format(self, client: TestClient) -> None:
        client.post("/api/v1/simulate", json=DEFAULT_CONFIG)
        client.post("/api/v1/step")
        resp = client.get("/metrics")
        assert resp.status_code == 200
        body = resp.text
        assert "gravtraffic_uptime_seconds" in body
        assert "gravtraffic_step_count 1" in body
        assert "gravtraffic_n_vehicles 20" in body
        assert "gravtraffic_mean_speed_kmh" in body

    def test_metrics_without_simulation(self, client: TestClient) -> None:
        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert "gravtraffic_step_count 0" in resp.text

    def test_openapi_schema_complete(self, client: TestClient) -> None:
        """Verify the OpenAPI schema includes all endpoints."""
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        schema = resp.json()
        paths = set(schema["paths"].keys())
        expected = {
            "/health", "/ready", "/metrics",
            "/api/v1/status", "/api/v1/simulate", "/api/v1/step",
            "/api/v1/network/state", "/api/v1/potential", "/api/v1/predict",
        }
        assert expected.issubset(paths), f"Missing: {expected - paths}"
        # Tags should be present
        tags = {t["name"] for t in schema.get("tags", [])}
        assert "infrastructure" in tags
        assert "simulation" in tags
        assert "analysis" in tags


class TestDashboard:
    def test_dashboard_returns_html(self, client: TestClient) -> None:
        resp = client.get("/dashboard")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "GravTraffic" in resp.text
        assert "simCanvas" in resp.text

    def test_dashboard_contains_controls(self, client: TestClient) -> None:
        resp = client.get("/dashboard")
        assert "btnPlay" in resp.text
        assert "scenarioSelect" in resp.text
        assert "sliderGs" in resp.text


class TestHTTPStatusCodes:
    def test_step_no_sim_returns_409(self, client: TestClient) -> None:
        resp = client.post("/api/v1/step")
        assert resp.status_code == 409

    def test_network_state_no_sim_returns_409(self, client: TestClient) -> None:
        resp = client.get("/api/v1/network/state")
        assert resp.status_code == 409

    def test_potential_no_sim_returns_409(self, client: TestClient) -> None:
        resp = client.get("/api/v1/potential")
        assert resp.status_code == 409


# ======================================================================
# Rate limiting
# ======================================================================


class TestRateLimiting:
    def test_rate_limiter_allows_normal_traffic(self, client: TestClient) -> None:
        """Under the limit, requests should succeed."""
        from gravtraffic.api.app import _rate_limiter

        # Reset the limiter state
        _rate_limiter._requests.clear()

        client.post("/api/v1/simulate", json=DEFAULT_CONFIG)
        for _ in range(5):
            resp = client.post("/api/v1/step")
            assert resp.status_code == 200

    def test_rate_limiter_blocks_excess_traffic(self) -> None:
        """Exceeding the limit should return 429."""
        from gravtraffic.api.app import _RateLimiter, _check_rate_limit

        limiter = _RateLimiter(max_requests=3, window_s=60.0)
        # Monkey-patch the module-level limiter temporarily
        import gravtraffic.api.app as app_module

        original = app_module._rate_limiter
        app_module._rate_limiter = limiter
        try:
            for _ in range(3):
                assert limiter.check("test_key")
            # Fourth should fail
            assert not limiter.check("test_key")
        finally:
            app_module._rate_limiter = original

    def test_rate_limiter_returns_429_on_excess(self, client: TestClient) -> None:
        """Exceeding rate limit should return HTTP 429 on a real endpoint."""
        import gravtraffic.api.app as app_module
        from gravtraffic.api.app import _RateLimiter

        original = app_module._rate_limiter
        app_module._rate_limiter = _RateLimiter(max_requests=2, window_s=60.0)
        try:
            client.post("/api/v1/simulate", json=DEFAULT_CONFIG)
            # First 2 requests should succeed
            resp1 = client.post("/api/v1/step")
            assert resp1.status_code == 200
            resp2 = client.post("/api/v1/step")
            assert resp2.status_code == 200
            # Third should be rate-limited
            resp3 = client.post("/api/v1/step")
            assert resp3.status_code == 429
        finally:
            app_module._rate_limiter = original

    def test_rate_limiter_prunes_empty_keys(self) -> None:
        """cleanup() should remove keys with no recent requests."""
        from gravtraffic.api.app import _RateLimiter

        limiter = _RateLimiter(max_requests=10, window_s=0.001)
        limiter.check("ephemeral_ip")
        import time
        time.sleep(0.01)  # let the window expire
        limiter.cleanup()
        assert "ephemeral_ip" not in limiter._requests
