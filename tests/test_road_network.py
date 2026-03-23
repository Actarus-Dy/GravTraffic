# SPDX-License-Identifier: MIT
"""Unit tests for gravtraffic.network.road_network.RoadNetwork.

All tests use ``from_grid`` so they run without network access or OSMnx.
"""

from __future__ import annotations

import numpy as np
import pytest

from gravtraffic.network.road_network import RoadNetwork

# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def grid_3x3() -> RoadNetwork:
    """A 3-row, 3-column grid with 200 m block size."""
    return RoadNetwork.from_grid(rows=3, cols=3, block_size=200.0)


@pytest.fixture
def grid_5x5() -> RoadNetwork:
    """A 5-row, 5-column grid with default 200 m block size."""
    return RoadNetwork.from_grid(rows=5, cols=5, block_size=200.0)


@pytest.fixture
def grid_1x1() -> RoadNetwork:
    """Edge case: single intersection, no edges."""
    return RoadNetwork.from_grid(rows=1, cols=1)


# ======================================================================
# Test 1: from_grid(3, 3) creates 9 nodes and correct number of edges
# ======================================================================


class TestFromGridTopology:
    """Verify that from_grid produces the correct graph topology."""

    def test_3x3_node_count(self, grid_3x3: RoadNetwork) -> None:
        assert grid_3x3.node_count == 9

    def test_3x3_edge_count(self, grid_3x3: RoadNetwork) -> None:
        # Horizontal edges: 3 rows * 2 = 6
        # Vertical edges:   2 rows * 3 cols = 6
        # Total = 12
        assert grid_3x3.edge_count == 12

    def test_5x5_node_count(self, grid_5x5: RoadNetwork) -> None:
        assert grid_5x5.node_count == 25

    def test_5x5_edge_count(self, grid_5x5: RoadNetwork) -> None:
        # Horizontal: 5 * 4 = 20, Vertical: 4 * 5 = 20 -> 40
        assert grid_5x5.edge_count == 40

    def test_generic_edge_formula(self) -> None:
        """For an R x C grid, edges = R*(C-1) + (R-1)*C."""
        for rows in range(1, 7):
            for cols in range(1, 7):
                net = RoadNetwork.from_grid(rows=rows, cols=cols, block_size=100.0)
                expected = rows * (cols - 1) + (rows - 1) * cols
                assert net.edge_count == expected, (
                    f"Failed for {rows}x{cols}: got {net.edge_count}, expected {expected}"
                )


# ======================================================================
# Test 2: intersections returns correct node count with coordinates
# ======================================================================


class TestIntersections:
    """Validate the intersections property."""

    def test_count(self, grid_3x3: RoadNetwork) -> None:
        assert len(grid_3x3.intersections) == 9

    def test_keys_present(self, grid_3x3: RoadNetwork) -> None:
        for node in grid_3x3.intersections:
            assert {"node_id", "x", "y", "degree"} <= set(node.keys())

    def test_coordinates_are_float64(self, grid_3x3: RoadNetwork) -> None:
        for node in grid_3x3.intersections:
            assert isinstance(node["x"], (float, np.floating))
            assert isinstance(node["y"], (float, np.floating))

    def test_coordinate_values(self) -> None:
        """Node (r=1, c=2) in a 3x3 grid with block_size=100 is at (200, 100)."""
        net = RoadNetwork.from_grid(rows=3, cols=3, block_size=100.0)
        node_map = {n["node_id"]: n for n in net.intersections}
        # nid = r*cols + c = 1*3 + 2 = 5
        assert np.isclose(node_map[5]["x"], 200.0)
        assert np.isclose(node_map[5]["y"], 100.0)

    def test_corner_degrees(self, grid_3x3: RoadNetwork) -> None:
        """Corners of a 3x3 grid have degree 2."""
        node_map = {n["node_id"]: n for n in grid_3x3.intersections}
        # Corners: (0,0)->nid 0, (0,2)->2, (2,0)->6, (2,2)->8
        for nid in [0, 2, 6, 8]:
            assert node_map[nid]["degree"] == 2, f"Corner node {nid} degree wrong"

    def test_edge_node_degrees(self, grid_3x3: RoadNetwork) -> None:
        """Edge (non-corner) nodes of 3x3 have degree 3."""
        node_map = {n["node_id"]: n for n in grid_3x3.intersections}
        for nid in [1, 3, 5, 7]:
            assert node_map[nid]["degree"] == 3, f"Edge node {nid} degree wrong"

    def test_center_degree(self, grid_3x3: RoadNetwork) -> None:
        """Centre node of 3x3 has degree 4."""
        node_map = {n["node_id"]: n for n in grid_3x3.intersections}
        assert node_map[4]["degree"] == 4


# ======================================================================
# Test 3: segments returns edges with valid lengths
# ======================================================================


class TestSegments:
    """Validate the segments property."""

    def test_count(self, grid_3x3: RoadNetwork) -> None:
        assert len(grid_3x3.segments) == 12

    def test_length_equals_block_size(self, grid_3x3: RoadNetwork) -> None:
        for seg in grid_3x3.segments:
            assert np.isclose(seg["length"], 200.0, rtol=1e-12), (
                f"Edge {seg['edge_id']} length {seg['length']} != 200.0"
            )

    def test_keys_present(self, grid_3x3: RoadNetwork) -> None:
        required = {"edge_id", "u", "v", "length", "speed_limit", "lanes"}
        for seg in grid_3x3.segments:
            assert required <= set(seg.keys())

    def test_length_dtype_float64(self, grid_3x3: RoadNetwork) -> None:
        for seg in grid_3x3.segments:
            assert isinstance(seg["length"], (float, np.float64))

    def test_unique_edge_ids(self, grid_5x5: RoadNetwork) -> None:
        ids = [s["edge_id"] for s in grid_5x5.segments]
        assert len(ids) == len(set(ids))


# ======================================================================
# Test 4: sample_positions returns (n, 2) array within network bounds
# ======================================================================


class TestSamplePositions:
    """Validate random position sampling on the network."""

    def test_shape_and_dtype(self, grid_5x5: RoadNetwork) -> None:
        rng = np.random.default_rng(42)
        pos = grid_5x5.sample_positions(100, rng=rng)
        assert pos.shape == (100, 2)
        assert pos.dtype == np.float64

    def test_within_bounds(self, grid_5x5: RoadNetwork) -> None:
        rng = np.random.default_rng(123)
        pos = grid_5x5.sample_positions(500, rng=rng)
        # 5x5 grid, block_size=200 -> x in [0, 800], y in [0, 800]
        assert np.all(pos[:, 0] >= 0.0 - 1e-12)
        assert np.all(pos[:, 0] <= 800.0 + 1e-12)
        assert np.all(pos[:, 1] >= 0.0 - 1e-12)
        assert np.all(pos[:, 1] <= 800.0 + 1e-12)

    def test_zero_samples(self, grid_3x3: RoadNetwork) -> None:
        pos = grid_3x3.sample_positions(0)
        assert pos.shape == (0, 2)
        assert pos.dtype == np.float64

    def test_reproducible_with_seed(self, grid_3x3: RoadNetwork) -> None:
        pos1 = grid_3x3.sample_positions(50, rng=np.random.default_rng(99))
        pos2 = grid_3x3.sample_positions(50, rng=np.random.default_rng(99))
        np.testing.assert_array_equal(pos1, pos2)

    def test_no_edges_raises(self, grid_1x1: RoadNetwork) -> None:
        with pytest.raises(ValueError, match="no edges"):
            grid_1x1.sample_positions(5)


# ======================================================================
# Test 5: nearest_edge returns valid edge_id and projected point
# ======================================================================


class TestNearestEdge:
    """Validate nearest-edge queries."""

    def test_point_on_node_returns_valid_edge(self, grid_3x3: RoadNetwork) -> None:
        """A point exactly on a node should project to a valid edge."""
        eid, px, py = grid_3x3.nearest_edge(0.0, 0.0)
        assert 0 <= eid < grid_3x3.edge_count
        # Projected point should be at or very near the query node.
        assert np.isclose(px, 0.0, atol=1e-10)
        assert np.isclose(py, 0.0, atol=1e-10)

    def test_point_on_midpoint(self, grid_3x3: RoadNetwork) -> None:
        """Midpoint of an edge should project to itself."""
        # Edge from node (0,0) to node (0,1) -> (0,0) to (200,0)
        eid, px, py = grid_3x3.nearest_edge(100.0, 0.0)
        assert np.isclose(px, 100.0, atol=1e-10)
        assert np.isclose(py, 0.0, atol=1e-10)

    def test_off_network_point_projects_onto_edge(self, grid_3x3: RoadNetwork) -> None:
        """A point away from the network should project to the nearest edge."""
        # Point at (100, -50) should project onto the bottom row (y=0).
        eid, px, py = grid_3x3.nearest_edge(100.0, -50.0)
        assert np.isclose(py, 0.0, atol=1e-10)
        assert 0.0 - 1e-10 <= px <= 400.0 + 1e-10

    def test_projected_dtype(self, grid_3x3: RoadNetwork) -> None:
        _, px, py = grid_3x3.nearest_edge(50.0, 50.0)
        assert isinstance(px, float)
        assert isinstance(py, float)

    def test_no_edges_raises(self, grid_1x1: RoadNetwork) -> None:
        with pytest.raises(ValueError, match="no edges"):
            grid_1x1.nearest_edge(0.0, 0.0)


# ======================================================================
# Test 6: get_speed_limit returns 13.9 m/s for grid networks
# ======================================================================


class TestGetSpeedLimit:
    """Validate speed limit queries."""

    def test_default_grid_speed(self, grid_3x3: RoadNetwork) -> None:
        for eid in range(grid_3x3.edge_count):
            assert np.isclose(grid_3x3.get_speed_limit(eid), 13.9, rtol=1e-12)

    def test_return_type(self, grid_3x3: RoadNetwork) -> None:
        val = grid_3x3.get_speed_limit(0)
        assert isinstance(val, float)

    def test_invalid_edge_id_raises(self, grid_3x3: RoadNetwork) -> None:
        with pytest.raises(IndexError):
            grid_3x3.get_speed_limit(999)

    def test_negative_edge_id_raises(self, grid_3x3: RoadNetwork) -> None:
        with pytest.raises(IndexError):
            grid_3x3.get_speed_limit(-1)


# ======================================================================
# Test 7: from_grid(1, 1) single node edge case
# ======================================================================


class TestSingleNodeGrid:
    """Edge case: a 1x1 grid has one node and zero edges."""

    def test_node_count(self, grid_1x1: RoadNetwork) -> None:
        assert grid_1x1.node_count == 1

    def test_edge_count(self, grid_1x1: RoadNetwork) -> None:
        assert grid_1x1.edge_count == 0

    def test_intersections(self, grid_1x1: RoadNetwork) -> None:
        ints = grid_1x1.intersections
        assert len(ints) == 1
        assert np.isclose(ints[0]["x"], 0.0)
        assert np.isclose(ints[0]["y"], 0.0)
        assert ints[0]["degree"] == 0

    def test_segments_empty(self, grid_1x1: RoadNetwork) -> None:
        assert grid_1x1.segments == []

    def test_speed_limit_raises(self, grid_1x1: RoadNetwork) -> None:
        with pytest.raises(IndexError):
            grid_1x1.get_speed_limit(0)


# ======================================================================
# Test 8 (bonus): manual dict construction
# ======================================================================


class TestManualConstruction:
    """Verify that constructing from raw dicts works correctly."""

    def test_simple_two_node_network(self) -> None:
        nodes = {
            "A": {"x": 0.0, "y": 0.0},
            "B": {"x": 300.0, "y": 0.0},
        }
        edges = [
            {
                "u": "A",
                "v": "B",
                "length": 300.0,
                "speed_limit": 16.7,
                "lanes": 3,
            }
        ]
        net = RoadNetwork(nodes, edges)
        assert net.node_count == 2
        assert net.edge_count == 1
        assert np.isclose(net.get_speed_limit(0), 16.7, rtol=1e-12)
        assert net.segments[0]["lanes"] == 3

    def test_string_node_ids(self) -> None:
        """Node IDs can be any hashable type."""
        nodes = {"alpha": {"x": 10.0, "y": 20.0}, "beta": {"x": 30.0, "y": 40.0}}
        edges = [{"u": "alpha", "v": "beta", "length": 28.28, "speed_limit": 10.0, "lanes": 1}]
        net = RoadNetwork(nodes, edges)
        eid, px, py = net.nearest_edge(20.0, 30.0)
        assert eid == 0
