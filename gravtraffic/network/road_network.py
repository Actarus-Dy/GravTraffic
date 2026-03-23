# SPDX-License-Identifier: MIT
"""Road network representation for GravTraffic simulation.

Supports loading from OpenStreetMap via OSMnx (optional dependency) or from
synthetic grid layouts for deterministic testing without network access.

All coordinates are in projected metres (UTM or equivalent). All floating-point
storage uses float64 to satisfy the Janus numerical-precision mandate.

Mathematical reference
----------------------
Point-to-segment projection uses the standard parametric formula:

    t = clamp(dot(P - A, B - A) / dot(B - A, B - A), 0, 1)
    proj = A + t * (B - A)

where A, B are segment endpoints and P is the query point.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class RoadNetwork:
    """Road network representation for GravTraffic simulation.

    Parameters
    ----------
    nodes : dict[Any, dict]
        Mapping ``{node_id: {'x': float, 'y': float}}`` in projected
        coordinates (metres).
    edges : list[dict]
        Each dict has keys ``'u'``, ``'v'`` (node ids), ``'length'`` (metres),
        ``'speed_limit'`` (m/s), ``'lanes'`` (int), and optionally
        ``'geometry'`` as a list of ``(x, y)`` tuples.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, nodes: dict[Any, dict], edges: list[dict]) -> None:
        # Defensive copies so callers cannot mutate internals.
        self._nodes: dict[Any, dict] = {
            nid: {"x": np.float64(data["x"]), "y": np.float64(data["y"])}
            for nid, data in nodes.items()
        }

        self._edges: list[dict] = []
        for i, e in enumerate(edges):
            rec: dict[str, Any] = {
                "edge_id": i,
                "u": e["u"],
                "v": e["v"],
                "length": np.float64(e["length"]),
                "speed_limit": np.float64(e.get("speed_limit", 13.9)),
                "lanes": int(e.get("lanes", 2)),
            }
            # Build geometry from nodes if not supplied explicitly.
            if "geometry" in e and e["geometry"]:
                rec["geometry"] = [(np.float64(x), np.float64(y)) for x, y in e["geometry"]]
            else:
                nu = self._nodes[e["u"]]
                nv = self._nodes[e["v"]]
                rec["geometry"] = [
                    (nu["x"], nu["y"]),
                    (nv["x"], nv["y"]),
                ]
            self._edges.append(rec)

        # Pre-compute degree per node (number of incident edges).
        self._degree: dict[Any, int] = {nid: 0 for nid in self._nodes}
        for e in self._edges:
            self._degree[e["u"]] = self._degree.get(e["u"], 0) + 1
            self._degree[e["v"]] = self._degree.get(e["v"], 0) + 1

    # ------------------------------------------------------------------
    # Class-method constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_osmnx(
        cls,
        place: str | None = None,
        bbox: tuple[float, float, float, float] | None = None,
        network_type: str = "drive",
    ) -> RoadNetwork:
        """Load a road network from OpenStreetMap via *osmnx*.

        Parameters
        ----------
        place : str, optional
            Geocodable place name (e.g. ``"Piedmont, California, USA"``).
        bbox : tuple, optional
            ``(north, south, east, west)`` in decimal degrees.
        network_type : str
            OSMnx network type -- ``'drive'``, ``'walk'``, ``'bike'``, etc.

        Returns
        -------
        RoadNetwork
            Network projected to the appropriate UTM zone.

        Raises
        ------
        ImportError
            If *osmnx* is not installed.
        ValueError
            If neither *place* nor *bbox* is provided.
        """
        try:
            import osmnx as ox  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "osmnx is required for from_osmnx(). Install it with: pip install osmnx"
            ) from exc

        if place is not None:
            G = ox.graph_from_place(place, network_type=network_type)
        elif bbox is not None:
            north, south, east, west = bbox
            G = ox.graph_from_bbox(north, south, east, west, network_type=network_type)
        else:
            raise ValueError("Provide either 'place' or 'bbox'.")

        G = ox.project_graph(G)

        nodes: dict[Any, dict] = {}
        for nid, data in G.nodes(data=True):
            nodes[nid] = {"x": float(data["x"]), "y": float(data["y"])}

        edges: list[dict] = []
        for u, v, data in G.edges(data=True):
            speed_raw = data.get("maxspeed", "50")
            if isinstance(speed_raw, list):
                speed_raw = speed_raw[0]
            try:
                speed_kmh = float(speed_raw)
            except (ValueError, TypeError):
                speed_kmh = 50.0
            speed_ms = speed_kmh / 3.6

            lanes = data.get("lanes", 2)
            if isinstance(lanes, list):
                lanes = lanes[0]
            try:
                lanes = int(lanes)
            except (ValueError, TypeError):
                lanes = 2

            length = float(data.get("length", 0.0))

            geom = None
            if "geometry" in data:
                geom = list(data["geometry"].coords)

            edges.append(
                {
                    "u": u,
                    "v": v,
                    "length": length,
                    "speed_limit": speed_ms,
                    "lanes": lanes,
                    "geometry": geom,
                }
            )

        return cls(nodes, edges)

    @classmethod
    def from_grid(
        cls,
        rows: int = 5,
        cols: int = 5,
        block_size: float = 200.0,
    ) -> RoadNetwork:
        """Create a synthetic rectangular grid network for testing.

        Parameters
        ----------
        rows : int
            Number of intersection rows.
        cols : int
            Number of intersection columns.
        block_size : float
            Distance in metres between adjacent intersections.

        Returns
        -------
        RoadNetwork
            A ``rows * cols`` intersection grid. All roads have
            ``speed_limit = 13.9 m/s`` (~50 km/h) and ``lanes = 2``.
        """
        block_size = np.float64(block_size)

        nodes: dict[int, dict] = {}
        for r in range(rows):
            for c in range(cols):
                nid = r * cols + c
                nodes[nid] = {
                    "x": np.float64(c) * block_size,
                    "y": np.float64(r) * block_size,
                }

        edges: list[dict] = []
        speed_limit = np.float64(13.9)  # 50 km/h in m/s

        for r in range(rows):
            for c in range(cols):
                nid = r * cols + c
                # Horizontal edge to the right neighbour.
                if c + 1 < cols:
                    edges.append(
                        {
                            "u": nid,
                            "v": nid + 1,
                            "length": block_size,
                            "speed_limit": speed_limit,
                            "lanes": 2,
                        }
                    )
                # Vertical edge to the neighbour below.
                if r + 1 < rows:
                    edges.append(
                        {
                            "u": nid,
                            "v": (r + 1) * cols + c,
                            "length": block_size,
                            "speed_limit": speed_limit,
                            "lanes": 2,
                        }
                    )

        return cls(nodes, edges)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def intersections(self) -> list[dict]:
        """Return intersection metadata.

        Returns
        -------
        list[dict]
            Each dict: ``{'node_id', 'x', 'y', 'degree'}``.
        """
        return [
            {
                "node_id": nid,
                "x": data["x"],
                "y": data["y"],
                "degree": self._degree.get(nid, 0),
            }
            for nid, data in self._nodes.items()
        ]

    @property
    def segments(self) -> list[dict]:
        """Return road-segment metadata.

        Returns
        -------
        list[dict]
            Each dict: ``{'edge_id', 'u', 'v', 'length', 'speed_limit', 'lanes'}``.
        """
        return [
            {
                "edge_id": e["edge_id"],
                "u": e["u"],
                "v": e["v"],
                "length": e["length"],
                "speed_limit": e["speed_limit"],
                "lanes": e["lanes"],
            }
            for e in self._edges
        ]

    @property
    def node_count(self) -> int:
        """Number of nodes (intersections) in the network."""
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        """Number of edges (road segments) in the network."""
        return len(self._edges)

    # ------------------------------------------------------------------
    # Sampling & queries
    # ------------------------------------------------------------------

    def sample_positions(
        self,
        n: int,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Sample *n* random positions lying on the network.

        Each position is placed at a uniformly random fraction along a
        uniformly random edge, weighted by edge length.

        Parameters
        ----------
        n : int
            Number of positions to sample.
        rng : numpy.random.Generator, optional
            PRNG for reproducibility.  Falls back to ``default_rng()``.

        Returns
        -------
        numpy.ndarray
            Shape ``(n, 2)``, dtype ``float64``.  Columns are ``(x, y)``
            in projected metres.
        """
        if n <= 0:
            return np.empty((0, 2), dtype=np.float64)

        if not self._edges:
            raise ValueError("Cannot sample positions: network has no edges.")

        if rng is None:
            rng = np.random.default_rng()

        # Length-weighted edge selection.
        lengths = np.array([e["length"] for e in self._edges], dtype=np.float64)
        probs = lengths / lengths.sum()

        edge_indices = rng.choice(len(self._edges), size=n, p=probs)
        fractions = rng.uniform(0.0, 1.0, size=n)

        positions = np.empty((n, 2), dtype=np.float64)
        for i, (eidx, t) in enumerate(zip(edge_indices, fractions)):
            geom = self._edges[eidx]["geometry"]
            # For multi-segment geometry, pick the sub-segment by cumulative
            # length then interpolate within it.
            if len(geom) == 2:
                ax, ay = geom[0]
                bx, by = geom[1]
                positions[i, 0] = ax + t * (bx - ax)
                positions[i, 1] = ay + t * (by - ay)
            else:
                positions[i] = self._interpolate_geometry(geom, t)

        return positions

    # ------------------------------------------------------------------

    def nearest_edge(self, x: float, y: float) -> tuple[int, float, float]:
        """Find the nearest edge to a query point.

        Parameters
        ----------
        x, y : float
            Query coordinates in projected metres.

        Returns
        -------
        tuple[int, float, float]
            ``(edge_id, proj_x, proj_y)`` -- the edge index and the
            projected point on that edge closest to ``(x, y)``.

        Raises
        ------
        ValueError
            If the network has no edges.
        """
        if not self._edges:
            raise ValueError("Network has no edges.")

        px, py = np.float64(x), np.float64(y)
        best_dist_sq = np.inf
        best_edge_id = 0
        best_proj = (px, py)

        for e in self._edges:
            for k in range(len(e["geometry"]) - 1):
                ax, ay = e["geometry"][k]
                bx, by = e["geometry"][k + 1]
                proj_x, proj_y = self._project_point_on_segment(px, py, ax, ay, bx, by)
                dsq = (proj_x - px) ** 2 + (proj_y - py) ** 2
                if dsq < best_dist_sq:
                    best_dist_sq = dsq
                    best_edge_id = e["edge_id"]
                    best_proj = (proj_x, proj_y)

        return best_edge_id, best_proj[0], best_proj[1]

    # ------------------------------------------------------------------

    def get_speed_limit(self, edge_id: int) -> float:
        """Speed limit in m/s for the given edge.

        Parameters
        ----------
        edge_id : int
            Index into the internal edge list.

        Returns
        -------
        float
            Speed limit in metres per second (float64).

        Raises
        ------
        IndexError
            If *edge_id* is out of range.
        """
        if edge_id < 0 or edge_id >= len(self._edges):
            raise IndexError(f"edge_id {edge_id} out of range [0, {len(self._edges)}).")
        return float(self._edges[edge_id]["speed_limit"])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _project_point_on_segment(
        px: float,
        py: float,
        ax: float,
        ay: float,
        bx: float,
        by: float,
    ) -> tuple[float, float]:
        """Project point P onto segment AB, clamped to [A, B].

        Returns the projected (x, y) as float64 values.
        """
        dx, dy = bx - ax, by - ay
        len_sq = dx * dx + dy * dy
        if len_sq == 0.0:
            return (ax, ay)
        t = ((px - ax) * dx + (py - ay) * dy) / len_sq
        t = max(0.0, min(1.0, t))
        return (ax + t * dx, ay + t * dy)

    @staticmethod
    def _interpolate_geometry(
        geom: list[tuple[float, float]], fraction: float
    ) -> tuple[float, float]:
        """Interpolate a position at *fraction* along a polyline geometry.

        Parameters
        ----------
        geom : list of (x, y)
            Ordered vertices of the polyline.
        fraction : float
            Value in [0, 1] along the total polyline length.

        Returns
        -------
        tuple[float, float]
        """
        # Compute cumulative segment lengths.
        seg_lengths: list[float] = []
        for k in range(len(geom) - 1):
            dx = geom[k + 1][0] - geom[k][0]
            dy = geom[k + 1][1] - geom[k][1]
            seg_lengths.append(np.sqrt(dx * dx + dy * dy))

        total = sum(seg_lengths)
        if total == 0.0:
            return geom[0]

        target = fraction * total
        cumul = 0.0
        for k, sl in enumerate(seg_lengths):
            if cumul + sl >= target or k == len(seg_lengths) - 1:
                local_t = (target - cumul) / sl if sl > 0.0 else 0.0
                ax, ay = geom[k]
                bx, by = geom[k + 1]
                return (ax + local_t * (bx - ax), ay + local_t * (by - ay))
            cumul += sl

        # Fallback (should not reach here).
        return geom[-1]  # pragma: no cover
