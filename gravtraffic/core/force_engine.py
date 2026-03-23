"""ForceEngine — Naive O(N^2) gravitational social force computation.

Reference implementation for the Janus traffic model (C-01, section 1.2).
This module computes pairwise gravitational social forces between all vehicle
pairs using a direct summation approach. It is intentionally O(N^2) to serve
as the ground-truth baseline for validating the Barnes-Hut O(N log N)
approximation developed later.

Force formula:
    F_ij = G_s * m_i * m_j / (d_ij^2 + epsilon^2)

Sign convention (Janus model section 1.2):
    - m_i * m_j > 0 (same sign)     -> attraction (force pulls i toward j)
    - m_i * m_j < 0 (opposite sign) -> repulsion  (force pushes i away from j)

Vector implementation:
    F_vec = +G_s * m_i * m_j / d^3 * (dx, dy)

where d = sqrt(dx^2 + dy^2 + epsilon^2) and dx = x_j - x_i, dy = y_j - y_i.
Same-sign masses (m_i*m_j > 0) produce a force toward j (attraction);
opposite-sign masses (m_i*m_j < 0) produce a force away from j (repulsion).

Author: Agent #01 Python Scientific Developer
Date: 2026-03-22
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt


class ForceEngine:
    """Naive O(N^2) gravitational social force engine.

    Parameters
    ----------
    G_s : float
        Social gravitational constant. Default 5.0 (unified calibration).
    softening : float
        Softening length epsilon in meters. Prevents singularity at d=0.
        Default 10.0 m per Janus model specification.

    Attributes
    ----------
    G_s : float
        Social gravitational constant.
    epsilon : float
        Softening length (meters).

    Notes
    -----
    All internal arithmetic uses float64 exclusively, as required by the
    Janus Civil numerical accuracy standard (relative error < 1e-12 on
    analytical test cases).
    """

    __slots__ = ("G_s", "epsilon")

    def __init__(self, G_s: float = 5.0, softening: float = 10.0) -> None:
        self.G_s: float = float(G_s)
        self.epsilon: float = float(softening)

    # ------------------------------------------------------------------
    # Single-pair force
    # ------------------------------------------------------------------
    def force_pair(
        self, m_i: float, m_j: float, dx: float, dy: float
    ) -> tuple[float, float]:
        """Compute the force on particle *i* due to particle *j*.

        Parameters
        ----------
        m_i : float
            Signed mass of particle i.
        m_j : float
            Signed mass of particle j.
        dx : float
            x_j - x_i (displacement from i to j along x).
        dy : float
            y_j - y_i (displacement from i to j along y).

        Returns
        -------
        tuple[float, float]
            (fx, fy) force components acting on particle i.

        Notes
        -----
        The softened distance is::

            d = sqrt(dx^2 + dy^2 + epsilon^2)

        The force vector on particle i due to particle j is::

            F_vec = +G_s * m_i * m_j / d^3 * (dx, dy)

        where ``(dx, dy) = (x_j - x_i, y_j - y_i)`` points from i to j.

        Sign analysis:

        - **Same-sign masses** (m_i * m_j > 0): the coefficient
          ``+G_s * m_i * m_j`` is positive, so the force points along
          (dx, dy), i.e., from i toward j = **attraction**.
        - **Opposite-sign masses** (m_i * m_j < 0): the coefficient is
          negative, so the force points opposite to (dx, dy) = **repulsion**.

        This formula is implemented literally as specified in the task
        definition.  The test suite validates directional correctness
        against known analytical cases.
        """
        eps2: float = self.epsilon * self.epsilon
        d2: float = dx * dx + dy * dy + eps2
        d: float = math.sqrt(d2)
        d3: float = d2 * d  # d^3 = d^2 * d, avoids extra sqrt

        coeff: float = self.G_s * m_i * m_j / d3

        return (coeff * dx, coeff * dy)

    # ------------------------------------------------------------------
    # Full N-body naive summation
    # ------------------------------------------------------------------
    def compute_all_naive(
        self,
        positions: npt.NDArray[np.float64],
        masses: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Compute total gravitational social force on every particle.

        This is the O(N^2) reference implementation. It uses an explicit
        double loop over all distinct pairs (i, j) with i != j.  Newton's
        third law is exploited: F_ji = -F_ij, so each pair is visited once.

        Parameters
        ----------
        positions : ndarray, shape (N, 2), dtype float64
            (x, y) coordinates of each particle.
        masses : ndarray, shape (N,), dtype float64
            Signed mass of each particle.

        Returns
        -------
        ndarray, shape (N, 2), dtype float64
            Total (fx, fy) force on each particle.

        Raises
        ------
        ValueError
            If positions and masses have incompatible shapes.
        """
        positions = np.asarray(positions, dtype=np.float64)
        masses = np.asarray(masses, dtype=np.float64)

        n: int = len(masses)
        if positions.shape != (n, 2):
            raise ValueError(
                f"positions shape {positions.shape} incompatible with "
                f"{n} masses; expected ({n}, 2)"
            )

        forces: npt.NDArray[np.float64] = np.zeros((n, 2), dtype=np.float64)

        eps2: float = self.epsilon * self.epsilon
        G_s: float = self.G_s

        for i in range(n):
            xi: float = positions[i, 0]
            yi: float = positions[i, 1]
            mi: float = masses[i]
            for j in range(i + 1, n):
                dx: float = positions[j, 0] - xi
                dy: float = positions[j, 1] - yi
                mj: float = masses[j]

                d2: float = dx * dx + dy * dy + eps2
                d: float = math.sqrt(d2)
                d3: float = d2 * d

                coeff: float = G_s * mi * mj / d3

                fx: float = coeff * dx
                fy: float = coeff * dy

                # Newton's third law: F_ji = -F_ij
                forces[i, 0] += fx
                forces[i, 1] += fy
                forces[j, 0] -= fx
                forces[j, 1] -= fy

        return forces

    # ------------------------------------------------------------------
    # Barnes-Hut O(N log N) force computation
    # ------------------------------------------------------------------
    def compute_all(
        self,
        positions: npt.NDArray[np.float64],
        masses: npt.NDArray[np.float64],
        theta: float = 0.5,
    ) -> npt.NDArray[np.float64]:
        """Compute total gravitational social force using Barnes-Hut.

        Builds a QuadTree, inserts all particles, then computes the
        approximate force on each particle using the opening-angle
        criterion *theta*.  Complexity is O(N log N) for typical
        particle distributions.

        Parameters
        ----------
        positions : ndarray, shape (N, 2), dtype float64
            (x, y) coordinates of each particle.
        masses : ndarray, shape (N,), dtype float64
            Signed mass of each particle.
        theta : float
            Barnes-Hut opening angle.  0 = exact (matches naive), 0.5 =
            default (< 1% error), up to ~1.0 for faster/rougher results.

        Returns
        -------
        ndarray, shape (N, 2), dtype float64
            Total (fx, fy) force on each particle.

        Raises
        ------
        ValueError
            If positions and masses have incompatible shapes.
        """
        from gravtraffic.core.quadtree import QuadTree

        positions = np.asarray(positions, dtype=np.float64)
        masses = np.asarray(masses, dtype=np.float64)

        n: int = len(masses)
        if positions.shape != (n, 2):
            raise ValueError(
                f"positions shape {positions.shape} incompatible with "
                f"{n} masses; expected ({n}, 2)"
            )

        if n == 0:
            return np.zeros((0, 2), dtype=np.float64)

        # Determine a *square* bounding box with a small margin.
        margin = 1.0
        x_min = float(positions[:, 0].min()) - margin
        y_min = float(positions[:, 1].min()) - margin
        x_max = float(positions[:, 0].max()) + margin
        y_max = float(positions[:, 1].max()) + margin

        # Make it square by expanding the shorter dimension.
        side = max(x_max - x_min, y_max - y_min)
        cx = (x_min + x_max) * 0.5
        cy = (y_min + y_max) * 0.5
        x_min = cx - side * 0.5
        x_max = cx + side * 0.5
        y_min = cy - side * 0.5
        y_max = cy + side * 0.5

        bbox = (x_min, y_min, x_max, y_max)

        # Dual-tree strategy for Janus signed masses.
        #
        # Standard Barnes-Hut with a single tree degrades badly when
        # positive and negative masses coexist because the signed-mass
        # center-of-mass can lie far from the actual particles, and net
        # mass cancellation makes the monopole approximation inaccurate.
        #
        # Instead, we build two separate trees: one for positive-mass
        # particles and one for negative-mass particles.  Each tree
        # contains only same-sign masses, so the standard monopole
        # approximation is well-behaved.  The total force on each
        # particle is the sum of forces from both trees.
        pos_mask = masses >= 0.0
        neg_mask = ~pos_mask

        pos_indices = np.where(pos_mask)[0]
        neg_indices = np.where(neg_mask)[0]

        tree_pos: QuadTree | None = None
        tree_neg: QuadTree | None = None

        # Leaf capacity of 8 balances accuracy and performance.  With
        # capacity=1, the tree is very deep and many borderline cells get
        # approximated; capacity=8 reduces tree depth and ensures small
        # clusters are computed exactly, giving <1% error at theta=0.5.
        _capacity = 8

        if len(pos_indices) > 0:
            tree_pos = QuadTree(bbox, capacity=_capacity)
            for i in pos_indices:
                ii = int(i)
                tree_pos.insert(
                    ii,
                    float(positions[ii, 0]),
                    float(positions[ii, 1]),
                    float(masses[ii]),
                )

        if len(neg_indices) > 0:
            tree_neg = QuadTree(bbox, capacity=_capacity)
            for i in neg_indices:
                ii = int(i)
                tree_neg.insert(
                    ii,
                    float(positions[ii, 0]),
                    float(positions[ii, 1]),
                    float(masses[ii]),
                )

        # Compute forces.
        forces: npt.NDArray[np.float64] = np.zeros((n, 2), dtype=np.float64)
        G_s = self.G_s
        eps = self.epsilon

        for i in range(n):
            px = float(positions[i, 0])
            py = float(positions[i, 1])
            mi = float(masses[i])
            fx_total = 0.0
            fy_total = 0.0

            if tree_pos is not None:
                fx, fy = tree_pos.compute_force(
                    px, py, mi, i, G_s, eps, theta
                )
                fx_total += fx
                fy_total += fy

            if tree_neg is not None:
                fx, fy = tree_neg.compute_force(
                    px, py, mi, i, G_s, eps, theta
                )
                fx_total += fx
                fy_total += fy

            forces[i, 0] = fx_total
            forces[i, 1] = fy_total

        return forces
