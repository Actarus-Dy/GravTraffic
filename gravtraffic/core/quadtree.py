"""Barnes-Hut QuadTree for O(N log N) gravitational force computation.

Implements the 2-D Barnes-Hut algorithm for the Janus traffic model (C-01).
The tree adaptively subdivides space into quadrants and approximates distant
particle groups by their center of mass, reducing force computation from
O(N^2) to O(N log N).

The opening angle parameter *theta* controls the accuracy-speed trade-off:
    - theta = 0   -> exact (equivalent to direct O(N^2) summation)
    - theta = 0.5 -> default, < 1% relative error vs. direct summation
    - theta = 1.0 -> fast but less accurate

For Janus signed masses, the ``ForceEngine.compute_all`` method builds
two separate trees (one for positive masses, one for negative masses) so
that each tree contains only same-sign particles.  This avoids the
catastrophic accuracy loss from signed-mass cancellation in a single
monopole approximation.

Force formula and sign convention match ``force_engine.py`` exactly::

    F_vec = +G_s * m_i * m_j / d^3 * (dx, dy)

where d = sqrt(dx^2 + dy^2 + epsilon^2).

Author: Agent #16 N-body Simulation Expert
Date: 2026-03-22
"""

from __future__ import annotations

import math

# Maximum tree depth to prevent infinite recursion on coincident particles.
_MAX_DEPTH: int = 64


class QuadTreeNode:
    """A single node in the Barnes-Hut quadtree.

    Each node represents a square region of 2-D space.  A leaf node holds
    at most ``capacity`` particles.  When a leaf exceeds capacity it is
    subdivided into four children (NW, NE, SW, SE), unless the maximum
    tree depth has been reached, in which case particles accumulate.

    Attributes
    ----------
    x_min, y_min, x_max, y_max : float
        Bounding box of this node.
    capacity : int
        Maximum particles in a leaf before subdivision.
    depth : int
        Depth of this node in the tree (root = 0).
    total_mass : float
        Sum of masses of all contained particles.
    com_x, com_y : float
        Center-of-mass coordinates.
    count : int
        Number of particles in this subtree.
    children : list[QuadTreeNode | None]
        [NW, NE, SW, SE] children, or None entries if not subdivided.
    is_leaf : bool
        True if the node has not been subdivided.
    indices : list[int]
        Particle indices stored in this leaf (empty for internal nodes).
    px, py, pm : list[float]
        Positions and masses of particles in this leaf.
    """

    __slots__ = (
        "x_min",
        "y_min",
        "x_max",
        "y_max",
        "capacity",
        "depth",
        "total_mass",
        "com_x",
        "com_y",
        "count",
        "children",
        "is_leaf",
        "indices",
        "px",
        "py",
        "pm",
    )

    def __init__(
        self,
        x_min: float,
        y_min: float,
        x_max: float,
        y_max: float,
        capacity: int = 1,
        depth: int = 0,
    ) -> None:
        self.x_min: float = x_min
        self.y_min: float = y_min
        self.x_max: float = x_max
        self.y_max: float = y_max
        self.capacity: int = capacity
        self.depth: int = depth

        self.total_mass: float = 0.0
        self.com_x: float = 0.0
        self.com_y: float = 0.0
        self.count: int = 0

        self.children: list[QuadTreeNode | None] = [None, None, None, None]
        self.is_leaf: bool = True

        # Leaf particle storage
        self.indices: list[int] = []
        self.px: list[float] = []
        self.py: list[float] = []
        self.pm: list[float] = []

    # ------------------------------------------------------------------
    # Insertion
    # ------------------------------------------------------------------
    def insert(self, index: int, x: float, y: float, mass: float) -> None:
        """Insert a particle into the tree.

        Parameters
        ----------
        index : int
            Global particle index (used to skip self-force).
        x, y : float
            Particle position.
        mass : float
            Signed particle mass.
        """
        # Update center-of-mass incrementally.
        old_total = self.total_mass
        new_total = old_total + mass
        if abs(new_total) > 1e-30:
            self.com_x = (old_total * self.com_x + mass * x) / new_total
            self.com_y = (old_total * self.com_y + mass * y) / new_total
        else:
            # Degenerate case: net mass is ~zero.  Use geometric mean.
            n = self.count + 1
            self.com_x = (self.com_x * self.count + x) / n
            self.com_y = (self.com_y * self.count + y) / n
        self.total_mass = new_total
        self.count += 1

        if self.is_leaf:
            self.indices.append(index)
            self.px.append(x)
            self.py.append(y)
            self.pm.append(mass)

            if len(self.indices) > self.capacity and self.depth < _MAX_DEPTH:
                self._subdivide()
        else:
            # Route to appropriate child.
            self._insert_into_child(index, x, y, mass)

    def _subdivide(self) -> None:
        """Split this leaf into four children and redistribute particles."""
        mx = (self.x_min + self.x_max) * 0.5
        my = (self.y_min + self.y_max) * 0.5
        cap = self.capacity
        d = self.depth + 1

        # NW, NE, SW, SE
        self.children[0] = QuadTreeNode(self.x_min, my, mx, self.y_max, cap, d)
        self.children[1] = QuadTreeNode(mx, my, self.x_max, self.y_max, cap, d)
        self.children[2] = QuadTreeNode(self.x_min, self.y_min, mx, my, cap, d)
        self.children[3] = QuadTreeNode(mx, self.y_min, self.x_max, my, cap, d)

        self.is_leaf = False

        # Redistribute existing particles.
        for i in range(len(self.indices)):
            self._insert_into_child(self.indices[i], self.px[i], self.py[i], self.pm[i])

        # Clear leaf storage.
        self.indices = []
        self.px = []
        self.py = []
        self.pm = []

    def _insert_into_child(self, index: int, x: float, y: float, mass: float) -> None:
        """Route a particle to the correct child quadrant."""
        mx = (self.x_min + self.x_max) * 0.5
        my = (self.y_min + self.y_max) * 0.5

        if y >= my:
            if x < mx:
                self.children[0].insert(index, x, y, mass)  # NW
            else:
                self.children[1].insert(index, x, y, mass)  # NE
        else:
            if x < mx:
                self.children[2].insert(index, x, y, mass)  # SW
            else:
                self.children[3].insert(index, x, y, mass)  # SE

    # ------------------------------------------------------------------
    # Force computation
    # ------------------------------------------------------------------
    def compute_force(
        self,
        x: float,
        y: float,
        mass: float,
        particle_index: int,
        G_s: float,
        softening: float,
        theta: float = 0.5,
    ) -> tuple[float, float]:
        """Compute the gravitational force on a query particle from this node.

        The opening-angle criterion uses the geometric (unsoftened)
        distance between the query particle and the node's center-of-mass
        to decide whether the node can be treated as a single body.
        Softening is applied only in the actual force computation.

        Parameters
        ----------
        x, y : float
            Position of the query particle.
        mass : float
            Signed mass of the query particle.
        particle_index : int
            Index of the query particle (to skip self-interaction).
        G_s : float
            Social gravitational constant.
        softening : float
            Softening length epsilon (meters).
        theta : float
            Barnes-Hut opening angle.  0 = exact, 0.5 = default.

        Returns
        -------
        tuple[float, float]
            (fx, fy) force on the query particle from all particles in
            this node's subtree.
        """
        if self.count == 0:
            return (0.0, 0.0)

        eps2 = softening * softening

        # Leaf node: compute direct forces from each stored particle.
        if self.is_leaf:
            fx_total = 0.0
            fy_total = 0.0
            for k in range(len(self.indices)):
                if self.indices[k] == particle_index:
                    continue  # skip self-force
                dx = self.px[k] - x
                dy = self.py[k] - y
                d2 = dx * dx + dy * dy + eps2
                d = math.sqrt(d2)
                d3 = d2 * d
                coeff = G_s * mass * self.pm[k] / d3
                fx_total += coeff * dx
                fy_total += coeff * dy
            return (fx_total, fy_total)

        # Internal node: opening-angle test.
        # First, if the query particle lies inside this cell, always recurse
        # (never use the COM approximation for a cell containing the query).
        if self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max:
            # Query is inside this cell -- must recurse.
            fx_total = 0.0
            fy_total = 0.0
            for child in self.children:
                if child is not None and child.count > 0:
                    cfx, cfy = child.compute_force(
                        x, y, mass, particle_index, G_s, softening, theta
                    )
                    fx_total += cfx
                    fy_total += cfy
            return (fx_total, fy_total)

        # Query is outside this cell.  Compute the distance from the
        # query to the *nearest edge* of the bounding box (Bmax criterion).
        # This is more conservative than using the COM distance and ensures
        # that the opening-angle test correctly accounts for cells whose COM
        # is far from the geometric center.
        # Use the cell diagonal as the effective size.  For a cell of
        # dimensions (w, h), the maximum distance from the cell center to
        # any corner is sqrt(w^2 + h^2) / 2.  Using the full diagonal as
        # the "size" is conservative and accounts for the spatial extent
        # of particles within the cell.
        w = self.x_max - self.x_min
        h = self.y_max - self.y_min
        diag = math.sqrt(w * w + h * h)

        # Distance from query to nearest point on the cell boundary.
        clamp_x = max(self.x_min, min(x, self.x_max))
        clamp_y = max(self.y_min, min(y, self.y_max))
        edge_dx = x - clamp_x
        edge_dy = y - clamp_y
        r_near = math.sqrt(edge_dx * edge_dx + edge_dy * edge_dy)

        dx = self.com_x - x
        dy = self.com_y - y

        if theta > 0.0 and r_near > 0.0 and (diag / r_near) < theta:
            # Use center-of-mass approximation with softened distance.
            d2 = dx * dx + dy * dy + eps2
            d = math.sqrt(d2)
            d3 = d2 * d
            coeff = G_s * mass * self.total_mass / d3
            return (coeff * dx, coeff * dy)

        # Otherwise, recurse into children.
        fx_total = 0.0
        fy_total = 0.0
        for child in self.children:
            if child is not None and child.count > 0:
                cfx, cfy = child.compute_force(x, y, mass, particle_index, G_s, softening, theta)
                fx_total += cfx
                fy_total += cfy
        return (fx_total, fy_total)


class QuadTree:
    """Barnes-Hut QuadTree for 2-D gravitational force computation.

    Parameters
    ----------
    bbox : tuple[float, float, float, float]
        Bounding box (x_min, y_min, x_max, y_max) for the entire domain.
    capacity : int
        Maximum particles per leaf node before subdivision.  Default 1.
    """

    __slots__ = ("root", "capacity")

    def __init__(
        self,
        bbox: tuple[float, float, float, float],
        capacity: int = 1,
    ) -> None:
        x_min, y_min, x_max, y_max = bbox
        self.capacity: int = capacity
        self.root: QuadTreeNode = QuadTreeNode(x_min, y_min, x_max, y_max, capacity, depth=0)

    def insert(self, index: int, x: float, y: float, mass: float) -> None:
        """Insert a particle into the tree.

        Parameters
        ----------
        index : int
            Unique particle index.
        x, y : float
            Particle position.
        mass : float
            Signed particle mass.
        """
        self.root.insert(index, x, y, mass)

    def compute_force(
        self,
        x: float,
        y: float,
        mass: float,
        particle_index: int,
        G_s: float,
        softening: float,
        theta: float = 0.5,
    ) -> tuple[float, float]:
        """Compute force on a particle from all other particles in the tree.

        Parameters
        ----------
        x, y : float
            Query particle position.
        mass : float
            Query particle signed mass.
        particle_index : int
            Query particle index (to skip self-interaction).
        G_s : float
            Social gravitational constant.
        softening : float
            Softening length epsilon.
        theta : float
            Opening angle.

        Returns
        -------
        tuple[float, float]
            (fx, fy) total force on the query particle.
        """
        return self.root.compute_force(x, y, mass, particle_index, G_s, softening, theta)
