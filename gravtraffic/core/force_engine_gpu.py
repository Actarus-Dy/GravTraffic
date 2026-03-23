"""GPU-accelerated O(N²) gravitational force computation using CuPy.

Fully vectorized naive force computation on GPU. For the Janus signed-mass
model, the naive O(N²) approach on GPU outperforms CPU Barnes-Hut O(N log N)
for N < ~50,000 due to massive parallelism on the distance matrix.

Force formula (matches force_engine.py and quadtree.py)::

    F_vec = +G_s * m_i * m_j / d^3 * (dx, dy)

where d = sqrt(dx² + dy² + epsilon²).

Falls back gracefully: if CuPy is not installed, ``GPU_AVAILABLE`` is False
and ``ForceEngineGPU`` raises ``RuntimeError`` on instantiation.

Author: Agent #01 Python Scientific Developer
Date: 2026-03-23
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from gravtraffic.core.force_engine import ForceEngine

try:
    import cupy as cp

    GPU_AVAILABLE = True
except ImportError:
    cp = None  # type: ignore[assignment]
    GPU_AVAILABLE = False

__all__ = ["ForceEngineGPU", "GPU_AVAILABLE"]


class ForceEngineGPU:
    """GPU-accelerated O(N²) gravitational social force engine.

    Parameters
    ----------
    G_s : float
        Social gravitational constant. Default 5.0 (unified calibration).
    softening : float
        Softening length epsilon in meters. Default 10.0 m.
    max_n : int
        Maximum N for GPU computation. Above this, falls back to CPU
        naive to avoid GPU out-of-memory. Default 10_000 (~1.5 GB VRAM).

    Raises
    ------
    RuntimeError
        If CuPy is not installed.
    """

    __slots__ = ("G_s", "epsilon", "max_n", "_cpu_fallback")

    def __init__(self, G_s: float = 5.0, softening: float = 10.0, max_n: int = 10_000) -> None:
        if not GPU_AVAILABLE:
            raise RuntimeError(
                "CuPy is required for GPU acceleration. Install with: pip install cupy-cuda12x"
            )
        self.G_s: float = float(G_s)
        self.epsilon: float = float(softening)
        self.max_n: int = int(max_n)
        # CPU fallback for large N (lazy import to avoid circular)
        self._cpu_fallback: ForceEngine | None = None

    def compute_all(
        self,
        positions: npt.NDArray[np.float64],
        masses: npt.NDArray[np.float64],
        theta: float = 0.5,  # ignored, kept for API compat
    ) -> npt.NDArray[np.float64]:
        """Compute total gravitational force on every particle using GPU.

        Fully vectorized O(N²) computation via CuPy broadcasting.
        The ``theta`` parameter is accepted for API compatibility with
        the CPU Barnes-Hut engine but is ignored (the computation is exact).

        Parameters
        ----------
        positions : ndarray, shape (N, 2), dtype float64
            (x, y) coordinates of each particle.
        masses : ndarray, shape (N,), dtype float64
            Signed mass of each particle.
        theta : float
            Ignored (kept for API compatibility).

        Returns
        -------
        ndarray, shape (N, 2), dtype float64
            Total (fx, fy) force on each particle (on CPU).
        """
        positions = np.asarray(positions, dtype=np.float64)
        masses = np.asarray(masses, dtype=np.float64)

        n = len(masses)
        if n == 0:
            return np.zeros((0, 2), dtype=np.float64)

        # Fallback to CPU for large N to avoid GPU OOM
        if n > self.max_n:
            if self._cpu_fallback is None:
                from gravtraffic.core.force_engine import ForceEngine

                self._cpu_fallback = ForceEngine(G_s=self.G_s, softening=self.epsilon)
            return self._cpu_fallback.compute_all(positions, masses, theta=theta)

        # Transfer to GPU
        pos_gpu = cp.asarray(positions)  # (N, 2)
        mass_gpu = cp.asarray(masses)  # (N,)

        eps2 = self.epsilon * self.epsilon
        G_s = self.G_s

        # Displacement matrix: dx[i,j] = pos[j,0] - pos[i,0]
        # Shape: (N, N, 2) via broadcasting (N,1,2) vs (1,N,2)
        diff = pos_gpu[cp.newaxis, :, :] - pos_gpu[:, cp.newaxis, :]  # (N, N, 2)

        # Softened distance: d[i,j] = sqrt(dx² + dy² + eps²)
        d2 = cp.sum(diff * diff, axis=2) + eps2  # (N, N)
        d = cp.sqrt(d2)  # (N, N)
        d3 = d2 * d  # (N, N)

        # Force coefficient: G_s * m_i * m_j / d³
        # mass_i: (N, 1), mass_j: (1, N) -> (N, N)
        mm = mass_gpu[:, cp.newaxis] * mass_gpu[cp.newaxis, :]  # (N, N)
        coeff = G_s * mm / d3  # (N, N)

        # Zero self-interaction (diagonal)
        cp.fill_diagonal(coeff, 0.0)

        # Force vectors: coeff * (dx, dy) -> sum over j
        # coeff: (N, N) -> (N, N, 1) * diff: (N, N, 2) -> sum axis=1
        forces_gpu = cp.sum(coeff[:, :, cp.newaxis] * diff, axis=1)  # (N, 2)

        # Transfer back to CPU
        return cp.asnumpy(forces_gpu)
