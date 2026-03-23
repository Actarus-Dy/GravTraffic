"""Comprehensive benchmark of all GravTraffic force engines.

Compares: Python naive, Python Barnes-Hut, Numba naive, Numba BH, GPU (CuPy).
Reports time/call and speedup vs Python naive at various N.

Usage:
    python benchmarks/bench_engines.py
"""

from __future__ import annotations

import time
import numpy as np

from gravtraffic.core.force_engine import ForceEngine


def _make_data(n: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0, 2000, (n, 2)).astype(np.float64)
    mass = rng.uniform(-5, 5, n).astype(np.float64)
    return pos, mass


def _bench(fn, pos, mass, warmup: int = 2, repeats: int = 5) -> float:
    """Run fn(pos, mass) with warmup, return median time in ms."""
    for _ in range(warmup):
        fn(pos, mass)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(pos, mass)
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


def main():
    sizes = [50, 100, 200, 500, 1000, 2000]

    engines: dict[str, callable] = {}

    # Python naive
    cpu = ForceEngine(G_s=5.0, softening=10.0)
    engines["Python naive"] = lambda p, m: cpu.compute_all_naive(p, m)
    engines["Python BH"] = lambda p, m: cpu.compute_all(p, m, theta=0.5)

    # Numba
    try:
        from gravtraffic.core.force_engine_numba import (
            ForceEngineNumba, ForceEngineBHNumba, NUMBA_AVAILABLE,
        )
        if NUMBA_AVAILABLE:
            nb = ForceEngineNumba(G_s=5.0, softening=10.0)
            nb_bh = ForceEngineBHNumba(G_s=5.0, softening=10.0)
            engines["Numba naive"] = lambda p, m: nb.compute_all(p, m)
            engines["Numba BH"] = lambda p, m: nb_bh.compute_all(p, m, theta=0.5)
    except ImportError:
        pass

    # GPU
    try:
        from gravtraffic.core.force_engine_gpu import ForceEngineGPU, GPU_AVAILABLE
        if GPU_AVAILABLE:
            gpu = ForceEngineGPU(G_s=5.0, softening=10.0)
            engines["GPU (CuPy)"] = lambda p, m: gpu.compute_all(p, m)
    except ImportError:
        pass

    # Header
    print(f"\n{'Engine':<18}", end="")
    for n in sizes:
        print(f"{'N='+str(n):>10}", end="")
    print(f"{'speedup':>10}")
    print("-" * (18 + len(sizes) * 10 + 10))

    # Benchmark
    baseline = {}  # N -> python naive time
    for name, fn in engines.items():
        print(f"{name:<18}", end="", flush=True)
        for n in sizes:
            pos, mass = _make_data(n)
            try:
                t = _bench(fn, pos, mass, warmup=1, repeats=3)
                if name == "Python naive":
                    baseline[n] = t
                print(f"{t:>8.1f}ms", end="")
            except Exception as e:
                print(f"{'ERR':>10}", end="")

        # Speedup at largest N
        n_max = sizes[-1]
        pos, mass = _make_data(n_max)
        try:
            t = _bench(fn, pos, mass, warmup=1, repeats=3)
            if n_max in baseline and baseline[n_max] > 0:
                speedup = baseline[n_max] / t
                print(f"{speedup:>8.1f}x", end="")
            else:
                print(f"{'--':>10}", end="")
        except Exception:
            print(f"{'--':>10}", end="")
        print()

    print()


if __name__ == "__main__":
    main()
