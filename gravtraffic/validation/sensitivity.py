"""Parameter sensitivity analysis for GravTraffic.

Sweeps G_s, beta, gamma and measures two objectives:
1. Fundamental diagram fit (R² vs Greenshields)
2. Emergence score (shock wave amplification)

Produces a parameter sensitivity grid for publication.

Author: Agent #23 Scientific Validation Tester
Date: 2026-03-23
"""

from __future__ import annotations

from gravtraffic.validation.emergence import run_emergence_analysis
from gravtraffic.validation.fundamental_diagram import run_fd_sweep


def run_sensitivity(
    G_s_values: list[float] | None = None,
    beta_values: list[float] | None = None,
    gamma_values: list[float] | None = None,
    fd_densities: list[float] | None = None,
    fd_n_steps: int = 300,
    emergence_n_steps: int = 300,
    seed: int = 42,
) -> dict:
    """Run parameter sensitivity sweep.

    Parameters
    ----------
    G_s_values : list[float], optional
        G_s values to sweep. Default: [1, 2, 5, 10, 15].
    beta_values : list[float], optional
        Beta values to sweep. Default: [0.25, 0.5, 1.0].
    gamma_values : list[float], optional
        Gamma (drag) values to sweep. Default: [0.1, 0.3, 0.5].

    Returns
    -------
    dict with keys:
        - grid: list[dict] — one entry per (G_s, beta, gamma) combo
        - best: dict — combo with highest combined score
        - summary: dict — statistics
    """
    if G_s_values is None:
        G_s_values = [1.0, 2.0, 5.0, 10.0, 15.0]
    if beta_values is None:
        beta_values = [0.25, 0.5, 1.0]
    if gamma_values is None:
        gamma_values = [0.1, 0.3, 0.5]
    if fd_densities is None:
        fd_densities = [20, 40, 60, 80, 100, 120]

    grid = []
    best_score = -float("inf")
    best_entry = None

    for G_s in G_s_values:
        for beta in beta_values:
            for gamma in gamma_values:
                # Fundamental diagram
                fd = run_fd_sweep(
                    densities=fd_densities,
                    G_s=G_s,
                    beta=beta,
                    gamma=gamma,
                    n_steps=fd_n_steps,
                    warmup_steps=100,
                    seed=seed,
                )

                # Emergence
                em = run_emergence_analysis(
                    G_s=G_s,
                    beta=beta,
                    gamma=gamma,
                    n_steps=emergence_n_steps,
                    seed=seed,
                )

                # Combined score: R² (0-1) + normalized emergence
                r2 = max(0.0, fd["r_squared"])
                em_score = em["emergence_score"]
                combined = r2 + min(em_score / 10.0, 1.0)  # cap emergence at 1.0

                entry = {
                    "G_s": G_s,
                    "beta": beta,
                    "gamma": gamma,
                    "r_squared": fd["r_squared"],
                    "rmse": fd["rmse"],
                    "emergence_score": em_score,
                    "combined_score": combined,
                }
                grid.append(entry)

                if combined > best_score:
                    best_score = combined
                    best_entry = entry

    return {
        "grid": grid,
        "best": best_entry,
        "summary": {
            "n_combos": len(grid),
            "best_combined_score": best_score,
            "parameters_tested": {
                "G_s": G_s_values,
                "beta": beta_values,
                "gamma": gamma_values,
            },
        },
    }
