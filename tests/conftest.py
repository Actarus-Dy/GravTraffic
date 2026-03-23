import numpy as np
import pytest


@pytest.fixture
def rng():
    """Reproducible random number generator."""
    return np.random.default_rng(seed=42)


@pytest.fixture
def sample_vehicles(rng):
    """100 vehicles with random speeds and positions on a 1km segment."""
    n = 100
    dtype = np.dtype(
        [
            ("position_x", np.float64),
            ("position_y", np.float64),
            ("speed", np.float64),
            ("local_density", np.float64),
            ("v_max", np.float64),
            ("mass", np.float64),
        ]
    )
    vehicles = np.zeros(n, dtype=dtype)
    vehicles["position_x"] = rng.uniform(0, 1000, n)
    vehicles["position_y"] = rng.uniform(-5, 5, n)  # 2 lanes ~10m wide
    vehicles["speed"] = rng.uniform(10, 40, n)  # 36-144 km/h
    vehicles["local_density"] = rng.uniform(10, 60, n)  # veh/km
    vehicles["v_max"] = 36.0  # 130 km/h
    vehicles["mass"] = 0.0  # to be assigned
    return vehicles
