# GravTraffic -- Physics Reference

> **Project:** GravTraffic (C-01) -- Actarus-Dy Software Catalogue
> **Version:** Phase 3 (2026-03-23)
> **Audience:** Developers, researchers, reviewers validating the Janus transposition

---

## Table of Contents

1. [Janus Cosmological Model](#1-janus-cosmological-model)
2. [Force Formula](#2-force-formula)
3. [Mass Assignment](#3-mass-assignment)
4. [Drag Enrichment](#4-drag-enrichment)
5. [Calibrated Parameters](#5-calibrated-parameters)
6. [Barnes-Hut Dual-Tree](#6-barnes-hut-dual-tree)
7. [Leapfrog KDK Integrator](#7-leapfrog-kdk-integrator)
8. [Fundamental Diagram Validation](#8-fundamental-diagram-validation)
9. [Emergence](#9-emergence)
10. [Key Design Decisions](#10-key-design-decisions)

---

## 1. Janus Cosmological Model

The Janus Cosmological Model (JCM), developed by Jean-Pierre Petit, is a
bimetric extension of general relativity in which the universe contains two
populations of matter with opposite gravitational properties:

- **Positive-mass matter** attracts other positive masses and repels negative
  masses.
- **Negative-mass matter** attracts other negative masses and repels positive
  masses.

This produces four interaction modes:

| Mass i | Mass j | m_i * m_j | Interaction |
|--------|--------|-----------|-------------|
| +      | +      | > 0       | Attraction  |
| -      | -      | > 0       | Attraction  |
| +      | -      | < 0       | Repulsion   |
| -      | +      | < 0       | Repulsion   |

### Transposition to Traffic

GravTraffic maps this cosmological framework onto vehicular traffic:

- **Slow vehicles** (speed below segment average) receive **positive mass**.
  They act as gravitational attractors, forming congestion seeds -- other
  vehicles are pulled toward them, nucleating jams.

- **Fast vehicles** (speed above segment average) receive **negative mass**.
  They act as repulsors, pushing nearby vehicles apart and creating fluid,
  free-flowing zones.

Traffic phenomena -- jam formation, convoy clustering, upstream shock waves,
and overtaking dynamics -- emerge naturally from these gravitational
interactions without explicit car-following rules.

### Source modules

- `gravtraffic/core/mass_assigner.py` -- mass computation
- `gravtraffic/core/force_engine.py` -- gravitational force computation
- `gravtraffic/core/simulation.py` -- full pipeline

---

## 2. Force Formula

### Vector form (Janus C-01, Section 1.2)

The gravitational social force exerted on vehicle *i* by vehicle *j* is:

```
F_vec_ij = +G_s * m_i * m_j / d^3 * (x_j - x_i, y_j - y_i)
```

where:

- `G_s` -- social gravitational constant (calibrated, see Section 5)
- `m_i`, `m_j` -- signed gravitational masses of vehicles i and j
- `(x_j - x_i, y_j - y_i)` -- displacement vector pointing from i toward j

### Softened distance

To prevent the force from diverging when two vehicles occupy nearly the same
position, a Plummer softening is applied:

```
d = sqrt(dx^2 + dy^2 + epsilon^2)
```

where `epsilon = 10 m` (softening length). The softened distance `d` replaces
the raw Euclidean distance in the denominator, bounding the maximum force
magnitude at `G_s * m_i * m_j / epsilon^3`.

### Sign convention

The sign convention follows directly from the formula -- no explicit negation
is applied:

```
coeff = +G_s * m_i * m_j / d^3
F_x = coeff * dx
F_y = coeff * dy
```

**Sign analysis:**

- **Same-sign masses** (`m_i * m_j > 0`): `coeff` is positive. The force
  vector points along `(dx, dy)`, which is from i toward j. This is
  **attraction**.

- **Opposite-sign masses** (`m_i * m_j < 0`): `coeff` is negative. The force
  vector points opposite to `(dx, dy)`, which is away from j. This is
  **repulsion**.

### Newton's third law

The naive O(N^2) implementation exploits antisymmetry: `F_ji = -F_ij`. Each
pair is visited once and both forces are accumulated simultaneously, halving
the work.

### Implementation

```python
# force_engine.py, ForceEngine.force_pair()
eps2 = self.epsilon * self.epsilon
d2   = dx * dx + dy * dy + eps2
d    = math.sqrt(d2)
d3   = d2 * d                      # d^3 via d^2 * d, avoids extra sqrt
coeff = self.G_s * m_i * m_j / d3
return (coeff * dx, coeff * dy)
```

---

## 3. Mass Assignment

### Formula (Janus C-01, Section 4.2)

Each vehicle *i* is assigned a signed gravitational mass at every timestep:

```
m_i = sgn(v_mean - v_i) * |v_mean - v_i|^beta * rho(x_i) / rho_0
```

where:

| Symbol       | Description                                  | Unit     |
|--------------|----------------------------------------------|----------|
| `v_i`        | Instantaneous speed of vehicle i             | m/s      |
| `v_mean`     | Population mean speed across all vehicles    | m/s      |
| `beta`       | Nonlinearity exponent (calibrated = 0.5)     | --       |
| `rho(x_i)`   | Local traffic density at vehicle position    | veh/km   |
| `rho_0`      | Reference density scale (`rho_scale = 30.0`) | veh/km   |

### Interpretation

- `v_i < v_mean` (slower than average): `v_mean - v_i > 0`, so `m_i > 0`
  (positive mass, attractor, congestion seed).
- `v_i > v_mean` (faster than average): `v_mean - v_i < 0`, so `m_i < 0`
  (negative mass, repulsor, fluid zone).
- `v_i = v_mean`: `m_i = 0` (neutral, no gravitational interaction).

### Role of beta

- `beta = 1.0`: linear relationship between speed deviation and mass.
- `beta < 1.0` (calibrated = 0.5): compresses large deviations, preventing
  extreme masses from destabilizing the simulation. The square-root scaling
  produces gentler gradients.
- `beta > 1.0`: amplifies large deviations.

### Density modulation

The `rho(x_i) / rho_0` factor scales mass by local density. In dense traffic,
masses are amplified, strengthening gravitational interactions where they
matter most. In sparse traffic, masses are suppressed.

### Classification thresholds

After mass computation, vehicles are classified for monitoring:

```
|m_i| > 0.1  and  m_i > 0  -->  "slow"    (positive mass)
|m_i| > 0.1  and  m_i < 0  -->  "fast"    (negative mass)
|m_i| <= 0.1               -->  "neutral"
```

### Implementation

```python
# mass_assigner.py, MassAssigner.assign()
delta        = v_mean - speeds                         # (N,)
abs_delta    = np.abs(delta)
signed_power = np.sign(delta) * np.power(abs_delta, self.beta)
masses       = signed_power * (local_densities / self.rho_scale)
```

Fully vectorized NumPy -- no Python loops.

---

## 4. Drag Enrichment

### Motivation

Pure gravity alone cannot generate a fundamental diagram. When all vehicles
travel at the mean speed, all masses are zero, all forces vanish, and any
uniform-speed state is a trivially stable fixed point. The model needs a
mechanism that drives speeds toward a density-dependent equilibrium.

The drag enrichment provides this. It is **physically motivated** (engine
thrust versus aerodynamic drag) and is explicitly **not** a car-following rule.
Gravity handles inter-vehicle interactions; drag provides the baseline
speed-density relationship.

### Formula

```
a_drag_i = gamma * (v_eq(rho_i) - |v_i|) * direction_i
```

where:

| Symbol         | Description                               | Unit   |
|----------------|-------------------------------------------|--------|
| `gamma`        | Drag coefficient (calibrated = 0.3)       | 1/s    |
| `v_eq(rho_i)`  | Greenshields equilibrium speed at local density | m/s |
| `\|v_i\|`     | Current speed magnitude of vehicle i      | m/s    |
| `direction_i`  | Unit vector along vehicle velocity        | --     |

### Greenshields equilibrium speed

```
v_eq(rho) = v_free * max(0, 1 - rho / rho_jam)
```

This is the classical Greenshields speed-density relationship:

- At `rho = 0` (empty road): `v_eq = v_free = 33.33 m/s` (120 km/h)
- At `rho = rho_jam = 150 veh/km`: `v_eq = 0` (gridlock)
- Linear interpolation between these extremes

### Behavior

- `|v_i| < v_eq`: drag is positive (acceleration toward equilibrium speed)
- `|v_i| > v_eq`: drag is negative (deceleration toward equilibrium speed)
- `|v_i| = v_eq`: drag is zero (at equilibrium)

### Total acceleration

The total acceleration on vehicle *i* combines gravity and drag:

```
a_i = F_gravity_i / max(|m_i|, 0.01) + a_drag_i
```

The mass floor of 0.01 prevents numerical instability for near-zero-mass
vehicles (which would otherwise produce extreme accelerations from small
forces).

### Implementation

```python
# simulation.py, GravSimulation._compute_accelerations()
v_eq = self._v_free * np.maximum(0.0, 1.0 - self.local_densities / self._rho_jam)
drag_scalar = self._drag_coefficient * (v_eq - speed_scalar)   # (N,)
accelerations += drag_scalar[:, np.newaxis] * direction         # (N, 2)
```

---

## 5. Calibrated Parameters

The following parameter set was determined by unified grid search
(`calibration_unified.py`), simultaneously satisfying the fundamental diagram
calibration (R^2 > 0.70) and emergence criterion (upstream deceleration
> 0.5 m/s).

| Parameter    | Symbol     | Value   | Unit    | Role                                   |
|------------- |------------|---------|---------|----------------------------------------|
| Gravity      | `G_s`      | 5.0     | --      | Social gravitational constant          |
| Mass exp.    | `beta`     | 0.5     | --      | Speed deviation exponent               |
| Softening    | `epsilon`  | 10.0    | m       | Force softening length                 |
| Drag coeff.  | `gamma`    | 0.3     | 1/s     | Greenshields drag strength             |
| Free-flow    | `v_free`   | 33.33   | m/s     | Free-flow speed (120 km/h)             |
| Jam density  | `rho_jam`  | 150.0   | veh/km  | Greenshields jam density               |
| Ref. density | `rho_0`    | 30.0    | veh/km  | Mass normalization scale               |
| BH theta     | `theta`    | 0.5     | --      | Barnes-Hut opening angle               |
| Speed limit  | `v_max`    | 36.0    | m/s     | Maximum vehicle speed (~130 km/h)      |
| Base dt      | `dt`       | 0.1     | s       | Integration timestep (adaptive)        |

### Calibration search space

```
G_s:       [1, 5, 10, 20, 50]
beta:      [0.3, 0.5, 1.0]
softening: [5, 10, 20]
gamma:     [0.1, 0.3, 0.5, 1.0]
```

180 configurations tested. The unified score combines calibration R^2 and
emergence deceleration:

```
unified_score = max(0, R^2) * min(1, upstream_decel / 2.0)
```

### Key findings from the search

- `G_s = 9.8` (original "Janus" value): too strong, destabilizes the system
  (R^2 = -0.9).
- `G_s = 2.0`: passes calibration but fails emergence (insufficient
  inter-vehicle coupling).
- `G_s = 5.0, gamma = 0.3`: satisfies both criteria -- the chosen operating
  point.

---

## 6. Barnes-Hut Dual-Tree

### Standard Barnes-Hut

The Barnes-Hut algorithm reduces force computation from O(N^2) to
O(N log N) by recursively subdividing space into a quadtree. Distant
particle groups are approximated by their center-of-mass monopole moment.
The **opening angle** parameter `theta` controls the accuracy-speed
trade-off:

```
criterion: cell_diagonal / distance_to_nearest_edge < theta
```

| theta | Behavior                                     |
|-------|----------------------------------------------|
| 0.0   | Exact (equivalent to O(N^2) direct sum)     |
| 0.5   | Default -- less than 1% relative error       |
| 1.0   | Fast but less accurate                       |

### Why dual-tree for signed masses

Standard Barnes-Hut assumes all masses are positive. With Janus signed masses,
a single tree suffers from **monopole cancellation**: a cell containing both
positive and negative masses has a small net mass, but the individual forces
do not cancel geometrically. The center-of-mass can lie far from any actual
particle, and the monopole approximation becomes wildly inaccurate.

**Solution:** Build two separate quadtrees:

1. **Positive tree**: contains only vehicles with `m_i >= 0`
2. **Negative tree**: contains only vehicles with `m_i < 0`

Within each tree, all masses have the same sign, so the standard monopole
approximation is well-behaved. The total force on any particle is the sum of
forces from both trees.

```
F_total_i = F_from_positive_tree(i) + F_from_negative_tree(i)
```

### Tree structure

- Leaf capacity: 8 particles (balances tree depth vs. leaf-level direct
  computation; reduces borderline approximation errors).
- Maximum depth: 64 (prevents infinite recursion on coincident particles).
- Bounding box: square enclosure of all particles with 1 m margin.
- Self-interaction exclusion: particles skip themselves via index comparison.
- Opening-angle test uses distance to nearest cell edge (Bmax criterion),
  which is more conservative than the COM distance and handles cells whose
  center-of-mass is offset from the geometric center.

### Implementation

```python
# force_engine.py, ForceEngine.compute_all()
pos_mask = masses >= 0.0
neg_mask = ~pos_mask

# Build separate trees
tree_pos = QuadTree(bbox, capacity=8)   # positive masses only
tree_neg = QuadTree(bbox, capacity=8)   # negative masses only

# Total force = sum from both trees
for i in range(n):
    fx, fy = tree_pos.compute_force(...) + tree_neg.compute_force(...)
```

---

## 7. Leapfrog KDK Integrator

### Scheme (Janus C-01, Section 2.1)

GravTraffic uses the **kick-drift-kick (KDK)** variant of the leapfrog
integrator, a second-order symplectic method. The KDK variant is preferred
because the forces at the end of the step are immediately available for the
next step without recomputation.

```
Given:  positions x, velocities v, accelerations a, timestep dt

1. v_half  = v     + 0.5 * a     * dt      (half-kick)
2. x_new   = x     + v_half      * dt      (drift)
3. a_new   = accel(x_new, v_half)           (recompute forces at new state)
4. v_new   = v_half + 0.5 * a_new * dt     (half-kick)
5. clip |v_new| to [0, v_max]              (speed limiter)
```

### Symplecticity

The leapfrog integrator preserves phase-space volume (Liouville's theorem),
which means it conserves a shadow Hamiltonian close to the true Hamiltonian
over exponentially long times. This prevents the artificial energy drift that
plagues non-symplectic integrators (e.g., forward Euler, RK4) in long-running
gravitational simulations.

The speed clipping step (step 5) breaks strict symplecticity but is physically
necessary to enforce the traffic speed limit. In practice, clipping events are
rare when the adaptive timestep is well-tuned.

### Use of v_half for mass and drag

The force callback in step 3 receives both the drifted positions `x_new` and
the half-kick velocities `v_half`. Both mass assignment and drag computation
use `v_half` (not the old or new velocities), which is essential for
maintaining the symplectic structure:

```python
# simulation.py, _compute_accelerations(positions, velocities)
# velocities = v_half from leapfrog
speeds = np.linalg.norm(velocities, axis=1)
masses = self._mass_assigner.assign(speeds, self._mean_speed, self.local_densities)
```

### Adaptive CFL timestep

An adaptive timestep based on the CFL (Courant-Friedrichs-Lewy) condition
prevents particles from crossing more than half the minimum inter-particle
distance per step:

```
dt <= d_min / (2 * v_max_system)
```

The minimum distance is approximated in O(N log N) by sorting particles by
x-coordinate and checking consecutive pairs. This is exact for 1-D-dominated
traffic layouts.

```
dt is clamped to [dt_min=0.01s, dt_max=0.2s]
```

### Implementation

```python
# integrator.py, leapfrog_step()
v_half         = velocities + 0.5 * forces * dt          # half-kick
positions_new  = positions  + v_half * dt                 # drift
forces_new     = force_fn(positions_new, v_half)          # recompute
velocities_new = v_half     + 0.5 * forces_new * dt      # half-kick
velocities_new = _clip_speed(velocities_new, v_max)       # limiter
```

---

## 8. Fundamental Diagram Validation

### Protocol

The calibration test verifies that the model reproduces the Greenshields
speed-density fundamental diagram. The test procedure is:

1. For each density in `[10, 30, 50, 70, 90, 110, 130]` veh/km:
   - Place vehicles uniformly on a 2 km highway segment.
   - Initialize **all vehicles at free-flow speed** (`v_free = 33.33 m/s`),
     NOT at the equilibrium speed. This avoids circular validation.
   - Run 300 steps (30 s) and measure the final mean speed.
2. Compare measured mean speeds against Greenshields predictions.
3. Compute R^2 and RMSE.

### Results (calibrated parameters)

With `G_s=5.0, beta=0.5, gamma=0.3`:

| Metric     | Value                  |
|------------|------------------------|
| R^2        | 0.9796                 |
| RMSE       | 0.96 m/s (3.5 km/h)   |
| Monotonic  | Yes (speed decreases with density) |
| Stable     | Yes (no NaN/Inf)       |

The model converges from a uniform free-flow initial condition to the
correct density-dependent equilibrium speed, driven by the drag enrichment
term. Gravity provides perturbations and inter-vehicle coupling but does not
dominate the equilibrium.

### Pass criteria

```
R^2 > 0.70    (calibration pass)
Stable         (no NaN or Inf in velocities)
Monotonic      (mean speed non-increasing with density, tolerance 1.0 m/s)
```

### Extended validation

The `fundamental_diagram.py` module runs a finer sweep (10 to 140 veh/km in
steps of 10) with 800 steps including 500 warmup steps to let transients
settle. This provides higher-fidelity validation data for plotting.

---

## 9. Emergence

Emergence is the hallmark of the Janus transposition: macroscopic traffic
phenomena arise from microscopic gravitational interactions without being
explicitly programmed. GravTraffic measures emergence by comparing
**gravity-on** (`G_s=5.0`) vs. **gravity-off** (`G_s=0`, drag only) scenarios.

### Test scenario

- 100-200 vehicles on a 2 km highway at uniform 25 m/s.
- One slow vehicle injected at the midpoint (5 m/s).
- Run 500 steps (50 s) and measure downstream effects.

### Metrics

#### Upstream deceleration

The gravitational well created by the slow vehicle (positive mass, attractor)
propagates a deceleration wave upstream. Vehicles approaching from behind
decelerate before reaching the slow vehicle -- a chain reaction mediated by
gravity.

```
upstream_decel = initial_speed - mean_final_speed(upstream_vehicles)
```

With calibrated parameters: upstream deceleration exceeds 15 m/s (gravity on)
vs. minimal deceleration with gravity off. The pass threshold is > 0.5 m/s.

#### Gini coefficient of spacings

The Gini coefficient measures clustering inequality in vehicle spacings.
A Gini of 0 means perfectly uniform spacing; a Gini approaching 1 means
extreme clustering (jam). Gravity increases the Gini over time as vehicles
cluster behind the slow vehicle.

```
Gini = (2 * sum(i * x_i) - (n+1) * sum(x_i)) / (n * sum(x_i))
```

where `x_i` are sorted vehicle spacings.

#### Shock wave speed

The furthest upstream vehicle that decelerated by more than 0.5 m/s defines
the wave front. The wave speed is:

```
wave_speed = wave_distance / elapsed_time
```

This corresponds to the backward-propagating kinematic wave observed in real
traffic.

#### Speed variance amplification

```
variance_ratio = std(final_speeds) / std(initial_speeds)
```

Gravity amplifies speed heterogeneity (variance_ratio >> 1), while drag alone
keeps it low.

### Composite emergence score

```
emergence_score = mean(
    min(decel_delta / 5.0, 1.0),       # extra upstream deceleration
    min(variance_delta / 2.0, 1.0),    # extra variance amplification
    min(gini_delta / 0.1, 1.0)         # extra clustering
)
```

Each component compares gravity-on minus gravity-off, normalized to [0, 1].

### Gravity ON vs OFF summary

| Metric                    | Gravity ON (G_s=5) | Gravity OFF (G_s=0) |
|---------------------------|--------------------|-----------------------|
| Upstream deceleration     | > 15 m/s           | minimal               |
| Variance ratio            | >> 1               | ~ 1                   |
| Gini increase             | significant         | near zero             |
| Backward wave             | observed            | not observed          |

---

## 10. Key Design Decisions

### 10.1 Sign convention history

**The force sign was initially inverted.** The original implementation used a
negative sign (`coeff = -G_s * m_i * m_j / d^3`), mimicking the Newtonian
convention where positive masses attract with a negative potential gradient.
However, in the Janus model the sign convention differs: same-sign masses
attract via a positive coefficient.

This was identified during the Devil's Advocate (DA) audit on 2026-03-22 and
corrected. The validated formula is:

```
coeff = +G_s * m_i * m_j / d^3     (NOT negative)
```

All tests were updated and the correction is validated by directional test
cases in `test_force_engine.py`.

### 10.2 Drag is NOT car-following

The drag enrichment `gamma * (v_eq(rho) - |v_i|)` depends only on the
vehicle's own speed and the local density field. It does **not** reference
the speed or position of any leading vehicle. It represents the macroscopic
tendency for vehicles to adopt a density-appropriate speed (engine thrust
vs. aerodynamic drag analogy).

Inter-vehicle interactions are handled exclusively by gravity. This separation
is a deliberate design choice: gravity provides the emergent, non-local
coupling that distinguishes GravTraffic from car-following models like
Intelligent Driver Model (IDM) or Krauss.

### 10.3 Pure gravity limitations

Pure gravity (`gamma = 0`) preserves speed equilibrium but cannot generate a
fundamental diagram. When all vehicles travel at the mean speed:

```
v_i = v_mean  for all i
  => delta = v_mean - v_i = 0
  => m_i = 0  for all i
  => F = 0    for all i
```

Any uniform-speed state is a fixed point regardless of density. The drag
enrichment breaks this degeneracy by pulling speeds toward the
density-dependent `v_eq(rho)`.

### 10.4 Obstacle injection

Red traffic lights are modeled as static positive-mass obstacles injected via
`simulation.set_obstacles()`. They participate in force computation (exerting
gravitational attraction on approaching vehicles, simulating deceleration) but
are not integrated -- their positions remain fixed.

### 10.5 Dynamic local densities

Local density `rho(x_i)` is recomputed every step using `scipy.spatial.cKDTree`
neighbor counting within a 100 m radius. This ensures masses respond to the
current traffic state rather than stale initial conditions.

### 10.6 VehicleAgent is passive

In the Mesa ABM layer, `VehicleAgent` does not compute its own physics. All
forces, accelerations, and position updates are computed centrally in
`GravSimulation` and pushed to agents. This prevents inconsistencies between
the physics engine and the agent state.

### 10.7 Mesa 3.5 compatibility

The Mesa ABM framework uses `super().__init__(model)` (no manual `unique_id`),
and `rng=` instead of `seed=` for random number generation.

---

## References

- Janus Civil C-01 GravTraffic Technical Plan, Sections 1.2, 1.3, 2.1, 4.2
- Greenshields, B.D. (1935). "A Study of Traffic Capacity." *Highway Research
  Board Proceedings*, 14, 448-477.
- Barnes, J. and Hut, P. (1986). "A hierarchical O(N log N) force-calculation
  algorithm." *Nature*, 324, 446-449.
- Petit, J.-P. "The Janus Cosmological Model." -- bimetric gravity framework.

---

*Document generated 2026-03-23. Source of truth: `gravtraffic/core/` modules.*
