# GPU Solver Architecture

This document describes the GPU-native solver architecture for VA-JAX,
including DC operating point and transient analysis implementations.

## Overview

VA-JAX provides two complementary GPU-native solvers:

1. **DC Solver** (`dc_gpu.py`) - Computes steady-state operating point
2. **Transient Solver** (`transient_gpu.py`) - Time-domain simulation with backward Euler

Both use automatic differentiation via `sparsejac` for sparse Jacobian computation,
eliminating the need for explicit analytical derivatives.

## Key Design Decisions

### Why Transient Instead of DC for Digital Circuits?

For digital MOSFET circuits (like the C6288 multiplier), **transient analysis with
`icmode='uic'` is preferred over DC operating point analysis**. This matches the
approach used by VACASK and other production simulators.

**The problem with DC analysis:**
- Digital circuits have many possible stable states (high/low combinations)
- Newton-Raphson can converge to wrong states or oscillate between them
- Floating gate nodes and feedback paths make convergence difficult
- Source stepping and GMIN stepping help but don't always work

**Why transient works better:**
- Natural settling behavior from initial conditions
- No ambiguity about which state to converge to
- The circuit "finds" its own solution through time evolution
- Capacitances (intrinsic or explicit) provide natural damping

**VACASK Example:**
```
// From c6288.sim - VACASK uses transient, not DC
analysis tranmul tran stop=2n step=2p icmode="uic"
// analysis op1 op  <- DC is commented out
```

### Initial Condition Modes (icmode)

The transient solver supports two initial condition modes:

1. **`icmode='op'`** (default): Compute DC operating point first, then run transient
   - Good for analog circuits where DC solution is well-defined
   - Can fail for digital circuits with multiple stable states

2. **`icmode='uic'`**: Use Initial Conditions directly (zeros + supply voltages)
   - Skip DC computation entirely
   - Let circuit settle through transient simulation
   - Preferred for digital logic circuits
   - Matches VACASK's behavior for benchmarks

## Solver Architecture

### Data Flow

```
MNASystem (devices)
      │
      ▼
build_device_groups()  ◄── Groups devices by type (MOSFET, R, V, C)
      │
      ▼
VectorizedDeviceGroup  ◄── Pre-computed JAX arrays per device type
      │
      ├──────────────────┐
      ▼                  ▼
build_gpu_residual_fn()  build_transient_circuit_data_fast()
      │                  │
      ▼                  ▼
  residual_fn()      TransientCircuitData
      │                  │
      ▼                  ▼
sparsejac.jacrev()   build_transient_residual_fn()
      │                  │
      ▼                  ▼
Sparse Jacobian      residual_fn(V_curr, V_prev, dt)
(BCOO format)             │
      │                   ▼
      ▼              sparsejac.jacrev()
Newton-Raphson            │
Iteration                 ▼
      │              Newton-Raphson per timestep
      ▼                   │
Solution                  ▼
                    Time-domain solution
```

### Key Components

#### 1. VectorizedDeviceGroup

Groups devices of the same type with pre-computed JAX arrays:

```python
class VectorizedDeviceGroup:
    device_type: DeviceType       # MOSFET, RESISTOR, VSOURCE, etc.
    device_names: List[str]
    node_indices: Array           # Shape: (n_devices, n_terminals)
    params: Dict[str, Array]      # Device parameters as batched arrays
```

This enables vectorized device evaluation instead of per-device Python loops.

#### 2. GPU Residual Function

The residual function computes KCL violations at each node:

```python
def gpu_residual_fn(V: Array) -> Array:
    """Compute f(V) where f=0 at solution.

    Returns:
        residual: Array of shape (n_nodes - 1,) - current imbalance at each node
    """
    # For each device type, compute currents vectorized:
    # - Voltage sources: I = G_big * (V_actual - V_target)
    # - MOSFETs: Ids = f(Vgs, Vds, W, L, model_params)
    # - Resistors: I = G * (Vp - Vn)
    # - Capacitors: I = C/dt * (V - V_prev)  [transient only]

    # Sum contributions at each node
    residual = sum_at_nodes(device_currents)

    # Add GMIN to ground (numerical stability)
    residual += gmin * V

    return residual
```

#### 3. Sparse Jacobian via sparsejac

Instead of manually computing Jacobian entries, we use automatic differentiation:

```python
# Create sparsity pattern (which entries are non-zero)
sparsity = jsparse.BCOO((data, indices), shape=(n, n))

# Create sparse Jacobian function via graph coloring
jacobian_fn = sparsejac.jacrev(residual_fn, sparsity=sparsity)

# Evaluate at a point
J_sparse = jacobian_fn(V)  # Returns BCOO sparse matrix
```

**Benefits:**
- No need for analytical derivatives
- Correct derivatives even for complex models
- Efficient evaluation via graph coloring

**Key insight:** The sparsity pattern comes from circuit topology - a device only
contributes to Jacobian entries connecting its terminal nodes.

#### 4. Newton-Raphson Iteration

Both solvers use Newton-Raphson with voltage limiting:

```python
for iteration in range(max_iterations):
    f = residual_fn(V)
    if max(abs(f)) < abstol:
        break  # Converged

    J = jacobian_fn(V)
    delta_V = solve(J, -f)  # Linear solve

    # Voltage limiting (prevent overshooting)
    max_step = 0.5 * vdd
    if max(abs(delta_V)) > max_step:
        delta_V *= max_step / max(abs(delta_V))

    V = V + delta_V
    V = clip(V, -2*vdd, 2*vdd)  # Safety clamp
```

### MOSFET Models

#### Transient Solver: Level-1 Model

The transient solver uses a simplified level-1 MOSFET model:

```python
beta = kp * W / L
Vgst = Vgs - Vth0

# Smooth subthreshold (avoids discontinuity at Vth)
Vgst_eff = log1p(exp(alpha * Vgst)) / alpha

# Saturation with soft clipping
Vdsat = max(Vgst_eff, epsilon)
Vds_eff = Vdsat * tanh(abs(Vds) / Vdsat)

# Drain current
Ids = beta * Vgst_eff * Vds_eff * (1 + lambda * abs(Vds))
```

#### DC Solver: BSIM-like Model

The DC solver uses a more sophisticated BSIM-like model from `mosfet_simple.py`:

- Body effect (gamma, phiB)
- Velocity saturation (vsat)
- Subthreshold conduction (n_sub)
- Short channel effects (theta, a0)
- Temperature dependence

## Performance Characteristics

### JIT Compilation

Both solvers benefit from JAX's JIT compilation:

```
First timestep (includes JIT): ~1.5s
Subsequent timesteps: ~0.002s (750x faster)
```

**Tip:** For benchmarking, always exclude the first iteration or use pre-warming.

### CPU vs GPU

| Circuit Size | CPU Time | GPU Time | Speedup |
|--------------|----------|----------|---------|
| Small (< 100 nodes) | Faster | Slower | 0.5x |
| Medium (1K nodes) | Similar | Similar | 1x |
| Large (5K+ nodes) | Slower | Faster | 3-10x |

GPU acceleration becomes beneficial at ~1000+ nodes due to data transfer overhead.

### Memory Usage

Sparse Jacobian memory scales with O(devices * terminals²) instead of O(nodes²):

```
C6288 (5123 nodes, 10112 MOSFETs):
- Dense Jacobian: ~200 MB
- Sparse Jacobian: ~1.3 MB
- Savings: 150x
```

## Usage Examples

### DC Operating Point

```python
from vajax.analysis.dc_gpu import dc_operating_point_gpu_vectorized

# Build and prepare system
system.build_device_groups()

# Run GPU DC solver
V, info = dc_operating_point_gpu_vectorized(
    system,
    vdd=1.2,
    max_iterations=100,
    abstol=1e-9,
)

print(f"Converged: {info['converged']}")
print(f"Iterations: {info['iterations']}")
```

### DC with Source Stepping (for difficult circuits)

```python
from vajax.analysis.dc_gpu import dc_operating_point_gpu_vectorized_source_stepping

V, info = dc_operating_point_gpu_vectorized_source_stepping(
    system,
    vdd_target=1.2,
    vdd_steps=10,  # Ramp VDD from 0 to 1.2V
    verbose=True,
)
```

### Transient Analysis

```python
from vajax.analysis.transient_gpu import transient_analysis_gpu

# For digital circuits - use icmode='uic' to skip DC
times, solutions, info = transient_analysis_gpu(
    system,
    t_stop=2e-9,      # 2ns
    t_step=2e-12,     # 2ps
    icmode='uic',     # Use Initial Conditions (like VACASK)
    vdd=1.2,
)

# For analog circuits - compute DC first
times, solutions, info = transient_analysis_gpu(
    system,
    t_stop=1e-6,
    t_step=1e-9,
    icmode='op',      # Compute DC operating point first
    vdd=1.2,
)
```

## Current Limitations

1. **Linear solver:** Currently uses dense solve on CPU, sparse solve on GPU
   - GPU sparse solve via `jax.experimental.sparse.linalg.spsolve`
   - For very large circuits, consider iterative methods (GMRES, BiCGSTAB)

2. **Fixed timestep:** No adaptive timestepping yet
   - Must choose dt based on fastest circuit dynamics
   - Too large: accuracy issues; too small: slow simulation

3. **Model coverage:**
   - Transient: Level-1 MOSFETs only
   - DC: BSIM-like model with more features
   - No capacitors in DC (would need AC analysis)

4. **AND gate convergence:** The AND gate test shows convergence issues with the
   level-1 model in transient. More sophisticated models or relaxed tolerances
   may be needed for complex logic gates.

## Future Work

1. **Adaptive timestepping** - Use LTE (Local Truncation Error) control
2. **Unified MOSFET model** - Same BSIM-like model for both DC and transient
3. **Iterative linear solvers** - For very large circuits
4. **Parallel timesteps** - Use `lax.scan` for GPU-resident time loops
5. **Capacitor support in DC** - For AC analysis

## References

1. VACASK simulator behavior for digital circuit analysis
2. sparsejac: https://github.com/mfschubert/sparsejac
3. JAX documentation: https://jax.readthedocs.io
4. Newton-Raphson for circuit simulation (Nagel, 1975)
