# GPU Solver Jacobian: Autodiff vs Analytical

## Status: SOLVED

The GPU solver has been fully migrated to analytical Jacobians, eliminating the convergence issues. This document describes the original problem and solution for reference.

## The Original Problem

The GPU solver originally used JAX autodiff (via `sparsejac`) to compute Jacobians. This caused convergence issues for circuits with floating nodes (like series NMOS stacks in AND gates).

### Root Cause

When a MOSFET is in cutoff or deep saturation, the autodiff-computed `dIds/dVds` (output conductance) can be extremely small:
- GPU autodiff: `gds ≈ 1e-16 S` in cutoff
- VACASK analytical: `gds = 1e-9 S` minimum (enforced)

This creates nearly-singular Jacobians that cause Newton-Raphson to diverge.

### Comparison at AND Gate `int` Node (Floating)

| Property | GPU (autodiff) | VACASK (analytical) |
|----------|----------------|---------------------|
| gds in cutoff | ~1e-16 S | 1e-9 S (min enforced) |
| Convergence | Fails (oscillates) | Succeeds (34 iters) |
| Final `int` voltage | Diverges to ±2.4V | Settles to 0.6V |

## How openvaf_jax Works

The `openvaf_jax.py` module translates Verilog-A models (via OpenVAF MIR) to JAX functions that return **both** residual and analytical Jacobian:

```python
# Generated function signature
def device_eval(inputs: List[float]) -> Tuple[Dict, Dict]:
    """
    Returns:
        residuals: Dict[node, {'resist': float, 'react': float}]
        jacobian: Dict[(row, col), {'resist': float, 'react': float}]
    """
```

### Key Code Location

In `openvaf_jax.py` lines 262-277:
```python
# Build output expressions
lines.append("    residuals = {")
for node, res in self.dae_data['residuals'].items():
    resist_val = res['resist'] if res['resist'] in defined_vars else '0.0'
    react_val = res['react'] if res['react'] in defined_vars else '0.0'
    lines.append(f"        '{node}': {{'resist': {resist_val}, 'react': {react_val}}},")

lines.append("    jacobian = {")
for entry in self.dae_data['jacobian']:
    key = f"('{entry['row']}', '{entry['col']}')"
    resist_val = entry['resist'] if entry['resist'] in defined_vars else '0.0'
    react_val = entry['react'] if entry['resist'] in defined_vars else '0.0'
    lines.append(f"        {key}: {{'resist': {resist_val}, 'react': {react_val}}},")
```

The Jacobian entries come from `dae_data['jacobian']` which is extracted from the OpenVAF MIR - these are **analytical derivatives** computed by OpenVAF's symbolic differentiation of the Verilog-A model.

## Current State

### VACASK Path (works)
```
c6288.sim → parser → psp103n/psp103p model
                          ↓
              create_simple_mosfet_eval() ← Uses explicit gm, gds stamps
                          ↓
              build_sparse_jacobian_and_residual() ← Stamps conductances into J
```

### GPU Path (broken)
```
c6288.sim → parser → psp103n/psp103p model
                          ↓
              create_simple_mosfet_eval() ← Same simplified model
                          ↓
              build_circuit_residual_fn() ← Residual only, no J
                          ↓
              sparsejac.jacrev(residual_fn) ← Autodiff gives wrong gds
```

## Solution Options

### Option 1: Add gds_min to GPU Model (Partial Fix)
Add minimum conductance leakage: `Ids += gds_min * Vds`

**Status**: Implemented but insufficient - gds_min only helps in cutoff, not saturation.

### Option 2: Use Analytical Jacobian from openvaf_jax (Preferred)

Create PSP103 device via openvaf_jax and use its analytical Jacobian:

```python
from openvaf_jax import OpenVAFToJAX

# Compile PSP103 model to JAX
translator = OpenVAFToJAX.from_file("psp103v4.va")
psp103_eval = translator.translate()

# In GPU residual function:
def residual_fn(V):
    for device in mosfets:
        inputs = build_inputs(V, device)
        residuals, jacobian = psp103_eval(inputs)  # Both from same call!
        # Stamp residuals into f
        # Jacobian available for building J analytically
```

This requires restructuring the GPU solver to:
1. Build residual AND Jacobian together (not separately via autodiff)
2. Use JAX's sparse matrix operations with explicit stamp values

### Option 3: Hybrid Approach

Use autodiff for most of the circuit but override MOSFET gds with minimum value:
```python
# After autodiff Jacobian computation
J_diag = jnp.diag(J)
J_diag = jnp.where(jnp.abs(J_diag) < gds_min, jnp.sign(J_diag) * gds_min, J_diag)
```

## Files Involved

- `jax_spice/analysis/dc_gpu.py` - GPU DC solver
- `jax_spice/analysis/transient_gpu.py` - GPU transient solver
- `jax_spice/benchmarks/c6288.py` - `create_simple_mosfet_eval()` at line 179
- `openvaf-py/openvaf_jax.py` - OpenVAF→JAX translator
- `jax_spice/devices/verilog_a.py` - VerilogADevice wrapper

## Test Case

The `and_test` circuit in `c6288.sim` is a good minimal test:
- 6 MOSFETs (NAND + inverter)
- Floating `int` node between series NMOS mn1/mn2
- VACASK converges in 34 iterations
- GPU solver diverges

## Verification: PSP103 via openvaf_jax Works

Confirmed that PSP103 can be compiled and evaluated via openvaf_jax:

```python
import openvaf_py
import openvaf_jax

va_path = 'openvaf-py/vendor/OpenVAF/integration_tests/PSP103/psp103.va'
modules = openvaf_py.compile_va(va_path)
module = modules[0]

translator = openvaf_jax.OpenVAFToJAX(module)
eval_fn = translator.translate()

# Evaluate - returns BOTH residual and analytical Jacobian
residuals, jacobian = eval_fn(inputs)
```

Results:
- Module: PSP103VA
- Nodes: 13 (internal nodes for the compact model)
- Parameters: 2616
- **Jacobian entries: 56** (analytical, not autodiff!)

The 56 Jacobian entries are computed symbolically by OpenVAF from the Verilog-A model equations - these are the well-conditioned conductance stamps that SPICE simulators rely on.

## How OpenVAF Handles Hidden States (Cache Slots)

OpenVAF's compilation pipeline distinguishes between:
1. **Model/Instance Parameters**: User-specified values (like VTH0, KP, W, L)
2. **Voltages**: Circuit node voltages from the solver
3. **Hidden States (Cache Slots)**: Intermediate values computed by the device model

### The Init/Eval Split in OSDI

OSDI models have two functions:
- **Init function**: Runs once per operating point, computes parameter-dependent values
- **Eval function**: Runs every Newton iteration, uses init outputs + voltages

The init function computes "cached" values that don't depend on voltages:
```c
// Example from a resistor model
// Init computes temperature-adjusted resistance:
res = R * pow(temperature / tnom, zeta);  // Cached, doesn't change during NR
```

These cached values are stored in "cache slots" and passed to the eval function.

### OpenVAF MIR Structure

In the MIR (Mid-level IR), cached values appear as:
```rust
// From openvaf/sim_back/src/init.rs
pub struct CacheSlot(u32);

// The init function outputs to cache slots
// The eval function reads from cache slots
```

The mapping between init outputs and eval inputs is tracked in `cache_mapping`:
```python
# From openvaf_py
cache_mapping = [
    {'init_value': 'init_v30', 'eval_param': 'v86'},  # sign
    {'init_value': 'init_v31', 'eval_param': 'v37'},  # beta
]
```

### How openvaf_jax Handles This

The JAX translator currently requires users to provide ALL parameters as inputs,
including those that should be computed by init:

```python
# What the user provides:
inputs = [
    0,        # PMOS flag
    1.0,      # sign (ideally computed from PMOS)
    0.0,      # V(s)
    1.2,      # V(g)
    1.2,      # Vgs (should be computed: V(g) - V(s))
    1.2,      # V(d)
    1.2,      # Vds (should be computed: V(d) - V(s))
    # ... etc
]
```

The eval function DOES compute voltage-derived values (Vgs, Vds) internally
from the terminal voltages. But it expects the init-computed values (like beta)
to be provided.

### Key Insight: Voltage-Derived Values

Looking at the MOSFET model:
```verilog
// These are computed EVERY iteration from voltages
Vgs = V(g) - V(s);
Vds = V(d) - V(s);
Vov = Vgs - VTH0;

// These are computed ONCE in init
beta = KP * W / L;  // Doesn't depend on voltages
```

The JAX-generated code correctly computes Vgs, Vds, Vov from the input
voltages on every call. The issue is that values like `beta` need to be
computed once and passed in.

### Solution for GPU Solver Integration

To use openvaf_jax models in the GPU solver:

1. **Run init once** at start of Newton iteration to compute cache slots
2. **Pass cache values** as part of the inputs array
3. **Eval function** computes residuals and Jacobians from voltages + cache

This matches how VACASK works: it runs init once, then calls eval repeatedly
with the cached values until convergence.

## Solution Implemented

The GPU solvers now use analytical Jacobians computed via the Shichman-Hodges MOSFET model:

1. **dc_gpu.py**: `build_analytical_residual_and_jacobian_fn()` computes both residual and Jacobian in one pass
2. **transient_gpu.py**: `build_transient_residual_and_jacobian_fn()` adds capacitor handling for transient analysis

### Key Implementation Details

```python
# Level-1 Shichman-Hodges MOSFET with explicit gm/gds computation
vth0 = 0.4
kp = 200e-6
lambda_ = 0.01
gds_min = 1e-9  # Minimum conductance for cutoff

# Analytical Jacobian stamps (Y-matrix)
stamps = {
    ('D', 'D'): gds,
    ('D', 'G'): gm,
    ('D', 'S'): -gds - gm,
    ('S', 'D'): -gds,
    ('S', 'G'): -gm,
    ('S', 'S'): gds + gm,
}
```

### Results

| Circuit | Before (autodiff) | After (analytical) |
|---------|-------------------|-------------------|
| Inverter | 6 iterations | 5 iterations |
| AND gate | FAILED | 78 iterations |
| NOR gate | ~15 iterations | ~10 iterations |

The `sparsejac` dependency has been removed from the GPU solver code path.
