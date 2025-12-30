# JAX-SPICE Development Guidelines

## Pure JAX Requirement

**CRITICAL**: This is a JAX-native circuit simulator. All numerical code MUST use JAX, not numpy or scipy.
**CRITICAL**: All functionality should be optimised to run as much on GPU as possible, minimising CPU to GPU context switching and data transfer

You should try to minimise the code base size:

 * Avoid duplication
 * Where possible unify critical paths.
 * Regularly check for duplications of code and look for any opportunities to minimise the code base without impacting the tested functionality. Conform with the user.

### Forbidden Patterns

Do NOT use these:
```python
# FORBIDDEN - numpy arrays
import numpy as np
np.array([1, 2, 3])
np.zeros(n)
np.linalg.solve(A, b)

# FORBIDDEN - scipy sparse
from scipy.sparse import csr_matrix, lil_matrix, coo_matrix
from scipy.sparse.linalg import spsolve, gmres

# FORBIDDEN - numpy linear algebra
np.linalg.inv(A)
np.linalg.lstsq(A, b)
```

### Required Patterns

Use these JAX equivalents:
```python
# REQUIRED - JAX arrays
import jax.numpy as jnp
from jax import Array
jnp.array([1, 2, 3])
jnp.zeros(n)
jax.scipy.linalg.solve(A, b)

# REQUIRED - JAX sparse (BCOO/BCSR formats)
from jax.experimental.sparse import BCOO, BCSR
from jax.experimental.sparse.linalg import spsolve

# REQUIRED - JAX linear algebra
jax.scipy.linalg.inv(A)
jax.scipy.linalg.lstsq(A, b)
```

### Why Pure JAX?

1. **GPU Acceleration**: JAX arrays stay on GPU, avoiding host-device transfers
2. **JIT Compilation**: All code can be traced and compiled
3. **Autodiff**: Jacobians via `jax.jacfwd` or matrix-free `jax.jvp`
4. **Consistency**: One API across CPU, GPU, and TPU

### Sparse Solver Strategy

The simulator supports two solver modes:

1. **Dense solver** (default): Uses `jax.scipy.linalg.solve()` for small-medium circuits
2. **Sparse solver**: Uses BCOO/BCSR + `spsolve` for large circuits (>1000 nodes)

For large circuits, use sparse mode with `use_sparse=True`:

```python
from jax.experimental.sparse import BCOO, BCSR
from jax.experimental.sparse.linalg import spsolve

# Build Jacobian as BCOO, convert to BCSR for solving
J_bcoo = BCOO((j_data, jnp.stack([j_rows, j_cols], axis=1)), shape=(n, n))
J_bcsr = BCSR.from_bcoo(J_bcoo)

# Solve J @ delta = -f using sparse direct solver
delta = spsolve(J_bcsr.data, J_bcsr.indices, J_bcsr.indptr, -f, (n, n))
```

Note: c6288 benchmark (~86k nodes) requires sparse mode as dense would need ~56GB memory.

## Test Commands

```bash
# Run all tests (CPU)
JAX_PLATFORMS=cpu uv run pytest tests/ -v

# Run VACASK benchmark tests
JAX_PLATFORMS=cpu uv run pytest tests/test_vacask_suite.py -v

# Run openvaf-py tests
cd openvaf-py && JAX_PLATFORMS=cpu ../.venv/bin/python -m pytest tests/ -v

# Run and profile GPU tests on non-CUDA systems (e.g. Apple silicon)
uv run scripts/profile_gpu_cloudrun.py --benchmark ring,c6288

# Profile GPU performance on CUDA systems
uv run python scripts/profile_gpu.py --benchmark ring,c6288

```

## Precision Configuration

Precision is auto-configured on import via `jax_spice/__init__.py`:
- **CPU/CUDA**: Float64 enabled (`jax_enable_x64=True`) for numerical accuracy
- **Metal/TPU**: Float32 (`jax_enable_x64=False`) since these backends don't support float64 natively

To check or override precision:
```python
import jax_spice

# Check current settings
info = jax_spice.get_precision_info()
print(f"x64 enabled: {info['x64_enabled']}, backend: {info['backend']}")

# Force specific precision (after import, before computation)
jax_spice.configure_precision(force_x64=True)   # Force float64
jax_spice.configure_precision(force_x64=False)  # Force float32
jax_spice.configure_precision()                  # Auto-detect again
```

## Key Architecture

```
jax_spice/analysis/
├── solver.py          # Newton-Raphson with lax.while_loop
├── system.py          # SystemBuilder for J,f construction
├── dc.py              # DC operating point analysis
├── transient.py       # Transient analysis (vectorized GPU path)
├── mna.py             # MNA system representation
└── sparse.py          # JAX sparse utilities (BCOO)

jax_spice/devices/
└── openvaf_device.py  # Batched OpenVAF device evaluation

jax_spice/benchmarks/
└── runner.py          # VACASK benchmark runner
```

## Device Routing

All devices are routed through OpenVAF except voltage/current sources:

- **OpenVAF path**: resistor, capacitor, diode, psp103, and other VA models
  - Batched evaluation via `vmap` for GPU efficiency
  - VA models from `vendor/VACASK/devices/` (resistor.va, capacitor.va, diode.va)
  - Complex models from `openvaf-py/vendor/OpenVAF/integration_tests/` (PSP103)

- **Source path**: vsource, isource only
  - Time-varying behavior (pulse, sine, DC)
  - Handled separately with vectorized stamping

## Migration Status: COMPLETE

All numpy/scipy dependencies have been removed from core computation:
- `jax_spice/analysis/sparse.py` - uses pure JAX for COO→CSR conversion via `jax.ops.segment_sum`
- `jax_spice/benchmarks/runner.py` - returns `jax.Array` types, uses `jnp` throughout
- `jax_spice/devices/openvaf_device.py` - uses `jax.Array` for batched inputs

The codebase is now pure JAX in all simulation paths.

## OpenVAF/VACASK PSP103 Parameter Alignment

### Overview

When compiling Verilog-A models with OpenVAF, the generated Python code has a specific input array structure with different parameter kinds:
- `param`: Model parameters from netlist (e.g., TOX, NSUBO, W, L)
- `voltage`: Computed at runtime from node voltages
- `hidden_state`: Intermediate computed values that OpenVAF expects to be pre-initialized
- `param_given`: Flags indicating which parameters were explicitly provided
- `temperature`: Device operating temperature
- `sysfun`: System functions like `mfactor`

### Key Discovery: hidden_state Parameters

OpenVAF's PSP103 model has ~2400 parameters, with ~1500 being `hidden_state` type. These are NOT just placeholders - they are intermediate values computed from model parameters that the model evaluation code expects to be already calculated.

**Three categories of hidden_state params:**

1. **Instance-level computed (`*_i` suffix)**: Clipped/bounded versions of process params
   - `TOX_i = CLIP_LOW(TOX_p, 1e-10)` - oxide thickness
   - `NEFF_i = CLIP_BOTH(NEFF_p, 1e20, 1e26)` - effective doping
   - `BETN_i = CLIP_LOW(BETN_p, 0)` - mobility factor

2. **Process-level computed (`*_p` suffix)**: Binning-adjusted values
   - `TOX_p = TOXO` when SWGEO=1
   - `NEFF_p = NSUB * (1 - FOL1*iLE - FOL2*iLE^2)`
   - `BETN_p = UO * WE / (GPE * LE) * GWE`

3. **Geometry intermediates**: Scaling factors computed from L, W
   - `GPE`, `GWE` - effective channel length/width
   - `iLE`, `iWE`, `iLEWE` - inverse geometry factors
   - `CoxPrime = EPSOX / TOX_i` - gate capacitance per area

### Parameter Lookup Pattern

The model card uses `*o` suffix for "global" parameters (e.g., `toxo`, `epsroxo`, `nsubo`).
When looking up values for `_i` suffix params:

```python
# Pattern for _i params
if name_lower.endswith('_i'):
    base_name = name_lower[:-2]  # e.g., 'tox' from 'tox_i'
    # Try direct match first
    base_val = params.get(base_name, None)
    if base_val is None:
        # Try with 'o' suffix (PSP103 pattern)
        base_name_o = base_name + 'o'  # e.g., 'toxo'
        base_val = params.get(base_name_o, None)
```

### NOI Node Handling

PSP103 has an internal NOI (noise correlation) node with extremely high conductance (1/mig where mig=1e-40):
- This creates G = 1e40 to ground
- If V(NOI) ≠ 0, residual = 1e40 * V(NOI) - massive numerical explosion
- **Solution**: Initialize NOI nodes to 0V and mask their residuals during NR convergence checking

```python
# NOI is internal node4 in PSP103
if 'node4' in internal_nodes:
    noi_idx = internal_nodes['node4']
    V = V.at[noi_idx].set(0.0)  # Initialize to 0V

# Mask NOI residuals in convergence check
residual_mask = jnp.ones(n_unknowns, dtype=jnp.bool_)
residual_mask = residual_mask.at[noi_indices - 1].set(False)
f_masked = jnp.where(residual_mask, f, 0.0)
max_f = jnp.max(jnp.abs(f_masked))
```

### VACASK vs JAX-SPICE Alignment Status

| Benchmark | Node Count Match | DC OP Match | Notes |
|-----------|------------------|-------------|-------|
| rc        | ✅ | ✅ | Simple RC circuit |
| graetz    | ✅ | ✅ | Diode bridge rectifier |
| ring      | ✅ (47 nodes) | ❌ (0.55V vs 0.66V) | PSP103 hidden_state params |
| c6288     | ✅ (~5k after collapse) | TBD | Large multiplier circuit |

### Remaining Work

The Ring benchmark DC operating point differs because many hidden_state params are still uninitialized or have wrong default values. Key areas:

1. **Surface potential params**: `eta_mu`, `E_eff0`, `qq` - affect channel charge
2. **Temperature factors**: `tmpx`, `temp0`, `temp00` - affect temperature scaling
3. **Stress/LOD params**: `Kstressu0`, `Kstressvth0` - affect mobility and threshold
