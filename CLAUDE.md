# JAX-SPICE Development Guidelines

## GPU Optimization

**CRITICAL**: All simulation hot paths should be optimized for GPU execution, minimizing CPU-GPU context switching and data transfer.

### Design Principles

1. **GPU-first simulation loops**: Use `lax.scan` or `lax.while_loop` for time-stepping
2. **Batched device evaluation**: Use `vmap` for parallel device evaluation
3. **Minimize host transfers**: Keep arrays on device throughout simulation
4. **JIT everything**: Ensure simulation functions are JIT-compilable

### Code Quality

- Avoid duplication - unify critical paths where possible
- Keep files under ~70KB (~20,000 tokens) - split large modules
- Regularly check for opportunities to simplify without impacting functionality
- *NO API/ABI Stability* - this is still v0.x. Do not keep old code paths, remove them agressively.
- Comprehendability - This code does some pretty complex maths and operations, try to keep the code as understandable as possible. 
  Any data structures should be well defined - dict keys should have meaning, arrays should have their index defined meaningfully.

### JAX vs NumPy/SciPy

Use JAX (`jnp`) for simulation hot paths that run on GPU. NumPy/SciPy are acceptable for:
- I/O and file parsing
- One-time setup/preprocessing
- Test utilities and validation
- Optional CPU-only solver backends (e.g., UMFPACK)

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
  - Complex models from `vendor/OpenVAF/integration_tests/` (PSP103)

- **Source path**: vsource, isource only
  - Time-varying behavior (pulse, sine, DC)
  - Handled separately with vectorized stamping

## Solver Architecture

The simulation hot paths use JAX for GPU acceleration:
- `jax_spice/analysis/sparse.py` - JAX BCOO/BCSR sparse matrix operations
- `jax_spice/analysis/solver.py` - Newton-Raphson with `lax.while_loop`
- `jax_spice/analysis/transient/scan.py` - Time-stepping with `lax.scan`

NumPy/SciPy are used appropriately for:
- File I/O (`rawfile.py`, `prn_reader.py`)
- Optional UMFPACK solver backend for CPU
- Test utilities and waveform comparison

## OpenVAF/VACASK PSP103 Parameter Alignment

### Overview

When compiling Verilog-A models with OpenVAF, the generated Python code has a specific input array structure with different parameter kinds:
- `param`: Model parameters from netlist (e.g., TOX, NSUBO, W, L)
- `voltage`: Computed at runtime from node voltages
- `hidden_state`: Intermediate computed values that OpenVAF expects to be pre-initialized
- `param_given`: Flags indicating which parameters were explicitly provided
- `temperature`: Device operating temperature
- `sysfun`: System functions like `mfactor`

### Key Discovery: OpenVAF Inlines hidden_state Parameters

**IMPORTANT**: OpenVAF's optimizer aggressively inlines hidden_state computations.
Analysis of MIR (Mid-level IR) shows that hidden_state params are **NEVER actually
used** in the eval function - they're all inlined into the cache values computed by init.

**Tested models (all show 0% hidden_state usage in eval MIR):**

| Model     | hidden_state params | Used in eval | Regular params | Used in eval |
|-----------|--------------------:|-------------:|---------------:|-------------:|
| resistor  |                   1 |            0 |              2 |            1 |
| capacitor |                   2 |            0 |              1 |            1 |
| diode     |                  16 |            0 |             13 |           13 |
| bsim4     |                2330 |            0 |            893 |            2 |
| psp103    |                1705 |            0 |            840 |            0 |

**What eval actually uses (PSP103 example):**
- 13 voltage params (terminal voltages)
- ~462 cache values (computed by init)
- 1 mfactor (system function)

**Implications:**
- Setting hidden_state to 0.0 is SAFE - those values are never read
- The `*_i` suffix params (like `TOX_i`, `NEFF_i`) are inlined into cache
- Debug warnings about "unmapped hidden_state" are informational only
- If simulation results differ from VACASK, look elsewhere (voltage mapping,
  cache computation, or solver issues) - NOT hidden_state initialization

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
| ring      | ✅ (47 nodes) | ~34% RMS error | Investigation ongoing |
| c6288     | ✅ (~5k after collapse) | TBD | Large multiplier circuit |

### Remaining Work

The Ring benchmark shows ~34% RMS error vs VACASK. Since hidden_state params are
inlined (see above), the issue must be elsewhere. Possible causes to investigate:

1. **Cache values**: Init computes ~462 cache values - verify these match VACASK
2. **Inf values in cache**: Found 2 inf values at indices 662, 741 - may cause issues
3. **Voltage mapping**: Check that terminal voltages are correctly extracted
4. **Transient integration**: Verify ddt() operator and charge integration
