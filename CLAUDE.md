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

### Performance Profile

JAX-SPICE has a per-step fixed overhead of ~10-15 us from adaptive timestep
machinery, `jnp.where` branching, vmap batching, and COO matrix assembly. This
overhead dominates for small circuits (6-11x slower than VACASK on CPU) but
becomes negligible for large circuits (c6288: 1.2x on CPU, **2.9x faster on GPU**).

When optimizing simulation performance:
- **Don't optimize for small-circuit CPU speed** unless it also helps GPU performance
- **Focus on reducing per-NR-iteration cost** (Jacobian build, linear solve) — these scale with circuit size
- **GPU threshold is 500 nodes** (`gpu_backend.py`) — circuits below this auto-route to CPU
- See `docs/performance_analysis.md` for the full overhead breakdown

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

## OSDI vs openvaf_jax Jacobian Format Differences

**CRITICAL**: OSDI and openvaf_jax return Jacobians in different formats. Direct array comparison will fail.

### Format Comparison

| Aspect | OSDI (osdi_py) | openvaf_jax |
|--------|----------------|-------------|
| **Ordering** | Column-major (Fortran-style) | Row-major (C-style) |
| **Sparsity** | Sparse (only `has_resist=True` entries) | Dense (all N×N entries) |
| **Array length** | Variable (sum of has_resist flags) | Fixed (N×N where N=num_nodes) |

### Example: 4-terminal device (D, G, S, B)

For a 4-terminal device with 2 internal nodes (6 total), OSDI returns ~20 entries while JAX returns 32 (for resistive Jacobian).

**OSDI sparse indices** (has_resist=True only):
```
[0, 1, 2, 3, 4, 6, 7, 8, 9, 11, 16, 17, 18, 19]
```

**JAX dense indices** (same physical entries):
```
[0, 1, 2, 3, 4, 10, 11, 12, 13, 15, 20, 21, 26, 31]
```

### Mapping Between Formats

Use `jax_spice.debug.jacobian` helpers for comparison:

```python
from jax_spice.debug.jacobian import (
    osdi_to_dense_jacobian,
    compare_jacobians,
)

# Convert OSDI sparse to dense for comparison
osdi_dense = osdi_to_dense_jacobian(osdi_jac, n_nodes, jacobian_keys)

# Or use compare helper directly
passed, report = compare_jacobians(osdi_jac, jax_jac, n_nodes, jacobian_keys)
```

### Why This Matters

When debugging openvaf_jax vs OSDI:
1. **Don't compare arrays directly** - indices don't match
2. **Non-zero counts should match** - both should have same sparsity pattern
3. **Value differences ~1%** are real computational differences
4. **Zeros at wrong positions** indicate PHI node or control flow issues

---

## MIR SSA Debugging

When debugging openvaf_jax discrepancies with OSDI, use the MIR CFG analysis script to trace
SSA value dependencies and PHI node resolution. This is especially useful for complex models
with conditional control flow (NMOS/PMOS branches, etc.).

### MIR Analysis Script

```bash
# Generate DOT and analyze CFG for a model
uv run scripts/analyze_mir_cfg.py vendor/OpenVAF/integration_tests/EKV/ekv.va --func eval

# Analyze existing DOT file
uv run scripts/analyze_mir_cfg.py --dot /tmp/model_eval.dot

# Find all PHI nodes (merge points in control flow)
uv run scripts/analyze_mir_cfg.py model.va --func eval --find-phis

# Trace paths to a specific block
uv run scripts/analyze_mir_cfg.py model.va --func eval --target block4654

# Trace dependencies of an SSA value through PHI nodes
uv run scripts/analyze_mir_cfg.py model.va --func eval --trace-value v12345

# Analyze a specific block with PHIs
uv run scripts/analyze_mir_cfg.py model.va --func eval --analyze-block block4654

# List all branch points (T/F conditional branches)
uv run scripts/analyze_mir_cfg.py model.va --func eval --branches
```

### Prerequisites

The script requires `openvaf-viz` to generate DOT files from Verilog-A:
```bash
cd vendor/OpenVAF && cargo build --release -p openvaf-viz
```

### Common Debugging Patterns

1. **PHI node gets wrong value for one branch**: Use `--analyze-block` to see which
   predecessor provides which value. Check if one branch uses `v3` (constant 0.0).

2. **Value computed incorrectly**: Use `--trace-value` to find where it's defined
   and what PHI sources feed into it.

3. **Understanding control flow**: Use `--target` to see all paths to a block,
   with branch labels (T/F) shown.

4. **Finding NMOS/PMOS split**: Use `--branches` to list all conditional branches,
   then trace paths to find where device type selection occurs.

---

## Debug Tools (`jax_spice.debug`)

The `jax_spice.debug` module provides utilities for debugging OSDI vs JAX discrepancies.
See `docs/debug_tools.md` for comprehensive documentation.

### When to Use Debug Tools

| Symptom | Tool | First Step |
|---------|------|------------|
| JAX returns wrong current | `ModelComparator` | Compare at specific bias point |
| Jacobian mismatch | `compare_jacobians()` | Check format-aware comparison |
| Near-zero outputs | `MIRInspector` | Check PHI nodes with zero operand |
| Cache looks suspicious | `CacheAnalysis` | Look for inf/nan, VT values |
| NMOS/PMOS issues | `MIRInspector.find_type_param()` | Verify TYPE parameter location |
| Step rejection / LTE issues | `capture_step_trace()` | Run with debug_steps, inspect LTE norms |
| Convergence varies with t_stop | `convergence_sweep()` | Sweep multiple durations |
| VACASK step mismatch | `parse_vacask_debug_output()` | Compare step-by-step with VACASK |

### Quick Start

```python
from jax_spice.debug import quick_compare, ModelComparator, MIRInspector

# One-shot comparison
result = quick_compare(va_path, osdi_path, params, voltages)
print(result)

# Detailed investigation
comparator = ModelComparator(va_path, osdi_path, params)
result = comparator.compare_at_bias([0.5, 0.6, 0.0, 0.0])
cache = comparator.analyze_cache()
comparator.print_residual_table([0.5, 0.6, 0.0, 0.0])

# MIR inspection
inspector = MIRInspector(va_path)
inspector.print_mir_stats()
inspector.print_phi_summary('eval')
inspector.print_type_param_info()
```

### Debugging Workflow

1. **Initial comparison**: Use `quick_compare()` to see if outputs match
2. **Cache analysis**: If mismatch, check `analyze_cache()` for inf/nan/temperature issues
3. **MIR inspection**: If cache looks OK, check `print_phi_summary()` for PHI node issues
4. **CFG analysis**: Use `scripts/analyze_mir_cfg.py` to trace control flow

### Key Modules

| Module | Purpose |
|--------|---------|
| `model_comparison` | Compare OSDI vs JAX outputs (residuals, Jacobians, cache) |
| `mir_inspector` | Inspect MIR data (params, PHI nodes, constants) |
| `jacobian` | Format-aware Jacobian comparison (OSDI sparse vs JAX dense) |
| `mir_tracer` | Trace value flow through MIR |
| `param_analyzer` | Analyze parameter kinds and OSDI comparison |
| `mir_analysis` | CFG analysis with networkx (optional dependency) |
| `transient_diagnostics` | Runtime transient step analysis (LTE, NR, step acceptance) |

### Transient Step Debugging

For runtime transient issues (step rejection, LTE behaviour, NR convergence):

```python
from jax_spice.debug import capture_step_trace, convergence_sweep, parse_vacask_debug_output

# 1. Sweep t_stop to find where convergence degrades
results = convergence_sweep("graetz", [1e-3, 5e-3, 7e-3, 10e-3])

# 2. Capture full step trace for detailed analysis
records, summary = capture_step_trace("ring", use_sparse=True)

# 3. Compare with VACASK tran_debug=1 output
vacask_records = parse_vacask_debug_output(vacask_stdout)
```

See `docs/debug_tools.md` for the full transient debugging workflow.

### Common Issues and Solutions

1. **JAX returns near-zero current** (~1e-15): PHI node resolution issue in NMOS/PMOS branching.
   Check `inspector.find_phi_nodes_with_value('v3')` for PHIs with zero operand.

2. **Jacobian sparsity mismatch**: OSDI has N non-zeros, JAX has M << N.
   Branch not taken, computations skipped. Trace control flow.

3. **~1% current difference**: Usually temperature-related. Check `cache.temperature_related`
   for VT values (should be ~0.02585 at 300K).

---

## OpenVAF CallbackKind Handling

OpenVAF compiles Verilog-A system functions into CallbackKind callbacks. These are translated
in `openvaf_jax/codegen/instruction.py:_translate_call()`. Analysis across 87 VACASK/OpenVAF
models shows the following callback usage:

### Callback Summary

| CallbackKind | Uses | Models | Handling |
|--------------|------|--------|----------|
| `WhiteNoise` | 424 | 47 | Returns 0 (noise analysis not supported) |
| `CollapseHint` | 253 | 43 | No-op (node collapse at build time) |
| `NodeDerivative` | 178 | 30 | Returns 0 (partial derivatives not supported) |
| `Print` | 131 | 33 | `jax.debug.print` (disabled by default for JIT speed) |
| `FlickerNoise` | 72 | 46 | Returns 0 (noise analysis not supported) |
| `TimeDerivative` | 49 | 49 | Returns charge (transient handles dQ/dt) |
| `StoreLimit` | 47 | 13 | Passthrough (no pnjlim/fetlim algorithms) |
| `SimParamOpt` | 28 | 28 | Via `simparams` array with default fallback |
| `SetRetFlag` | 19 | 19 | No-op ($finish/$stop not supported) |
| `Analysis` | 14 | 14 | Via `simparams[$analysis_type]` |
| `LimDiscontinuity` | 13 | 13 | Ignored for DC analysis |
| `SimParam` | 2 | 2 | Via `simparams` array |

### Implementation Notes

**Limiting Functions (`$limit`):**
- `StoreLimit` and `BuiltinLimit` (pnjlim, fetlim) are used for Newton-Raphson convergence help
- Current implementation passes through the input voltage unchanged
- This may result in slightly different convergence behavior compared to OSDI

**Noise Functions:**
- `white_noise()`, `flicker_noise()`, `noise_table()` return 0.0
- Full noise analysis would require frequency-domain analysis

**$simparam:**
- Registered dynamically via `ctx.register_simparam(name)`
- Caller builds `simparams` array using `build_simparams(eval_meta, values)`
- See `jax_spice/__init__.py:build_simparams()` for helper

**$display/$strobe:**
- Disabled by default (`emit_debug_prints=False`)
- Enable via `InstructionTranslator(..., emit_debug_prints=True)`
- Warning: `jax.debug.print` causes slow JIT tracing
