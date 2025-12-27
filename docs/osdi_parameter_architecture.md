# OSDI Parameter Architecture

This document describes how OSDI (Open Simulator Device Interface) handles parameters, the distinction between different parameter types, and how this relates to JAX-SPICE's parameter handling through `openvaf_jax`.

## 1. OSDI Parameter Kinds

OSDI defines three parameter kinds in the descriptor:

```c
// From OSDI header
#define PARA_KIND_MODEL 0   // Model-level parameters (shared across instances)
#define PARA_KIND_INST  1   // Instance-level parameters (per-device)
#define PARA_KIND_OPVAR 2   // Output variables (computed, read-only)
```

### Model Parameters (`PARA_KIND_MODEL`)

- Shared across all instances of a model (e.g., all psp103n devices)
- Typically process parameters, physical constants
- Examples: `toxe`, `vth0`, `k1`, `k2`
- Stored in model data structure at `descriptor->model_size` bytes

### Instance Parameters (`PARA_KIND_INST`)

- Per-device parameters
- Geometry, placement, multipliers
- Examples: `w`, `l`, `nf`, `mult`, `sa`, `sb`
- Stored in instance data structure at `descriptor->instance_size` bytes

### Output Variables (`PARA_KIND_OPVAR`)

- Computed during evaluation
- Read-only to the simulator
- Examples: `ids`, `gm`, `gds`, `vth`
- Extracted after eval for probing/measurement

## 2. OSDI Internal States (NOT "hidden_state")

**Critical distinction:** OSDI has a concept of "internal states" that is separate from parameters. These are used for:

- `$limit()` function state (voltage limiting for convergence)
- `ddt()` integration state
- Other time-dependent constructs

### How Internal States Work

```c
typedef struct OsdiSimInfo {
    // ...
    double *prev_state;   // Previous internal state array
    double *next_state;   // Next internal state array
    // ...
} OsdiSimInfo;
```

During evaluation:
1. Model reads from `prev_state` (previous timestep/iteration)
2. Model writes to `next_state` (current computation)
3. After convergence, simulator swaps: `prev_state = next_state`

These are for convergence aids and time integration, NOT for geometry calculations.

## 3. What openvaf_py Calls "hidden_state"

In `openvaf_py` (our Rust bindings for OpenVAF's MIR interpreter), `hidden_state` is a parameter kind that refers to something entirely different from OSDI internal states.

### openvaf_py's hidden_state

These are values computed by the model's **init function** and cached for use in the **eval function**:

```rust
// From openvaf_py/src/lib.rs
pub enum ParamKind {
    Param,        // Regular parameters
    DepBreak,     // Dependency break points
    ParamHidden,  // Hidden model params (not in OSDI)
    Voltage,      // Node voltages
    Current,      // Branch currents
    HiddenState,  // <-- Values computed by init, cached for eval
    StateLim,     // Limit function state
    Temperature,  // Temperature
}
```

### Examples of hidden_state Values

For PSP103, the init function computes ~462 cached values:

| Value | Computation | Purpose |
|-------|-------------|---------|
| `invNF` | `1.0 / NF` | Inverse number of fingers |
| `LE`, `WE` | Effective length/width after binning | Geometry |
| `iL`, `iW` | `1.0 / L`, `1.0 / W` | Inverse geometry |
| `iLE`, `iWE` | `1.0 / LE`, `1.0 / WE` | Inverse effective geometry |
| `inv_phit` | `1.0 / (kT/q)` | Inverse thermal voltage |
| `inv_phita` | Adjusted thermal voltage inverse | Temperature-adjusted |
| `chnl_type` | 1 or -1 | NMOS vs PMOS |
| `lcinv2` | `1.0 / (LC * LC)` | Inverse LC squared |

### Why These Exist

OpenVAF optimizes Verilog-A code by:
1. Running "init" phase once per parameter change
2. Caching computed values that don't depend on voltages
3. Passing cached values to "eval" phase for each iteration

This avoids recomputing expensive expressions every NR iteration.

## 4. The Translation Pipeline

### OpenVAF Compilation

```
Verilog-A source
    ↓
OpenVAF Compiler
    ↓
OSDI shared library (.so/.dylib)
```

The OSDI library contains native code with the init/eval split built-in.

### openvaf_jax Translation

```
Verilog-A source
    ↓
OpenVAF Compiler (partial - to MIR)
    ↓
openvaf_py (Rust MIR interpreter)
    ↓
openvaf_jax.py (MIR → JAX translator)
    ↓
JAX Python functions
```

The JAX translator must replicate the init/eval split:

1. **Init phase**: Computes cached values from parameters
2. **Cache mapping**: Passes cached values to eval via variable assignments
3. **Eval phase**: Uses cached values + voltages to compute currents/Jacobians

### Cache Mapping in Generated Code

```python
# Example from generated JAX code
def eval_func(voltages, params, cache_values):
    # Cache values assigned from init results
    v84982 = init_v12340   # invNF
    v84983 = init_v12341   # LE
    v84984 = init_v12342   # WE
    # ...

    # Eval uses these cached values
    ids = v84982 * some_expression(voltages)
```

## 5. JAX-SPICE Parameter Handling

### The Problem

JAX-SPICE's `runner.py` batches devices for GPU efficiency. The vectorized path must set:

1. **Regular parameters** (`l`, `w`, `nf`, etc.) - from netlist
2. **Hidden_state parameters** - computed geometry values

### What runner.py Does

```python
# From runner.py lines 650-745

# Set regular parameters
l_vals = np.array([float(p.get('l', 1e-6)) for p in all_dev_params])
w_vals = np.array([float(p.get('w', 1e-6)) for p in all_dev_params])
nf_vals = np.maximum(np.array([float(p.get('nf', 1.0)) for p in all_dev_params]), 1.0)

# Compute and set hidden_state params
we_vals = np.maximum(w_vals / nf_vals, 1e-9)  # Effective width
le_vals = np.maximum(l_vals, 1e-9)            # Effective length

if 'invnf' in hidden_to_col:
    all_inputs[:, hidden_to_col['invnf']] = 1.0 / nf_vals
if 'le' in hidden_to_col:
    all_inputs[:, hidden_to_col['le']] = le_vals
# ... ~40 more hidden_state params
```

### Current Issues

1. **1705 hidden_state params** in PSP103, only ~40 handled
2. **Many values still 0**, causing division by zero
3. **NaN propagation** through model calculations
4. **Should JAX code compute these?** The generated JAX init function should compute them, but the vectorized runner bypasses this

## 6. Comparison: VACASK vs JAX-SPICE

| Aspect | VACASK | JAX-SPICE |
|--------|--------|-----------|
| Init execution | Native code, once per setup | Should use generated JAX init |
| Cache passing | Automatic in native code | Manual via cache_mapping |
| Vectorization | Per-device sequential | Batched vmap over devices |
| Hidden_state | Computed by init function | Currently computed in runner.py |
| GPU support | None (CPU only) | Full JAX GPU acceleration |

## 7. Recommended Architecture

### Current (Problematic)

```
runner.py manually sets ~40 hidden_state params
    ↓
Generated eval function expects all 1705 values
    ↓
Missing values = 0 → Division by zero → NaN
```

### Proposed Solution

```
runner.py sets only regular params (l, w, nf, etc.)
    ↓
Generated init function computes all hidden_state
    ↓
Cache passed to eval function
    ↓
No manual geometry computation needed
```

This requires:
1. `openvaf_jax.py` to generate a proper init function
2. `runner.py` to call init before eval
3. Batched init execution via vmap

## 8. Analysis: Does openvaf-py's hidden_state Model Make Sense?

### The Architectural Issue

openvaf-py exposes `hidden_state` as a parameter kind, but this is architecturally confused:

1. **Eval function's hidden_state params are UNUSED**
   - `insert_var_init` (in `sim_back/src/state.rs`) replaces all HiddenState references with their computed values
   - After this pass, HiddenState params have 0 operand references in the eval function
   - They exist in the param list but aren't actually read by any instruction

2. **Init function's hidden_state params ARE used**
   - These represent variable initialization expressions
   - The init function computes them and they become cache values

3. **The cache_mapping is the correct mechanism**
   - `compiled.init.cached_vals` maps init outputs → eval inputs
   - This is what should be used to pass values from init to eval

### What OpenVAF Actually Does (in sim_back/src/init.rs)

```rust
// From Initialization::new()
while let Some(bb) = blocks.next(&builder.func.layout) {
    // Copy instructions that are NOT op dependent to instance setup MIR
    // and zap them in module MIR.
    builder.split_block(bb);
}
```

The init/eval split:
1. Instructions NOT operating-point dependent → moved to init function
2. Instructions operating-point dependent → kept in eval function
3. Values computed in init that eval needs → become cache slots

### The Hidden State Flow

```
Verilog-A variable 'real LE;'
    ↓
ParamKind::HiddenState(LE) created in MIR
    ↓
insert_var_init() replaces HiddenState with actual computation
    ↓
Initialization::new() moves computation to init function
    ↓
Value becomes a cache slot in cached_vals
    ↓
cache_mapping connects init output → eval input param
```

After this flow, **HiddenState is no longer needed** - it's been replaced by the cache system.

### Why openvaf-py's Model is Problematic

1. **Misleading API**: Exposing hidden_state in eval params suggests they're inputs, but they're unused

2. **Fragile workarounds**: openvaf_jax.py uses value-number matching (`_build_hidden_state_assignments`) to work around the architecture:
   ```python
   # If eval uses vN for a hidden_state param and init computed init_vN,
   # add assignment vN = init_vN
   ```
   This relies on OpenVAF using same value numbers, which isn't guaranteed.

3. **Manual computation in runner.py**: Because the workarounds fail for batched execution, runner.py manually computes ~40 hidden_state values, missing 1600+.

### The Correct Solution

**Short term**: runner.py should call the init function (via vmap) for each device, using the output via cache_mapping.

**Long term**: openvaf-py should:
1. Not expose hidden_state as eval params (or mark them as "internal/unused")
2. Provide a clear API: `init(params) → cache`, `eval(voltages, cache) → residuals`
3. Make both functions easily vmappable for batched GPU execution

### Summary

| Aspect | Current | Correct |
|--------|---------|---------|
| hidden_state in eval | Exposed as params | Should be internal |
| Init function | Generated but not used in batched mode | Should be called via vmap |
| Cache values | Workaround via value-number matching | Proper cache_mapping API |
| Runner computation | Manual ~40 params | None needed |

## 9. Related Documentation

- [VACASK OSDI Inputs](vacask_osdi_inputs.md) - How VACASK interfaces with OSDI
- [Architecture Overview](architecture_overview.md) - Overall JAX-SPICE architecture
- [OpenVAF CLAUDE.md](../openvaf-py/vendor/OpenVAF/CLAUDE.md) - OpenVAF compiler architecture

## 9. File References

| Purpose | File |
|---------|------|
| OSDI header definitions | `openvaf-py/vendor/OpenVAF/openvaf/osdi/header/osdi.h` |
| Rust MIR interpreter | `openvaf-py/src/lib.rs` |
| JAX code translator | `openvaf-py/openvaf_jax.py` |
| Benchmark runner | `jax_spice/benchmarks/runner.py` |
| VACASK OSDI interface | `vendor/VACASK/lib/osdiinstance.cpp` |
