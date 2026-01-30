# Transient Analysis Options

JAX-SPICE supports VACASK-compatible options for controlling transient analysis behavior.
Options can be specified in the netlist or passed programmatically via `AdaptiveConfig`.

## Quick Start

Options in netlist are automatically used:

```spice
// In your .sim file
options tran_lteratio=1.5 nr_convtol=1.0 reltol=1e-4
analysis top tran step=1p stop=10n
```

```python
from jax_spice import CircuitEngine
from jax_spice.analysis.transient import FullMNAStrategy

engine = CircuitEngine('circuit.sim')
engine.parse()

# Strategy automatically uses netlist options
strategy = FullMNAStrategy(engine, use_sparse=True)
# strategy.config.lte_ratio == 1.5  (from tran_lteratio)
```

## Supported Options

### LTE (Local Truncation Error) Control

| Netlist Option | AdaptiveConfig | Default | Description |
|----------------|----------------|---------|-------------|
| `tran_lteratio` | `lte_ratio` | 3.5 | LTE tolerance multiplier. Higher values allow larger LTE before reducing timestep. |
| `tran_redofactor` | `redo_factor` | 2.5 | Rejection threshold. If `current_dt / new_dt > redo_factor`, the step is rejected and retried. |

### Tolerances

| Netlist Option | AdaptiveConfig | Default | Description |
|----------------|----------------|---------|-------------|
| `reltol` | `reltol` | 1e-3 | Relative tolerance for LTE and NR convergence. |
| `abstol` | `abstol` | 1e-12 | Absolute tolerance (current) for LTE and NR convergence. |
| `nr_convtol` | `nr_convtol` | 1.0 | NR convergence tolerance factor (multiplier on abstol). Values < 1.0 are stricter. |

### Timestep Control

| Netlist Option | AdaptiveConfig | Default | Description |
|----------------|----------------|---------|-------------|
| `tran_method` | `integration_method` | `trap` | Integration method: `be` (backward Euler), `trap` (trapezoidal), `gear2` (Gear order 2). |
| `tran_fs` | `tran_fs` | 0.25 | Initial timestep scale factor. Scales user-specified `step` to start smaller. |
| `tran_minpts` | `tran_minpts` | 50 | Minimum output points. Automatically caps `max_dt` to `t_stop / tran_minpts`. |
| `maxstep` | `max_dt` | inf | Maximum allowed timestep (from tran analysis line). |

### Convergence Aids

| Netlist Option | AdaptiveConfig | Default | Description |
|----------------|----------------|---------|-------------|
| `tran_gshunt` | `gshunt_init` | 0.0 | Initial conductance to ground for all nodes. Helps with singular matrices in UIC mode. |
| - | `gshunt_steps` | 5 | Steps to ramp gshunt from `gshunt_init` to `gshunt_target`. |
| - | `gshunt_target` | 0.0 | Final gshunt value after ramping. |

### Debug Options

| AdaptiveConfig | Default | Description |
|----------------|---------|-------------|
| `debug_steps` | False | Print per-step info (time, dt, NR iterations, LTE). |
| `debug_lte` | False | Print detailed LTE debug info (top contributing nodes). |
| `progress_interval` | 100 | Report progress every N steps. Set to 0 to disable. |

### Internal Parameters

| AdaptiveConfig | Default | Description |
|----------------|---------|-------------|
| `min_dt` | 1e-18 | Minimum allowed timestep (seconds). |
| `max_order` | 2 | Maximum polynomial order for predictor (0=constant, 1=linear, 2=quadratic). |
| `grow_factor` | 2.0 | Maximum factor by which timestep can grow per step. |
| `warmup_steps` | 2 | Fixed-dt steps before enabling LTE control (need history for predictor). |
| `max_consecutive_rejects` | 5 | Force accept after this many consecutive LTE rejects. |

## Default Behavior

When no explicit `AdaptiveConfig` is passed to `FullMNAStrategy`:

1. **Netlist options are parsed** during `engine.parse()` and stored in `engine.analysis_params`
2. **Strategy builds config** from `analysis_params` automatically
3. **Defaults apply** for any options not specified in the netlist

Example with defaults:
```python
# No options in netlist - uses all defaults
strategy = FullMNAStrategy(engine)
# lte_ratio=3.5, redo_factor=2.5, reltol=1e-3, etc.
```

## Programmatic Override

Explicit `AdaptiveConfig` completely overrides netlist options:

```python
from jax_spice.analysis.transient import AdaptiveConfig, FullMNAStrategy

# Override specific options
config = AdaptiveConfig(
    lte_ratio=2.0,      # Stricter than default 3.5
    tran_fs=0.1,        # Even smaller initial timesteps
    debug_steps=True,   # Enable per-step debugging
)
strategy = FullMNAStrategy(engine, config=config)
```

## VACASK Compatibility

### Supported (Functional)

| VACASK Option | Status | Notes |
|---------------|--------|-------|
| `tran_lteratio` | ✅ Full | Maps to `lte_ratio` |
| `tran_redofactor` | ✅ Full | Maps to `redo_factor` |
| `tran_method` | ✅ Full | Supports be, trap, gear2 |
| `tran_fs` | ✅ Full | Initial timestep scaling |
| `tran_minpts` | ✅ Full | Auto-computes max_dt |
| `reltol` | ✅ Full | Relative tolerance |
| `abstol` | ✅ Full | Absolute tolerance |
| `nr_convtol` | ✅ Full | NR tolerance factor |

### Not Yet Implemented

| VACASK Option | Status | Notes |
|---------------|--------|-------|
| `nr_bypass` | ❌ | Device bypass for unchanged nodes |
| `nr_bypasstol` | ❌ | Bypass tolerance threshold |
| `nr_contbypass` | ❌ | Bypass in continuation mode |
| `tran_predictor` | ❌ | We always use predictor |
| `tran_maxord` | Partial | Fixed at order 2 |
| `tran_itl` | ❌ | Max NR iterations per step |
| `tran_ft` | ❌ | Timestep cut factor on NR failure |
| `tran_xmu` | ❌ | Trap/Euler mixture parameter |

## Example: C6288 Benchmark

The c6288 multiplier benchmark uses these options:

```spice
options nr_convtol=1 tran_lteratio=1.5
analysis top tran step=2p stop=2n icmode="uic"
```

This sets:
- `nr_convtol=1.0` - Standard NR tolerance
- `tran_lteratio=1.5` - Tighter LTE control than default (3.5)
- `icmode="uic"` - Use Initial Conditions (skip DC operating point)

```python
engine = CircuitEngine('c6288.sim')
engine.parse()

# These options are automatically applied
strategy = FullMNAStrategy(engine, use_sparse=True)
assert strategy.config.lte_ratio == 1.5
assert strategy.config.nr_convtol == 1.0
```
