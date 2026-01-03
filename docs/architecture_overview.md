# JAX-SPICE Architecture Overview

This document provides a high-level overview of JAX-SPICE's architecture for developers new to the codebase.

## Design Philosophy

JAX-SPICE is built on three core principles:

1. **Functional Device Models**: Devices are pure JAX functions compiled from Verilog-A
2. **Automatic Differentiation**: Jacobians computed via JAX autodiff, no explicit derivatives
3. **Vectorization**: Same-type devices evaluated in parallel via `jax.vmap`

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                           User Code                                  │
│   from jax_spice import CircuitEngine                               │
│   engine = CircuitEngine("circuit.sim")                             │
│   engine.parse()                                                     │
│   result = engine.run_transient(...)                                │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    CircuitEngine (analysis/engine.py)                │
│  ┌─────────────┐  ┌──────────────────┐  ┌───────────────────────┐  │
│  │ run_transient│ │     run_ac       │  │    run_noise         │  │
│  │ (lax.scan)  │  │ (AC analysis)    │  │  (noise analysis)    │  │
│  └─────────────┘  └──────────────────┘  └───────────────────────┘  │
│  ┌─────────────┐  ┌──────────────────┐  ┌───────────────────────┐  │
│  │ run_corners │  │   run_dcinc      │  │    run_hb            │  │
│  │ (PVT sweep) │  │ (transfer funcs) │  │ (harmonic balance)   │  │
│  └─────────────┘  └──────────────────┘  └───────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          Device Layer                                │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │              OpenVAF Compiled Verilog-A Models                │  │
│  │    resistor.va  capacitor.va  diode.va  psp103.va  ...       │  │
│  └───────────────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │              Built-in Sources (vsource.py)                    │  │
│  │    DC, Pulse, Sine, PWL voltage/current sources               │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                           JAX Runtime                                │
│     ┌──────────────┐  ┌─────────────┐  ┌────────────────────┐      │
│     │     JIT      │  │    vmap     │  │   Autodiff         │      │
│     │ Compilation  │  │ Batched     │  │ (Jacobians via     │      │
│     │ (lax.scan)   │  │ Device Eval │  │  jacfwd/jvp)       │      │
│     └──────────────┘  └─────────────┘  └────────────────────┘      │
└─────────────────────────────────────────────────────────────────────┘
```

## Data Flow: Circuit Parsing and Setup

```
1. Load Circuit File
   └── engine = CircuitEngine("circuit.sim")
       └── engine.parse()

2. Netlist Processing
   ├── Parse VACASK .sim file or SPICE netlist
   ├── Load device models (.osdi files via OpenVAF)
   ├── Build node map (node name → index)
   └── Flatten hierarchical instances

3. Device Compilation
   ├── Compile Verilog-A models with OpenVAF
   ├── Generate JAX-compatible device functions
   └── Batch devices by type for vmap evaluation

4. System Builder
   └── _make_gpu_resident_build_system_fn()
       ├── Creates JIT-compiled residual/Jacobian builder
       ├── Batches device evaluations via vmap
       └── Handles sparse vs dense matrix assembly
```

## Data Flow: Transient Analysis

```
1. Initial Conditions
   └── Run DC operating point via Newton-Raphson

2. Time Integration (lax.scan)
   │
   ├── For each time step t = dt, 2*dt, ..., t_stop:
   │   │
   │   ├── Update source waveforms (pulse, sine, PWL)
   │   │
   │   ├── Build system (residual f, Jacobian J)
   │   │   ├── Evaluate all devices via batched vmap
   │   │   ├── Stamp currents → residual vector
   │   │   └── Stamp conductances → Jacobian matrix
   │   │
   │   ├── Newton-Raphson iteration (lax.while_loop):
   │   │   ├── Solve: delta_V = solve(J, -f)
   │   │   │   ├── Dense: jax.scipy.linalg.solve()
   │   │   │   └── Sparse: jax.experimental.sparse.linalg.spsolve()
   │   │   ├── Update: V = V + delta_V
   │   │   └── Check: max(|f|) < abstol?
   │   │
   │   └── Store solution: V[t] appended to trajectory
   │
   └── Return TransientResult(times, voltages)

3. GPU Efficiency
   └── lax.scan enables full GPU execution without Python callbacks
```

## Key Classes

### CircuitEngine (`analysis/engine.py`)

The central class that manages circuit parsing, device compilation, and analysis.

```python
class CircuitEngine:
    # Core data
    circuit_file: str                  # Path to .sim or SPICE file
    node_map: Dict[str, int]           # Node name → index
    models: Dict[str, CompiledModel]   # OpenVAF compiled models
    device_data: Dict[str, DeviceInfo] # Device instances and parameters

    # Parsing
    def parse() -> None                # Parse netlist, compile models

    # Analysis methods
    def run_transient(t_stop, dt, ...) -> TransientResult
    def run_ac(fstart, fstop, ...) -> ACResult
    def run_noise(fstart, fstop, ...) -> NoiseResult
    def run_corners(corners) -> List[CornerResult]
    def run_dcinc() -> DCINCResult
    def run_dcxf(input_source, output_node) -> DCXFResult
    def run_acxf(input_source, output_node) -> ACXFResult
    def run_hb(...) -> HBResult

    # Internal system building
    def _make_gpu_resident_build_system_fn() -> Callable
```

### TransientResult

The output of transient analysis.

```python
@dataclass
class TransientResult:
    times: Array                       # Shape: (n_steps,) time points
    voltages: Dict[str, Array]         # node_name → voltage array
    currents: Dict[str, Array]         # Optional: device currents
    converged: bool                    # True if all steps converged
```

### OpenVAF Device Interface

Devices are compiled from Verilog-A and evaluated in batches:

```python
# OpenVAF compiles Verilog-A to JAX-compatible functions
model = compile_va("resistor.va")

# Device evaluation function signature (simplified):
def device_fn(
    voltages: Array,     # Terminal voltages [n_devices, n_terminals]
    params: Array,       # Device parameters [n_devices, n_params]
    temperature: float,  # Operating temperature
) -> Tuple[Array, Array]:
    # Returns (currents, conductances) for MNA stamping
    ...

# Batched evaluation via vmap
currents, G = jax.vmap(device_fn)(V_terminals, params_batch, temp)
```

## Device Model Interface

### OpenVAF Compilation Pipeline

All devices are compiled from Verilog-A source:

```
resistor.va  →  OpenVAF Compiler  →  MIR/OSDI  →  JAX Function
```

Example Verilog-A source (`resistor.va`):
```verilog
module resistor(p, n);
    inout p, n;
    electrical p, n;
    parameter real r = 1k;
    parameter real tc1 = 0.0;
    parameter real tc2 = 0.0;

    analog begin
        I(p, n) <+ V(p, n) / r * (1 + tc1*dT + tc2*dT*dT);
    end
endmodule
```

OpenVAF compiles this to a pure JAX function that:
- Takes terminal voltages and parameters as input
- Returns currents and conductance matrix
- Is automatically differentiable for Jacobian computation
- Can be batched with `jax.vmap` for parallel evaluation

### Why Verilog-A + OpenVAF?

1. **PDK Compatibility**: Use production models (PSP103, BSIM4) directly
2. **Standardization**: Industry-standard compact model format
3. **Validation**: Models tested against commercial simulators
4. **Maintainability**: One source for all backends (JAX, VACASK, ngspice)

## Sparse Matrix Support

For large circuits (>1000 nodes), JAX-SPICE uses JAX's native sparse formats:

```python
from jax.experimental.sparse import BCOO, BCSR
from jax.experimental.sparse.linalg import spsolve

# Build sparse Jacobian from COO triplets
def build_sparse_jacobian(rows, cols, values, shape):
    # Use pure JAX for COO→CSR conversion
    data, indices, indptr = build_csr_arrays(rows, cols, values, shape)
    return data, indices, indptr

# Solve sparse system
# JAX spsolve works on CPU and GPU (via cuSOLVER)
delta_V = spsolve(data, indices, indptr, -residual, tol=0)
```

### Sparse Formats

| Format | Usage | Notes |
|--------|-------|-------|
| BCOO | Matrix construction | JAX native COO, efficient for building |
| BCSR | Linear solve | CSR required by spsolve |

### When Sparse is Used

| Circuit Size | Solver | Reason |
|--------------|--------|--------|
| < 1000 nodes | Dense | Lower overhead, `jax.scipy.linalg.solve()` |
| ≥ 1000 nodes | Sparse | Memory efficiency, `spsolve()` |

The switch is controlled by `use_sparse=True` in analysis methods:

```python
result = engine.run_transient(t_stop=1e-6, dt=1e-9, use_sparse=True)
```

## OpenVAF Integration

### Compilation Pipeline

```
Verilog-A (.va)
      │
      ▼
OpenVAF Compiler
      │
      ▼
MIR (Mid-level IR)
      │
      ▼
openvaf_jax Translator
      │
      ▼
JAX Function
```

### VerilogADevice Wrapper

```python
class VerilogADevice:
    def __init__(self, compiled_model, params):
        self.eval_fn = openvaf_jax.translate(compiled_model)
        self.params = params
        self.n_internal = compiled_model.n_internal_nodes

    def evaluate(self, V, params, context):
        # Call the JAX-translated function
        outputs = self.eval_fn(V, params)

        # Extract currents and conductances from outputs
        return DeviceStamps(
            currents=outputs.currents,
            conductances=outputs.jacobian
        )
```

## Performance Optimization

### JIT Compilation

```python
@jax.jit
def newton_step(V, system, context):
    residual, J = system.build_jacobian_and_residual(V, context)
    delta_V = jnp.linalg.solve(J, -residual)
    return V + delta_V
```

First call: ~1-5 seconds (compilation)
Subsequent calls: ~1-20 ms

### Vectorized Evaluation

```python
# Without vectorization (slow)
currents = []
for i in range(n_devices):
    I = device_fn(V[nodes[i]], params[i])
    currents.append(I)

# With vectorization (fast)
currents = jax.vmap(device_fn)(V[node_indices], batched_params)
```

Speedup: 10-100x depending on device count

### Batched Parameter Arrays

Pre-computing parameter arrays eliminates Python loops:

```python
# During build_device_groups()
params = {
    "r": jnp.array([d.params["r"] for d in resistors]),
    "tc1": jnp.array([d.params.get("tc1", 0) for d in resistors]),
}
# Now vmap can use these directly
```

## Convergence Strategies

### Source Stepping

Gradually ramps supply voltage from 0 to target:

```
VDD steps: 0.0 → 0.12 → 0.24 → ... → 1.2V

At each step:
1. Solve DC with current VDD
2. Use solution as initial guess for next step
```

Helps with digital circuits that have multiple stable states.

### GMIN Stepping

Adds decreasing conductance to ground at each node:

```
GMIN steps: 1e-3 → 1e-6 → 1e-9 → 1e-12 S

At each step:
1. Add GMIN*V to residual (pulls nodes toward 0)
2. Solve DC
3. Reduce GMIN and repeat
```

Prevents floating nodes and improves conditioning.

## File Reference

| File | Purpose |
|------|---------|
| `analysis/engine.py` | CircuitEngine - main simulation API, parsing, all analyses |
| `analysis/solver.py` | Newton-Raphson solver with `lax.while_loop` |
| `analysis/transient/` | Transient analysis (scan/loop strategies) |
| `analysis/ac.py` | AC small-signal analysis |
| `analysis/noise.py` | Noise analysis |
| `analysis/hb.py` | Harmonic balance analysis |
| `analysis/xfer.py` | Transfer function (DCINC, DCXF, ACXF) |
| `analysis/corners.py` | PVT corner analysis |
| `analysis/homotopy.py` | Convergence aids (GMIN, source stepping) |
| `analysis/sparse.py` | JAX sparse utilities (BCOO/BCSR, spsolve) |
| `devices/vsource.py` | Voltage/current source waveforms |
| `devices/verilog_a.py` | OpenVAF Verilog-A device wrapper |
| `netlist/parser.py` | VACASK netlist parser |
| `benchmarks/runner.py` | VACASK benchmark runner |
| `benchmarks/registry.py` | Auto-discovery of benchmark circuits |

## Common Patterns

### Loading and Running a Circuit

```python
from jax_spice import CircuitEngine

# Load circuit from VACASK .sim file
engine = CircuitEngine("vendor/VACASK/sim/ring.sim")
engine.parse()

# Run transient analysis
result = engine.run_transient(
    t_stop=1e-6,
    dt=1e-9,
    use_scan=True,     # GPU-efficient
    use_sparse=True,   # For large circuits
)
```

### Extracting Results

```python
# Get all node voltages at final time
for node_name, voltages in result.voltages.items():
    print(f"{node_name}: {voltages[-1]:.3f}V")

# Get specific node over time
vout = result.voltages["out"]  # Array of voltages

# Time array
times = result.times
```

### Running Multiple Analysis Types

```python
# Transient
tran_result = engine.run_transient(t_stop=1e-6, dt=1e-9)

# AC analysis
ac_result = engine.run_ac(fstart=1e3, fstop=1e9, num_points=100)

# Noise analysis
noise_result = engine.run_noise(
    fstart=1e3, fstop=1e9,
    input_source="vin", output_node="vout"
)

# PVT corners
from jax_spice.analysis import create_pvt_corners
corners = create_pvt_corners(
    process=['tt', 'ff', 'ss'],
    voltage=[0.9, 1.0, 1.1],
    temperature=[233, 300, 398],
)
corner_results = engine.run_corners(corners)
```

## Debugging Entry Points

When investigating issues, start here:

1. **Parsing issues**: Check `CircuitEngine.parse()` in `engine.py`
2. **Device compilation**: Check OpenVAF model loading in `_compile_openvaf_models()`
3. **System building**: Check `_make_gpu_resident_build_system_fn()` for J/f construction
4. **Convergence issues**: Look at Newton-Raphson loop in `solver.py`
5. **Sparse solver**: Check `sparse.py` for BCOO/BCSR operations
6. **Source waveforms**: Check `vsource.py` for pulse/sine/PWL evaluation
