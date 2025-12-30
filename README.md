# JAX-SPICE: GPU-Accelerated Analog Circuit Simulator

A proof-of-concept GPU-accelerated analog circuit simulator built on JAX, demonstrating:
- **Automatic differentiation** for computing device Jacobians without explicit derivatives
- **GPU acceleration** for large circuits using JAX's JIT compilation
- **Verilog-A model integration** via OpenVAF/OSDI bindings for PDK compatibility
- **SAX-inspired functional device model API**

## Current Status

JAX-SPICE is in active development as a proof-of-concept. All VACASK benchmark circuits are passing:

| Benchmark | Devices | Nodes | Status |
|-----------|---------|-------|--------|
| rc | Resistor, Capacitor, VSource | ~10 | Passing |
| graetz | Diode bridge rectifier | ~20 | Passing |
| mul | Multiplier circuit | ~100 | Passing |
| ring | Ring oscillator (PSP103) | ~150 | Passing (~20ms/step after JIT) |
| c6288 | 16-bit multiplier (PSP103) | 86,000 | Passing (~1s/step, sparse solver) |

**Performance highlights:**
- 34x speedup on ring oscillator via JIT-compiled vmap batching
- Sparse matrix support auto-activates for circuits >1000 nodes
- OpenVAF-compiled PSP103 MOSFETs validated against MIR interpreter

## Quick Start

```bash
# Install with uv (recommended)
uv sync

# Run tests
JAX_PLATFORMS=cpu uv run pytest tests/ -v

# Run a benchmark
JAX_PLATFORMS=cpu uv run python -m jax_spice.benchmarks.runner vendor/VACASK/sim/ring.sim
```

### Installation Options

```bash
# CPU only (default)
uv sync

# With CUDA 12 support (Linux)
uv sync --extra cuda12

# With SAX integration
uv sync --extra sax
```

## Example: Simple DC Analysis

```python
from jax_spice.devices import Resistor, VoltageSource
from jax_spice.analysis import MNASystem, dc_operating_point

# Create MNA system
system = MNASystem()

# Add a voltage source and resistor
system.add_device("V1", VoltageSource(dc=5.0), ["vdd", "gnd"])
system.add_device("R1", Resistor(r=1000.0), ["vdd", "out"])
system.add_device("R2", Resistor(r=1000.0), ["out", "gnd"])

# Solve for DC operating point
V, info = dc_operating_point(system)
print(f"Output voltage: {V[system.get_node_index('out')]:.3f}V")
# Output voltage: 2.500V
```

## Architecture Overview

```
jax_spice/
├── devices/              # Device models (pure JAX functions)
│   ├── base.py          # Device protocol and DeviceStamps
│   ├── resistor.py      # Temperature-dependent resistor
│   ├── capacitor.py     # Capacitor with companion model
│   ├── mosfet_simple.py # Simplified BSIM-like MOSFET
│   ├── verilog_a.py     # Verilog-A wrapper
│   └── openvaf_device.py # OpenVAF integration
│
├── analysis/             # Circuit solvers
│   ├── mna.py           # Modified Nodal Analysis system
│   ├── dc.py            # DC operating point (Newton-Raphson)
│   ├── transient.py     # Transient analysis (Backward Euler)
│   ├── sparse.py        # Sparse matrix operations
│   └── context.py       # Analysis context (time, temperature)
│
├── netlist/              # Circuit representation
│   ├── parser.py        # VACASK netlist parser
│   └── circuit.py       # Circuit data structures
│
└── benchmarks/           # Benchmark infrastructure
    └── runner.py        # VACASK benchmark runner
```

### Key Design Principles

1. **Functional devices**: All device models are pure JAX functions that take terminal voltages and parameters, returning current/conductance stamps
2. **Automatic differentiation**: Jacobians computed via JAX autodiff - no explicit derivatives needed
3. **Vectorized evaluation**: Devices grouped by type and evaluated in parallel with `jax.vmap`
4. **Sparse scalability**: Auto-switches to sparse matrices for large circuits

### Device Model Interface

Every device implements the `Device` protocol:

```python
from jax_spice.devices import DeviceStamps

def evaluate(terminal_voltages: Array, params: dict, context: AnalysisContext) -> DeviceStamps:
    """Evaluate device at given terminal voltages.

    Returns:
        DeviceStamps containing:
        - currents: Current into each terminal
        - conductances: Conductance matrix (dI/dV)
        - charges: (optional) Charge at each terminal for transient
    """
```

## Supported Devices

| Device | Type | Description |
|--------|------|-------------|
| `Resistor` | Built-in | Temperature-dependent resistor (R, tc1, tc2) |
| `Capacitor` | Built-in | Ideal capacitor with Backward Euler companion |
| `VoltageSource` | Built-in | DC and time-varying (pulse, PWL) |
| `CurrentSource` | Built-in | DC and pulse current sources |
| `MOSFETSimple` | Built-in | Simplified BSIM-like N/PMOS model |
| `VerilogADevice` | OpenVAF | Any Verilog-A model (PSP103, BSIM4, etc.) |

## Analysis Types

### DC Operating Point

```python
from jax_spice.analysis import dc_operating_point, dc_operating_point_sparse

# Standard Newton-Raphson (for small circuits)
V, info = dc_operating_point(system, max_iterations=50, abstol=1e-9)

# Sparse solver (for large circuits, auto-selected >1000 nodes)
V, info = dc_operating_point_sparse(system, vdd=1.2)
```

For difficult circuits, use homotopy methods:

```python
from jax_spice.analysis.homotopy import run_homotopy_chain, HomotopyConfig
from jax_spice.analysis.solver import NRConfig

# Build residual/jacobian functions with gmin/gshunt/source_scale parameters
result = run_homotopy_chain(build_residual_fn, build_jacobian_fn, V_init,
                            HomotopyConfig(), NRConfig())
```

### Transient Analysis

```python
from jax_spice.analysis import transient_analysis
from jax_spice.analysis.transient import transient_analysis_jit

# Flexible Python loop
times, solutions = transient_analysis(system, t_stop=1e-6, t_step=1e-9)

# JIT-compiled for performance (preferred)
times, solutions = transient_analysis_jit(system, t_stop=1e-6, t_step=1e-9, icmode='uic')
```

**Initial condition modes:**
- `icmode='op'`: Compute DC operating point first (good for analog)
- `icmode='uic'`: Use Initial Conditions directly (preferred for digital)

## Verilog-A Integration

JAX-SPICE can use production PDK models via OpenVAF:

```python
from jax_spice.devices import VerilogADevice, compile_va

# Compile a Verilog-A model
model = compile_va("psp103.va")

# Create device with model card parameters
m1 = VerilogADevice(model, params={"type": 1, "vth0": 0.4, ...})
system.add_device("M1", m1, ["d", "g", "s", "b"])
```

See `docs/vacask_osdi_inputs.md` for details on the OpenVAF integration.

## Running Benchmarks

```bash
# Run specific benchmark
JAX_PLATFORMS=cpu uv run python -m jax_spice.benchmarks.runner vendor/VACASK/sim/ring.sim

# Profile with GPU
JAX_PLATFORMS=cuda uv run python scripts/profile_gpu.py --benchmark ring

# Run all VACASK suite tests
JAX_PLATFORMS=cpu uv run pytest tests/test_vacask_suite.py -v
```

## Platform Notes

- **macOS**: Metal GPU backend doesn't support `triangular_solve`, automatically falls back to CPU
- **Linux + CUDA**: CUDA libraries are auto-preloaded for GPU detection
- **Precision**: Auto-configured based on backend:
  - CPU/CUDA: Float64 enabled for numerical precision
  - Metal/TPU: Float32 (backends don't support float64 natively)
  - Use `jax_spice.configure_precision(force_x64=True/False)` to override

## Documentation

- `docs/gpu_solver_architecture.md` - Detailed solver design and optimization
- `docs/gpu_solver_jacobian.md` - Jacobian computation details
- `docs/vacask_osdi_inputs.md` - OpenVAF/OSDI input handling
- `docs/vacask_sim_format.md` - VACASK simulation file format
- `TODO.md` - Development roadmap and known issues

## Contributing

See `CONTRIBUTING.md` for development setup and guidelines.

## License

MIT (prototype/research code)
