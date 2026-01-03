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

## Example: Transient Simulation

```python
from jax_spice import CircuitEngine

# Load and parse a VACASK circuit file
engine = CircuitEngine("path/to/circuit.sim")
engine.parse()

# Run transient analysis
result = engine.run_transient(t_stop=1e-6, dt=1e-9, use_scan=True)

# Access results
print(f"Simulated {len(result.times)} time points")
for node_name, voltages in result.voltages.items():
    print(f"  {node_name}: {voltages[-1]:.3f}V (final)")
```

## Architecture Overview

```
jax_spice/
├── analysis/             # Circuit solvers and analysis engines
│   ├── engine.py        # CircuitEngine - main simulation API
│   ├── solver.py        # Newton-Raphson with lax.while_loop
│   ├── transient/       # Transient analysis (scan/loop strategies)
│   ├── ac.py            # AC small-signal analysis
│   ├── noise.py         # Noise analysis
│   ├── hb.py            # Harmonic balance
│   ├── xfer.py          # Transfer function (DCINC, DCXF, ACXF)
│   ├── corners.py       # PVT corner analysis
│   ├── homotopy.py      # Convergence aids (GMIN, source stepping)
│   └── sparse.py        # JAX sparse matrix operations (BCOO/BCSR)
│
├── devices/              # Device models
│   ├── vsource.py       # Voltage/current source waveforms
│   └── verilog_a.py     # OpenVAF Verilog-A wrapper
│
├── netlist/              # Circuit representation
│   ├── parser.py        # VACASK netlist parser
│   └── circuit.py       # Circuit data structures
│
└── benchmarks/           # Benchmark infrastructure
    ├── registry.py      # Auto-discovery of benchmarks
    └── runner.py        # VACASK benchmark runner
```

### Key Design Principles

1. **Functional devices**: All device models are pure JAX functions that take terminal voltages and parameters, returning current/conductance stamps
2. **Automatic differentiation**: Jacobians computed via JAX autodiff - no explicit derivatives needed
3. **Vectorized evaluation**: Devices grouped by type and evaluated in parallel with `jax.vmap`
4. **Sparse scalability**: Auto-switches to sparse matrices for large circuits

### Device Model Interface

All devices are compiled from Verilog-A sources using OpenVAF. Device models are batched
and evaluated in parallel using `jax.vmap` for GPU efficiency.

```python
# Devices are loaded from Verilog-A via OpenVAF
# Example from a VACASK .sim file:
load "resistor.osdi"    # Compiled from resistor.va
load "capacitor.osdi"   # Compiled from capacitor.va
load "psp103.osdi"      # PSP103 MOSFET model

model r sp_resistor
model c sp_capacitor
model nmos psp103va
```

## Supported Devices

| Device | Source | Description |
|--------|--------|-------------|
| Resistor | `resistor.va` | SPICE resistor with temperature coefficients |
| Capacitor | `capacitor.va` | Ideal capacitor |
| Diode | `diode.va` | SPICE diode model |
| VSource | Built-in | DC, pulse, sine, PWL voltage sources |
| ISource | Built-in | DC, pulse current sources |
| PSP103 | `psp103.va` | Production MOSFET model (OpenVAF) |
| Any VA | OpenVAF | Any Verilog-A model via OSDI interface |

## Analysis Types

All analyses are accessed through `CircuitEngine`:

```python
from jax_spice import CircuitEngine

engine = CircuitEngine("circuit.sim")
engine.parse()
```

### Transient Analysis

```python
# Run transient simulation
result = engine.run_transient(
    t_stop=1e-6,      # Stop time
    dt=1e-9,          # Time step
    use_scan=True,    # Use lax.scan for GPU efficiency
    use_sparse=True,  # Use sparse solver for large circuits
)

# Access results
times = result.times           # Array of time points
voltages = result.voltages     # Dict of node_name -> voltage array
```

### AC Analysis

```python
# Small-signal frequency response
ac_result = engine.run_ac(
    fstart=1e3,       # Start frequency (Hz)
    fstop=1e9,        # Stop frequency (Hz)
    num_points=100,   # Number of frequency points
    sweep_type='dec', # 'dec', 'lin', or 'oct'
)
```

### Noise Analysis

```python
# Compute noise figure across frequency
noise_result = engine.run_noise(
    fstart=1e3,
    fstop=1e9,
    input_source="vin",
    output_node="vout",
)
```

### Corner Analysis (PVT Sweep)

```python
from jax_spice.analysis import create_pvt_corners

# Create PVT corners
corners = create_pvt_corners(
    process=['tt', 'ff', 'ss'],
    voltage=[0.9, 1.0, 1.1],
    temperature=[233, 300, 398],  # Kelvin
)

# Run across all corners
corner_results = engine.run_corners(corners)
```

### Transfer Function Analysis

```python
# DC incremental (small-signal gain)
dcinc_result = engine.run_dcinc()

# DC transfer function
dcxf_result = engine.run_dcxf(input_source="vin", output_node="vout")

# AC transfer function
acxf_result = engine.run_acxf(input_source="vin", output_node="vout")
```

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
