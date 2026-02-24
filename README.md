# VA-JAX: GPU-Accelerated Analog Circuit Simulator

[![Tests](https://github.com/ChipFlow/va-jax/actions/workflows/test.yml/badge.svg)](https://github.com/ChipFlow/va-jax/actions/workflows/test.yml)
[![GPU Tests](https://github.com/ChipFlow/va-jax/actions/workflows/test-gpu.yml/badge.svg)](https://github.com/ChipFlow/va-jax/actions/workflows/test-gpu.yml)
[![Lint](https://github.com/ChipFlow/va-jax/actions/workflows/lint.yml/badge.svg)](https://github.com/ChipFlow/va-jax/actions/workflows/lint.yml)
[![Benchmark](https://github.com/ChipFlow/va-jax/actions/workflows/benchmark-comparison.yml/badge.svg)](https://github.com/ChipFlow/va-jax/actions/workflows/benchmark-comparison.yml)

A proof-of-concept GPU-accelerated analog circuit simulator built on JAX, demonstrating:
- **Automatic differentiation** for computing device Jacobians without explicit derivatives
- **GPU acceleration** for large circuits using JAX's JIT compilation
- **Verilog-A model integration** via OpenVAF/OSDI bindings for PDK compatibility
- **SAX-inspired functional device model API**

## Current Status

VA-JAX is in active development as a proof-of-concept. All VACASK benchmark circuits are passing.

**[Full benchmark results and test coverage →](https://chipflow.github.io/va-jax/)**

## Validation: Three-Way Comparison

VA-JAX results are validated against VACASK (reference simulator) and ngspice.
All simulators use identical netlists and device models (PSP103 MOSFETs via OSDI).

### RC Low-Pass Filter
Simple RC circuit demonstrating basic transient behavior. VA-JAX matches VACASK and ngspice exactly.

![RC Comparison](docs/images/rc_three_way_comparison.png)

### PSP103 Ring Oscillator
7-stage ring oscillator with production PSP103 MOSFET models. Shows excellent agreement in oscillation frequency and waveform shape.

![Ring Oscillator Comparison](docs/images/ring_three_way_comparison.png)

### C6288 16-bit Multiplier
Large-scale benchmark with ~86,000 nodes. Uses sparse solver for memory efficiency. Demonstrates VA-JAX scaling to production-sized circuits.

![C6288 Comparison](docs/images/c6288_three_way_comparison.png)

Generate comparison plots:
```bash
uv run scripts/plot_three_way_comparison.py --benchmark ring --output-dir docs/images
uv run scripts/plot_three_way_comparison.py --benchmark c6288 --output-dir docs/images --skip-ngspice
```

## Performance

VA-JAX is designed for GPU acceleration of large circuits. The table below shows
per-step timing against VACASK (C++ reference simulator) on CI runners.

### CPU Performance (vs VACASK)

| Benchmark | Nodes | Steps | JAX (ms/step) | VACASK (ms/step) | Ratio | RMS Error |
|-----------|------:|------:|--------------:|-----------------:|------:|-----------|
| rc        |     4 |    1M |         0.012 |            0.002 |  6.6x | 0.00%     |
| graetz    |     6 |    1M |         0.020 |            0.004 |  5.4x | 0.00%     |
| mul       |     8 |  500k |         0.041 |            0.004 | 10.9x | 0.00%     |
| ring      |    47 |   20k |         0.511 |            0.109 |  4.7x | -         |
| c6288     | ~5000 |    1k |        88.060 |           76.390 |  1.2x | 2.01%     |

### GPU Performance

| Benchmark | Nodes | JAX GPU (ms/step) | JAX CPU (ms/step) | GPU Speedup | vs VACASK CPU |
|-----------|------:|-------------------:|-------------------:|------------:|--------------:|
| c6288     | ~5000 |             19.81  |             88.06  |      4.4x   |   **2.9x faster** |
| ring      |    47 |              1.49  |              0.51  |      0.3x   | below threshold |
| rc        |     4 |              0.24  |              0.01  |      0.05x  | below threshold |

GPU results for circuits below ~500 nodes are shown for completeness but are not
meaningful performance comparisons — GPU kernel launch overhead dominates when the
per-step computation is tiny. The GPU auto-threshold (`gpu_threshold=500` nodes)
prevents this in normal usage.

### Performance Characteristics

**Where VA-JAX excels:** Large circuits (1000+ nodes) on GPU, where matrix
operations dominate and GPU parallelism pays off. The c6288 benchmark (16-bit
multiplier, ~5000 nodes) runs **2.9x faster than VACASK** on GPU.

**Where VACASK is faster:** Small circuits on CPU. VA-JAX carries a per-step
fixed overhead of ~5-12 microseconds from:

- **Adaptive timestep machinery**: LTE estimation, voltage prediction, and
  variable-step BDF2 coefficient computation run every step regardless of
  circuit size.
- **Functional array updates**: JAX requires `jnp.where` for conditional updates
  inside `lax.while_loop`, which evaluates both branches. VACASK uses C++ runtime
  branching that skips unused work.
- **Vmap batching**: Device evaluation is vectorized with `jax.vmap` for GPU
  parallelism, but this adds overhead when evaluating only 2-4 device instances.
- **COO matrix assembly**: Jacobian construction from coordinate format adds
  indirection that VACASK avoids with direct matrix stamping.

This overhead is negligible for large circuits (c6288: 0.01% of step time) but
dominates for small ones (rc: ~80% of step time). See `docs/performance_analysis.md`
for the full analysis.

## Quick Start

```bash
# Install with uv (recommended)
uv sync

# Run tests
JAX_PLATFORMS=cpu uv run pytest tests/ -v

# Run a benchmark
JAX_PLATFORMS=cpu uv run python -m vajax.benchmarks.runner vendor/VACASK/sim/ring.sim
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

## Command-Line Interface

VA-JAX provides an ngspice-style CLI:

```bash
# Run simulation on a circuit file
va-jax circuit.sim

# Specify output file and format
va-jax circuit.sim -o results.raw
va-jax circuit.sim -o results.csv --format csv

# Override analysis parameters
va-jax circuit.sim --tran 1n 100u
va-jax circuit.sim --ac dec 100 1k 1G

# Run benchmarks
va-jax benchmark ring --profile

# System info
va-jax info
```

See [docs/cli_reference.md](docs/cli_reference.md) for full documentation.

## Example: Transient Simulation

```python
from vajax import CircuitEngine

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
vajax/
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
from vajax import CircuitEngine

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
from vajax.analysis import create_pvt_corners

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

VA-JAX can use production PDK models via OpenVAF:

```python
from vajax.devices import VerilogADevice, compile_va

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
JAX_PLATFORMS=cpu uv run python -m vajax.benchmarks.runner vendor/VACASK/sim/ring.sim

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
  - Use `vajax.configure_precision(force_x64=True/False)` to override

## Documentation

- `docs/api_reference.md` - **API Reference** (CircuitEngine, result types, I/O)
- `docs/cli_reference.md` - Command-line interface reference
- `docs/architecture_overview.md` - System architecture and design
- `docs/performance_analysis.md` - Performance analysis and overhead breakdown
- `docs/gpu_solver_architecture.md` - Detailed solver design and optimization
- `docs/gpu_solver_jacobian.md` - Jacobian computation details
- `docs/debug_tools.md` - Debug utilities reference
- `docs/vacask_osdi_inputs.md` - OpenVAF/OSDI input handling
- `docs/vacask_sim_format.md` - VACASK simulation file format
- `TODO.md` - Development roadmap and known issues

## Contributing

See `CONTRIBUTING.md` for development setup and guidelines.

## License

MIT (prototype/research code)
