# JAX-SPICE CLI Reference

The `jax-spice` command-line interface provides ngspice-style access to circuit simulation.

## Installation

After installing jax-spice, the CLI is available:

```bash
# Install with uv
uv sync

# Verify installation
jax-spice --version
jax-spice --help
```

## Basic Usage

```bash
# Run simulation on a circuit file
jax-spice circuit.sim

# Specify output file
jax-spice circuit.sim -o results.raw

# Run with specific output format
jax-spice circuit.sim -o results.csv --format csv
```

## Commands

### `run` (default)

Run circuit simulation. This is the default command when a circuit file is provided.

```bash
# These are equivalent:
jax-spice circuit.sim
jax-spice run circuit.sim
```

**Options:**

| Option | Description |
|--------|-------------|
| `-o, --output FILE` | Output file path (default: circuit.raw) |
| `-f, --format FMT` | Output format: `raw`, `csv`, `json` (default: raw) |
| `--tran DT TSTOP` | Override transient analysis parameters |
| `--ac TYPE PTS FSTART FSTOP` | AC analysis (dec/lin/oct, points, freq range) |
| `--sparse` | Force sparse solver (for large circuits) |
| `--no-scan` | Disable lax.scan (use Python loop) |
| `--gpu` | Force GPU backend |
| `--cpu` | Force CPU backend |
| `--x64` | Force float64 precision |
| `--x32` | Force float32 precision |
| `--profile` | Enable profiling |

**Examples:**

```bash
# Transient analysis with custom parameters
jax-spice circuit.sim --tran 1n 100u

# AC analysis
jax-spice circuit.sim --ac dec 100 1k 1G

# Large circuit with sparse solver on GPU
jax-spice large_circuit.sim --sparse --gpu

# Force float64 precision on CPU
jax-spice circuit.sim --cpu --x64
```

### `benchmark`

Run benchmark circuits from the VACASK test suite.

```bash
# List available benchmarks
jax-spice benchmark --list

# Run a specific benchmark
jax-spice benchmark ring

# Run with profiling
jax-spice benchmark ring --profile
```

**Options:**

| Option | Description |
|--------|-------------|
| `-l, --list` | List available benchmarks |
| `--sparse` | Force sparse solver |
| `--x64` | Force float64 precision |
| `--x32` | Force float32 precision |
| `--profile` | Enable timing profiling |

### `convert`

Convert SPICE netlists to VACASK format.

```bash
jax-spice convert input.sp output.sim
```

### `info`

Display system information.

```bash
jax-spice info
```

Shows:
- JAX backend (CPU, CUDA, Metal)
- Float64 support status
- Available devices

## Output Formats

### Raw (ngspice-compatible)

Binary format compatible with ngspice tools and gwave waveform viewer.

```bash
jax-spice circuit.sim -o results.raw --format raw
```

Read with ngspice:
```
ngspice> load results.raw
ngspice> plot v(out)
```

Read with gwave:
```bash
gwave results.raw
```

### CSV

Comma-separated values, compatible with spreadsheets and data analysis tools.

```bash
jax-spice circuit.sim -o results.csv --format csv
```

Format:
```csv
time,node1,node2,...
0.0,0.0,1.2,...
1e-9,0.1,1.1,...
```

### JSON

JSON format for programmatic processing.

```bash
jax-spice circuit.sim -o results.json --format json
```

Format:
```json
{
  "times": [0.0, 1e-9, ...],
  "voltages": {
    "node1": [0.0, 0.1, ...],
    "node2": [1.2, 1.1, ...]
  }
}
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `JAX_PLATFORMS` | Force JAX platform: `cpu`, `cuda`, `gpu` |
| `JAX_ENABLE_X64` | Enable float64: `1` or `0` |
| `NGSPICE_BIN` | Path to ngspice binary (for convert) |

## Examples

### Transient Analysis

```bash
# Basic transient simulation
jax-spice ring_oscillator.sim --tran 1n 10u

# Large circuit with sparse solver
jax-spice c6288.sim --tran 1n 100n --sparse

# GPU acceleration
jax-spice ring_oscillator.sim --gpu --tran 1n 10u
```

### AC Analysis

```bash
# Decade sweep, 100 points per decade
jax-spice amplifier.sim --ac dec 100 1k 1G

# Linear sweep
jax-spice filter.sim --ac lin 1000 1k 10k
```

### Benchmarking

```bash
# Run ring oscillator benchmark with profiling
jax-spice benchmark ring --profile

# Compare sparse vs dense solver
jax-spice benchmark mul --profile
jax-spice benchmark mul --profile --sparse
```

### Batch Processing

```bash
# Process multiple circuits
for f in circuits/*.sim; do
    jax-spice "$f" -o "results/$(basename "$f" .sim).raw"
done
```

## Troubleshooting

### GPU Not Detected

```bash
# Check system info
jax-spice info

# Force CPU fallback
jax-spice circuit.sim --cpu
```

### Convergence Issues

For difficult circuits, try:
```bash
# Use sparse solver for better conditioning
jax-spice circuit.sim --sparse

# Force double precision
jax-spice circuit.sim --x64
```

### Out of Memory

For large circuits:
```bash
# Use sparse solver (required for >1000 nodes)
jax-spice large_circuit.sim --sparse
```

## See Also

- [README.md](../README.md) - Getting started guide
- [Architecture Overview](architecture_overview.md) - System design
- [GPU Solver Architecture](gpu_solver_architecture.md) - Performance optimization
