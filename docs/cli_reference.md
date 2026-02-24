# VA-JAX CLI Reference

The `va-jax` command-line interface provides ngspice-style access to circuit simulation.

## Installation

After installing va-jax, the CLI is available:

```bash
# Install with uv
uv sync

# Verify installation
va-jax --version
va-jax --help
```

## Basic Usage

```bash
# Run simulation on a circuit file
va-jax circuit.sim

# Specify output file
va-jax circuit.sim -o results.raw

# Run with specific output format
va-jax circuit.sim -o results.csv --format csv
```

## Commands

### `run` (default)

Run circuit simulation. This is the default command when a circuit file is provided.

```bash
# These are equivalent:
va-jax circuit.sim
va-jax run circuit.sim
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
va-jax circuit.sim --tran 1n 100u

# AC analysis
va-jax circuit.sim --ac dec 100 1k 1G

# Large circuit with sparse solver on GPU
va-jax large_circuit.sim --sparse --gpu

# Force float64 precision on CPU
va-jax circuit.sim --cpu --x64
```

### `benchmark`

Run benchmark circuits from the VACASK test suite.

```bash
# List available benchmarks
va-jax benchmark --list

# Run a specific benchmark
va-jax benchmark ring

# Run with profiling
va-jax benchmark ring --profile
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
va-jax convert input.sp output.sim
```

### `info`

Display system information.

```bash
va-jax info
```

Shows:
- JAX backend (CPU, CUDA, Metal)
- Float64 support status
- Available devices

## Output Formats

### Raw (ngspice-compatible)

Binary format compatible with ngspice tools and gwave waveform viewer.

```bash
va-jax circuit.sim -o results.raw --format raw
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
va-jax circuit.sim -o results.csv --format csv
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
va-jax circuit.sim -o results.json --format json
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
va-jax ring_oscillator.sim --tran 1n 10u

# Large circuit with sparse solver
va-jax c6288.sim --tran 1n 100n --sparse

# GPU acceleration
va-jax ring_oscillator.sim --gpu --tran 1n 10u
```

### AC Analysis

```bash
# Decade sweep, 100 points per decade
va-jax amplifier.sim --ac dec 100 1k 1G

# Linear sweep
va-jax filter.sim --ac lin 1000 1k 10k
```

### Benchmarking

```bash
# Run ring oscillator benchmark with profiling
va-jax benchmark ring --profile

# Compare sparse vs dense solver
va-jax benchmark mul --profile
va-jax benchmark mul --profile --sparse
```

### Batch Processing

```bash
# Process multiple circuits
for f in circuits/*.sim; do
    va-jax "$f" -o "results/$(basename "$f" .sim).raw"
done
```

## Troubleshooting

### GPU Not Detected

```bash
# Check system info
va-jax info

# Force CPU fallback
va-jax circuit.sim --cpu
```

### Convergence Issues

For difficult circuits, try:
```bash
# Use sparse solver for better conditioning
va-jax circuit.sim --sparse

# Force double precision
va-jax circuit.sim --x64
```

### Out of Memory

For large circuits:
```bash
# Use sparse solver (required for >1000 nodes)
va-jax large_circuit.sim --sparse
```

## See Also

- [README.md](../README.md) - Getting started guide
- [Architecture Overview](architecture_overview.md) - System design
- [GPU Solver Architecture](gpu_solver_architecture.md) - Performance optimization
