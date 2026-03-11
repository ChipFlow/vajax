# VAJAX CLI Reference

The `vajax` command-line interface provides ngspice-style access to circuit simulation.

## Installation

After installing vajax, the CLI is available:

```bash
# Install with uv
uv sync

# Verify installation
vajax --version
vajax --help
```

## Basic Usage

```bash
# Run simulation on a circuit file
vajax circuit.sim

# Specify output file
vajax circuit.sim -o results.raw

# Run with specific output format
vajax circuit.sim -o results.csv --format csv
```

## Commands

### `run` (default)

Run circuit simulation. This is the default command when a circuit file is provided.

```bash
# These are equivalent:
vajax circuit.sim
vajax run circuit.sim
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
vajax circuit.sim --tran 1n 100u

# AC analysis
vajax circuit.sim --ac dec 100 1k 1G

# Large circuit with sparse solver on GPU
vajax large_circuit.sim --sparse --gpu

# Force float64 precision on CPU
vajax circuit.sim --cpu --x64
```

### `benchmark`

Run benchmark circuits from the VACASK test suite.

```bash
# List available benchmarks
vajax benchmark --list

# Run a specific benchmark
vajax benchmark ring

# Run with profiling
vajax benchmark ring --profile
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
vajax convert input.sp output.sim
```

### `info`

Display system information.

```bash
vajax info
```

Shows:
- JAX backend (CPU, CUDA, Metal)
- Float64 support status
- Available devices

## Output Formats

### Raw (ngspice-compatible)

Binary format compatible with ngspice tools and gwave waveform viewer.

```bash
vajax circuit.sim -o results.raw --format raw
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
vajax circuit.sim -o results.csv --format csv
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
vajax circuit.sim -o results.json --format json
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
| `VAJAX_MODEL_PATH` | Additional model search paths (colon-separated, `;` on Windows) |
| `VAJAX_CACHE_DIR` | Override cache directory (default: `~/.cache/vajax/`) |
| `VAJAX_NO_PROGRESS` | Set to `1` to disable progress bars |
| `NGSPICE_BIN` | Path to ngspice binary (for convert) |

See [Configuration](configuration.md) for config file support and model search order.

## Examples

### Transient Analysis

```bash
# Basic transient simulation
vajax ring_oscillator.sim --tran 1n 10u

# Large circuit with sparse solver
vajax c6288.sim --tran 1n 100n --sparse

# GPU acceleration
vajax ring_oscillator.sim --gpu --tran 1n 10u
```

### AC Analysis

```bash
# Decade sweep, 100 points per decade
vajax amplifier.sim --ac dec 100 1k 1G

# Linear sweep
vajax filter.sim --ac lin 1000 1k 10k
```

### Benchmarking

```bash
# Run ring oscillator benchmark with profiling
vajax benchmark ring --profile

# Compare sparse vs dense solver
vajax benchmark mul --profile
vajax benchmark mul --profile --sparse
```

### Batch Processing

```bash
# Process multiple circuits
for f in circuits/*.sim; do
    vajax "$f" -o "results/$(basename "$f" .sim).raw"
done
```

## Troubleshooting

### GPU Not Detected

```bash
# Check system info
vajax info

# Force CPU fallback
vajax circuit.sim --cpu
```

### Convergence Issues

For difficult circuits, try:
```bash
# Use sparse solver for better conditioning
vajax circuit.sim --sparse

# Force double precision
vajax circuit.sim --x64
```

### Out of Memory

For large circuits:
```bash
# Use sparse solver (required for >1000 nodes)
vajax large_circuit.sim --sparse
```

## See Also

- [Configuration](configuration.md) - Config files and model search paths
- [README.md](../README.md) - Getting started guide
- [Architecture Overview](architecture_overview.md) - System design
- [GPU Solver Architecture](gpu_solver_architecture.md) - Performance optimization
