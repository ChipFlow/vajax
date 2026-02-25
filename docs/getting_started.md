# Getting Started with VAJAX

This guide walks you through installing VAJAX and running your first circuit simulation.

## Prerequisites

- Python 3.11-3.13
- [uv](https://docs.astral.sh/uv/) package manager

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd vajax

# Install dependencies
uv sync
```

For GPU support on NVIDIA:

```bash
uv sync --extra cuda12
```

## Your First Simulation: RC Low-Pass Filter

Here's a simple RC circuit excited by a pulse train. Create a file called `rc.sim`:

```spice
RC circuit excited by a pulse train

load "spice/resistor.osdi"
load "spice/capacitor.osdi"

model r sp_resistor
model c sp_capacitor
model vsource vsource

vs (1 0) vsource dc=0 type="pulse" val0=0 val1=1 rise=1u fall=1u width=1m period=2m
r1 (1 2) r r=1k
c1 (2 0) c c=1u

control
  options tran_method="trap"
  analysis tran1 tran step=1u stop=10m maxstep=1u
endc
```

This defines:
- A pulse voltage source `vs` switching between 0V and 1V
- A 1k resistor `r1`
- A 1uF capacitor `c1`
- The RC time constant is 1ms

### Running with Python

```python
from pathlib import Path
from vajax import CircuitEngine

# Load and parse the circuit
engine = CircuitEngine(Path("rc.sim"))
engine.parse()

# Configure and run transient analysis
engine.prepare(t_stop=10e-3, dt=1e-6)
result = engine.run_transient()

# Access results
times = result.times
v_in = result.voltage("1")    # Input node
v_out = result.voltage("2")   # Output node (across capacitor)

print(f"Simulated {result.num_steps} timesteps")
print(f"Final output voltage: {float(v_out[-1]):.4f} V")
```

### Plotting Results

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(times * 1000, v_in, label="V(in)")
plt.plot(times * 1000, v_out, label="V(out)")
plt.xlabel("Time [ms]")
plt.ylabel("Voltage [V]")
plt.title("RC Low-Pass Filter Response")
plt.legend()
plt.grid(True)
plt.show()
```

## Running from the Command Line

VAJAX includes a CLI for running simulations:

```bash
# Run a VACASK benchmark
uv run vajax run vendor/VACASK/benchmark/rc/vacask/runme.sim

# See all CLI options
uv run vajax --help
```

## Available Analysis Types

| Analysis | Method | Description |
|----------|--------|-------------|
| Transient | `run_transient()` | Time-domain simulation with adaptive timestep |
| AC | `run_ac()` | Small-signal frequency sweep |
| Noise | `run_noise()` | Thermal, shot, and flicker noise analysis |
| DC Transfer | `run_dcinc()` | DC incremental (small-signal) response |
| DC XF | `run_dcxf()` | DC transfer function and input impedance |
| AC XF | `run_acxf()` | AC transfer function over frequency |
| PVT Corners | `run_corners()` | Process/voltage/temperature sweep |

## Built-in Benchmark Circuits

VAJAX ships with several benchmark circuits in `vendor/VACASK/benchmark/`:

| Circuit | Description | Nodes | Complexity |
|---------|-------------|-------|------------|
| `rc/` | RC low-pass filter | 4 | Beginner |
| `graetz/` | Diode bridge rectifier | 6 | Beginner |
| `mul/` | Diode voltage multiplier | 8 | Intermediate |
| `ring/` | 7-stage PSP103 ring oscillator | 47 | Intermediate |
| `c6288/` | 16-bit multiplier (PSP103) | ~5000 | Advanced (GPU) |

## Sparse Solver for Large Circuits

For circuits with more than ~1000 nodes, enable the sparse solver:

```python
engine.prepare(t_stop=1e-6, dt=1e-9, use_sparse=True)
result = engine.run_transient()
```

## Known Limitations

- **No DC sweep analysis** yet (only DC operating point as part of other analyses)
- **Float32 only on Metal/TPU** backends (float64 on CPU/CUDA)
- **No interactive waveform viewer** built-in (use matplotlib or export to raw files)

## Next Steps

- See [Architecture Overview](architecture_overview.md) for how VAJAX works internally
- See [Supported Devices](supported_devices.md) for available device models
- See [API Reference](api_reference.md) for full API documentation
- See [Transient Options](transient_options.md) for advanced transient configuration
