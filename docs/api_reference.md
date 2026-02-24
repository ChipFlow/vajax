# VAJAX API Reference

## Core Classes

### CircuitEngine

The main class for circuit simulation. Parses VACASK/SPICE netlists and runs various analyses.

```python
from vajax import CircuitEngine

engine = CircuitEngine("circuit.sim")
engine.parse()
engine.prepare(t_stop=1e-6, dt=1e-9)
result = engine.run_transient()
```

#### Constructor

```python
CircuitEngine(sim_path: Path | str)
```

**Parameters:**
- `sim_path`: Path to circuit file (`.sim` VACASK format; use `vajax convert` for SPICE netlists)

#### Methods

##### parse()

Parse the circuit file and compile device models.

```python
engine.parse()
```

Loads the netlist, compiles Verilog-A models via OpenVAF, and prepares the circuit for simulation.

---

##### prepare()

Prepare for transient analysis by configuring all parameters. Call this once
before `run_transient()`. If `run_transient()` is called without `prepare()`,
it will auto-prepare from netlist defaults.

```python
engine.prepare(
    *,
    t_stop: float = None,              # Stop time (seconds)
    dt: float = None,                  # Time step / initial timestep (seconds)
    use_sparse: bool = None,           # Force sparse (True) or dense (False) solver
    backend: str = None,               # 'gpu', 'cpu', or None (auto-select)
    temperature: float = 300.15,       # Simulation temperature (Kelvin)
    adaptive_config: AdaptiveConfig = None,  # Adaptive timestep configuration
    checkpoint_interval: int = None,   # GPU memory checkpointing interval
)
```

**Parameters:**
- `t_stop`: Simulation stop time. If None, uses value from netlist analysis params
- `dt`: Time step. If None, uses value from netlist. For adaptive mode, this is the initial timestep
- `use_sparse`: Use sparse matrix solver (recommended for >1000 nodes). Defaults to False (dense)
- `backend`: Force GPU or CPU backend. If None, auto-selects based on circuit size
- `temperature`: Simulation temperature in Kelvin (default: 300.15K)
- `adaptive_config`: Override adaptive timestep settings (LTE ratio, tolerances, etc.)
- `checkpoint_interval`: If set, use GPU memory checkpointing with this many steps per buffer

---

##### run_transient()

Run transient (time-domain) simulation. Requires `prepare()` to have been
called first (or will auto-prepare from netlist defaults).

```python
result = engine.run_transient() -> TransientResult
```

**Returns:** `TransientResult`

---

##### run_ac()

Run AC (small-signal frequency response) analysis.

```python
result = engine.run_ac(
    freq_start: float = 1.0,      # Start frequency (Hz)
    freq_stop: float = 1e6,       # Stop frequency (Hz)
    mode: str = 'dec',            # 'dec', 'lin', 'oct', or 'list'
    points: int = 10,             # Points per decade/octave
    step: float = None,           # Frequency step for 'lin' mode
    values: List[float] = None,   # Explicit frequency list for 'list' mode
) -> ACResult
```

**Parameters:**
- `freq_start`: Start frequency in Hz
- `freq_stop`: Stop frequency in Hz
- `mode`: Sweep mode - `'dec'` (decade), `'lin'` (linear), `'oct'` (octave), or `'list'`
- `points`: Points per decade/octave (for `'dec'`/`'oct'` modes)
- `step`: Frequency step for `'lin'` mode
- `values`: Explicit frequency list for `'list'` mode

**Returns:** `ACResult`

---

##### run_noise()

Run noise analysis.

```python
result = engine.run_noise(
    out: str | int = 1,          # Output node (name or index)
    input_source: str = "",      # Name of input source (e.g., "vin")
    freq_start: float = 1.0,    # Start frequency (Hz)
    freq_stop: float = 1e6,     # Stop frequency (Hz)
    mode: str = 'dec',          # 'dec', 'lin', 'oct', or 'list'
    points: int = 10,           # Points per decade/octave
    temperature: float = 300.15, # Temperature (Kelvin)
) -> NoiseResult
```

**Returns:** `NoiseResult`

---

##### run_corners()

Run PVT (Process-Voltage-Temperature) corner analysis.

```python
from vajax.analysis.corners import create_pvt_corners

corners = create_pvt_corners(
    processes=['FF', 'TT', 'SS'],
    voltages=[0.9, 1.0, 1.1],
    temperatures=['cold', 'room', 'hot'],
)

engine.prepare(t_stop=1e-6, dt=1e-9)
results = engine.run_corners(corners)
```

**Returns:** `CornerSweepResult` containing results for all corners.

---

##### run_dcinc()

Run DC incremental (small-signal gain) analysis.

```python
result = engine.run_dcinc() -> DCIncResult
```

**Returns:** `DCIncResult` with DC operating point and incremental gains.

---

##### run_dcxf()

Run DC transfer function analysis.

```python
result = engine.run_dcxf(
    out: str | int = 1,  # Output node (name or index)
) -> DCXFResult
```

**Returns:** `DCXFResult` with transfer function magnitude and phase.

---

##### run_acxf()

Run AC transfer function analysis.

```python
result = engine.run_acxf(
    out: str | int = 1,              # Output node (name or index)
    freq_start: float = 1.0,        # Start frequency (Hz)
    freq_stop: float = 1e6,         # Stop frequency (Hz)
    mode: str = 'dec',              # 'dec', 'lin', 'oct', or 'list'
    points: int = 10,               # Points per decade/octave
    step: float = None,             # Frequency step for 'lin' mode
    values: List[float] = None,     # Explicit frequency list for 'list' mode
) -> ACXFResult
```

**Returns:** `ACXFResult` with frequency-dependent transfer function.

---

#### Properties

##### node_names

Dict mapping node name strings to integer indices.

```python
engine.node_names  # {'0': 0, 'vdd': 1, 'out': 2, ...}
```

##### devices

List of device information dicts.

```python
engine.devices  # [{'name': 'R1', 'model': 'resistor', ...}, ...]
```

---

## Result Types

### TransientResult

Result of transient simulation.

```python
@dataclass
class TransientResult:
    times: Array                    # Time points array
    voltages: Dict[str, Array]      # Node voltages over time
    currents: Dict[str, Array]      # Source currents over time
    stats: Dict[str, Any]           # Simulation statistics
```

**Properties:**
- `num_steps`: Number of time steps
- `node_names`: List of node names
- `source_names`: List of voltage source names with current data

**Methods:**
- `voltage(node: str) -> Array`: Get voltage waveform at a specific node
- `current(source: str) -> Array`: Get current waveform through a voltage source

**Example:**
```python
engine.prepare(t_stop=1e-6, dt=1e-9)
result = engine.run_transient()

# Access results
print(f"Simulated {result.num_steps} time points")
print(f"Nodes: {result.node_names}")

# Get specific node voltage
vout = result.voltage("out")
print(f"Final output: {vout[-1]:.3f}V")

# Iterate over all nodes
for name, voltages in result.voltages.items():
    print(f"  {name}: {voltages[-1]:.3f}V")
```

---

### ACResult

Result of AC analysis.

```python
@dataclass
class ACResult:
    frequencies: Array           # Frequency points (Hz)
    voltages: Dict[str, Array]   # Complex voltages at each frequency
    currents: Dict[str, Array]   # Complex currents at each frequency
```

**Example:**
```python
result = engine.run_ac(freq_start=1e3, freq_stop=1e9, points=100)

# Get transfer function magnitude
vout = result.voltages["out"]
magnitude_db = 20 * jnp.log10(jnp.abs(vout))
phase_deg = jnp.angle(vout) * 180 / jnp.pi
```

---

### NoiseResult

Result of noise analysis.

```python
@dataclass
class NoiseResult:
    frequencies: Array           # Frequency points (Hz)
    input_noise: Array           # Input-referred noise spectral density
    output_noise: Array          # Output noise spectral density
    noise_figure: Array          # Noise figure (dB)
```

---

### CornerResult

Result of a single PVT corner simulation.

```python
@dataclass
class CornerResult:
    corner: CornerConfig         # Corner configuration used
    result: Any                  # Simulation result (TransientResult, etc.) or None if failed
    converged: bool              # Whether simulation converged successfully
    stats: Dict[str, Any]        # Additional statistics and metadata
```

---

## Utility Functions

### create_pvt_corners()

Create PVT corner combinations for corner analysis.

```python
from vajax.analysis.corners import create_pvt_corners

corners = create_pvt_corners(
    processes=['FF', 'TT', 'SS'],          # Process corners
    voltages=[0.9, 1.0, 1.1],             # VDD scale factors
    temperatures=['cold', 'room', 'hot'],  # Temperature corners
)
```

**Returns:** List of `CornerConfig` objects for all combinations (default: 27 = 3x3x3).

---

### configure_precision()

Configure JAX floating-point precision.

```python
from vajax import configure_precision

configure_precision(force_x64=True)   # Force float64
configure_precision(force_x64=False)  # Force float32
configure_precision()                  # Auto-detect based on backend
```

---

### get_precision_info()

Get current precision configuration.

```python
from vajax import get_precision_info

info = get_precision_info()
print(f"Backend: {info['backend']}")
print(f"Float64 enabled: {info['x64_enabled']}")
```

---

## I/O Functions

### write_rawfile()

Write results to ngspice-compatible raw file.

```python
from vajax.io import write_rawfile

write_rawfile(result, "output.raw", binary=True)
```

**Parameters:**
- `result`: TransientResult or ACResult
- `output_path`: Path to output file
- `binary`: If True, write binary format (more compact)

---

### write_csv()

Write results to CSV file.

```python
from vajax.io import write_csv

write_csv(result, "output.csv", precision=9)
```

**Parameters:**
- `result`: Simulation result
- `output_path`: Path to output file
- `precision`: Decimal places for scientific notation

---

### read_csv()

Read simulation results from CSV file.

```python
from vajax.io import read_csv

data = read_csv("output.csv")
times = data['times']
voltages = data['voltages']
```

---

## Sparse Solver Functions

For large circuits (>1000 nodes), use sparse matrix solvers:

```python
from vajax.analysis.sparse import (
    build_csr_arrays,    # COO to CSR conversion
    sparse_solve_csr,    # Sparse linear solve
)
```

See `vajax/analysis/sparse.py` for details.

---

## Example: Complete Workflow

```python
from vajax import CircuitEngine, configure_precision
from vajax.io import write_rawfile

# Configure precision
configure_precision(force_x64=True)

# Load and parse circuit
engine = CircuitEngine("ring_oscillator.sim")
engine.parse()

# Run transient analysis
engine.prepare(t_stop=100e-9, dt=1e-9, use_sparse=True)
tran = engine.run_transient()
print(f"Transient: {tran.num_steps} steps, final Vout={tran.voltage('out')[-1]:.3f}V")

# Run AC analysis
ac = engine.run_ac(freq_start=1e3, freq_stop=1e9, points=100)
print(f"AC: {len(ac.frequencies)} frequency points")

# Run noise analysis
noise = engine.run_noise(freq_start=1e3, freq_stop=1e9, input_source="vin", out="vout")

# Save results
write_rawfile(tran, "transient.raw")
```

---

## See Also

- [CLI Reference](cli_reference.md) - Command-line interface
- [Architecture Overview](architecture_overview.md) - System design
- [GPU Solver Architecture](gpu_solver_architecture.md) - Performance optimization
