# VA-JAX API Reference

## Core Classes

### CircuitEngine

The main class for circuit simulation. Parses VACASK/SPICE netlists and runs various analyses.

```python
from vajax import CircuitEngine

engine = CircuitEngine("circuit.sim")
engine.parse()
result = engine.run_transient(t_stop=1e-6, dt=1e-9)
```

#### Constructor

```python
CircuitEngine(sim_path: Path | str)
```

**Parameters:**
- `sim_path`: Path to circuit file (`.sim` VACASK format or SPICE netlist)

#### Methods

##### parse()

Parse the circuit file and compile device models.

```python
engine.parse()
```

Loads the netlist, compiles Verilog-A models via OpenVAF, and prepares the circuit for simulation.

---

##### run_transient()

Run transient (time-domain) simulation.

```python
result = engine.run_transient(
    t_stop: float = None,      # Stop time (seconds)
    dt: float = None,          # Time step (seconds)
    use_scan: bool = True,     # Use lax.scan for GPU efficiency
    use_sparse: bool = False,  # Use sparse solver (for large circuits)
    max_steps: int = None,     # Maximum number of steps
    abstol: float = 1e4,       # Absolute tolerance for NR convergence
) -> TransientResult
```

**Parameters:**
- `t_stop`: Simulation stop time. If None, uses value from netlist `.control` section
- `dt`: Time step. If None, uses value from netlist
- `use_scan`: Use `jax.lax.scan` for fully-traced GPU execution
- `use_sparse`: Use sparse matrix solver (required for >1000 nodes)
- `max_steps`: Override maximum number of time steps
- `abstol`: Convergence tolerance for Newton-Raphson

**Returns:** `TransientResult`

---

##### run_ac()

Run AC (small-signal frequency response) analysis.

```python
result = engine.run_ac(
    fstart: float,            # Start frequency (Hz)
    fstop: float,             # Stop frequency (Hz)
    num_points: int = 100,    # Number of frequency points
    sweep_type: str = 'dec',  # 'dec', 'lin', or 'oct'
) -> ACResult
```

**Parameters:**
- `fstart`: Start frequency in Hz
- `fstop`: Stop frequency in Hz
- `num_points`: Number of frequency points
- `sweep_type`: Sweep type - `'dec'` (decade), `'lin'` (linear), or `'oct'` (octave)

**Returns:** `ACResult`

---

##### run_noise()

Run noise analysis.

```python
result = engine.run_noise(
    fstart: float,           # Start frequency (Hz)
    fstop: float,            # Stop frequency (Hz)
    input_source: str,       # Name of input source (e.g., "vin")
    output_node: str,        # Name of output node (e.g., "vout")
    num_points: int = 100,   # Number of frequency points
    sweep_type: str = 'dec', # 'dec', 'lin', or 'oct'
) -> NoiseResult
```

**Returns:** `NoiseResult`

---

##### run_corners()

Run PVT (Process-Voltage-Temperature) corner analysis.

```python
from vajax.analysis import create_pvt_corners

corners = create_pvt_corners(
    process=['tt', 'ff', 'ss'],
    voltage=[0.9, 1.0, 1.1],
    temperature=[233, 300, 398],
)

results = engine.run_corners(corners) -> List[CornerResult]
```

**Returns:** List of `CornerResult` objects, one per corner combination.

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
    input_source: str,  # Name of input source
    output_node: str,   # Name of output node
) -> DCXFResult
```

**Returns:** `DCXFResult` with transfer function magnitude and phase.

---

##### run_acxf()

Run AC transfer function analysis.

```python
result = engine.run_acxf(
    input_source: str,       # Name of input source
    output_node: str,        # Name of output node
    fstart: float = 1.0,     # Start frequency (Hz)
    fstop: float = 1e9,      # Stop frequency (Hz)
    num_points: int = 100,   # Number of frequency points
) -> ACXFResult
```

**Returns:** `ACXFResult` with frequency-dependent transfer function.

---

##### run_hb()

Run harmonic balance analysis.

```python
result = engine.run_hb(
    freq: float,            # Fundamental frequency (Hz)
    num_harmonics: int = 7, # Number of harmonics
) -> HBResult
```

**Returns:** `HBResult` with harmonic amplitudes and phases.

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
    stats: Dict[str, Any]           # Simulation statistics
```

**Properties:**
- `num_steps`: Number of time steps
- `node_names`: List of node names

**Methods:**
- `voltage(node: str) -> Array`: Get voltage waveform at a specific node

**Example:**
```python
result = engine.run_transient(t_stop=1e-6, dt=1e-9)

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
result = engine.run_ac(fstart=1e3, fstop=1e9, num_points=100)

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
    corner_name: str             # Corner identifier (e.g., "tt_1.0V_300K")
    process: str                 # Process corner (tt, ff, ss, ...)
    voltage: float               # Supply voltage
    temperature: float           # Temperature (Kelvin)
    result: TransientResult      # Simulation result for this corner
```

---

## Utility Functions

### create_pvt_corners()

Create PVT corner combinations for corner analysis.

```python
from vajax.analysis import create_pvt_corners

corners = create_pvt_corners(
    process=['tt', 'ff', 'ss'],     # Process corners
    voltage=[0.9, 1.0, 1.1],        # Supply voltages
    temperature=[233, 300, 398],    # Temperatures (Kelvin)
)
```

**Returns:** List of `PVTCorner` objects for all combinations.

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
tran = engine.run_transient(t_stop=100e-9, dt=1e-9, use_sparse=True)
print(f"Transient: {tran.num_steps} steps, final Vout={tran.voltage('out')[-1]:.3f}V")

# Run AC analysis
ac = engine.run_ac(fstart=1e3, fstop=1e9, num_points=100)
print(f"AC: {len(ac.frequencies)} frequency points")

# Run noise analysis
noise = engine.run_noise(fstart=1e3, fstop=1e9, input_source="vin", output_node="vout")
print(f"Noise figure at 1MHz: {noise.noise_figure[50]:.2f} dB")

# Save results
write_rawfile(tran, "transient.raw")
```

---

## See Also

- [CLI Reference](cli_reference.md) - Command-line interface
- [Architecture Overview](architecture_overview.md) - System design
- [GPU Solver Architecture](gpu_solver_architecture.md) - Performance optimization
