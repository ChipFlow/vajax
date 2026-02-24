# Contributing to VA-JAX

This guide covers development setup, code organization, and contribution guidelines.

## Development Setup

### Prerequisites

- Python 3.11-3.13
- [uv](https://docs.astral.sh/uv/) package manager
- Rust toolchain (for building openvaf-py)
- LLVM 18 (for OpenVAF, macOS only)

### Quick Setup

```bash
# Clone and enter directory
git clone <repo-url>
cd va-jax

# Install dependencies with uv
uv sync

# Run tests to verify setup
JAX_PLATFORMS=cpu uv run pytest tests/ -v
```

### Building OpenVAF (macOS)

OpenVAF requires LLVM 18 on macOS:

```bash
# Install LLVM via Homebrew
brew install llvm@18

# Build OpenVAF
./scripts/build_openvaf.sh

# Build openvaf-py
cd openvaf-py
LLVM_SYS_181_PREFIX=/opt/homebrew/opt/llvm@18 uv run maturin develop
```

### GPU Support (Linux + CUDA)

```bash
# Install with CUDA support
uv sync --extra cuda12

# Test GPU availability
JAX_PLATFORMS=cuda uv run python -c "import jax; print(jax.devices())"
```

## Project Structure

```
va-jax/
├── vajax/              # Main library
│   ├── devices/            # Device models
│   ├── analysis/           # Circuit solvers
│   ├── netlist/            # Netlist parsing
│   └── benchmarks/         # Benchmark infrastructure
├── openvaf-py/             # OpenVAF Python bindings (Rust)
├── tests/                  # Test suite
├── scripts/                # Build and profiling scripts
├── docs/                   # Architecture documentation
└── vendor/                 # External dependencies (VACASK)
```

### Key Modules

| Module | Purpose | Key Files |
|--------|---------|-----------|
| `vajax.devices` | Device models | `base.py`, `mosfet_simple.py`, `openvaf_device.py` |
| `vajax.analysis` | Solvers | `mna.py`, `dc.py`, `transient.py` |
| `vajax.netlist` | Parsing | `parser.py`, `circuit.py` |
| `vajax.benchmarks` | Benchmarks | `runner.py` |

## Running Tests

```bash
# All tests (force CPU for reproducibility)
JAX_PLATFORMS=cpu uv run pytest tests/ -v

# Specific test file
JAX_PLATFORMS=cpu uv run pytest tests/test_vacask_suite.py -v

# Single test
JAX_PLATFORMS=cpu uv run pytest tests/test_resistor.py::test_ohms_law -v

# With coverage
JAX_PLATFORMS=cpu uv run pytest tests/ --cov=vajax --cov-report=html

# openvaf-py tests (separate environment)
cd openvaf-py && JAX_PLATFORMS=cpu ../.venv/bin/python -m pytest tests/ -v
```

### Test Organization

- `tests/test_*.py` - Main library tests
- `tests/test_vacask_*.py` - VACASK benchmark integration tests
- `openvaf-py/tests/` - OpenVAF translator tests

## Code Style

### Python

```bash
# Format with ruff
uv run ruff format vajax tests

# Lint
uv run ruff check vajax tests

# Type check
uv run pyright vajax
```

Configuration is in `pyproject.toml`:
- Line length: 100 characters
- Target: Python 3.11
- Lints: E, F, I (isort), N (naming), W

### Guidelines

1. **Pure functions preferred**: Device models should be pure JAX functions without side effects
2. **Type hints**: Use type hints for public APIs
3. **Docstrings**: Document public functions with parameter descriptions
4. **No trailing whitespace**: Configure editor to trim trailing whitespace
5. **Snake case**: Use `snake_case` for functions and variables

## Architecture Concepts

### Device Model Protocol

All devices implement a common interface:

```python
from vajax.devices import DeviceStamps
from vajax.analysis import AnalysisContext
from jax import Array

def evaluate(
    terminal_voltages: Array,  # Voltages at device terminals
    params: dict,              # Device parameters
    context: AnalysisContext   # Time, temperature, etc.
) -> DeviceStamps:
    """Return currents and conductances for MNA stamping."""
    ...
```

### Modified Nodal Analysis (MNA)

The simulator uses MNA to form the circuit equations:

```
G*V + C*dV/dt = I

Where:
- G: Conductance matrix (from device stamps)
- V: Node voltages (unknowns)
- C: Capacitance matrix (for transient)
- I: Current vector (from sources)
```

### Newton-Raphson Iteration

DC and transient solvers use Newton-Raphson:

```
V_new = V - J^(-1) * f(V)

Where:
- f(V): Residual (KCL violations)
- J: Jacobian (df/dV, computed via JAX autodiff)
```

### Vectorized Device Evaluation

For performance, devices are grouped by type and evaluated in parallel:

```python
# Instead of:
for device in mosfets:
    I = device.evaluate(V)

# We do:
I_all = jax.vmap(mosfet_evaluate)(V_batched, params_batched)
```

## Adding a New Device

1. Create `vajax/devices/your_device.py`:

```python
import jax.numpy as jnp
from vajax.devices.base import DeviceStamps
from vajax.analysis.context import AnalysisContext

def your_device(
    V: jnp.ndarray,           # Terminal voltages [V1, V2, ...]
    params: dict,              # Device parameters
    context: AnalysisContext
) -> DeviceStamps:
    """Your device model.

    Parameters:
        V: Terminal voltages
        params: dict with keys 'param1', 'param2', etc.
        context: Analysis context

    Returns:
        DeviceStamps with currents and conductances
    """
    # Extract parameters
    param1 = params.get('param1', default_value)

    # Compute currents
    I = ...  # Current into each terminal

    # Compute conductances (dI/dV)
    G = ...  # 2D array of shape (n_terminals, n_terminals)

    return DeviceStamps(currents=I, conductances=G)
```

2. Export from `vajax/devices/__init__.py`

3. Add tests in `tests/test_your_device.py`

## Adding a New Analysis Type

1. Create `vajax/analysis/your_analysis.py`
2. Use `MNASystem` for device management
3. Build on existing patterns from `dc.py` or `transient.py`
4. Export from `vajax/analysis/__init__.py`

## Debugging Tips

### JAX Debugging

```python
# Print intermediate values (breaks JIT)
jax.debug.print("Value: {x}", x=my_array)

# Check for NaN
jax.config.update("jax_debug_nans", True)

# Disable JIT for debugging
with jax.disable_jit():
    result = my_function(inputs)
```

### Common Issues

1. **JIT compilation errors**: Usually from Python conditionals on traced values
   - Use `jax.lax.cond` instead of `if`
   - Use `jax.lax.select` for element-wise conditionals

2. **NaN in results**: Check for:
   - Division by zero (add small epsilon)
   - Log of non-positive values
   - Invalid device parameters

3. **Convergence failures**: Try:
   - Homotopy chain: `run_homotopy_chain()` from `vajax.analysis.homotopy`
   - GMIN stepping: `gmin_stepping()` with mode="gdev" or "gshunt"
   - Source stepping: `source_stepping()`
   - Increased iteration limit
   - Relaxed tolerances

### Profiling

```bash
# CPU profiling
JAX_PLATFORMS=cpu uv run python scripts/profile_gpu.py --benchmark ring

# GPU profiling with Perfetto traces
uv run python scripts/profile_gpu_cloudrun.py --benchmark ring --timesteps 50
```

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes with clear commit messages
3. Ensure tests pass: `JAX_PLATFORMS=cpu uv run pytest tests/ -v`
4. Run linter: `uv run ruff check vajax tests`
5. Push and create PR
6. Wait for CI checks to pass

### Commit Message Format

```
<type>: <short description>

<optional longer description>

Types:
- feat: New feature
- fix: Bug fix
- refactor: Code restructuring
- test: Adding tests
- docs: Documentation
- perf: Performance improvement
```

## Resources

- [JAX Documentation](https://jax.readthedocs.io/)
- [OpenVAF GitHub](https://github.com/pascalkuthe/OpenVAF)
- [SPICE Theory](https://en.wikipedia.org/wiki/SPICE)
- `docs/` folder for architecture details
