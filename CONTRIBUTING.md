# Contributing to VAJAX

This guide covers development setup, code organization, and contribution guidelines.

## Start Here

If you're new to VAJAX, start with these resources:

1. **[Getting Started](docs/getting_started.md)** - Install and run your first simulation
2. **[Architecture Overview](docs/architecture_overview.md)** - How the pieces fit together
3. **[Supported Devices](docs/supported_devices.md)** - What device models are available

### Key Concepts

VAJAX is a **SPICE-class circuit simulator** built on JAX. If you're unfamiliar with
circuit simulation, here are the core concepts:

- **Modified Nodal Analysis (MNA)**: Formulates circuit equations as `G*V = I` where G is a
  conductance matrix, V is node voltages, and I is current sources. Each device "stamps" its
  contributions into G and I. See [MNA on Wikipedia](https://en.wikipedia.org/wiki/Modified_nodal_analysis).

- **Newton-Raphson iteration**: Nonlinear circuits require iterative solving. At each step,
  we linearize the circuit around the current solution and solve `J * delta = -f(V)` where
  J is the Jacobian and f is the residual (KCL violations).

- **Transient analysis**: Time-domain simulation using numerical integration (trapezoidal rule
  or Gear's method). Capacitors and inductors introduce time derivatives that are discretized
  into equivalent conductances.

- **Verilog-A / OpenVAF**: Device models (transistors, diodes, etc.) are written in Verilog-A,
  compiled by OpenVAF to machine code, then translated to JAX functions. This gives us
  production-quality device models with automatic differentiation.

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
cd vajax

# Install dependencies with uv (--extra test includes pytest and test utilities)
uv sync --extra test

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
vajax/
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
| `vajax.devices` | Device models | `verilog_a.py`, `vsource.py` |
| `vajax.analysis` | Solvers | `engine.py`, `mna.py`, `dc_operating_point.py`, `solver.py`, `transient/` |
| `vajax.netlist` | Parsing | `parser.py`, `circuit.py` |
| `vajax.benchmarks` | Benchmarks | `registry.py`, `runner.py` |

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

### Device Model Architecture

All devices except voltage/current sources are compiled from Verilog-A via
OpenVAF and wrapped by `VerilogADevice` (`vajax/devices/verilog_a.py`).
Device instances are grouped by type and evaluated in parallel using `jax.vmap`
for GPU efficiency. The engine handles MNA stamping automatically.

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

All devices (resistors, capacitors, diodes, MOSFETs, etc.) are routed through
OpenVAF Verilog-A compilation via `VerilogADevice` in `vajax/devices/verilog_a.py`.
The only exceptions are voltage and current sources, which are handled separately
in `vajax/devices/vsource.py`.

To add a new device:

1. Write or obtain a Verilog-A model (`.va` file)
2. Compile it with OpenVAF to produce an `.osdi` module
3. Reference it in a VACASK `.sim` netlist with `load "your_model.osdi"`
4. The engine will automatically wrap it via `VerilogADevice` with batched
   `jax.vmap` evaluation for GPU efficiency

There is no need to write Python device code — OpenVAF handles the compilation
from Verilog-A to JAX-compatible functions.

To add tests, create `tests/test_your_device.py` using the public
`CircuitEngine` API with a `.sim` netlist that exercises the device.

## Adding a New Analysis Type

1. Create `vajax/analysis/your_analysis.py`
2. Use `DeviceInfo` and `DeviceType` from `vajax/analysis/mna.py` for device management
3. For transient-style analyses, subclass `TransientStrategy` ABC from `vajax/analysis/transient/base.py`
4. Build on existing patterns from `dc_operating_point.py` or the `transient/` package
5. Export from `vajax/analysis/__init__.py`

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
