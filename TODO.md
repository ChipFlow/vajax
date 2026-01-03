# jax-spice TODO

Central tracking for development tasks and known issues.

## High Priority

### External Simulator Regression Suites

**Goal**: Compare JAX-SPICE against ngspice and Xyce reference implementations to validate correctness.

#### ngspice Regression Suite (`vendor/ngspice/tests/`)

**Current Status (2025-01)**:
- 113 test files across categories: resistance, filters, general, bsim*, hisim, jfet, etc.
- Infrastructure: `tests/test_ngspice_regression.py`, `tests/ngspice_test_registry.py`
- Passing: 2 tests (res_simple, test_rc_benchmark)
- Skipped: 2 tests (lowpass - AC analysis, rtlinv - BJT device)

**Supported test types**:
- [x] Resistor circuits (res_simple, res_array, res_partition)
- [x] RC circuits (rc in general/, RC benchmark comparison)
- [ ] RLC circuits (need inductor support)
- [ ] Diode circuits (basic diode tests)
- [ ] Filter circuits (lowpass - needs AC analysis)

**Blocking features needed**:
- [ ] AC analysis support (for lowpass.cir and filter tests)
- [ ] Inductor device support (for RLC tests)
- [ ] BJT device support (for rtlinv.cir and analog tests)
- [ ] MOSFET tests via PSP103 (for general/mosamp.cir, fourbitadder.cir)

**Next steps**:
- [ ] Add res_array and res_partition tests (resistor arrays)
- [ ] Add general/rc.cir test (basic RC transient)
- [ ] Implement inductor device for RLC tests
- [ ] Add diode-only transient tests from ngspice suite

#### Xyce Regression Suite (`vendor/Xyce_Regression/`)

**Current Status (2025-01)**:
- 267+ test directories covering all device types and analysis modes
- Infrastructure: `tests/test_xyce_regression.py`, `jax_spice/io/prn_reader.py`
- Passing: 0 tests
- xfail: 1 test (DIODE/diode.cir - model parameter differences)

**Simple tests to add (using supported devices)**:
- [ ] RESISTOR/resistor.cir - basic resistor DC/transient
- [ ] CAPACITOR/capacitor.cir - capacitor transient
- [ ] CAPACITOR/rc_osc.cir - RC oscillator
- [ ] RLC/rlc.cir - RLC circuit (needs inductor)
- [ ] SOURCES/sources.cir - various source types

**Blocking features needed**:
- [ ] Fix diode model parameter alignment with Xyce defaults
- [ ] Inductor device support (for RLC tests)
- [ ] Controlled sources (VCVS, CCCS, VCCS, CCVS) for ABM tests

**Infrastructure improvements**:
- [ ] Auto-discover compatible Xyce tests (like ngspice registry)
- [ ] Add tolerance override per test (some need looser tolerances)
- [ ] Generate coverage report showing pass/fail/skip by category

### Convert Homotopy Loops to JAX

**Status**: Homotopy (continuation) methods use Python loops that could be JIT-compiled.

**Medium priority (outer loops, inner solves already use JAX)**:
- [ ] `homotopy.py:132` - GMIN stepping uses `for step in range()`
- [ ] `homotopy.py:348` - Source stepping uses `for step in range()`

**Goal**: Convert to `lax.fori_loop` or `lax.while_loop` for full JIT compilation.

**Note**: These are rarely-used fallback methods for difficult convergence cases. Low priority.

### Simplify Transient Analysis Paths

**Current state**: Multiple overlapping implementations exist:
- `PythonLoopStrategy` - Python for-loop with per-step debugging
- `ScanStrategy` - lax.scan for full JIT compilation (5x faster)
- `_run_transient_hybrid` - another Python loop (duplicates PythonLoopStrategy)
- `_run_transient_while_loop` - lax.while_loop version

**GPU status**: GPU is ~4x faster than CPU (working correctly).

**Tasks**:
- [ ] Change default from `use_while_loop=False` to `use_while_loop=True`
- [ ] Remove duplicated Python loop code in `_run_transient_hybrid`
- [ ] Keep `PythonLoopStrategy` as opt-in debugging mode only

## Low Priority

### Documentation
- [ ] **Update README** with current project status
- [ ] **Add architecture overview** diagram
- [ ] **Document openvaf_jax API** for external users

### Code Cleanup
- [x] ~~**Remove xfail markers** from PSP/JUNCAP/diode_cmc tests~~ (all pass)
- [ ] **Consolidate test files** in `openvaf-py/tests/` (some at root level)

### Technical Debt: Code Duplication

**Status (2025-12):** Unified solver and system builder infrastructure complete.

**Completed:**
- [x] Created `jax_spice/analysis/solver.py` with unified NR iteration loop
- [x] Created `jax_spice/analysis/system.py` with SystemBuilder for unified J,f construction
- [x] Added `CompiledModelBatch` to `jax_spice/devices/openvaf_device.py` for batched OpenVAF evaluation
- [x] Added `dc_operating_point_analytical()` using analytical Jacobians
- [x] Added `transient_analysis_analytical()` using analytical Jacobians
- [x] Both functions use SystemBuilder with OpenVAF analytical Jacobians (no autodiff)

**Current architecture:**
```
jax_spice/analysis/
├── solver.py          # Core NR loop with lax.while_loop
│   ├── NRConfig       # Solver configuration
│   ├── NRResult       # Result with convergence info
│   ├── newton_solve() # Residual+Jacobian functions
│   └── solve_dc_with_builder()  # SystemBuilder wrapper
│
├── system.py          # Unified system building
│   ├── SystemBuilder  # Manages simple + OpenVAF devices
│   ├── SimpleDevice   # Simple device info
│   └── build_system() # Returns (J, f) with analytical Jacobians
│
├── dc.py              # DC analysis
│   ├── dc_operating_point()           # Original (autodiff)
│   └── dc_operating_point_analytical() # NEW: Uses SystemBuilder
│
├── transient.py       # Transient analysis
│   ├── transient_analysis_jit()         # Original (hardcoded devices)
│   └── transient_analysis_analytical()  # NEW: Uses SystemBuilder
│
jax_spice/devices/
├── openvaf_device.py  # OpenVAF device support
│   ├── VADevice           # Single device evaluation
│   ├── CompiledModelBatch # Batched vmapped evaluation
│   └── compile_model()    # Model compilation with caching

jax_spice/benchmarks/
└── runner.py          # VACASK benchmark runner (separate for now)
```

**Remaining work:**
- [ ] Migrate runner.py to use SystemBuilder (non-blocking, works as-is)
- [ ] Add sparse solver support to SystemBuilder

### Build System
- [ ] **Upstream VACASK macOS fixes** to original repo
  - Current workaround: `robtaylor/VACASK` fork with `macos-fixes` branch
  - Fixes: C++20, PTBlockSequence, VLA→vector, KLU destructor, <numbers> header, CMake var escaping

## Completed

- [x] ~~ngspice regression suite infrastructure~~ (2025-01)
  - Added `test_ngspice_regression.py` with curated test list
  - Added `ngspice_test_registry.py` for test discovery and categorization
  - Fixed SPICE→VACASK converter for pulse sources with DC prefix
  - Fixed SI suffix parsing in source parameters (e.g., "1u" → 1e-6)
  - Passing: res_simple, test_rc_benchmark (2/4, others skipped for missing features)

- [x] ~~Xyce regression suite infrastructure~~ (2025-01)
  - Added `test_xyce_regression.py` with PRN comparison framework
  - Added `jax_spice/io/prn_reader.py` for Xyce output parsing
  - One test added (DIODE/diode.cir) - xfail due to model parameter differences

- [x] ~~openvaf_jax Complex Model Support~~ (2025-12)
  - JAX translator matches MIR interpreter for all models
  - Added missing opcodes: fbcast, irem, idiv
  - Working models (no xfail): PSP102, PSP103, JUNCAP, diode_cmc, EKV
  - NaN models need proper model cards (BSIM*, HiSIM*, HICUM, etc.)

- [x] ~~VACASK benchmark testing~~ (2025-12)
  - All 5 benchmarks passing: rc, graetz, mul, ring, c6288
  - Device support: resistor, capacitor, vsource, isource, diode, PSP103 MOSFET
  - PSP103 JIT-compiled vmap evaluation: 34x speedup (680ms→20ms per step)
  - c6288 working with sparse solver (~1s/step, 86k nodes)

- [x] ~~Sparse matrix support for large circuits~~ (2025-12)
  - Implemented sparse Jacobian assembly using `scipy.sparse.lil_matrix`
  - Sparse linear solve via `scipy.sparse.linalg.spsolve`
  - Auto-detects when to use sparse (>1000 nodes)
  - c6288 benchmark: 86k nodes, 490k non-zeros (0.007% density), ~1s/step

- [x] ~~VACASKBenchmarkRunner module~~ (`jax_spice/benchmarks/`)
  - Generic runner for VACASK benchmark circuits
  - Subcircuit flattening with parameter expression evaluation
  - Uses production `transient_analysis_jit()` for simulation
  - Supports: resistor, capacitor, vsource, isource, diode, PSP103 MOSFET
  - Sparse and dense solver modes with automatic selection
  - Parses analysis params from .sim files (step, stop, icmode)
  - All benchmarks passing: rc, graetz, mul, ring, c6288

- [x] ~~Removed legacy GPU solvers~~ (dc_gpu.py and transient_gpu.py)
  - Removed sparsejac dependency
  - VACASKBenchmarkRunner now handles all benchmarking

- [x] ~~Create test suite using VACASK sim files~~ (`tests/test_vacask_jax.py`)
  - Parses actual VACASK `.sim` files
  - Compiles VA models with openvaf_jax
  - Tests: resistor (Ohm's law, mfactor), diode, capacitor, inductor, op
  - 9 tests passing

- [x] ~~Fix VACASK netlist parser~~ (all 37 test files pass)
  - Added @if/@endif directive handling
  - Added vector parameter `[...]` support
  - Fixed title parsing for keywords

- [x] ~~Fix multi-way PHI nodes in openvaf_jax~~
  - MOSFET JAX output now matches MIR interpreter to 6 significant figures
  - Added `_build_multi_way_phi()` for >2 predecessor blocks

- [x] ~~Fix PMOS current sign convention~~ (historical)

- [x] ~~Add gds_min leakage~~ to MOSFET model (historical)

- [x] ~~Document VACASK OSDI input handling~~ (`docs/vacask_osdi_inputs.md`)

- [x] ~~Add OpenVAF/VACASK build scripts~~ for macOS
  - `scripts/build_openvaf.sh`
  - `scripts/build_vacask.sh`

## Reference

### Key Files
| Purpose | Location |
|---------|----------|
| **Newton-Raphson solver** | `jax_spice/analysis/solver.py` |
| **System builder** | `jax_spice/analysis/system.py` |
| DC operating point | `jax_spice/analysis/dc.py` |
| Transient solver | `jax_spice/analysis/transient.py` |
| MNA system | `jax_spice/analysis/mna.py` |
| GPU backend selection | `jax_spice/analysis/gpu_backend.py` |
| Benchmark runner | `jax_spice/benchmarks/runner.py` |
| Benchmark profiling | `scripts/profile_gpu.py` |
| Cloud Run profiling | `scripts/profile_gpu_cloudrun.py` |
| **OpenVAF batched eval** | `jax_spice/devices/openvaf_device.py` |
| OpenVAF→JAX translator | `openvaf-py/openvaf_jax.py` |
| VACASK parser | `jax_spice/netlist/parser.py` |
| VACASK suite tests | `tests/test_vacask_suite.py` |
| **ngspice regression tests** | `tests/test_ngspice_regression.py` |
| ngspice test registry | `tests/ngspice_test_registry.py` |
| **Xyce regression tests** | `tests/test_xyce_regression.py` |
| PRN file reader | `jax_spice/io/prn_reader.py` |
| SPICE→VACASK converter | `jax_spice/netlist_converter/ng2vclib/` |

### Test Commands
```bash
# Run all tests
JAX_PLATFORMS=cpu uv run pytest tests/ -v

# Run VACASK suite tests
JAX_PLATFORMS=cpu uv run pytest tests/test_vacask_suite.py -v

# Run ngspice regression tests
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 uv run pytest tests/test_ngspice_regression.py -v

# Run Xyce regression tests
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 uv run pytest tests/test_xyce_regression.py -v

# Run openvaf-py tests
cd openvaf-py && JAX_PLATFORMS=cpu ../.venv/bin/python -m pytest tests/ -v

# Run local benchmark profiling
JAX_PLATFORMS=cpu uv run python scripts/profile_gpu.py --benchmark ring

# Run Cloud Run GPU profiling with Perfetto traces
uv run scripts/profile_gpu_cloudrun.py --benchmark ring --timesteps 50
```
