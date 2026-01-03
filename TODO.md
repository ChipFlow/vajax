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

### Convert remaining non-JAX analysis to JAX

**Status**: Several analysis functions still use numpy/Python loops instead of JAX.

**Critical (numpy in core solver path)**:
- [ ] `transient_analysis_analytical()` in `transient.py` - uses `np.linalg.solve()` and Python `for` loops
- [ ] `_transient_analysis_python()` in `transient.py` - deprecated but still present

**Medium (outer loops, inner solves use JAX)**:
- [ ] `dc_operating_point_with_source_stepping()` - Python `while` loop for VDD stepping
- [ ] `dc_operating_point_with_gmin_stepping()` - Python `while` loop for GMIN stepping

**Goal**: Convert to use `jax.lax.scan` for time/source stepping and JAX linear solvers throughout.

### GPU Performance Issue (HIGH PRIORITY)

**Root cause identified (2025-12):** GPU is slower than CPU because `transient_analysis_jit` processes devices sequentially, not in parallel.

**Current implementation** (`_build_system_jit` in `transient.py`):
```python
# Sequential device processing - NOT GPU-friendly
jacobian, residual = lax.fori_loop(
    0, circuit.num_devices,
    stamp_one_device,
    (jacobian, residual)
)
```

This results in:
- Thousands of tiny GPU kernel launches (one per device)
- No parallelism to offset kernel launch overhead
- GPU adds overhead without speedup

**Solution**: Use vectorized device evaluation (already implemented for DC analysis):
- `build_device_groups()` groups devices by type
- `VectorizedDeviceGroup` evaluates all same-type devices in parallel via vmap
- Batched matrix scatter operations for Jacobian stamping

**Tasks**:
- [ ] Refactor `transient_analysis_jit` to use `VectorizedDeviceGroup` instead of `lax.fori_loop`
- [ ] Benchmark GPU vs CPU after vectorization
- [ ] Consider using `transient_analysis_analytical()` as starting point (uses SystemBuilder)

### openvaf_jax Complex Model Support
The JAX translator now matches the MIR interpreter for all models. NaN outputs are caused by model-specific parameter requirements, not translator bugs.

**Status**: JAX translator is **complete and consistent** with MIR interpreter.

**Completed**:
- [x] ~~Re-test affected models, compare against VACASK running same models~~
- [x] ~~Update xfail markers~~
- [x] ~~Add missing opcodes~~ (fbcast, irem, idiv)
- [x] ~~Verified JAX output matches MIR interpreter~~

**Working models** (xfail markers removed):
- PSP102, PSP103, JUNCAP
- diode_cmc
- EKV

**Models with NaN (same in JAX and interpreter)**:
These models produce NaN with default parameters but work with proper model card setup:
- BSIM3, BSIM4, BSIM6, BSIMBulk, BSIMCMG, BSIMSOI
- HiSIM2, HiSIMHV
- HICUM L2, MEXTRAM
- ASMHEMT, MVSG

**Note**: The NaN issues are due to model parameter requirements (e.g., division by zero in cutoff regions), not JAX translator bugs. Proper model cards (`.lib` files with device parameters) should resolve these.

### Complete testing of VACASK benchmarks

**Current status** (as of 2025-12):
| Benchmark | Device Types | Status |
|-----------|--------------|--------|
| rc | resistor, capacitor, vsource | ✅ Passing |
| graetz | resistor, capacitor, vsource, diode | ✅ Passing |
| mul | resistor, capacitor, vsource, diode | ✅ Passing |
| ring | vsource, isource, PSP103 MOSFET | ✅ Fast (~20ms/step after JIT warmup) |
| c6288 | vsource, isource, PSP103 MOSFET | ✅ Working with sparse solver (~1s/step, 86k nodes) |

**Transient solver device support**:
- [x] Resistor
- [x] Capacitor
- [x] Voltage source (DC and time-varying)
- [x] Current source (DC and pulse)
- [x] Diode (Shockley equation with limiting)
- [x] OpenVAF-compiled models (PSP103) - **fast with JIT compilation**

**PSP103/OpenVAF Integration Status**:
The hybrid solver is functionally complete:
- [x] PSP103 model compilation via OpenVAF
- [x] Model card parameters (N/P type, 280+ params) properly parsed
- [x] Internal nodes allocated (8 per MOSFET, 144 total for ring)
- [x] Voltage parameter mapping for internal node voltages
- [x] Residual/Jacobian stamping into expanded system matrix
- [x] Newton-Raphson converges (3 iterations for t=0)

**Performance Optimization (2025-12)**:
Implemented JIT-compiled vmap-based batched evaluation for OpenVAF devices:
- [x] Added `translate_array()` method to openvaf_jax.py for vmap-compatible output
- [x] Fixed boolean constants to use `jnp.bool_()` for JIT compatibility
- [x] Fixed `bnot` opcode to use `jnp.logical_not()` instead of Python `not`
- [x] Pre-computed static inputs (parameters) once per simulation
- [x] Fast voltage-only update path for NR iterations
- [x] JIT-compiled vmapped function for near-instant device evaluation

**Performance results** (18 PSP103 MOSFETs on ring oscillator):
| Metric | Before | After | Speedup |
|--------|--------|-------|---------|
| Device evaluation (18 MOSFETs) | ~2.2s | ~0ms | ∞ |
| Input preparation | ~39ms | ~1ms | 39x |
| Per-timestep total | ~680ms | ~20ms | **34x** |

**Warmup time**: ~4.8s (includes JAX JIT compilation, done once per model)

**Future optimization options**:
1. ~~Modify openvaf_jax to use `jax.lax.cond` for JIT compatibility~~ ✅ Done
2. Use GPU acceleration (now possible with JIT compilation)

## Low Priority

### Documentation
- [ ] **Update README** with current project status
- [ ] **Add architecture overview** diagram
- [ ] **Document openvaf_jax API** for external users

### Code Cleanup
- [ ] **Remove xfail markers** from PSP/JUNCAP/diode_cmc tests (they pass)
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
