# jax-spice TODO

Central tracking for development tasks and known issues.

## Critical - Blocking Transient Analysis

### ddt() Operator Returns Zero

**Status**: CRITICAL BUG - All capacitive/charge contributions are ignored

**Problem**: The `ddt()` operator in openvaf_jax.py always returns zero regardless of analysis type.
This means:
- Capacitors don't charge/discharge in transient
- MOSFET charge contributions (Qg, Qd, Qb) are zero
- Ring oscillator settles to DC instead of oscillating
- Any circuit relying on energy storage fails

**Locations** (TWO places!):
- `openvaf-py/openvaf_jax.py:3069-3071` - `_translate_callback_result()` returns `'_ZERO'`
- `openvaf-py/openvaf_jax.py:3280-3281` - eval function translation returns `expr_builder.zero()`

**Fix required**: Return the charge expression instead of zero. The transient solver computes `dQ/dt = (Q - Q_prev) / dt`.

**Test**: Run ring benchmark - should show oscillation with ~578 zero crossings (currently shows 1).

## High Priority

### Make jitted function to *not* change with number of steps

When used as an API, we should expect that a user would ask for different lengths of simuation of the same model.
This issue is likely caused by the array sizes for the while loop (scan mode)

### Branch Current Computation

**Status**: Not implemented - JAX-SPICE only returns node voltages

**Problem**: Many ngspice tests print `I(V1)` (source currents) which map to `v1#branch` in output.
JAX-SPICE's `TransientResult` only contains `voltages` dict, not branch currents.

**Impact**: Tests with only current outputs are skipped (no comparable signals).

**Fix needed**: Compute and return branch currents through voltage sources.

### Analysis Type Support

**Implemented in `CircuitEngine`**:
- `run_transient()` - transient analysis ✅
- `run_ac()` - AC small-signal frequency sweep ✅
- `run_dcinc()` - DC incremental (small-signal) ✅
- `run_dcxf()` / `run_acxf()` - transfer functions ✅
- `run_noise()` - noise analysis ✅
- `run_corners()` - corner/monte-carlo ✅
- DC operating point - internal, used by other analyses ✅

**Missing API**:
- `run_dc_sweep()` - Sweep a source value, re-solve DC at each step
  - ngspice: `.dc Vsource START STOP STEP`
  - Different from `run_dcinc()` which is small-signal around one OP
  - 817 tests blocked (25 ngspice + 792 Xyce)

**Test runner integration needed**:
| Type | Tests | Status |
|------|-------|--------|
| `.tran` | 1018 | ✅ Integrated |
| `.ac` | 85 | API exists, runner needs to call `run_ac()` |
| `.op` | 79 | API exists internally, expose to runner |
| `.dc` | 817 | **Missing `run_dc_sweep()` API** |

### External Simulator Regression Suites

**Goal**: Compare JAX-SPICE against ngspice and Xyce reference implementations to validate correctness.

#### ngspice Regression Suite (`vendor/ngspice/tests/`)

**Current Status (2025-01)**:
- 70 tests discovered via auto-discovery
- 65 tests have reference files (`.out` or `.standard` format)
- 541 `.standard` reference files available (HiSIM, BSIM, etc.)
- Infrastructure: `tests/test_ngspice_regression.py`, `tests/ngspice_test_registry.py`
- Reference parsers: `jax_spice/io/ngspice_out_reader.py`

**Test Discovery**:
- Auto-discovers all `.sp`, `.cir`, `.spice` files in `vendor/ngspice/tests/`
- Detects analysis type, device types, expected output signals
- Finds reference files in same directory (`.out`) or `reference/` subdir (`.standard`)

**Current Blockers**:
- [ ] **ddt() bug** - Capacitors/MOSFETs don't work in transient
- [ ] **Branch currents** - Many tests print `I(source)`, not `V(node)`
- [ ] **Behavioral sources** - 'b' devices not supported by converter
- [ ] **ASCII plot format** - Some `.out` files use plot format, not tabular

**CMC QA Framework** (`tests/bin/runQaTests.pl`):
- 13 qaSpec test suites: BSIM3, BSIM4, BSIMSOI, HiSIM, HiSIM-HV1/2, HICUM2
- 541 `.standard` reference files in `reference/` subdirectories
- Defines DC sweeps, AC, noise tests with bias conditions
- Generates netlists via Perl scripts (`modelQaTestRoutines.pm`)
- Future: Parse qaSpec format to generate JAX-SPICE tests directly

**Device Support Gaps**:
- [ ] BJT (`q` devices) - rtlinv.cir, analog tests
- [ ] JFET (`j` devices) - jfet tests
- [ ] Controlled sources (VCVS, CCCS, VCCS, CCVS) - `e`, `f`, `g`, `h` devices
- [ ] Behavioral sources (`b` devices) - mesa/mesosc.cir, etc.
- [ ] Transmission lines (`t` devices)

#### Xyce Regression Suite (`vendor/Xyce_Regression/`)

**Current Status (2025-01)**:
- 1929 tests discovered via auto-discovery
- Infrastructure: `tests/test_xyce_regression.py`, `tests/xyce_test_registry.py`
- Reference files: `vendor/Xyce_Regression/OutputData/*.prn`
- Reference parser: `jax_spice/io/prn_reader.py`

**Test Discovery**:
- Auto-discovers all `.cir` files in `vendor/Xyce_Regression/Netlists/`
- Matches with expected output in `OutputData/<category>/<file>.cir.prn`
- Detects analysis type and device types

**Current Blockers** (same as ngspice):
- [ ] **ddt() bug** - Capacitor/MOSFET transients broken
- [ ] **Device gaps** - BJT, JFET, controlled sources, PDE devices

**Xyce-specific devices**:
- [ ] PDE devices (`y` prefix) - Xyce-specific
- [ ] Digital devices (`p` prefix) - Xyce-specific
- [ ] Mutual inductors (`u` prefix)
- [ ] Coupling (`k` prefix)

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

## Needed for release

### Documentation
- [ ] **Update README** with current project status
- [ ] **Add architecture overview** diagram
- [ ] **Document openvaf_jax API** for external users

## Nice to have

### Code Cleanup
- [x] ~~**Remove xfail markers** from PSP/JUNCAP/diode_cmc tests~~ (all pass)
- [ ] **Consolidate test files** in `openvaf-py/tests/` (some at root level)

## Completed

- [x] ~~Test suite auto-discovery~~ (2025-01)
  - ngspice: 70 tests discovered, 65 with reference files
  - Xyce: 1929 tests discovered via registry
  - Full parametrized test generation without manual curation

- [x] ~~Reference file parsers~~ (2025-01)
  - Added `jax_spice/io/ngspice_out_reader.py` for `.out` and `.standard` formats
  - Added `jax_spice/io/prn_reader.py` for Xyce `.prn` format
  - Signal name mapping: `i(v1)` → `v1#branch` in ngspice output

- [x] ~~Test registries~~ (2025-01)
  - `tests/ngspice_test_registry.py` - discovers tests, detects devices/analysis/signals
  - `tests/xyce_test_registry.py` - discovers Xyce tests with output matching
  - Reference file detection in same directory or `reference/` subdir

- [x] ~~Fixed SI suffix parsing in safe_eval.py~~ (2025-01)
  - Added time units (ms, us, ns, ps, fs) to `jax_spice/utils/safe_eval.py`
  - Added voltage/current units (mv, uv, nv, ma, ua, na, pa, fa)
  - Fixes PULSE source parameter parsing (delay, rise, fall, width, period)
  - Required for Xyce DIODE test and any netlists using time units

- [x] ~~ngspice regression suite infrastructure~~ (2025-01)
  - Added `test_ngspice_regression.py` with parametrized tests
  - Added `ngspice_test_registry.py` for test discovery and categorization
  - Fixed SPICE→VACASK converter for pulse sources with DC prefix
  - Fixed SI suffix parsing in source parameters (e.g., "1u" → 1e-6)

- [x] ~~Xyce regression suite infrastructure~~ (2025-01)
  - Added `test_xyce_regression.py` with PRN comparison framework
  - Added `jax_spice/io/prn_reader.py` for Xyce output parsing

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
| **OpenVAF→JAX translator** | `openvaf-py/openvaf_jax.py` |
| VACASK parser | `jax_spice/netlist/parser.py` |
| VACASK suite tests | `tests/test_vacask_suite.py` |
| **ngspice regression tests** | `tests/test_ngspice_regression.py` |
| ngspice test registry | `tests/ngspice_test_registry.py` |
| ngspice .out/.standard parser | `jax_spice/io/ngspice_out_reader.py` |
| **Xyce regression tests** | `tests/test_xyce_regression.py` |
| Xyce test registry | `tests/xyce_test_registry.py` |
| Xyce .prn parser | `jax_spice/io/prn_reader.py` |
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
