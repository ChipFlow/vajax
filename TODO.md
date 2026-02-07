# jax-spice TODO

Central tracking for development tasks and known issues.

## High Priority

### Graetz Benchmark - NR Convergence During Diode Turn-On

**Status**: Partially fixed. Bugs in DEVpnjlim and iniLim corrected. With abstol=1e-9, simulation reaches 99.5% completion. Residual explosions still occur at some stiff transition points.

**Bugs fixed**:
1. **DEVpnjlim algorithm bug** - diode.va used ngspice's buggy `log(arg-2)` formula which produces negative values when `|vnew-vold| < 3*vt`. Fixed to use VACASK's corrected `log(1+arg)` formula.
2. **iniLim simparam bug** - The check `nr_iteration == 1` was wrong since iteration starts at 0. Fixed to `nr_iteration == 0`.
3. **isource COOVector bug** - isource stamping appended a plain tuple instead of COOVector to f_resist_parts.

**Remaining issues**:
- With tight abstol (1e-12), NR still fails at stiff transitions
- With looser abstol (1e-9), simulation completes but force-accepts some steps with high residuals
- Residual explosions (1e+5) at some transition points suggest numerical instability

**Workaround**: Use `engine.options.abstol = 1e-9` for graetz benchmark.

**Related files**:
- `vendor/VACASK/devices/spice/sn/diode.va` - DEVpnjlim function (lines 403-440)
- `jax_spice/analysis/mna_builder.py` - iniLim simparam setting (line 355)
- `jax_spice/analysis/mna_builder.py` - isource stamping (line 316)

### Spineax Solver Missing limit_state Threading

**Bug**: The Spineax solver (`make_spineax_full_mna_solver` in `solver_factories.py`) does not thread `limit_state` through NR iterations. It always passes `None` for `limit_state_in`, breaking device-level $limit callbacks (pnjlim/fetlim).

**Impact**: Device limiting (pnjlim for diodes, fetlim for MOSFETs) does not work with Spineax backend.

**Fix needed**: Add `limit_state` to the state tuple (like dense solver does at line 219) and thread `limit_state_out` through iterations.

**Workaround**: Use `backend="dense"` for circuits requiring device limiting.

### Make jitted function to *not* change with number of steps

When used as an API, we should expect that a user would ask for different lengths of simulation of the same model.
This issue is likely caused by the array sizes for the while loop (scan mode)

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

**Current Blockers**:
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

### NR Device Bypass Infrastructure

**Status**: Not implemented. VACASK has device bypass optimization that we don't support.

**VACASK options** (from `simulation_options.md`):
- `nr_bypass` (default 0): Enable instance bypass - skip device re-evaluation when inputs barely changed
- `nr_contbypass` (default 1): Allow forced bypass in first NR iteration of continuation mode
- `nr_bypasstol` (default 0.01): Bypass tolerance factor for instance input check

**Current behavior**: We always evaluate all devices every iteration, which matches `nr_bypass=0`.

**Impact**: Performance only. With `nr_bypass=0` (disabled), correctness is unaffected.
The graetz benchmark uses `nr_bypass=0 nr_contbypass=1`, so our current behavior is correct.

**Implementation notes**:
- Would require tracking previous device inputs and comparing against tolerance
- GPU batched evaluation may not benefit much from bypass (branch divergence)
- Low priority unless profiling shows device evaluation as bottleneck

### Clean Up Legacy Transient Code Paths

**Status**: COMPLETED

The transient module uses `FullMNAStrategy` as the primary implementation (`jax_spice/analysis/transient/`).
Legacy code paths (`_run_transient_hybrid`, `_run_transient_while_loop`, `_make_gpu_resident_build_system_fn`)
have been deleted from `engine.py`.

**GPU status**: GPU is ~4x faster than CPU (working correctly). Uses Spineax/cuDSS when available.

**Completed**:
- [x] Consolidated into the strategy pattern using `FullMNAStrategy`
- [x] Removed legacy solver factories (make_dense_solver, make_sparse_solver, make_spineax_solver, make_umfpack_solver)
- [x] Added GPU sparse support via `make_spineax_full_mna_solver`
- [x] AC analysis updated to use `_make_full_mna_build_system_fn`

## Needed for release

### Documentation
- [ ] **Update README** with current project status
- [ ] **Add architecture overview** diagram
- [ ] **Document openvaf_jax API** for external users

## Nice to have

### Code Cleanup
- [x] ~~**Remove xfail markers** from PSP/JUNCAP/diode_cmc tests~~ (all pass)
- [ ] **Consolidate test files** in `openvaf_jax/openvaf_py/` (some at root level, some in tests/)

## Completed

- [x] ~~DEVpnjlim algorithm bug in diode.va~~ (2025-02)
  - diode.va used ngspice's buggy `log(arg-2)` formula instead of VACASK's `log(1+arg)`
  - When `|vnew-vold| < 3*vt`, the ngspice formula produces negative log arguments
  - Fixed in `vendor/VACASK/devices/spice/sn/diode.va` lines 416-419

- [x] ~~iniLim simparam off-by-one bug~~ (2025-02)
  - `nr_iteration` starts at 0, but iniLim check was `== 1`, so iniLim was never 1
  - Fixed to `nr_iteration == 0` in `jax_spice/analysis/mna_builder.py:355`

- [x] ~~isource COOVector stamping bug~~ (2025-02)
  - isource stamping appended plain tuple instead of COOVector
  - Fixed to use `mask_coo_vector()` in `jax_spice/analysis/mna_builder.py:316`

- [x] ~~ddt() operator fixed~~ (2025-01)
  - `openvaf_jax/codegen/instruction.py` now returns charge value instead of zero
  - Ring oscillator shows proper oscillation behavior
  - All VACASK benchmarks pass including transient with capacitors/MOSFETs

- [x] ~~Branch current computation~~ (2025-01)
  - `TransientResult` now includes `currents: Dict[str, Array]` field
  - Branch currents through voltage sources are computed and returned
  - Full MNA with explicit branch current unknowns implemented

- [x] ~~Transient analysis refactored~~ (2025-01)
  - Unified `FullMNAStrategy` in `jax_spice/analysis/transient/`
  - Adaptive timestep control based on local truncation error
  - JIT-compiled simulation loop using lax.while_loop

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
| **Circuit engine (main API)** | `jax_spice/analysis/engine.py` |
| Transient strategies | `jax_spice/analysis/transient/` |
| MNA system | `jax_spice/analysis/mna.py` |
| GPU backend selection | `jax_spice/analysis/gpu_backend.py` |
| Benchmark runner | `jax_spice/benchmarks/runner.py` |
| Benchmark profiling | `scripts/profile_gpu.py` |
| Cloud Run profiling | `scripts/profile_gpu_cloudrun.py` |
| **Verilog-A device wrapper** | `jax_spice/devices/verilog_a.py` |
| **OpenVAF→JAX codegen** | `openvaf_jax/codegen/` |
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

# Run openvaf_jax tests
cd openvaf_jax/openvaf_py && JAX_PLATFORMS=cpu ../../.venv/bin/python -m pytest tests/ -v

# Run local benchmark profiling
JAX_PLATFORMS=cpu uv run python scripts/profile_gpu.py --benchmark ring

# Run Cloud Run GPU profiling with Perfetto traces
uv run scripts/profile_gpu_cloudrun.py --benchmark ring --timesteps 50
```
