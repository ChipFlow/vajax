# jax-spice TODO

Central tracking for development tasks and known issues.

## High Priority

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

**Status (2025-12):** Partially addressed with unified solver module.

**Completed:**
- [x] Created `jax_spice/analysis/solver.py` with unified NR iteration loop
- [x] Refactored `dc.py` to use unified solver with autodiff Jacobian
- [x] Added legacy fallback for MNASystem without device_groups (test compatibility)

**Remaining duplication:**
| Category | Status | Notes |
|----------|--------|-------|
| NR iteration (DC) | ✅ Unified | Uses `solver.newton_solve()` |
| NR iteration (transient) | Kept separate | Needs V_prev for backward Euler |
| NR iteration (runner) | Kept separate | Uses hybrid OpenVAF + simple stamping |
| Device stamping | Moderate | Different for each use case |

**Design decision:** Transient and runner NR loops kept separate because:
- Transient: Needs `V_prev` for capacitor companion model (backward Euler)
- Runner: Uses hybrid approach with OpenVAF vmapped functions + simple device stamping
- Both are already JIT-compiled and optimized for their specific use cases

**Architecture after refactoring:**
```
solver.py          # Core NR loop (DC analysis)
├── NRConfig       # Solver configuration
├── NRResult       # Result with convergence info
└── newton_solve() # Main entry point

dc.py              # Uses solver.newton_solve() + autodiff Jacobian
transient.py       # Separate NR with V_prev for backward Euler
runner.py          # Separate hybrid NR for OpenVAF + simple devices
```

### Build System
- [ ] **Upstream VACASK macOS fixes** to original repo
  - Current workaround: `robtaylor/VACASK` fork with `macos-fixes` branch
  - Fixes: C++20, PTBlockSequence, VLA→vector, KLU destructor, <numbers> header, CMake var escaping

## Completed

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
| DC operating point | `jax_spice/analysis/dc.py` |
| Transient solver | `jax_spice/analysis/transient.py` |
| MNA system | `jax_spice/analysis/mna.py` |
| GPU backend selection | `jax_spice/analysis/gpu_backend.py` |
| Benchmark runner | `jax_spice/benchmarks/runner.py` |
| Benchmark profiling | `scripts/profile_gpu.py` |
| Cloud Run profiling | `scripts/profile_gpu_cloudrun.py` |
| OpenVAF device wrapper | `jax_spice/devices/openvaf_device.py` |
| OpenVAF→JAX translator | `openvaf-py/openvaf_jax.py` |
| VACASK parser | `jax_spice/netlist/parser.py` |
| VACASK suite tests | `tests/test_vacask_suite.py` |

### Test Commands
```bash
# Run all tests
JAX_PLATFORMS=cpu uv run pytest tests/ -v

# Run VACASK suite tests
JAX_PLATFORMS=cpu uv run pytest tests/test_vacask_suite.py -v

# Run openvaf-py tests
cd openvaf-py && JAX_PLATFORMS=cpu ../.venv/bin/python -m pytest tests/ -v

# Run local benchmark profiling
JAX_PLATFORMS=cpu uv run python scripts/profile_gpu.py --benchmark ring

# Run Cloud Run GPU profiling with Perfetto traces
uv run scripts/profile_gpu_cloudrun.py --benchmark ring --timesteps 50
```
