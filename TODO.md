# jax-spice TODO

Central tracking for development tasks and known issues.

## High Priority

### GPU Solver Convergence
The GPU solver has convergence issues with circuits containing floating nodes (e.g., AND gates with series NMOS stacks).

**Root cause**: Autodiff-computed Jacobians give extremely small `gds` (~1e-16 S) in cutoff, while VACASK enforces a minimum `gds` (~1e-9 S). This creates near-singular Jacobians.

**Progress**:
- [x] Created `jax_spice/devices/openvaf_device.py` - VADevice wrapper for openvaf_jax models
- [x] Added `get_vacask_resistor()`, `get_vacask_diode()`, etc. cached model accessors
- [x] Implemented `stamp_device_into_system()` for stamping analytical Jacobians
- [x] Added `dc_operating_point_analytical()` and `build_analytical_residual_and_jacobian_fn()` (WIP)

**Current Issue**:
- MOSFET model mismatch between analytical and autodiff:
  - Autodiff uses smooth `tanh`-based formulation
  - Analytical uses piecewise Shichman-Hodges
  - Results in different residuals AND Jacobian signs
- Need to either:
  1. Update analytical model to match autodiff smooth formulation
  2. Or use openvaf_jax VADevice directly (returns analytical Jacobians from Verilog-A)

**Remaining Tasks**:
- [ ] **Fix MOSFET model mismatch**
  - Option A: Derive analytical Jacobian for smooth tanh-based model
  - Option B: Integrate VADevice.evaluate() for MOSFET (uses openvaf_jax)

- [ ] **Test AND gate convergence with analytical Jacobian**
  - Test circuit: `and_test` from `c6288.sim` (6 MOSFETs, floating `int` node)
  - VACASK converges in 34 iterations; GPU solver diverges

- [ ] **Benchmark full C6288 multiplier** on GPU with analytical Jacobians
  - 5123 nodes, 10112 MOSFETs
  - Currently ~30min on CPU, should be much faster on GPU

**Files**:
- `jax_spice/devices/openvaf_device.py` - NEW: VADevice wrapper
- `jax_spice/analysis/dc_gpu.py` - needs analytical Jacobian variant
- `jax_spice/analysis/transient_gpu.py` - needs same treatment
- `docs/gpu_solver_jacobian.md` - analysis of the issue

## Medium Priority

### openvaf_jax Complex Model Support
The JAX translator produces NaN outputs for complex models due to init variable handling.

**Root cause**: Complex models (BSIM3/4/6, HiSIM, HICUM, etc.) have init functions that compute many cached values. The JAX translator expects these as inputs, but with default values they're wrong → NaN.

**Tasks**:
- [ ] **Use equivalent approach as OSDI compile uses**
  - see docs/vacask_osdi_inputs.md

**Affected models** (currently xfailed in tests):
- BSIM3, BSIM4, BSIM6, BSIMBulk, BSIMCMG, BSIMSOI
- HiSIM2, HiSIMHV
- HICUM L2
- MEXTRAM

**Working models** (should remove xfail markers):
- PSP102, PSP103, JUNCAP
- diode_cmc
- EKV

## Low Priority

### Documentation
- [ ] **Update README** with current project status
- [ ] **Add architecture overview** diagram
- [ ] **Document openvaf_jax API** for external users

### Code Cleanup
- [ ] **Remove xfail markers** from PSP/JUNCAP/diode_cmc tests (they pass)
- [ ] **Consolidate test files** in `openvaf-py/tests/` (some at root level)

### Build System
- [ ] **Upstream VACASK macOS fixes** to original repo
  - Current workaround: `robtaylor/VACASK` fork with `macos-fixes` branch
  - Fixes: C++20, PTBlockSequence, VLA→vector, KLU destructor, <numbers> header, CMake var escaping

## Completed

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

- [x] ~~Fix PMOS current sign convention~~ in GPU solvers
  - Both `dc_gpu.py` and `transient_gpu.py` updated

- [x] ~~Add gds_min leakage~~ to GPU MOSFET model (partial fix)

- [x] ~~Document VACASK OSDI input handling~~ (`docs/vacask_osdi_inputs.md`)

- [x] ~~Add OpenVAF/VACASK build scripts~~ for macOS
  - `scripts/build_openvaf.sh`
  - `scripts/build_vacask.sh`

## Reference

### Key Files
| Purpose | Location |
|---------|----------|
| GPU DC solver | `jax_spice/analysis/dc_gpu.py` |
| GPU transient solver | `jax_spice/analysis/transient_gpu.py` |
| OpenVAF device wrapper | `jax_spice/devices/openvaf_device.py` |
| OpenVAF→JAX translator | `openvaf-py/openvaf_jax.py` |
| VACASK parser | `jax_spice/netlist/parser.py` |
| VACASK JAX tests | `tests/test_vacask_jax.py` |
| Jacobian issue analysis | `docs/gpu_solver_jacobian.md` |

### Test Commands
```bash
# Run all tests
JAX_PLATFORMS=cpu uv run pytest tests/ -v

# Run VACASK JAX tests specifically
JAX_PLATFORMS=cpu uv run pytest tests/test_vacask_jax.py -v

# Run openvaf-py tests
cd openvaf-py && JAX_PLATFORMS=cpu ../.venv/bin/python -m pytest tests/ -v

# Run GPU benchmarks (slow)
JAX_PLATFORMS=cpu RUN_GPU_BENCHMARKS=1 uv run pytest tests/test_transient_gpu.py -v

# Run Cloud Run GPU tests
uv run scripts/run_gpu_tests.py --watch
```
