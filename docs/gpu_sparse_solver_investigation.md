# GPU Sparse Solver Performance Investigation

**Date:** 2025-12-20
**Status:** In Progress
**Authors:** Robert Taylor, Claude

## Executive Summary

JAX-SPICE GPU performance for large circuits (c6288, ~15k nodes) is bottlenecked by sparse linear solve overhead. This document captures our investigation, measurements, and proposed solutions.

## Problem Statement

When running the c6288 benchmark on GPU with sparse solver, we observed that **92% of simulation time was spent in CPU/driver overhead**, not GPU computation. The core issue is that JAX's `spsolve` re-does symbolic analysis on every call.

## Measurements

### Baseline Performance (c6288, 10 timesteps)

| Solver | Platform | Time/Step | Notes |
|--------|----------|-----------|-------|
| Dense | CPU | 306 ms | **1.6x faster than VACASK** |
| VACASK (sparse) | CPU | 503 ms | Reference C implementation |
| Dense | GPU (L4) | 8,509 ms | 28x slower than CPU |
| Sparse (spsolve) | GPU (L4) | ~2,800 ms | From trace analysis |
| **Sparse (Spineax)** | **GPU (L4)** | **442 ms** | **6.3x faster than spsolve** |

**Source:** Cloud Run GPU jobs, Perfetto traces from `gs://jax-spice-cuda-test-traces/`

### Spineax Results (2025-12-20)

With Spineax (cuDSS with cached symbolic factorization):
- **c6288 (20 steps):** 442.169 ms/step (8.843s total)
- **Improvement vs spsolve:** 6.3x faster (2800ms → 442ms)
- **Comparison to CPU dense:** 1.4x slower (306ms vs 442ms)

The improvement comes from:
1. Symbolic analysis (METIS reordering) done **once** during solver initialization
2. Subsequent solves use only numeric refactorization + triangular solve
3. Eliminated the ~100ms per-call symbolic overhead that dominated spsolve

### Trace Analysis (Sparse on GPU)

From Perfetto trace of c6288 sparse run:
- `spsolve.0` total: **28.7s** (253 calls, 113ms average)
- GPU kernel time: **380ms** (1.3% of total)
- Memory copies: **1.9s** (6.8%)
- CPU/driver overhead: **26.4s** (91.9%)

**Key Finding:** The ~100ms per-call overhead is dominated by symbolic analysis (METIS reordering, fill-in pattern computation), which is redundant since our sparsity pattern never changes.

### Why Dense Works on CPU

Surprising result: Dense solver on CPU (306ms/step) outperforms VACASK's cached sparse solver (503ms/step).

**Hypothesized reasons:**
1. BLAS (OpenBLAS/MKL) is extremely optimized for dense operations
2. XLA JIT compiles entire Newton loop - no Python overhead between iterations
3. Dense has perfect memory locality vs sparse's indirect indexing
4. 15k×15k matrix fits in CPU cache hierarchy

**Caveat:** This is hypothesis based on general knowledge of BLAS optimization, not direct profiling.

### Why Dense Fails on GPU

Dense on GPU takes 8.5 seconds/step because:
- Matrix is 15k×15k × 8 bytes = **1.8 GB**
- LU factorization is O(n³) = 3.4 trillion operations
- GPU LU (getrf) has poor parallelism due to sequential pivoting
- Trace showed constant folding warnings taking 2+ seconds

**Source:** Cloud Run job logs, XLA slow operation warnings

## Root Cause Analysis

### JAX spsolve Implementation

JAX's `jax.experimental.sparse.linalg.spsolve`:
- Uses cuSOLVER's `csrlsvqr` on GPU
- Performs **full symbolic analysis on every call**
- No API to cache/reuse symbolic factorization
- This is a known limitation: [JAX Issue #22500](https://github.com/jax-ml/jax/issues/22500)

### What SPICE Simulators Do

Traditional SPICE (ngspice, Xyce) uses KLU/SuperLU with:
1. **Symbolic analysis** (once): METIS reordering, fill-in prediction
2. **Numeric factorization** (each matrix): Compute L, U values
3. **Triangular solves** (each RHS): Forward/back substitution

For Newton-Raphson on circuits, the sparsity pattern is fixed, so step 1 is done once and cached.

## Solutions Explored

### Option 1: NVIDIA cuDSS via Spineax (Implemented)

[Spineax](https://github.com/johnviljoen/Spineax) wraps NVIDIA cuDSS with:
- Explicit three-phase API: analysis → factorization → solve
- Refactorization support for fixed sparsity patterns
- JAX JIT/vmap compatible via XLA FFI

**Implementation:**
- Added `spineax` to cuda12 optional dependencies
- Created `jax_spice/analysis/spineax_solver.py` wrapper
- Modified `runner.py` to auto-detect and use Spineax on GPU

**Expected improvement:** 10-100x speedup on linear solve phase (eliminating ~100ms symbolic overhead per call).

**Caveat:** Not yet measured - Cloud Run job pending.

### Option 2: Pure JAX GMRES + Preconditioner (Not Implemented)

```python
jax.scipy.sparse.linalg.gmres(lambda v: J @ v, b, M=preconditioner)
```

**Pros:**
- GPU/TPU/CPU agnostic
- No custom ops needed
- Matrix-free (only needs matvec)

**Cons:**
- Iterative - convergence not guaranteed
- Needs good preconditioner (ILU, block-Jacobi, AMG)
- ILU is inherently sequential, hard to parallelize

**Status:** Identified as fallback for TPU or if Spineax unavailable.

### Option 3: XLA/MLIR Sparse Extensions (Future Research)

MLIR Sparse Tensor dialect has ~40 ops but:
- No direct LU factorization
- No sparse triangular solve
- Focus is on SpMM/SpMV, not direct solvers

Could potentially:
1. Build sparse LU from MLIR primitives
2. Add symbolic factorization caching as JAX primitive
3. Contribute to JAX sparse solver infrastructure

**Status:** Long-term research direction, not actionable now.

### Option 4: Hybrid Strategy (Recommended)

```python
if jax.default_backend() == 'gpu':
    solver = SpineaxSolver(...)  # cuDSS with cached factorization
elif jax.default_backend() == 'tpu':
    solver = GMRESSolver(...)    # Iterative with preconditioner
else:
    solver = DenseSolver(...)    # CPU BLAS is fast enough
```

## Current Implementation Status

### Completed
- [x] Added spineax to pyproject.toml cuda12 extras
- [x] Updated uv.lock with CUDA dependencies
- [x] Created `jax_spice/analysis/spineax_solver.py`
- [x] Integrated Spineax into `runner.py` sparse solver path
- [x] Auto-detection: uses Spineax when available on GPU
- [x] Fixed Spineax XLA FFI build issues (forked to robtaylor/spineax)
- [x] Fixed CUDA kernel compilation issue (disabled pbatch_solve module)
- [x] Fixed LD_LIBRARY_PATH for NVIDIA pip packages on Cloud Run
- [x] Cloud Run GPU benchmark with Spineax: **442 ms/step** (6.3x improvement)

### Pending
- [ ] Report fix upstream to Spineax maintainer
- [ ] GMRES fallback implementation for TPU

## Assumptions and Uncertainties

### Validated Assumptions
1. **Sparsity pattern is fixed** - Confirmed by code review. Circuit topology determines Jacobian structure; only values change during Newton-Raphson.

2. **cuDSS supports refactorization** - Confirmed by [NVIDIA documentation](https://docs.nvidia.com/cuda/cudss/index.html) and CUDSS.jl usage.

3. **Spineax is JIT-compatible** - Claimed in Spineax README ("full FFI jit/vmap integration").

### Unvalidated Assumptions
1. **Spineax will provide 10-100x speedup** - Based on eliminating symbolic overhead, but not yet measured.

2. **BCSR pattern is deterministic** - We assume `BCSR.from_bcoo(J.sum_duplicates(nse=nse))` produces consistent indptr/indices. Needs verification.

3. **No numerical stability issues** - cuDSS uses different algorithms than cuSOLVER. May have different behavior for ill-conditioned matrices.

### Known Risks
1. **Spineax is young project** - Version 0.0.2, single maintainer. May have bugs or API changes.

2. **GPU-only** - Spineax requires CUDA. No TPU or AMD GPU support.

3. **Build complexity** - Spineax uses scikit-build-core + nanobind. May have build issues on some systems.

4. **JAX version compatibility** - Spineax uses XLA FFI headers which can have breaking changes between JAX versions. We encountered build failures on Cloud Run with jaxlib warnings being treated as errors in the XLA FFI API headers.

### Build Issues Fixed (2025-12-20)

We encountered and fixed two build issues:

1. **XLA FFI Header Warnings** - The XLA FFI headers in jaxlib have `-Wreturn-type` warnings that were treated as errors. Fixed by adding compiler flags in CMakeLists.txt:
   ```cmake
   target_compile_options(${TARGET} PRIVATE
       $<$<COMPILE_LANGUAGE:CXX>:-Wno-return-type -Wno-attributes>
   )
   ```

2. **CUDA Kernel Version Mismatch** - The `pbatch_solve.cu` file uses internal CUDA API symbols (`__cudaGetKernel`) that differ between nvcc versions. Fixed by disabling the pbatch_solve module (we don't need it for single-matrix solves).

Both fixes are in the ChipFlow fork at `robtaylor/spineax`.

## Next Steps

1. **Immediate:** Report fixes upstream to Spineax maintainer
2. **Short-term:** Implement GMRES + block-Jacobi fallback for TPU
3. **Long-term:** Investigate contributing sparse solver improvements to JAX

## References

- [Spineax GitHub](https://github.com/johnviljoen/Spineax)
- [NVIDIA cuDSS Documentation](https://docs.nvidia.com/cuda/cudss/index.html)
- [JAX spsolve Documentation](https://docs.jax.dev/en/latest/_autosummary/jax.experimental.sparse.linalg.spsolve.html)
- [JAX Issue #22500 - No pre-factorization](https://github.com/jax-ml/jax/issues/22500)
- [MLIR Sparse Tensor Dialect](https://mlir.llvm.org/docs/Dialects/SparseTensorOps/)
- [Extending JAX with Custom CUDA](https://dfm.io/posts/extending-jax/)
