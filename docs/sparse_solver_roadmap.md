# Roadmap: GPU/TPU-Agnostic Sparse Linear Solvers in JAX

**Date:** 2025-12-20
**Status:** Draft
**Authors:** Robert Taylor, Claude

## Goal

Enable efficient sparse linear solving in JAX-SPICE that works across CPU, GPU (NVIDIA, AMD), and TPU, with cached symbolic factorization for Newton-Raphson iterations.

## Current State of JAX Sparse Support

### What Exists

| Feature | Status | Source |
|---------|--------|--------|
| BCOO format | Stable | [PR #6824](https://github.com/google/jax/pull/6824) |
| BCSR format | Stable | [jax.experimental.sparse](https://docs.jax.dev/en/latest/jax.experimental.sparse.html) |
| `spsolve` | Experimental | Uses cuSOLVER on GPU, no caching |
| `gmres`, `cg`, `bicgstab` | Experimental | [Issue #11376](https://github.com/jax-ml/jax/issues/11376) |
| Sparse matmul (SpMV, SpMM) | Stable | Uses cusparse on GPU |

### What's Missing

1. **Cached symbolic factorization** - Every `spsolve` call redoes METIS ordering
2. **TPU sparse support** - Limited to dense operations
3. **Sparse triangular solve** - Not exposed as primitive
4. **Pre-factorization API** - [JAX Issue #22500](https://github.com/jax-ml/jax/issues/22500)

## Relevant JAX/XLA/MLIR Work

### Closed/Stale PRs (Potential Revival)

| PR | Description | Status | Relevance |
|----|-------------|--------|-----------|
| [#6555](https://github.com/jax-ml/jax/pull/6555) | MLIR/TACO-like sparse representation | Closed (stale) | **HIGH** - Explored MLIR sparse tensor integration |
| [#4422](https://github.com/google/jax/pull/4422) | Add experimental sparse support | Merged | Foundation for current sparse module |
| [#2566](https://github.com/jax-ml/jax/pull/2566) | scipy.sparse.linalg.cg | Merged | Iterative solver baseline |

### Active/Recent Work

| PR/Issue | Description | Status | Relevance |
|----------|-------------|--------|-----------|
| [#25958](https://github.com/jax-ml/jax/pull/25958) | Performance note on sparse docs | Merged (Jan 2025) | Documents current limitations |
| [#11376](https://github.com/jax-ml/jax/issues/11376) | Development of scipy.sparse.linalg | Open | Tracking issue for sparse solvers |
| [LLVM #151885](https://github.com/llvm/llvm-project/pull/151885) | MLIR sparse loop ordering heuristics | Active | 30% speedup on sparse workloads |

### MLIR Sparse Tensor Dialect

The [MLIR Sparse Tensor Dialect](https://mlir.llvm.org/docs/Dialects/SparseTensorOps/) provides:
- ~40 sparse tensor operations
- Multiple storage formats (COO, CSR, CSC, etc.)
- Automatic code generation from sparsity-agnostic Linalg ops
- GPU codegen (experimental)

**Gap:** No direct LU factorization or triangular solve ops. Focus is on SpMM/SpMV.

Reference: [Compiler Support for Sparse Tensor Computations in MLIR](https://dl.acm.org/doi/10.1145/3544559)

### XLA Sparse Support

XLA has limited sparse support:
- `SparseTensorDotGeneral` for sparse matmul
- Custom calls to cuSPARSE for GPU
- No native sparse LU or triangular solve

Relevant issues:
- [openxla/xla#6834](https://github.com/openxla/xla/issues/6834) - Triangular solve integer overflow (fixed)
- [openxla/xla#6871](https://github.com/openxla/xla/pull/6871) - Fix for above

## Proposed Architecture

### Phase 1: Hybrid Backend (Immediate)

```
                    ┌─────────────────┐
                    │  JAX-SPICE NR   │
                    │    Solver       │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
        ┌─────────┐   ┌─────────┐   ┌─────────┐
        │ Spineax │   │  GMRES  │   │  Dense  │
        │ (cuDSS) │   │ + Prec  │   │  BLAS   │
        └─────────┘   └─────────┘   └─────────┘
            GPU          TPU/CPU       CPU
```

**Implementation:** Done (Spineax integration committed)

### Phase 2: Pure JAX Iterative Solver (Short-term)

Add GMRES with block-Jacobi preconditioner as agnostic fallback:

```python
# jax_spice/analysis/iterative_solver.py

def build_block_jacobi_preconditioner(J_diag_blocks):
    """Build M^-1 from diagonal blocks of Jacobian."""
    inv_blocks = [jnp.linalg.inv(b) for b in J_diag_blocks]
    def apply_M_inv(v):
        # Apply block-diagonal inverse
        return jnp.concatenate([inv @ v_i for inv, v_i in zip(inv_blocks, split(v))])
    return apply_M_inv

def gmres_solve(J, b, preconditioner=None, restart=30, maxiter=100):
    """Solve Jx = b using preconditioned GMRES."""
    return jax.scipy.sparse.linalg.gmres(
        lambda v: J @ v,
        b,
        M=preconditioner,
        restart=restart,
        maxiter=maxiter,
    )
```

**Effort:** 1-2 weeks
**Benefit:** Works on all backends, no custom ops

### Phase 3: XLA Sparse Primitive (Medium-term)

Add sparse triangular solve to XLA via custom call:

```
XLA HLO:
  SparseLU(sparse_matrix) -> (L, U, perm)  # Factorize
  SparseTriangularSolve(L, b) -> x         # Forward sub
  SparseTriangularSolve(U, x) -> y         # Back sub
```

This requires:
1. Define HLO ops for sparse LU components
2. Implement CPU lowering (via SuiteSparse/KLU)
3. Implement GPU lowering (via cuDSS)
4. Add JAX bindings

**Effort:** 2-3 months (significant XLA contribution)
**Benefit:** Native caching, works with XLA optimizations

### Phase 4: MLIR Sparse Integration (Long-term)

Leverage MLIR Sparse Tensor Dialect:

1. **Define sparse LU in Linalg dialect** - Express as sparse tensor operations
2. **Use MLIR sparsifier** - Generate optimized code automatically
3. **Multi-backend codegen** - CPU, GPU, TPU from single definition

Reference: [MLIR Sparsifier JAX Colab](https://developers.google.com/mlir-sparsifier/colabs/Sparse_JAX_Colab_for_GPU_(custom_ops))

**Effort:** 6-12 months (research project)
**Benefit:** True platform independence, compiler-level optimization

## Technical Challenges

### 1. Symbolic Factorization Caching in XLA

XLA's functional model doesn't naturally support mutable state (cached factorization).

**Options:**
- Use XLA state tokens (like RNG state)
- Store factorization as "constant" in compiled program
- Use external state via custom call

### 2. TPU Sparse Support

TPUs are optimized for dense, regular computation. Sparse operations have overhead.

**Options:**
- Dense solver for small matrices (our current approach works)
- Padded block-sparse for structured sparsity
- Iterative solvers (GMRES) which are more TPU-friendly

### 3. Dynamic Sparsity Patterns

Some applications have varying sparsity. Our circuit simulation has fixed patterns.

**Options:**
- Re-analyze on pattern change (expensive but rare)
- Approximate with superset pattern
- Fall back to iterative solver

## Recommendations

### For JAX-SPICE (Immediate)
1. ✅ Use Spineax on NVIDIA GPUs (done)
2. Add GMRES fallback for TPU/other GPUs
3. Keep dense option for small circuits

### For JAX Ecosystem (Contribution)
1. Propose `spsolve_prefactor` API to JAX ([Issue #22500](https://github.com/jax-ml/jax/issues/22500))
2. Add sparse triangular solve primitive
3. Document hybrid sparse strategy

### For Long-term Research
1. Explore MLIR sparsifier for LU
2. Work with XLA team on sparse primitive design
3. Investigate AMD ROCm sparse solver integration

## References

### JAX Sparse
- [jax.experimental.sparse docs](https://docs.jax.dev/en/latest/jax.experimental.sparse.html)
- [JAX Issue #6544 - Sparse matrices](https://github.com/jax-ml/jax/issues/6544) (CLOSED)
- [JAX Issue #11376 - scipy.sparse.linalg](https://github.com/jax-ml/jax/issues/11376) (OPEN)
- [JAX PR #6555 - MLIR/TACO sparse](https://github.com/jax-ml/jax/pull/6555) (CLOSED - stale)

### MLIR Sparse
- [MLIR Sparse Tensor Dialect](https://mlir.llvm.org/docs/Dialects/SparseTensorOps/)
- [MLIR Sparsifier Guide](https://developers.google.com/mlir-sparsifier/guides/intro)
- [Compiler Support Paper](https://dl.acm.org/doi/10.1145/3544559)

### GPU Sparse Solvers
- [NVIDIA cuDSS](https://docs.nvidia.com/cuda/cudss/index.html)
- [Spineax](https://github.com/johnviljoen/Spineax)
- [cuSOLVER Sparse](https://docs.nvidia.com/cuda/cusolver/index.html)

### Iterative Solvers
- [JAX GMRES](https://docs.jax.dev/en/latest/_autosummary/jax.scipy.sparse.linalg.gmres.html)
- [Lineax](https://docs.kidger.site/lineax/api/solvers/)
