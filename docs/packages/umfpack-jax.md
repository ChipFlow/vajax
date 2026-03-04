# umfpack-jax

High-performance UMFPACK sparse direct solver with XLA FFI bindings for JAX.

`umfpack-jax` provides UMFPACK sparse solver operations as native XLA custom calls,
eliminating the ~100ms `pure_callback` overhead when solving large sparse linear systems
within JAX's JIT-compiled code.

## Installation

```bash
pip install umfpack-jax
```

Pre-built wheels are available for:

- Linux x86_64 and aarch64 (statically linked SuiteSparse)
- macOS arm64 (Apple Silicon, uses Accelerate framework)

No system dependencies needed when installing from wheels.

## Usage

```python
from vajax.sparse import umfpack_jax

# Check if FFI version is available
if umfpack_jax.is_available():
    print("Using FFI-based UMFPACK")

# Solve sparse system Ax = b (CSR format)
x = umfpack_jax.solve(csr_indptr, csr_indices, csr_data, b)
```

## Performance

| Operation | pure_callback | FFI |
|-----------|--------------|-----|
| Solve overhead | ~100ms | ~0.1ms |
| c6288 solve | ~117ms | ~17ms |
| Newton-Raphson iteration | ~120ms | ~20ms |

## API

### `solve(indptr, indices, data, b) -> x`
Solve Ax = b where A is in CSR format.

### `dot(indptr, indices, data, x) -> b`
Compute b = A @ x (sparse matrix-vector multiply).

### `solve_transpose(indptr, indices, data, b) -> x`
Solve A^T x = b (transpose solve, needed for autodiff).

### `clear_cache()`
Clear the cached symbolic factorization. Call when switching between
matrices with different sparsity patterns.

## Architecture

The extension uses XLA FFI (Foreign Function Interface) to register UMFPACK operations
directly as XLA custom calls. Key optimizations:

- Symbolic factorization cached for repeated solves
- CSR to CSC conversion done in C++ (UMFPACK requires column-major)
- No Python callback overhead within JIT-compiled code
- Thread-safe with mutex-protected cache

## Supported Platforms

| Platform | Architecture | Status |
|----------|-------------|--------|
| Linux    | x86_64      | Supported |
| Linux    | aarch64     | Supported |
| macOS    | arm64       | Supported |
| Windows  | x86_64      | Not yet supported |

## Links

- [Source code](https://github.com/ChipFlow/vajax/tree/main/vajax/sparse)
- [PyPI](https://pypi.org/project/umfpack-jax/)
- [Sparse Solver Roadmap](../sparse_solver_roadmap.md) — solver strategy overview
- [GPU Solver Architecture](../gpu_solver_architecture.md) — how sparse solvers fit into the simulation pipeline
