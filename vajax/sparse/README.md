# UMFPACK JAX FFI Extension

High-performance UMFPACK sparse direct solver with XLA FFI bindings for JAX.
This eliminates the ~100ms pure_callback overhead when using UMFPACK within
JAX's JIT-compiled code.

## Prerequisites

### macOS (Homebrew)
```bash
brew install suite-sparse nanobind
```

### Ubuntu/Debian
```bash
sudo apt install libsuitesparse-dev
pip install nanobind
```

### Fedora/RHEL
```bash
sudo dnf install suitesparse-devel
pip install nanobind
```

## Build and Install

From this directory:
```bash
pip install .
```

Or for development:
```bash
pip install -e . -v
```

## Usage

Once installed, VA-JAX will automatically detect and use the FFI-based
UMFPACK solver instead of the pure_callback version:

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

The FFI version reduces per-solve overhead from ~100ms to ~0.1ms by
eliminating Python callback marshaling, GIL acquisition, and host
synchronization.

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

The extension uses XLA FFI (Foreign Function Interface) to register
UMFPACK operations directly as XLA custom calls, following the same
pattern as klujax (KLU solver for JAX).

Key optimizations:
- Symbolic factorization cached for repeated solves
- CSR→CSC conversion done in C++ (UMFPACK requires column-major)
- No Python callback overhead within JIT-compiled code
- Thread-safe with mutex-protected cache

## License

Same license as VA-JAX (Apache 2.0).
