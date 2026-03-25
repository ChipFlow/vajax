"""Sprux JAX FFI — Metal-accelerated sparse LU solver for JAX on Apple Silicon.

This module provides Sprux sparse linear solver with XLA FFI bindings.
Internally, Sprux factors in float32 on the Metal GPU and uses CPU-side
float64 iterative refinement to recover near-f64 accuracy.

Example usage:
    import jax.numpy as jnp
    from vajax.sprux import sprux_jax

    # Sparse matrix in CSR format
    indptr = jnp.array([0, 2, 4, 6], dtype=jnp.int32)
    indices = jnp.array([0, 1, 0, 1, 0, 1], dtype=jnp.int32)
    data = jnp.array([4., 1., 1., 3., 2., 2.], dtype=jnp.float64)
    b = jnp.array([1., 2., 3.], dtype=jnp.float64)

    # Solve Ax = b
    x = sprux_jax.solve(indptr, indices, data, b)
"""

from __future__ import annotations

import os
import sys

import jax
import jax.extend.core
import jax.numpy as jnp
from jax.core import ShapedArray
from jax.interpreters import mlir
from jaxtyping import Array

__version__ = "0.1.0"
__all__ = ["solve", "dot", "clear_cache", "is_available"]

# =============================================================================
# Configuration
# =============================================================================

DEBUG = os.environ.get("SPRUX_JAX_DEBUG", False)


def _debug(s: str) -> None:
    if DEBUG:
        print(s, file=sys.stderr)


# =============================================================================
# Check if the FFI extension is available
# =============================================================================

_SPRUX_FFI_AVAILABLE = False
_sprux_jax_cpp = None

try:
    # Set SPRUX_METALLIB_PATH before importing the C extension, so the Metal
    # context singleton finds the shader library installed alongside the .so.
    if "SPRUX_METALLIB_PATH" not in os.environ:
        import importlib.util

        _spec = importlib.util.find_spec("sprux_jax_cpp")
        if _spec and _spec.origin:
            from pathlib import Path

            _metallib = Path(_spec.origin).parent / "MetalKernels.metallib"
            if _metallib.exists():
                os.environ["SPRUX_METALLIB_PATH"] = str(_metallib)
                _debug(f"Set SPRUX_METALLIB_PATH={_metallib}")

    import sprux_jax_cpp as _sprux_jax_cpp

    _SPRUX_FFI_AVAILABLE = True
    _debug("Sprux FFI extension available")
except ImportError:
    _debug("Sprux FFI extension not available")


def is_available() -> bool:
    """Check if the Sprux FFI extension is available."""
    return _SPRUX_FFI_AVAILABLE


def clear_cache() -> None:
    """Clear the cached solver.

    Call this when switching between circuits with different sparsity patterns.
    """
    if _SPRUX_FFI_AVAILABLE:
        _sprux_jax_cpp.clear_cache()


# =============================================================================
# High-level API
# =============================================================================


def solve(
    indptr: Array,
    indices: Array,
    data: Array,
    b: Array,
) -> Array:
    """Solve for x in the sparse linear system Ax = b.

    Uses Metal GPU for f32 LU factorization with CPU f64 iterative
    refinement. Returns near-f64 accuracy.

    Args:
        indptr: CSR row pointers (length n+1), int32
        indices: CSR column indices (length nnz), int32
        data: CSR non-zero values (length nnz), float64
        b: Right-hand side vector (length n), float64

    Returns:
        x: Solution vector (length n), float64
    """
    if not _SPRUX_FFI_AVAILABLE:
        raise RuntimeError(
            "Sprux FFI extension not available. Install with: cd vajax/sprux && pip install ."
        )

    indptr = jnp.asarray(indptr, dtype=jnp.int32)
    indices = jnp.asarray(indices, dtype=jnp.int32)
    data = jnp.asarray(data, dtype=jnp.float64)
    b = jnp.asarray(b, dtype=jnp.float64)

    return solve_f64.bind(indptr, indices, data, b)


def dot(
    indptr: Array,
    indices: Array,
    data: Array,
    x: Array,
) -> Array:
    """Compute b = A @ x (sparse matrix-vector multiply).

    Args:
        indptr: CSR row pointers (length n+1), int32
        indices: CSR column indices (length nnz), int32
        data: CSR non-zero values (length nnz), float64
        x: Input vector (length n), float64

    Returns:
        b: Result vector (length n), float64
    """
    if not _SPRUX_FFI_AVAILABLE:
        raise RuntimeError("Sprux FFI extension not available")

    indptr = jnp.asarray(indptr, dtype=jnp.int32)
    indices = jnp.asarray(indices, dtype=jnp.int32)
    data = jnp.asarray(data, dtype=jnp.float64)
    x = jnp.asarray(x, dtype=jnp.float64)

    return dot_f64.bind(indptr, indices, data, x)


# =============================================================================
# JAX Primitives
# =============================================================================

solve_f64 = jax.extend.core.Primitive("sprux_solve_f64")
dot_f64 = jax.extend.core.Primitive("sprux_dot_f64")

# =============================================================================
# Primitive Implementations
# =============================================================================


def _ffi_call(name: str, indptr: Array, indices: Array, data: Array, x: Array) -> Array:
    """Call an FFI function."""
    call = jax.ffi.ffi_call(
        name,
        jax.ShapeDtypeStruct(x.shape, x.dtype),
    )
    return call(indptr, indices, data, x)


@solve_f64.def_impl
def solve_f64_impl(indptr: Array, indices: Array, data: Array, b: Array) -> Array:
    return _ffi_call("sprux_solve_f64", indptr, indices, data, b)


@dot_f64.def_impl
def dot_f64_impl(indptr: Array, indices: Array, data: Array, x: Array) -> Array:
    return _ffi_call("sprux_dot_f64", indptr, indices, data, x)


# =============================================================================
# FFI Target Registration
# =============================================================================

if _SPRUX_FFI_AVAILABLE:
    jax.ffi.register_ffi_target(
        "sprux_solve_f64",
        _sprux_jax_cpp.sprux_solve_f64(),
        platform="cpu",
    )

    jax.ffi.register_ffi_target(
        "sprux_dot_f64",
        _sprux_jax_cpp.sprux_dot_f64(),
        platform="cpu",
    )

# =============================================================================
# MLIR Lowerings
# =============================================================================

if _SPRUX_FFI_AVAILABLE:
    solve_f64_low = mlir.lower_fun(solve_f64_impl, multiple_results=False)
    mlir.register_lowering(solve_f64, solve_f64_low)

    dot_f64_low = mlir.lower_fun(dot_f64_impl, multiple_results=False)
    mlir.register_lowering(dot_f64, dot_f64_low)

# =============================================================================
# Abstract Evaluations
# =============================================================================


@solve_f64.def_abstract_eval
@dot_f64.def_abstract_eval
def general_abstract_eval(
    indptr: ShapedArray, indices: ShapedArray, data: ShapedArray, x: ShapedArray
) -> ShapedArray:
    """Abstract evaluation - returns shape/dtype info without computing."""
    return ShapedArray(x.shape, x.dtype)
