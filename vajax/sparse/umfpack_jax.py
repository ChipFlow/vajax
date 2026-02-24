"""UMFPACK JAX FFI - High-performance sparse direct solver for JAX.

This module provides UMFPACK sparse linear solver with XLA FFI bindings,
eliminating the ~100ms pure_callback overhead. It follows the same pattern
as klujax for JAX FFI integration.

Example usage:
    import jax.numpy as jnp
    from vajax.sparse import umfpack_jax

    # Sparse matrix in CSR format
    indptr = jnp.array([0, 2, 4, 6], dtype=jnp.int32)
    indices = jnp.array([0, 1, 0, 1, 0, 1], dtype=jnp.int32)
    data = jnp.array([4., 1., 1., 3., 2., 2.], dtype=jnp.float64)
    b = jnp.array([1., 2., 3.], dtype=jnp.float64)

    # Solve Ax = b
    x = umfpack_jax.solve(indptr, indices, data, b)
"""

from __future__ import annotations

import os
import sys
from typing import Tuple

import jax
import jax.extend.core
import jax.numpy as jnp
from jax import lax
from jax.core import ShapedArray
from jax.interpreters import ad, batching, mlir
from jaxtyping import Array

__version__ = "0.1.0"
__all__ = ["solve", "dot", "solve_transpose", "clear_cache", "is_available"]

# =============================================================================
# Configuration
# =============================================================================

DEBUG = os.environ.get("UMFPACK_JAX_DEBUG", False)


def _debug(s: str) -> None:
    if DEBUG:
        print(s, file=sys.stderr)


# =============================================================================
# Check if the FFI extension is available
# =============================================================================

_UMFPACK_FFI_AVAILABLE = False
_umfpack_jax_cpp = None

try:
    import umfpack_jax_cpp as _umfpack_jax_cpp

    _UMFPACK_FFI_AVAILABLE = True
    _debug("UMFPACK FFI extension available")
except ImportError:
    _debug("UMFPACK FFI extension not available - using fallback")


def is_available() -> bool:
    """Check if the UMFPACK FFI extension is available."""
    return _UMFPACK_FFI_AVAILABLE


def clear_cache() -> None:
    """Clear the symbolic factorization cache.

    Call this when switching between matrices with different sparsity patterns.
    """
    if _UMFPACK_FFI_AVAILABLE:
        _umfpack_jax_cpp.clear_cache()


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

    Args:
        indptr: CSR row pointers (length n+1), int32
        indices: CSR column indices (length nnz), int32
        data: CSR non-zero values (length nnz), float64
        b: Right-hand side vector (length n), float64

    Returns:
        x: Solution vector (length n), float64
    """
    _debug("umfpack_jax.solve")

    if not _UMFPACK_FFI_AVAILABLE:
        raise RuntimeError(
            "UMFPACK FFI extension not available. "
            "Install with: cd vajax/sparse && pip install ."
        )

    # Ensure correct dtypes
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
    _debug("umfpack_jax.dot")

    if not _UMFPACK_FFI_AVAILABLE:
        raise RuntimeError("UMFPACK FFI extension not available")

    indptr = jnp.asarray(indptr, dtype=jnp.int32)
    indices = jnp.asarray(indices, dtype=jnp.int32)
    data = jnp.asarray(data, dtype=jnp.float64)
    x = jnp.asarray(x, dtype=jnp.float64)

    return dot_f64.bind(indptr, indices, data, x)


def solve_transpose(
    indptr: Array,
    indices: Array,
    data: Array,
    b: Array,
) -> Array:
    """Solve for x in A^T x = b (transpose solve).

    This is needed for reverse-mode automatic differentiation.

    Args:
        indptr: CSR row pointers (length n+1), int32
        indices: CSR column indices (length nnz), int32
        data: CSR non-zero values (length nnz), float64
        b: Right-hand side vector (length n), float64

    Returns:
        x: Solution vector (length n), float64
    """
    _debug("umfpack_jax.solve_transpose")

    if not _UMFPACK_FFI_AVAILABLE:
        raise RuntimeError("UMFPACK FFI extension not available")

    indptr = jnp.asarray(indptr, dtype=jnp.int32)
    indices = jnp.asarray(indices, dtype=jnp.int32)
    data = jnp.asarray(data, dtype=jnp.float64)
    b = jnp.asarray(b, dtype=jnp.float64)

    return solve_transpose_f64.bind(indptr, indices, data, b)


# =============================================================================
# JAX Primitives
# =============================================================================

solve_f64 = jax.extend.core.Primitive("umfpack_solve_f64")
dot_f64 = jax.extend.core.Primitive("umfpack_dot_f64")
solve_transpose_f64 = jax.extend.core.Primitive("umfpack_solve_transpose_f64")

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
    return _ffi_call("umfpack_solve_f64", indptr, indices, data, b)


@dot_f64.def_impl
def dot_f64_impl(indptr: Array, indices: Array, data: Array, x: Array) -> Array:
    return _ffi_call("umfpack_dot_f64", indptr, indices, data, x)


@solve_transpose_f64.def_impl
def solve_transpose_f64_impl(indptr: Array, indices: Array, data: Array, b: Array) -> Array:
    return _ffi_call("umfpack_solve_transpose_f64", indptr, indices, data, b)


# =============================================================================
# FFI Target Registration
# =============================================================================

if _UMFPACK_FFI_AVAILABLE:
    jax.ffi.register_ffi_target(
        "umfpack_solve_f64",
        _umfpack_jax_cpp.umfpack_solve_f64(),
        platform="cpu",
    )

    jax.ffi.register_ffi_target(
        "umfpack_dot_f64",
        _umfpack_jax_cpp.umfpack_dot_f64(),
        platform="cpu",
    )

    jax.ffi.register_ffi_target(
        "umfpack_solve_transpose_f64",
        _umfpack_jax_cpp.umfpack_solve_transpose_f64(),
        platform="cpu",
    )

# =============================================================================
# MLIR Lowerings
# =============================================================================

if _UMFPACK_FFI_AVAILABLE:
    solve_f64_low = mlir.lower_fun(solve_f64_impl, multiple_results=False)
    mlir.register_lowering(solve_f64, solve_f64_low)

    dot_f64_low = mlir.lower_fun(dot_f64_impl, multiple_results=False)
    mlir.register_lowering(dot_f64, dot_f64_low)

    solve_transpose_f64_low = mlir.lower_fun(solve_transpose_f64_impl, multiple_results=False)
    mlir.register_lowering(solve_transpose_f64, solve_transpose_f64_low)

# =============================================================================
# Abstract Evaluations
# =============================================================================


@solve_f64.def_abstract_eval
@dot_f64.def_abstract_eval
@solve_transpose_f64.def_abstract_eval
def general_abstract_eval(
    indptr: ShapedArray, indices: ShapedArray, data: ShapedArray, x: ShapedArray
) -> ShapedArray:
    """Abstract evaluation - returns shape/dtype info without computing."""
    return ShapedArray(x.shape, x.dtype)


# =============================================================================
# Forward-mode Differentiation (JVP)
# =============================================================================


def solve_f64_jvp(
    arg_values: Tuple[Array, Array, Array, Array],
    arg_tangents: Tuple[Array, Array, Array, Array],
) -> Tuple[Array, Array]:
    """JVP rule for solve.

    For Ax = b, the JVP is:
        dx = A^{-1} (db - dA @ x)
           = A^{-1} db - A^{-1} dA @ x

    Args:
        arg_values: (indptr, indices, data, b)
        arg_tangents: (d_indptr, d_indices, d_data, db)

    Returns:
        (x, dx): Primal and tangent outputs
    """
    indptr, indices, data, b = arg_values
    _, _, d_data, db = arg_tangents

    # Handle zero tangents
    d_data = d_data if not isinstance(d_data, ad.Zero) else lax.zeros_like_array(data)
    db = db if not isinstance(db, ad.Zero) else lax.zeros_like_array(b)

    # Primal: x = A^{-1} b
    x = solve_f64.bind(indptr, indices, data, b)

    # dA @ x
    dA_x = dot_f64.bind(indptr, indices, d_data, x)

    # A^{-1} (db - dA @ x)
    dx = solve_f64.bind(indptr, indices, data, db - dA_x)

    return x, dx


ad.primitive_jvps[solve_f64] = solve_f64_jvp


def dot_f64_jvp(
    arg_values: Tuple[Array, Array, Array, Array],
    arg_tangents: Tuple[Array, Array, Array, Array],
) -> Tuple[Array, Array]:
    """JVP rule for dot.

    For b = A @ x, the JVP is:
        db = dA @ x + A @ dx

    Args:
        arg_values: (indptr, indices, data, x)
        arg_tangents: (d_indptr, d_indices, d_data, dx)

    Returns:
        (b, db): Primal and tangent outputs
    """
    indptr, indices, data, x = arg_values
    _, _, d_data, dx = arg_tangents

    d_data = d_data if not isinstance(d_data, ad.Zero) else lax.zeros_like_array(data)
    dx = dx if not isinstance(dx, ad.Zero) else lax.zeros_like_array(x)

    # Primal: b = A @ x
    b = dot_f64.bind(indptr, indices, data, x)

    # dA @ x
    dA_x = dot_f64.bind(indptr, indices, d_data, x)

    # A @ dx
    A_dx = dot_f64.bind(indptr, indices, data, dx)

    return b, dA_x + A_dx


ad.primitive_jvps[dot_f64] = dot_f64_jvp


# =============================================================================
# Reverse-mode Differentiation (Transpose/VJP)
# =============================================================================


def solve_f64_transpose(
    ct: Array,
    indptr: Array,
    indices: Array,
    data: Array,
    b: Array,
) -> Tuple[Array, Array, Array, Array]:
    """Transpose rule for solve (used in VJP).

    For x = A^{-1} b, the VJP is:
        For db: db_bar = A^{-T} x_bar
        For dA: dA_bar[i,j] = -(A^{-T} x_bar)[i] * x[j]

    Args:
        ct: Cotangent (x_bar)
        indptr, indices, data: Matrix structure
        b: Original RHS

    Returns:
        Cotangents for (indptr, indices, data, b)
    """
    if ad.is_undefined_primal(indptr) or ad.is_undefined_primal(indices):
        raise ValueError("Sparse indices should not require gradients")

    if ad.is_undefined_primal(b):
        # db_bar = A^{-T} x_bar
        db_bar = solve_transpose_f64.bind(indptr, indices, data, ct)
        return None, None, None, db_bar

    if ad.is_undefined_primal(data):
        # Need to compute -A^{-T} ct and original solution
        # For simplicity, we compute this using the fact that
        # dA_bar[k] = -y[row[k]] * x[col[k]] where y = A^{-T} ct
        #
        # This requires access to the row/col structure, which we have
        # in CSR format. However, this is complex to implement correctly.
        # For now, we raise an error suggesting to use klujax if
        # differentiation w.r.t. matrix values is needed.
        raise NotImplementedError(
            "Differentiation w.r.t. matrix values not yet implemented. "
            "Use klujax.solve() if you need this functionality."
        )

    raise ValueError("No undefined primals in transpose")


ad.primitive_transposes[solve_f64] = solve_f64_transpose


def dot_f64_transpose(
    ct: Array,
    indptr: Array,
    indices: Array,
    data: Array,
    x: Array,
) -> Tuple[Array, Array, Array, Array]:
    """Transpose rule for dot.

    For b = A @ x, the transpose is:
        For dx: A^T @ ct (sparse matrix transpose-vector multiply)
        For dA: dA[i,j] = ct[i] * x[j]
    """
    if ad.is_undefined_primal(indptr) or ad.is_undefined_primal(indices):
        raise ValueError("Sparse indices should not require gradients")

    if ad.is_undefined_primal(x):
        # dx = A^T @ ct
        # For CSR format, A^T @ ct is equivalent to iterating over columns
        # We can compute this by treating the CSR as a CSC and doing matvec
        # For now, use a simple loop
        ct.shape[0]

        # A^T @ ct: for each column j, sum over rows i: A[i,j] * ct[i]
        # In CSR: for each row i, A[i,j] contributes to result[j]
        def compute_at_ct(carry, inputs):
            row_idx, start, end = inputs
            row_indices = lax.dynamic_slice(indices, [start], [end - start])
            row_data = lax.dynamic_slice(data, [start], [end - start])
            result = carry
            # Scatter add: result[row_indices] += row_data * ct[row_idx]
            result = result.at[row_indices].add(row_data * ct[row_idx])
            return result, None

        # This is tricky in JAX due to dynamic shapes. Use a simpler approach.
        # For now, raise not implemented.
        raise NotImplementedError("Differentiation w.r.t. x in dot not yet implemented")

    if ad.is_undefined_primal(data):
        # dA[k] = ct[row[k]] * x[col[k]]
        # In CSR format, for position k in row i, dA[k] = ct[i] * x[indices[k]]
        raise NotImplementedError("Differentiation w.r.t. matrix values not yet implemented")

    raise ValueError("No undefined primals in transpose")


ad.primitive_transposes[dot_f64] = dot_f64_transpose


# =============================================================================
# Batching (vmap support)
# =============================================================================

# For now, batching is not supported. Users can manually vmap over
# the b vector if needed.


def solve_f64_vmap(
    vector_arg_values: Tuple[Array, Array, Array, Array],
    batch_axes: Tuple[int | None, int | None, int | None, int | None],
) -> Tuple[Array, int]:
    """Batching rule for solve."""
    indptr, indices, data, b = vector_arg_values
    ax_indptr, ax_indices, ax_data, ax_b = batch_axes

    if ax_indptr is not None or ax_indices is not None:
        raise ValueError("Cannot vmap over sparse matrix structure")

    if ax_data is not None and ax_b is not None:
        # Both data and b are batched - handle this case
        raise NotImplementedError("Batching over both matrix values and RHS not yet implemented")

    if ax_data is not None:
        # Different matrices, same RHS
        raise NotImplementedError("Batching over matrix values not yet implemented")

    if ax_b is not None:
        # Same matrix, different RHS vectors
        # This is the common case for multiple solves
        # For efficiency, we should batch these together
        # For now, use a sequential fallback
        raise NotImplementedError(
            "Batching over RHS vectors not yet implemented. "
            "Use jax.lax.map() or a for loop instead."
        )

    raise ValueError("vmap failed - no batch axis specified")


batching.primitive_batchers[solve_f64] = solve_f64_vmap
batching.primitive_batchers[dot_f64] = solve_f64_vmap  # Same logic
batching.primitive_batchers[solve_transpose_f64] = solve_f64_vmap
