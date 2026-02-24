"""Unified sparse linear solver for circuit simulation

This module provides a differentiable sparse linear solver that works
identically across CPU, GPU, and TPU using JAX's native sparse operations.

The solver uses jax.experimental.sparse.linalg.spsolve which:
- Works on CPU via JAX's built-in sparse solver
- Works on GPU via cuSOLVER
- Provides a unified API across all platforms

NaN/Inf Safety:
    This module does NOT perform explicit NaN/Inf checking on inputs.
    Safety is guaranteed by upstream code in the simulation pipeline:

    1. Device models (OpenVAF/Verilog-A) clamp terminal voltages to prevent
       numerical overflow (e.g., exp(V) for diodes is clamped).

    2. The Jacobian builder in solver.py/transient.py validates that device
       contributions are finite before assembly.

    3. Newton-Raphson convergence checks in solver.py detect and handle
       divergence before it propagates to the linear solve.

    This separation of concerns keeps the sparse solver focused on its
    core responsibility (linear algebra) while upstream components ensure
    numerical validity.

Note: For CPU-optimized circuit simulation, see VACASK which uses
native sparse solvers. This module prioritizes code consistency
across platforms over maximum single-platform performance.

Note: numpy is imported for type hints (ArrayLike) and array protocol compatibility.
All sparse operations use JAX's BCOO/BCSR formats for GPU acceleration.
"""

from typing import Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.experimental.sparse.linalg import spsolve as jax_spsolve

# ArrayLike accepts both numpy and JAX arrays (converted to JAX internally)
ArrayLike = Union[Array, np.ndarray]


def sparse_solve_csr(
    data: Array,
    indices: Array,
    indptr: Array,
    b: Array,
    shape: Tuple[int, int],
) -> Array:
    """Solve sparse linear system Ax = b with CSR format matrix

    Uses JAX's native sparse solver which works on all platforms.

    Args:
        data: Non-zero values of sparse matrix A (CSR format)
        indices: Column indices for each value (CSR format)
        indptr: Row pointers (CSR format)
        b: Right-hand side vector
        shape: Matrix shape (n, n)

    Returns:
        Solution vector x
    """
    # Ensure inputs are JAX arrays with the right dtype
    data = jnp.asarray(data)
    indices = jnp.asarray(indices)
    indptr = jnp.asarray(indptr)
    b = jnp.asarray(b)

    x = jax_spsolve(data, indices, indptr, b, tol=0)
    return x


def build_csr_arrays(
    rows: ArrayLike,
    cols: ArrayLike,
    values: ArrayLike,
    shape: Tuple[int, int],
) -> Tuple[Array, Array, Array]:
    """Convert COO triplets to CSR format arrays using pure JAX.

    Handles duplicate entries by summing them using segment_sum.

    Args:
        rows: Row indices (COO format) as JAX array
        cols: Column indices (COO format) as JAX array
        values: Non-zero values (COO format) as JAX array
        shape: Matrix shape (n, n)

    Returns:
        Tuple of (data, indices, indptr) in CSR format as JAX arrays
    """
    rows = jnp.asarray(rows)
    cols = jnp.asarray(cols)
    values = jnp.asarray(values)

    n_rows = shape[0]

    # Create linear indices for sorting (row-major order)
    linear_idx = rows * shape[1] + cols

    # Sort by linear index to group duplicates
    sort_order = jnp.argsort(linear_idx)
    sorted_linear = linear_idx[sort_order]
    sorted_values = values[sort_order]

    # Find unique entries and sum duplicates using segment_sum
    unique_linear, unique_inverse = jnp.unique(sorted_linear, return_inverse=True)
    n_unique = len(unique_linear)
    summed_values = jax.ops.segment_sum(sorted_values, unique_inverse, num_segments=n_unique)

    # Get unique row/col indices
    unique_rows = unique_linear // shape[1]
    unique_cols = unique_linear % shape[1]

    # Sort by row for CSR format
    row_order = jnp.argsort(unique_rows)
    csr_rows = unique_rows[row_order]
    csr_cols = unique_cols[row_order]
    csr_data = summed_values[row_order]

    # Build indptr (row pointers)
    # Count entries per row
    row_counts = jnp.zeros(n_rows + 1, dtype=jnp.int32)
    row_counts = row_counts.at[csr_rows + 1].add(1)
    indptr = jnp.cumsum(row_counts)

    return csr_data, csr_cols.astype(jnp.int32), indptr


def build_csc_arrays(
    rows: Array,
    cols: Array,
    values: Array,
    shape: Tuple[int, int],
) -> Tuple[Array, Array, Array]:
    """Convert COO triplets to CSC format arrays using pure JAX.

    Note: Prefer build_csr_arrays for new code as JAX spsolve uses CSR format.
    """
    # CSC is CSR of the transpose
    data, row_indices, colptr = build_csr_arrays(cols, rows, values, (shape[1], shape[0]))
    return data, row_indices, colptr


# Legacy alias
def sparse_solve(
    data: Array,
    indices: Array,
    indptr: Array,
    b: Array,
    shape: Tuple[int, int],
) -> Array:
    """Legacy alias for sparse_solve_csr"""
    return sparse_solve_csr(data, indices, indptr, b, shape)


# GPU-optimized sparse operations using BCOO format


def build_bcoo_from_coo(
    rows: Array,
    cols: Array,
    values: Array,
    shape: Tuple[int, int],
) -> "jax.experimental.sparse.BCOO":
    """Build BCOO sparse matrix from COO triplets.

    BCOO (Batched COO) is JAX's native sparse format, efficient for GPU.

    Args:
        rows: Row indices as JAX array
        cols: Column indices as JAX array
        values: Non-zero values as JAX array
        shape: Matrix shape (n, n)

    Returns:
        BCOO sparse matrix
    """
    from jax.experimental.sparse import BCOO

    # Stack indices into (nnz, 2) array
    indices = jnp.stack([rows, cols], axis=1)
    return BCOO((values, indices), shape=shape)


def sparse_solve_bcoo(
    bcoo_matrix: "jax.experimental.sparse.BCOO",
    b: Array,
) -> Array:
    """Solve sparse linear system Ax = b with BCOO format matrix.

    This is the GPU-native path - keeps data on GPU throughout.

    Args:
        bcoo_matrix: Sparse matrix A in BCOO format
        b: Right-hand side vector

    Returns:
        Solution vector x
    """
    # Convert BCOO to CSR for the solver
    # JAX's spsolve expects CSR format
    csr = bcoo_matrix.tocsr()
    return jax_spsolve(csr.data, csr.indices, csr.indptr, b, tol=0)


def dense_to_sparse_gpu(
    dense_matrix: Array,
) -> "jax.experimental.sparse.BCOO":
    """Convert dense matrix to BCOO sparse format on GPU.

    Useful for converting dense Jacobians to sparse format for solving.

    Args:
        dense_matrix: Dense matrix as JAX array

    Returns:
        BCOO sparse matrix
    """
    from jax.experimental.sparse import BCOO

    return BCOO.fromdense(dense_matrix, nse=None)


def sparse_solve_from_dense_gpu(
    dense_matrix: Array,
    b: Array,
) -> Array:
    """Solve Ax = b where A is provided as dense but solved as sparse.

    Converts dense matrix to sparse and solves on GPU. This is useful
    when the Jacobian is built densely but we want sparse solve efficiency.

    Args:
        dense_matrix: Dense matrix A
        b: Right-hand side vector

    Returns:
        Solution vector x
    """
    bcoo = dense_to_sparse_gpu(dense_matrix)
    return sparse_solve_bcoo(bcoo, b)
