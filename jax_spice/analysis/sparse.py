"""Unified sparse linear solver for circuit simulation

This module provides a differentiable sparse linear solver that works
identically across CPU, GPU, and TPU using JAX's native sparse operations.

The solver uses jax.experimental.sparse.linalg.spsolve which:
- Works on CPU via JAX's built-in sparse solver
- Works on GPU via cuSOLVER
- Provides a unified API across all platforms

Note: For CPU-optimized circuit simulation, see VACASK which uses
native sparse solvers. This module prioritizes code consistency
across platforms over maximum single-platform performance.
"""

from typing import Tuple
import jax
import jax.numpy as jnp
from jax import Array
from jax.experimental.sparse.linalg import spsolve as jax_spsolve
import numpy as np
from scipy.sparse import coo_matrix


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
    rows: np.ndarray,
    cols: np.ndarray,
    values: np.ndarray,
    shape: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert COO triplets to CSR format arrays

    Uses scipy for reliable handling of duplicate entries (sums them).

    Args:
        rows: Row indices (COO format)
        cols: Column indices (COO format)
        values: Non-zero values (COO format)
        shape: Matrix shape (n, n)

    Returns:
        Tuple of (data, indices, indptr) in CSR format
    """
    # Build COO matrix and convert to CSR
    # scipy handles duplicate summing correctly
    A_coo = coo_matrix((values, (rows, cols)), shape=shape)
    A_csr = A_coo.tocsr()
    A_csr.sum_duplicates()  # Combine duplicate entries

    return A_csr.data, A_csr.indices, A_csr.indptr


def build_csc_arrays(
    rows: np.ndarray,
    cols: np.ndarray,
    values: np.ndarray,
    shape: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert COO triplets to CSC format arrays

    Note: Prefer build_csr_arrays for new code as JAX uses CSR format.
    """
    A_coo = coo_matrix((values, (rows, cols)), shape=shape)
    A_csc = A_coo.tocsc()
    A_csc.sum_duplicates()

    return A_csc.data, A_csc.indices, A_csc.indptr


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
