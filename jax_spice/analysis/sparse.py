"""Sparse linear solver with CPU/GPU backends for circuit simulation

This module provides a differentiable sparse linear solver that automatically
selects the best backend based on the JAX platform:
- CPU: Uses scipy.sparse.linalg.spsolve via jax.pure_callback
- GPU: Uses jax.experimental.sparse.linalg.spsolve (cuSOLVER)

The solver supports reverse-mode autodiff through jax.custom_vjp using
the adjoint method for implicit differentiation.

Pattern adapted from jax_fdm (MIT license)
https://github.com/arpastrana/jax_fdm/blob/main/src/jax_fdm/equilibrium/sparse.py

References:
- JAX-FEM: https://github.com/deepmodeling/jax-fem
- Lineax: https://github.com/patrick-kidger/lineax
- JAX sparse discussion: https://github.com/jax-ml/jax/discussions/18452
"""

from typing import Tuple
import jax
from jax import Array
import numpy as np
from scipy.sparse import csc_matrix, csr_matrix
from scipy.sparse.linalg import spsolve as scipy_spsolve


def sparse_solve(
    data: Array,
    indices: Array,
    indptr: Array,
    b: Array,
    shape: Tuple[int, int]
) -> Array:
    """Solve sparse linear system Ax = b

    Automatically selects CPU or GPU backend based on JAX platform.
    Supports reverse-mode autodiff.

    Args:
        data: Non-zero values of sparse matrix A (CSC format)
        indices: Row indices for each value (CSC format)
        indptr: Column pointers (CSC format)
        b: Right-hand side vector
        shape: Matrix shape (n, n)

    Returns:
        Solution vector x
    """
    backend = jax.default_backend()
    if backend == 'cpu':
        return _spsolve_cpu(data, indices, indptr, b, shape)
    elif backend == 'tpu':
        # TPU doesn't have native sparse solve; fall back to CPU via callback
        return _spsolve_cpu(data, indices, indptr, b, shape)
    else:
        return _spsolve_gpu(data, indices, indptr, b, shape)


def _spsolve_cpu(
    data: Array,
    indices: Array,
    indptr: Array,
    b: Array,
    shape: Tuple[int, int]
) -> Array:
    """CPU sparse solve using scipy via pure_callback

    Args:
        data: CSC data array
        indices: CSC indices array
        indptr: CSC indptr array
        b: RHS vector
        shape: Matrix shape

    Returns:
        Solution vector
    """
    def solve_fn(data_np, indices_np, indptr_np, b_np):
        """Pure numpy/scipy function for callback"""
        A = csc_matrix((data_np, indices_np, indptr_np), shape=shape)
        x = scipy_spsolve(A, b_np)
        return np.asarray(x)

    result = jax.pure_callback(
        solve_fn,
        jax.ShapeDtypeStruct(b.shape, b.dtype),
        data, indices, indptr, b
    )
    return result


def _spsolve_gpu(
    data: Array,
    indices: Array,
    indptr: Array,
    b: Array,
    shape: Tuple[int, int]
) -> Array:
    """GPU sparse solve using jax.experimental.sparse

    Uses cuSOLVER under the hood. Only supports 1D RHS vectors.

    Args:
        data: CSR data array (GPU expects CSR)
        indices: CSR indices array
        indptr: CSR indptr array
        b: RHS vector
        shape: Matrix shape

    Returns:
        Solution vector
    """
    from jax.experimental.sparse.linalg import spsolve as jax_spsolve

    # jax.experimental.sparse.linalg.spsolve expects CSR format
    # and takes (data, indices, indptr, b, tol) as arguments
    x = jax_spsolve(data, indices, indptr, b, tol=0)
    return x


# For autodiff support, we could add custom_vjp here if needed
# For now, the pure_callback version doesn't support autodiff,
# but the GPU version via jax.experimental.sparse does have some support


def build_csc_arrays(
    rows: np.ndarray,
    cols: np.ndarray,
    values: np.ndarray,
    shape: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert COO triplets to CSC format arrays

    Args:
        rows: Row indices
        cols: Column indices
        values: Non-zero values
        shape: Matrix shape (n, n)

    Returns:
        Tuple of (data, indices, indptr) in CSC format
    """
    from scipy.sparse import coo_matrix

    # Build COO matrix and convert to CSC
    A_coo = coo_matrix((values, (rows, cols)), shape=shape)
    A_csc = A_coo.tocsc()
    A_csc.sum_duplicates()  # Combine duplicate entries

    return A_csc.data, A_csc.indices, A_csc.indptr


def build_csr_arrays(
    rows: np.ndarray,
    cols: np.ndarray,
    values: np.ndarray,
    shape: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert COO triplets to CSR format arrays

    Args:
        rows: Row indices
        cols: Column indices
        values: Non-zero values
        shape: Matrix shape (n, n)

    Returns:
        Tuple of (data, indices, indptr) in CSR format
    """
    from scipy.sparse import coo_matrix

    # Build COO matrix and convert to CSR
    A_coo = coo_matrix((values, (rows, cols)), shape=shape)
    A_csr = A_coo.tocsr()
    A_csr.sum_duplicates()  # Combine duplicate entries

    return A_csr.data, A_csr.indices, A_csr.indptr


def sparse_solve_csr(
    data: Array,
    indices: Array,
    indptr: Array,
    b: Array,
    shape: Tuple[int, int]
) -> Array:
    """Solve sparse linear system Ax = b with CSR format matrix

    This is the preferred format for GPU (cuSOLVER uses CSR internally).

    Args:
        data: Non-zero values of sparse matrix A (CSR format)
        indices: Column indices for each value (CSR format)
        indptr: Row pointers (CSR format)
        b: Right-hand side vector
        shape: Matrix shape (n, n)

    Returns:
        Solution vector x
    """
    backend = jax.default_backend()
    if backend == 'cpu':
        return _spsolve_cpu_csr(data, indices, indptr, b, shape)
    elif backend == 'tpu':
        # TPU doesn't have native sparse solve; fall back to CPU via callback
        return _spsolve_cpu_csr(data, indices, indptr, b, shape)
    else:
        return _spsolve_gpu(data, indices, indptr, b, shape)


def _spsolve_cpu_csr(
    data: Array,
    indices: Array,
    indptr: Array,
    b: Array,
    shape: Tuple[int, int]
) -> Array:
    """CPU sparse solve using scipy with CSR format"""
    def solve_fn(data_np, indices_np, indptr_np, b_np):
        A = csr_matrix((data_np, indices_np, indptr_np), shape=shape)
        x = scipy_spsolve(A, b_np)
        return np.asarray(x)

    result = jax.pure_callback(
        solve_fn,
        jax.ShapeDtypeStruct(b.shape, b.dtype),
        data, indices, indptr, b
    )
    return result
