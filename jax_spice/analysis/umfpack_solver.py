"""UMFPACK-based sparse solver with cached symbolic factorization.

This module wraps scikit-umfpack to provide fast sparse linear solves
with cached symbolic factorization on CPU. On the first call, AMD reordering
and symbolic analysis are performed. Subsequent calls reuse this analysis
and only perform numeric refactorization + solve.

This provides 10-100x speedup over scipy's spsolve for repeated
solves with the same sparsity pattern (like Newton-Raphson iterations).

Requires: scikit-umfpack package (CPU only)
Falls back to JAX spsolve if scikit-umfpack is not available.
"""

from typing import Optional, Tuple

import numpy as np
import jax
import jax.numpy as jnp
from jax import Array
from scipy.sparse import csc_matrix

from jax_spice._logging import logger

# Check if scikit-umfpack is available
_UMFPACK_AVAILABLE = False
_UmfpackContext = None

try:
    from scikits.umfpack import UmfpackContext
    _UmfpackContext = UmfpackContext
    _UMFPACK_AVAILABLE = True
    logger.info("UMFPACK solver available (scikit-umfpack)")
except ImportError:
    logger.debug("scikit-umfpack not available, will use JAX spsolve fallback")


def is_umfpack_available() -> bool:
    """Check if UMFPACK is available for use."""
    return _UMFPACK_AVAILABLE


class UMFPACKSolver:
    """Sparse solver with cached symbolic factorization using UMFPACK.

    This solver is designed to match Spineax's interface for drop-in replacement
    on CPU. It uses UMFPACK via scikits.umfpack.UmfpackContext for fast LU
    factorization with cached symbolic analysis.

    The solver caches:
    1. CSR→CSC transpose mapping (computed once at init)
    2. Symbolic factorization (computed on first solve)

    Subsequent solves only perform numeric factorization + solve, giving
    10-100x speedup over scipy's spsolve for repeated solves.

    Usage:
        # Create solver with CSR sparsity pattern (like Spineax)
        solver = UMFPACKSolver(bcsr_indptr, bcsr_indices)

        # Solve from within JIT-compiled code
        # CSR data comes from pre-computed COO→CSR mapping
        delta, info = solver(b, csr_data)

    Attributes:
        n: Matrix dimension
        nnz: Number of non-zero entries
    """

    def __init__(
        self,
        bcsr_indptr: Array,
        bcsr_indices: Array,
        device_id: int = 0,  # Ignored, for Spineax API compatibility
        mtype_id: int = 1,   # Ignored, for Spineax API compatibility
        mview_id: int = 0,   # Ignored, for Spineax API compatibility
    ):
        """Initialize solver with CSR sparsity pattern.

        Args:
            bcsr_indptr: CSR row pointers (length n+1)
            bcsr_indices: CSR column indices
            device_id: Ignored (Spineax compatibility)
            mtype_id: Ignored (Spineax compatibility)
            mview_id: Ignored (Spineax compatibility)
        """
        if not _UMFPACK_AVAILABLE:
            raise RuntimeError(
                "scikit-umfpack is not available. Install with: "
                "pip install scikit-umfpack"
            )

        # Store CSR structure as contiguous numpy arrays
        self._csr_indptr = np.array(bcsr_indptr, dtype=np.int32, copy=True)
        self._csr_indices = np.array(bcsr_indices, dtype=np.int32, copy=True)
        self.n = len(self._csr_indptr) - 1
        self.nnz = len(self._csr_indices)

        # Pre-compute CSR→CSC transpose mapping
        # This avoids repeated COO→CSC conversion at solve time
        self._csc_indptr, self._csc_indices, self._csr_to_csc_map = \
            self._compute_csr_to_csc_mapping()

        # Create UMFPACK context (will cache symbolic factorization)
        self._umf = _UmfpackContext()
        self._symbolic_done = False

        logger.debug(f"UMFPACKSolver initialized: {self.n}×{self.n}, {self.nnz} nnz")

    def _compute_csr_to_csc_mapping(self):
        """Pre-compute the CSR→CSC transpose mapping.

        Returns:
            csc_indptr: CSC column pointers
            csc_indices: CSC row indices
            csr_to_csc_map: Index mapping from CSR data to CSC data
        """
        # Create a dummy CSR matrix to get the CSC structure
        dummy_data = np.ones(self.nnz, dtype=np.float64)
        from scipy.sparse import csr_matrix
        csr = csr_matrix(
            (dummy_data, self._csr_indices, self._csr_indptr),
            shape=(self.n, self.n)
        )
        csc = csr.tocsc()

        # Store CSC structure
        csc_indptr = np.array(csc.indptr, dtype=np.int32, copy=True)
        csc_indices = np.array(csc.indices, dtype=np.int32, copy=True)

        # Compute mapping: for each CSC data position, which CSR data position?
        # This is done by tracking the original positions through the conversion
        positions = np.arange(self.nnz, dtype=np.int32)
        csr_with_pos = csr_matrix(
            (positions.astype(np.float64), self._csr_indices, self._csr_indptr),
            shape=(self.n, self.n)
        )
        csc_with_pos = csr_with_pos.tocsc()
        csr_to_csc_map = csc_with_pos.data.astype(np.int32)

        return csc_indptr, csc_indices, csr_to_csc_map

    def _solve_impl(self, b, csr_data) -> np.ndarray:
        """Internal solve implementation (called via pure_callback).

        Args:
            b: RHS vector
            csr_data: CSR data values

        Returns:
            Solution vector x
        """
        # Convert inputs to contiguous numpy arrays
        b = np.ascontiguousarray(np.asarray(b), dtype=np.float64)
        csr_data = np.ascontiguousarray(np.asarray(csr_data), dtype=np.float64)

        # Reorder CSR data to CSC order using pre-computed mapping
        csc_data = np.ascontiguousarray(csr_data[self._csr_to_csc_map])

        # Create CSC matrix with the reordered data
        A_csc = csc_matrix(
            (csc_data, self._csc_indices.copy(), self._csc_indptr.copy()),
            shape=(self.n, self.n)
        )
        # Ensure arrays are contiguous
        A_csc.data = np.ascontiguousarray(A_csc.data)
        A_csc.indices = np.ascontiguousarray(A_csc.indices, dtype=np.intc)
        A_csc.indptr = np.ascontiguousarray(A_csc.indptr, dtype=np.intc)

        # Do symbolic factorization on first call (cached by UmfpackContext)
        if not self._symbolic_done:
            self._umf.symbolic(A_csc)
            self._symbolic_done = True
            logger.debug(f"UMFPACK symbolic factorization done for {self.n}x{self.n} matrix")

        # Numeric factorization (must be done each time values change)
        self._umf.numeric(A_csc)

        # Solve
        x = self._umf.solve(UMFPACK_A, A_csc, b, autoTranspose=True)

        return np.ascontiguousarray(x)

    def __call__(self, b: Array, csr_data: Array) -> Tuple[Array, dict]:
        """Solve Ax = b using UMFPACK.

        This method can be called from within JIT-compiled JAX code.
        Uses jax.pure_callback internally.

        Args:
            b: Right-hand side vector (JAX array)
            csr_data: CSR non-zero values (JAX array, same order as indices)

        Returns:
            Tuple of (solution x, info dict)
        """
        # Define callback that JAX can call
        def solve_callback(b_np, data_np):
            return self._solve_impl(b_np, data_np)

        # Use pure_callback to call numpy/UMFPACK from JAX
        x = jax.pure_callback(
            solve_callback,
            jax.ShapeDtypeStruct((self.n,), jnp.float64),
            b,
            csr_data,
        )

        info = {'solver': 'umfpack', 'n': self.n, 'nnz': self.nnz}
        return x, info


# UMFPACK solve type constant
UMFPACK_A = 0  # Solve Ax = b


def create_umfpack_solver(
    bcsr_indptr: Array,
    bcsr_indices: Array,
) -> Optional[UMFPACKSolver]:
    """Create an UMFPACK solver if available, else return None.

    This is the recommended way to create a solver with graceful fallback.

    Args:
        bcsr_indptr: CSR row pointers
        bcsr_indices: CSR column indices

    Returns:
        UMFPACKSolver instance if scikit-umfpack available, else None
    """
    if not _UMFPACK_AVAILABLE:
        return None

    try:
        return UMFPACKSolver(bcsr_indptr, bcsr_indices)
    except Exception as e:
        logger.warning(f"Failed to create UMFPACKSolver: {e}")
        return None
