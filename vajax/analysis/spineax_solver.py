"""Spineax-based sparse solver with cached symbolic factorization.

This module wraps Spineax (cuDSS) to provide fast sparse linear solves
with cached symbolic factorization. On the first call, METIS reordering
and symbolic analysis are performed. Subsequent calls reuse this analysis
and only perform numeric refactorization + solve.

This provides 10-100x speedup over JAX's built-in spsolve for repeated
solves with the same sparsity pattern (like Newton-Raphson iterations).

Requires: spineax package (GPU only, Linux)
Falls back to JAX spsolve if spineax is not available.
"""

import logging
from typing import Optional, Tuple

from jax import Array

logger = logging.getLogger(__name__)

# Check if spineax is available
_SPINEAX_AVAILABLE = False
_CuDSSSolver = None

try:
    from spineax.cudss.solver import CuDSSSolver as _CuDSSSolver

    _SPINEAX_AVAILABLE = True
    logger.info("Spineax (cuDSS) solver available")
except ImportError:
    logger.debug("Spineax not available, will use JAX spsolve fallback")


def is_spineax_available() -> bool:
    """Check if Spineax is available for use."""
    return _SPINEAX_AVAILABLE


class SpineaxSolver:
    """Sparse solver with cached symbolic factorization using Spineax/cuDSS.

    This solver caches the symbolic analysis (METIS reordering, fill-in pattern)
    from the first solve and reuses it for subsequent solves with the same
    sparsity pattern.

    Usage:
        # Create solver with sparsity pattern
        solver = SpineaxSolver(indptr, indices, n)

        # Solve many times (only first call does symbolic analysis)
        for step in range(1000):
            x = solver.solve(data, b)

    Attributes:
        n: Matrix dimension
        _solver: Underlying CuDSSSolver instance (created on first solve)
    """

    def __init__(
        self,
        indptr: Array,
        indices: Array,
        n: int,
        device_id: int = 0,
        matrix_type: str = "general",
    ):
        """Initialize solver with sparsity pattern.

        Args:
            indptr: CSR row pointers (length n+1)
            indices: CSR column indices
            n: Matrix dimension
            device_id: CUDA device ID
            matrix_type: One of "general", "symmetric", "spd", "hermitian", "hpd"
        """
        if not _SPINEAX_AVAILABLE:
            raise RuntimeError(
                "Spineax is not available. Install with: "
                "pip install git+https://github.com/johnviljoen/Spineax.git"
            )

        self.n = n
        self.indptr = indptr
        self.indices = indices
        self.device_id = device_id

        # Map matrix type to spineax constants
        mtype_map = {
            "general": 1,  # CUDSS_MTYPE_GENERAL
            "symmetric": 2,  # CUDSS_MTYPE_SYMMETRIC
            "spd": 3,  # CUDSS_MTYPE_SPD
            "hermitian": 4,  # CUDSS_MTYPE_HERMITIAN
            "hpd": 5,  # CUDSS_MTYPE_HPD
        }
        self.mtype_id = mtype_map.get(matrix_type, 1)
        self.mview_id = 0  # Full matrix (not lower/upper triangular view)

        self._solver = None
        self._initialized = False

    def _ensure_initialized(self):
        """Create the underlying solver on first use."""
        if not self._initialized:
            self._solver = _CuDSSSolver(
                self.indptr,
                self.indices,
                self.device_id,
                self.mtype_id,
                self.mview_id,
            )
            self._initialized = True
            logger.debug(f"SpineaxSolver initialized for {self.n}x{self.n} matrix")

    def solve(self, data: Array, b: Array) -> Tuple[Array, dict]:
        """Solve Ax = b using cached symbolic factorization.

        On the first call, performs METIS reordering, symbolic analysis,
        and numeric factorization. Subsequent calls only do refactorization
        and triangular solves.

        Args:
            data: CSR non-zero values (must match sparsity pattern from __init__)
            b: Right-hand side vector

        Returns:
            Tuple of (solution x, info dict with solver statistics)
        """
        self._ensure_initialized()
        x, info = self._solver(b, data)
        return x, info


def create_spineax_solver(
    indptr: Array,
    indices: Array,
    n: int,
    device_id: int = 0,
) -> Optional[SpineaxSolver]:
    """Create a Spineax solver if available, else return None.

    This is the recommended way to create a solver with graceful fallback.

    Args:
        indptr: CSR row pointers
        indices: CSR column indices
        n: Matrix dimension
        device_id: CUDA device ID

    Returns:
        SpineaxSolver instance if spineax available, else None
    """
    if not _SPINEAX_AVAILABLE:
        return None

    try:
        return SpineaxSolver(indptr, indices, n, device_id)
    except Exception as e:
        logger.warning(f"Failed to create SpineaxSolver: {e}")
        return None


def sparse_solve_with_spineax(
    data: Array,
    indices: Array,
    indptr: Array,
    b: Array,
    solver: Optional[SpineaxSolver] = None,
) -> Array:
    """Solve sparse system, using Spineax if solver provided, else JAX spsolve.

    This is a drop-in replacement for jax.experimental.sparse.linalg.spsolve
    that can use a cached Spineax solver for better performance.

    Args:
        data: CSR non-zero values
        indices: CSR column indices
        indptr: CSR row pointers
        b: Right-hand side vector
        solver: Optional SpineaxSolver with cached symbolic factorization

    Returns:
        Solution vector x
    """
    if solver is not None:
        x, _ = solver.solve(data, b)
        return x
    else:
        # Fallback to JAX spsolve
        from jax.experimental.sparse.linalg import spsolve

        return spsolve(data, indices, indptr, b, tol=1e-6)
