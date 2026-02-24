"""VA-JAX Sparse Solver Extensions.

This module provides high-performance sparse solvers with XLA FFI bindings
for use within JAX-compiled code.

Available solvers:
- umfpack_jax: UMFPACK FFI bindings (requires building the C++ extension)
"""

from vajax.sparse import umfpack_jax

__all__ = ["umfpack_jax"]
