"""Pytest configuration for JAX-SPICE tests

Forces CPU backend for tests since Metal doesn't support all operations.
"""

import os

# Force CPU backend before any JAX imports
# Metal doesn't support triangular_solve needed for linear solvers
os.environ['JAX_PLATFORMS'] = 'cpu'

import jax

# Enable float64 for numerical precision in tests
jax.config.update('jax_enable_x64', True)
