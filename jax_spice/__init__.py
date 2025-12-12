"""JAX-SPICE: GPU-Accelerated Analog Circuit Simulator"""

import jax

# Enable 64-bit precision by default (required for circuit simulation accuracy)
jax.config.update("jax_enable_x64", True)

__version__ = "0.1.0"
