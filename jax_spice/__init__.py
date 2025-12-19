"""JAX-SPICE: GPU-Accelerated Analog Circuit Simulator"""

import jax

# Enable 64-bit precision by default (required for circuit simulation accuracy)
jax.config.update("jax_enable_x64", True)

__version__ = "0.1.0"

# Profiling utilities (lazy import to avoid loading unless needed)
from jax_spice.profiling import (
    profile,
    profile_section,
    enable_profiling,
    disable_profiling,
    ProfileConfig,
    ProfileTimer,
)

__all__ = [
    "profile",
    "profile_section",
    "enable_profiling",
    "disable_profiling",
    "ProfileConfig",
    "ProfileTimer",
]
