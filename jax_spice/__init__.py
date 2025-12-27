"""JAX-SPICE: GPU-Accelerated Analog Circuit Simulator"""

import jax

# Import backend detection before enabling x64
from jax_spice.analysis.gpu_backend import is_metal_backend, _init_default_dtype

# Enable 64-bit precision only on non-Metal backends
# Metal backends (jax-metal, iree-metal) only support float32
if not is_metal_backend():
    jax.config.update("jax_enable_x64", True)

# Initialize the default dtype based on backend
_init_default_dtype()

__version__ = "0.1.0"

# Core simulation API
from jax_spice.simulator import Simulator, TransientResult

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
    # Core API
    "Simulator",
    "TransientResult",
    # Profiling
    "profile",
    "profile_section",
    "enable_profiling",
    "disable_profiling",
    "ProfileConfig",
    "ProfileTimer",
]
