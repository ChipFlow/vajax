"""JAX-SPICE: GPU-Accelerated Analog Circuit Simulator"""

# Force CPU backend on Apple Silicon to avoid Metal backend compatibility issues
# This must be done before any JAX imports
import os as _os
import platform as _platform

if _platform.system() == "Darwin" and _platform.machine() == "arm64":
    # Apple Silicon - force CPU backend to avoid Metal/GPU issues
    _os.environ.setdefault("JAX_PLATFORMS", "cpu")

from jax_spice.circuit import Circuit
from jax_spice.devices.mosfet_simple import MOSFETSimple

__version__ = "0.1.0"
__all__ = ["Circuit", "MOSFETSimple"]
