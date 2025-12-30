"""JAX-SPICE: GPU-Accelerated Analog Circuit Simulator"""

import jax

__version__ = "0.1.0"


def _backend_supports_x64() -> bool:
    """Check if the current JAX backend supports 64-bit floats.

    Returns:
        True if backend supports float64, False otherwise.

    Note:
        - Metal (Apple Silicon) does not support float64
        - TPU does not natively support float64
        - CPU and CUDA support float64
    """
    try:
        backend = jax.default_backend().lower()
        # Metal and TPU don't support 64-bit floats natively
        # IREE Metal backend reports as various names
        if backend in ("metal", "tpu", "iree_metal"):
            return False
        # Check for IREE backends that might be Metal
        devices = jax.devices()
        for d in devices:
            platform = getattr(d, "platform", "").lower()
            if "metal" in platform or "iree_metal" in platform:
                return False
        return True
    except Exception:
        # If we can't determine, default to enabling x64
        return True


def configure_precision(force_x64: bool | None = None) -> bool:
    """Configure JAX precision based on backend capabilities.

    Args:
        force_x64: If True, force x64 even on unsupported backends (may fail).
                   If False, force x32. If None (default), auto-detect.

    Returns:
        True if x64 is enabled, False otherwise.

    This function is called automatically on import, but can be called again
    to reconfigure after changing backends.
    """
    if force_x64 is not None:
        enable_x64 = force_x64
    else:
        enable_x64 = _backend_supports_x64()

    jax.config.update("jax_enable_x64", enable_x64)
    return enable_x64


def get_precision_info() -> dict:
    """Get information about the current precision configuration.

    Returns:
        Dict with precision settings and backend info.
    """
    return {
        "x64_enabled": jax.config.jax_enable_x64,
        "backend": jax.default_backend(),
        "backend_supports_x64": _backend_supports_x64(),
    }


# Auto-configure precision on import
_x64_enabled = configure_precision()

# Core simulation API
from jax_spice.analysis import CircuitEngine, TransientResult

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
    "CircuitEngine",
    "TransientResult",
    # Precision configuration
    "configure_precision",
    "get_precision_info",
    # Profiling
    "profile",
    "profile_section",
    "enable_profiling",
    "disable_profiling",
    "ProfileConfig",
    "ProfileTimer",
]
