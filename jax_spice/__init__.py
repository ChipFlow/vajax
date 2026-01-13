"""JAX-SPICE: GPU-Accelerated Analog Circuit Simulator"""
import logging

import jax

__version__ = "0.1.0"


logger = logging.getLogger("jax_spice")

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

    if enable_x64:
        logger.info("Using 64-bit float precision")
        print("Using 64-bit float precision")
    else:
        logger.warn("Using 32-bit float precision")
        print("Using 32-bit float precision")

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


# =============================================================================
# Simparam helpers (VAMS-LRM Table 9-27)
# =============================================================================

# Default simparam values per VAMS-LRM
SIMPARAM_DEFAULTS = {
    '$analysis_type': 0.0,    # 0=DC, 1=AC, 2=transient, 3=noise
    'gmin': 1e-12,            # Minimum conductance (S)
    'abstol': 1e-12,          # Absolute current tolerance (A)
    'vntol': 1e-6,            # Absolute voltage tolerance (V)
    'reltol': 1e-3,           # Relative tolerance
    'tnom': 300.15,           # Nominal temperature (K)
    'scale': 1.0,             # Scale factor
    'shrink': 0.0,            # Shrink factor
    'imax': 1.0,              # Branch current limit (A)
    '$abstime': 0.0,          # Absolute simulation time (s)
    '$mfactor': 1.0,          # Device multiplicity factor
}


def build_simparams(metadata: dict, values: dict | None = None) -> list:
    """Build simparams array from eval metadata and user values.

    This helper creates the simparams array that eval functions expect,
    based on the simparam_indices from translate_eval() metadata.

    Args:
        metadata: The metadata dict from translate_eval(), containing:
            - simparams_used: List of simparam names in index order
            - simparam_indices: Dict mapping name -> index
            - simparam_count: Total number of simparams
        values: Optional dict of simparam values to override defaults.
            Keys should match simparam names (e.g., 'gmin', '$abstime').

    Returns:
        List of simparam values in the correct order for the eval function.

    Example:
        >>> eval_fn, meta = translator.translate_eval(params=..., temperature=300.0)
        >>> simparams = build_simparams(meta, {'gmin': 1e-12, '$analysis_type': 0})
        >>> result = eval_fn(shared, varying, cache, jnp.array(simparams))
    """
    simparams_used = metadata.get('simparams_used', ['$analysis_type'])
    simparam_count = metadata.get('simparam_count', 1)

    # Merge user values with defaults
    effective_values = dict(SIMPARAM_DEFAULTS)
    if values:
        effective_values.update(values)

    # Build array in index order
    simparams = []
    for name in simparams_used:
        if name in effective_values:
            simparams.append(float(effective_values[name]))
        else:
            # Unknown simparam - use 0.0 (shouldn't happen if metadata is correct)
            logger.warning(f"Unknown simparam '{name}' - using 0.0")
            simparams.append(0.0)

    return simparams


# Core simulation API
from jax_spice.analysis import CircuitEngine, TransientResult

# Profiling utilities (lazy import to avoid loading unless needed)
from jax_spice.profiling import (
    ProfileConfig,
    ProfileTimer,
    disable_profiling,
    enable_profiling,
    profile,
    profile_section,
)

__all__ = [
    # Core API
    "CircuitEngine",
    "TransientResult",
    # Precision configuration
    "configure_precision",
    "get_precision_info",
    # Simparam helpers
    "build_simparams",
    "SIMPARAM_DEFAULTS",
    # Profiling
    "profile",
    "profile_section",
    "enable_profiling",
    "disable_profiling",
    "ProfileConfig",
    "ProfileTimer",
]
