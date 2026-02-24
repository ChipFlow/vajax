"""VA-JAX: GPU-Accelerated Analog Circuit Simulator"""

import logging
import os
from pathlib import Path

import jax

__version__ = "0.1.0"


logger = logging.getLogger("vajax")


def _get_xdg_cache_dir() -> Path:
    """Get XDG cache directory, following XDG Base Directory Specification.

    Returns:
        Path to cache directory ($XDG_CACHE_HOME or ~/.cache)
    """
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache:
        return Path(xdg_cache)
    return Path.home() / ".cache"


def configure_xla_cache(cache_dir: Path | str | None = None) -> Path | None:
    """Configure JAX XLA compilation cache directory.

    This enables caching of compiled XLA programs across Python sessions,
    providing significant speedup for repeated simulations with the same
    circuit/model configurations.

    Args:
        cache_dir: Cache directory path. If None (default), uses
            $XDG_CACHE_HOME/vajax/xla or ~/.cache/vajax/xla.
            Set to empty string "" to disable caching.

    Returns:
        Path to the cache directory, or None if caching is disabled.

    Note:
        This must be called before any JAX compilation occurs.
        It's called automatically on import with default settings.
    """
    # Check if user has already set the cache dir via environment
    if "JAX_COMPILATION_CACHE_DIR" in os.environ:
        existing = os.environ["JAX_COMPILATION_CACHE_DIR"]
        if existing:
            logger.debug(f"Using existing JAX_COMPILATION_CACHE_DIR: {existing}")
            # Still configure JAX to use the cache
            jax.config.update("jax_compilation_cache_dir", existing)
            jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
            return Path(existing)
        return None

    # Disable caching if explicitly requested
    if cache_dir == "":
        logger.debug("XLA compilation caching disabled")
        return None

    # Use provided path or default XDG location
    if cache_dir is None:
        cache_path = _get_xdg_cache_dir() / "vajax" / "xla"
    else:
        cache_path = Path(cache_dir)

    # Create directory if it doesn't exist
    cache_path.mkdir(parents=True, exist_ok=True)

    # Configure JAX compilation cache via jax.config (required for persistence)
    # Environment variable alone is not sufficient
    jax.config.update("jax_compilation_cache_dir", str(cache_path))
    # Cache all compilations, not just those taking >1 second
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

    # Also set environment variable for compatibility
    os.environ["JAX_COMPILATION_CACHE_DIR"] = str(cache_path)
    logger.debug(f"XLA compilation cache: {cache_path}")

    return cache_path


# Note: XLA cache is configured lazily in CircuitEngine.__init__
# to avoid side effects on import. Call configure_xla_cache() explicitly
# if you need caching without using CircuitEngine.


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


def get_float_dtype():
    """Get the appropriate float dtype based on x64 configuration.

    Returns:
        jnp.float64 if x64 is enabled, jnp.float32 otherwise.

    Example:
        >>> import vajax
        >>> dtype = vajax.get_float_dtype()
        >>> V = jnp.zeros(n_nodes, dtype=dtype)
    """
    import jax.numpy as jnp

    return jnp.float64 if jax.config.jax_enable_x64 else jnp.float32


# =============================================================================
# Simparam helpers (VAMS-LRM Table 9-27)
# =============================================================================

# Default simparam values per VAMS-LRM and VACASK extensions
SIMPARAM_DEFAULTS = {
    "$analysis_type": 0.0,  # 0=DC, 1=AC, 2=transient, 3=noise
    "gmin": 1e-12,  # Minimum conductance (S)
    "abstol": 1e-12,  # Absolute current tolerance (A)
    "vntol": 1e-6,  # Absolute voltage tolerance (V)
    "reltol": 1e-3,  # Relative tolerance
    "tnom": 27.0,  # Nominal temperature (°C, VACASK convention)
    "scale": 1.0,  # Scale factor
    "shrink": 0.0,  # Shrink factor
    "imax": 1.0,  # Branch current limit (A)
    "$abstime": 0.0,  # Absolute simulation time (s)
    "$mfactor": 1.0,  # Device multiplicity factor
    # VACASK extensions for limiting and iteration control
    "iniLim": 0.0,  # 1=initialize limiting (first NR iter), 0=normal eval
    "iteration": 1.0,  # Current NR iteration number (1-based)
    "gdev": 0.0,  # Extra conductance during homotopy (S)
    "sourceScaleFactor": 1.0,  # Source scaling during homotopy
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
    simparams_used = metadata.get("simparams_used", ["$analysis_type"])
    metadata.get("simparam_count", 1)

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
from vajax.analysis import CircuitEngine, TransientResult, warmup_models

# Profiling utilities (lazy import to avoid loading unless needed)
from vajax.profiling import (
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
    "warmup_models",
    # Precision configuration
    "configure_precision",
    "get_precision_info",
    # XLA cache configuration
    "configure_xla_cache",
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
