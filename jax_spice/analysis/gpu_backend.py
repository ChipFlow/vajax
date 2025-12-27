"""GPU backend selection and configuration for JAX-SPICE

Provides automatic backend selection based on circuit size,
GPU availability detection, and device management.
"""

from dataclasses import dataclass
from typing import Optional
import jax


@dataclass
class BackendConfig:
    """Configuration for simulation backend selection.

    Attributes:
        gpu_threshold: Minimum number of nodes to use GPU (default 500)
        force_backend: Force specific backend ('cpu', 'gpu', or None for auto)
    """

    gpu_threshold: int = 500
    force_backend: Optional[str] = None


# Default configuration
_default_config = BackendConfig()


def is_gpu_available() -> bool:
    """Check if a GPU backend is available.

    Returns:
        True if CUDA or other GPU devices are available, False otherwise.
    """
    try:
        devices = jax.devices()
        return any(d.platform != "cpu" for d in devices)
    except Exception:
        return False


def get_gpu_devices() -> list:
    """Get list of available GPU devices.

    Returns:
        List of non-CPU JAX devices.
    """
    try:
        return [d for d in jax.devices() if d.platform != "cpu"]
    except Exception:
        return []


def select_backend(
    num_nodes: int, config: Optional[BackendConfig] = None
) -> str:
    """Select optimal backend based on circuit size and availability.

    Args:
        num_nodes: Number of nodes in the circuit
        config: Backend configuration (uses default if None)

    Returns:
        'gpu' or 'cpu' string indicating selected backend
    """
    if config is None:
        config = _default_config

    # Honor forced backend if specified
    if config.force_backend is not None:
        if config.force_backend == "gpu" and not is_gpu_available():
            # Silently fall back to CPU if GPU forced but not available
            return "cpu"
        return config.force_backend

    # Auto-select based on circuit size and GPU availability
    if num_nodes >= config.gpu_threshold and is_gpu_available():
        return "gpu"

    return "cpu"


def get_device(backend: str) -> jax.Device:
    """Get JAX device for the selected backend.

    Args:
        backend: 'gpu' or 'cpu'

    Returns:
        JAX device object for the selected backend

    Raises:
        RuntimeError: If GPU requested but not available
    """
    if backend == "gpu":
        gpu_devices = get_gpu_devices()
        if not gpu_devices:
            raise RuntimeError("GPU backend requested but no GPU available")
        return gpu_devices[0]
    else:
        # Explicitly request CPU device - this works even when GPU is default
        cpu_devices = jax.devices("cpu")
        if cpu_devices:
            return cpu_devices[0]
        # This shouldn't happen - CPU should always be available
        raise RuntimeError("CPU backend requested but no CPU device available")


def is_metal_backend() -> bool:
    """Check if the current backend is Metal (Apple GPU).

    Returns:
        True if using jax-metal or iree-metal backend.
    """
    backend = jax.default_backend().lower()
    return backend in ("metal", "iree_metal")


def get_default_dtype(backend: str = None):
    """Get default dtype for the selected backend.

    Args:
        backend: 'gpu' or 'cpu' (optional, auto-detects if None)

    Returns:
        jax.numpy dtype (float64 for CPU/CUDA, float32 for Metal)
    """
    import jax.numpy as jnp

    # Metal backends (jax-metal and iree-metal) only support float32
    if is_metal_backend():
        return jnp.float32

    return jnp.float64


# Module-level dtype that gets set at import time
# This allows code to use `from jax_spice.analysis.gpu_backend import default_dtype`
default_dtype = None


def _init_default_dtype():
    """Initialize the default dtype based on current backend."""
    global default_dtype
    import jax.numpy as jnp
    default_dtype = get_default_dtype()
    return default_dtype


def backend_info() -> dict:
    """Get information about available backends.

    Returns:
        Dict with backend availability and device info
    """
    devices = jax.devices()

    return {
        "default_backend": jax.default_backend(),
        "gpu_available": is_gpu_available(),
        "gpu_devices": [
            {"name": d.device_kind, "platform": d.platform}
            for d in devices
            if d.platform != "cpu"
        ],
        "cpu_devices": [
            {"name": d.device_kind, "platform": d.platform}
            for d in devices
            if d.platform == "cpu"
        ],
        "default_threshold": _default_config.gpu_threshold,
    }
