"""Pytest configuration for JAX-SPICE tests

Handles platform-specific JAX configuration:
- macOS: Forces CPU backend since Metal doesn't support triangular_solve
- Linux with CUDA: Preloads CUDA libraries to help JAX discover them

Enables jaxtyping runtime checks with beartype for array shape validation.

Uses pytest_configure hook to ensure CUDA setup happens before any test imports.
"""

import os
import sys


def _setup_cuda_libraries():
    """Preload CUDA libraries before JAX import for proper GPU detection."""
    import ctypes

    cuda_libs = [
        "libcuda.so.1",
        "libcudart.so.12",
        "libnvrtc.so.12",
        "libnvJitLink.so.12",
        "libcusparse.so.12",
        "libcublas.so.12",
        "libcusolver.so.11",
    ]

    for lib in cuda_libs:
        try:
            ctypes.CDLL(lib)
        except OSError:
            pass  # Some libraries may not be available


def _setup_jaxtyping():
    """Enable jaxtyping runtime checking with beartype."""
    try:
        from jaxtyping import install_import_hook
        # Enable runtime type checking for jax_spice modules
        install_import_hook("jax_spice", "beartype.beartype")
    except ImportError:
        pass  # beartype not installed, skip runtime checking


def pytest_configure(config):
    """
    Pytest hook that runs before test collection.

    This ensures CUDA libraries are preloaded and JAX is configured
    BEFORE any test modules are imported.
    """
    # Platform-specific configuration BEFORE importing JAX
    if sys.platform == 'darwin':
        # macOS: Force CPU backend - Metal doesn't support triangular_solve
        os.environ['JAX_PLATFORMS'] = 'cpu'
    elif sys.platform == 'linux' and os.environ.get('JAX_PLATFORMS', '').startswith('cuda'):
        # Linux with CUDA: Preload CUDA libraries before JAX import
        _setup_cuda_libraries()

    # Enable jaxtyping runtime checking (must be before jax_spice imports)
    _setup_jaxtyping()

    # Import JAX and configure it
    import jax

    # Enable float64 for numerical precision in tests
    jax.config.update('jax_enable_x64', True)
