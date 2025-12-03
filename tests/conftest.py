"""Pytest configuration for JAX-SPICE tests

Handles platform-specific JAX configuration:
- macOS: Forces CPU backend since Metal doesn't support triangular_solve
- Linux with CUDA: Preloads CUDA libraries to help JAX discover them
"""

import os
import sys

# Platform-specific configuration BEFORE importing JAX
if sys.platform == 'darwin':
    # macOS: Force CPU backend - Metal doesn't support triangular_solve
    os.environ['JAX_PLATFORMS'] = 'cpu'
elif sys.platform == 'linux' and os.environ.get('JAX_PLATFORMS', '').startswith('cuda'):
    # Linux with CUDA: Preload CUDA libraries before JAX import
    # JAX's library discovery needs libraries to be loaded first
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
            pass  # Library not available, that's OK

import jax

# Enable float64 for numerical precision in tests
jax.config.update('jax_enable_x64', True)
