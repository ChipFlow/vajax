"""Pytest configuration for JAX-SPICE tests

Handles platform-specific JAX configuration:
- macOS: Forces CPU backend since Metal doesn't support triangular_solve
- Linux with CUDA: Preloads CUDA libraries to help JAX discover them

Enables jaxtyping runtime checks with beartype for array shape validation.

Uses pytest_configure hook to ensure CUDA setup happens before any test imports.

Also provides shared test utilities:
- parse_embedded_python: Extract expected values from VACASK test files
- parse_si_value: Parse SPICE values with SI suffixes
"""

import os
import re
import sys
from pathlib import Path
from typing import Any, Dict


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

    # Import jax_spice to auto-configure precision based on backend
    # This will detect Metal/TPU and disable x64 if needed
    import jax_spice  # noqa: F401


# =============================================================================
# Shared Test Utilities
# =============================================================================


def parse_si_value(s: str) -> float:
    """Parse a SPICE value with SI suffix.

    Handles standard SI prefixes like:
    - f (femto, 1e-15), p (pico, 1e-12), n (nano, 1e-9)
    - u (micro, 1e-6), m (milli, 1e-3)
    - k (kilo, 1e3), meg (mega, 1e6), g (giga, 1e9), t (tera, 1e12)

    Args:
        s: String value like "2k", "100n", "1meg"

    Returns:
        Float value with SI scaling applied
    """
    s = s.strip().lower()
    suffixes = {
        'f': 1e-15, 'p': 1e-12, 'n': 1e-9, 'u': 1e-6, 'm': 1e-3,
        'k': 1e3, 'meg': 1e6, 'g': 1e9, 't': 1e12
    }
    # Check longer suffixes first (meg before m)
    for suffix, mult in sorted(suffixes.items(), key=lambda x: -len(x[0])):
        if s.endswith(suffix):
            return float(s[:-len(suffix)]) * mult
    return float(s)


def parse_embedded_python(sim_path: Path) -> Dict[str, Any]:
    """Extract expected values from embedded Python test script in VACASK .sim files.

    Parses patterns like:
        v = op1["2"]
        exact = 10*0.9

    Args:
        sim_path: Path to VACASK .sim file

    Returns:
        Dict with:
        - 'expectations': List of (variable_name, expected_value, tolerance)
        - 'analysis_type': 'op' or 'tran'
    """
    import numpy as np

    text = sim_path.read_text()

    # Find embedded Python between <<<FILE and >>>FILE
    match = re.search(r'<<<FILE\n(.*?)>>>FILE', text, re.DOTALL)
    if not match:
        return {'expectations': [], 'analysis_type': 'op'}

    py_code = match.group(1)
    lines = py_code.split('\n')

    expectations = []
    current_var = None

    for line in lines:
        # Match: v = op1["node_name"] or i = op1["device.i"]
        m = re.match(r'\s*(\w+)\s*=\s*op1\["([^"]+)"\]', line)
        if m:
            current_var = m.group(2)
            continue

        # Match: exact = <expression>
        m = re.match(r'\s*exact\s*=\s*(.+)', line)
        if m and current_var:
            try:
                # Safe evaluation of numeric expressions
                expr = m.group(1).strip()
                val = eval(expr, {"__builtins__": {}, "np": np}, {})
                expectations.append((current_var, float(val), 1e-3))
            except Exception:
                pass
            current_var = None

    # Determine analysis type
    analysis_type = 'op'
    if 'tran1' in py_code or 'rawread(\'tran' in py_code:
        analysis_type = 'tran'

    return {
        'expectations': expectations,
        'analysis_type': analysis_type
    }
