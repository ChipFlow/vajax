"""PDK test configuration

This module handles proprietary PDK paths via environment variables,
and ensures PDK paths don't leak into CI logs.
"""

import sys
from pathlib import Path

# Add this directory to path to find pdk_utils
sys.path.insert(0, str(Path(__file__).parent))

import pytest
from pdk_utils import PDK_PATHS, compile_pdk_model


def pytest_configure(config):
    """Register custom markers"""
    config.addinivalue_line(
        "markers", "requires_pdk(name): skip test if PDK not available"
    )


def pytest_collection_modifyitems(config, items):
    """Skip PDK tests if PDK not available"""
    for item in items:
        for marker in item.iter_markers("requires_pdk"):
            pdk_name = marker.args[0]
            if PDK_PATHS.get(pdk_name) is None:
                env_var = f"PDK_{pdk_name.upper()}_PATH"
                item.add_marker(pytest.mark.skip(
                    reason=f"PDK '{pdk_name}' not available (set {env_var})"
                ))


@pytest.fixture(scope="module")
def compile_pdk_model_fixture():
    """Factory fixture to compile PDK VA models

    Returns a function that compiles a model with PDK path sanitization.
    Uses caching to avoid recompiling the same model.
    """
    _cache = {}

    def _compile(model_path: Path, allow_analog_in_cond: bool = True):
        cache_key = (str(model_path), allow_analog_in_cond)
        if cache_key not in _cache:
            _cache[cache_key] = compile_pdk_model(model_path, allow_analog_in_cond)
        return _cache[cache_key]

    return _compile
