"""PDK utilities for test configuration and path handling

This module provides utilities for working with proprietary PDKs,
including path handling and error message sanitization.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Optional

# Add paths to find openvaf modules
# tests_pdk/ -> openvaf_py/ -> openvaf_jax/ -> jax-spice/
REPO_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import openvaf_py

import openvaf_jax


def get_pdk_path(env_var: str) -> Optional[Path]:
    """Get PDK path from environment variable

    Returns None if env var is not set or path doesn't exist.
    """
    env_path = os.environ.get(env_var, "")
    if env_path:
        path = Path(env_path)
        if path.exists():
            return path
    return None


# PDK paths from environment variables
PDK_PATHS: Dict[str, Optional[Path]] = {
    "gf130": get_pdk_path("PDK_GF130_PATH"),
    "ihp": get_pdk_path("PDK_IHP_PATH"),
    "skywater": get_pdk_path("PDK_SKYWATER_PATH"),
}


def sanitize_pdk_path(msg: str) -> str:
    """Remove PDK paths from error messages to avoid leaking in CI logs"""
    for key, path in PDK_PATHS.items():
        if path:
            msg = msg.replace(str(path), f"<{key.upper()}_PDK>")
    return msg


class CompiledPDKModel:
    """Wrapper for a compiled PDK Verilog-A model with JAX function"""

    def __init__(self, module, translator, jax_fn, model_name: str):
        self.module = module
        self.translator = translator
        self.jax_fn = jax_fn
        self._display_name = model_name  # Use sanitized name for display

    @property
    def name(self) -> str:
        return self._display_name

    @property
    def nodes(self):
        return list(self.module.nodes)

    @property
    def param_names(self):
        return list(self.module.param_names)

    @property
    def param_kinds(self):
        return list(self.module.param_kinds)

    def build_default_inputs(self):
        """Build input array with sensible defaults"""
        inputs = []
        for name, kind in zip(self.param_names, self.param_kinds):
            if kind == 'voltage':
                inputs.append(0.0)
            elif kind == 'param':
                if 'temperature' in name.lower() or name == '$temperature':
                    inputs.append(300.15)
                elif name.lower() in ('tnom', 'tref'):
                    inputs.append(300.0)
                elif name.lower() == 'mfactor':
                    inputs.append(1.0)
                else:
                    inputs.append(1.0)
            elif kind == 'hidden_state':
                inputs.append(0.0)
            else:
                inputs.append(0.0)
        return inputs


def compile_pdk_model(model_path: Path, allow_analog_in_cond: bool = True) -> CompiledPDKModel:
    """Compile a PDK VA model with path sanitization"""
    try:
        modules = openvaf_py.compile_va(str(model_path), allow_analog_in_cond)
        if not modules:
            raise ValueError(f"No modules found in {sanitize_pdk_path(str(model_path))}")
        module = modules[0]
        translator = openvaf_jax.OpenVAFToJAX(module)
        jax_fn = translator.translate()
        # Use only the filename for display, not full path
        display_name = model_path.stem
        return CompiledPDKModel(module, translator, jax_fn, display_name)
    except Exception as e:
        # Sanitize any error message before raising
        sanitized_msg = sanitize_pdk_path(str(e))
        raise type(e)(sanitized_msg) from None
