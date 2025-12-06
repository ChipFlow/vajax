"""Shared fixtures and utilities for OpenVAF JAX tests"""

import sys
from pathlib import Path

# Add parent directory to path to find openvaf_jax module
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
from typing import Tuple, Dict, Any, List, Callable

import openvaf_py
import openvaf_jax


# Path to OpenVAF integration tests (in vendor submodule)
INTEGRATION_PATH = Path(__file__).parent.parent / "vendor" / "OpenVAF" / "integration_tests"

# All integration test models with their paths
INTEGRATION_MODELS = [
    # Simple models
    ("resistor", "RESISTOR/resistor.va"),
    ("diode", "DIODE/diode.va"),
    ("diode_cmc", "DIODE_CMC/diode_cmc.va"),
    ("isrc", "CURRENT_SOURCE/current_source.va"),
    ("vccs", "VCCS/vccs.va"),
    ("cccs", "CCCS/cccs.va"),

    # MOSFETs
    ("ekv", "EKV/ekv.va"),
    ("bsim3", "BSIM3/bsim3.va"),
    ("bsim4", "BSIM4/bsim4.va"),
    ("bsim6", "BSIM6/bsim6.va"),
    ("bsimbulk", "BSIMBULK/bsimbulk.va"),
    ("bsimcmg", "BSIMCMG/bsimcmg.va"),
    ("bsimsoi", "BSIMSOI/bsimsoi.va"),

    # PSP
    ("psp102", "PSP102/psp102.va"),
    ("psp103", "PSP103/psp103.va"),
    ("juncap", "PSP103/juncap200.va"),

    # HiSIM
    ("hisim2", "HiSIM2/hisim2.va"),
    ("hisimhv", "HiSIMHV/hisimhv.va"),

    # BJT
    ("hicum", "HICUML2/hicuml2.va"),
    ("mextram", "MEXTRAM/mextram.va"),

    # HEMT
    ("asmhemt", "ASMHEMT/asmhemt.va"),
    ("mvsg", "MVSG_CMC/mvsg_cmc.va"),
]


# Default tolerances for comparisons
TOLERANCES = {
    'residual': {'rtol': 1e-6, 'atol': 1e-15},
    'jacobian': {'rtol': 1e-5, 'atol': 1e-12},
    'current': {'rtol': 1e-6, 'atol': 1e-15},
    'conductance': {'rtol': 1e-5, 'atol': 1e-12},
}


class CompiledModel:
    """Wrapper for a compiled Verilog-A model with JAX function"""

    def __init__(self, module, translator, jax_fn):
        self.module = module
        self.translator = translator
        self.jax_fn = jax_fn

    @property
    def name(self) -> str:
        return self.module.name

    @property
    def nodes(self) -> List[str]:
        return list(self.module.nodes)

    @property
    def param_names(self) -> List[str]:
        return list(self.module.param_names)

    @property
    def param_kinds(self) -> List[str]:
        return list(self.module.param_kinds)

    def build_default_inputs(self) -> List[float]:
        """Build input array with sensible defaults"""
        inputs = []
        for name, kind in zip(self.param_names, self.param_kinds):
            if kind == 'voltage':
                inputs.append(0.0)
            elif kind == 'param':
                # Use defaults from common parameters
                if 'temperature' in name.lower() or name == '$temperature':
                    inputs.append(300.15)
                elif name.lower() in ('tnom', 'tref'):
                    inputs.append(300.0)
                elif name.lower() == 'mfactor':
                    inputs.append(1.0)
                elif name.lower() == 'r':
                    inputs.append(1000.0)
                else:
                    inputs.append(1.0)
            elif kind == 'hidden_state':
                inputs.append(0.0)
            else:
                inputs.append(0.0)
        return inputs

    def evaluate(self, inputs: List[float]) -> Tuple[Dict, Dict]:
        """Evaluate the JAX function"""
        return self.jax_fn(inputs)

    def run_interpreter(self, params: Dict[str, float]) -> Tuple[List, List]:
        """Run the MIR interpreter"""
        return self.module.run_init_eval(params)


@pytest.fixture(scope="module")
def compile_model():
    """Factory fixture to compile VA models

    Returns a function that compiles a model and returns (module, translator, jax_fn)
    """
    _cache = {}

    def _compile(model_path, allow_analog_in_cond=False) -> CompiledModel:
        cache_key = (str(model_path), allow_analog_in_cond)
        if cache_key not in _cache:
            modules = openvaf_py.compile_va(str(model_path), allow_analog_in_cond)
            if not modules:
                raise ValueError(f"No modules found in {model_path}")
            module = modules[0]
            translator = openvaf_jax.OpenVAFToJAX(module)
            jax_fn = translator.translate()
            _cache[cache_key] = CompiledModel(module, translator, jax_fn)
        return _cache[cache_key]

    return _compile


@pytest.fixture(scope="module")
def resistor_model(compile_model) -> CompiledModel:
    """Compiled resistor model"""
    return compile_model(INTEGRATION_PATH / "RESISTOR/resistor.va")


@pytest.fixture(scope="module")
def diode_model(compile_model) -> CompiledModel:
    """Compiled diode model"""
    return compile_model(INTEGRATION_PATH / "DIODE/diode.va")


@pytest.fixture(scope="module")
def diode_cmc_model(compile_model) -> CompiledModel:
    """Compiled CMC diode model"""
    return compile_model(INTEGRATION_PATH / "DIODE_CMC/diode_cmc.va")


@pytest.fixture(scope="module")
def isrc_model(compile_model) -> CompiledModel:
    """Compiled current source model"""
    return compile_model(INTEGRATION_PATH / "CURRENT_SOURCE/current_source.va")


@pytest.fixture(scope="module")
def vccs_model(compile_model) -> CompiledModel:
    """Compiled VCCS model"""
    return compile_model(INTEGRATION_PATH / "VCCS/vccs.va")


@pytest.fixture(scope="module")
def cccs_model(compile_model) -> CompiledModel:
    """Compiled CCCS model"""
    return compile_model(INTEGRATION_PATH / "CCCS/cccs.va")


def assert_allclose(actual, expected, rtol=1e-6, atol=1e-12, err_msg=""):
    """Assert that actual and expected are close within tolerances

    Like numpy.testing.assert_allclose but with better error messages
    """
    actual = float(actual)
    expected = float(expected)

    # Handle special cases
    if np.isnan(actual) or np.isnan(expected):
        if np.isnan(actual) and np.isnan(expected):
            return  # Both NaN is OK
        raise AssertionError(f"{err_msg}: actual={actual}, expected={expected} (NaN mismatch)")

    if np.isinf(actual) or np.isinf(expected):
        if actual == expected:
            return  # Same infinity is OK
        raise AssertionError(f"{err_msg}: actual={actual}, expected={expected} (inf mismatch)")

    # Check relative and absolute tolerance
    diff = abs(actual - expected)
    if expected != 0:
        rel_diff = diff / abs(expected)
    else:
        rel_diff = diff  # Relative diff not meaningful for zero

    if diff <= atol or rel_diff <= rtol:
        return

    raise AssertionError(
        f"{err_msg}: actual={actual:.9g}, expected={expected:.9g}, "
        f"diff={diff:.9g}, rel_diff={rel_diff:.9g}"
    )


def build_param_dict(model: CompiledModel, inputs: List[float]) -> Dict[str, float]:
    """Build parameter dictionary from input list

    Maps inputs to their parameter names for the interpreter
    """
    params = {}
    for i, (name, value) in enumerate(zip(model.param_names, inputs)):
        params[name] = value
    return params
