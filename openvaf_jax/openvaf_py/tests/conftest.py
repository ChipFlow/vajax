"""Shared fixtures and utilities for OpenVAF JAX tests"""

import sys
from pathlib import Path

# Add project root to path to find openvaf_jax module
# Path: tests/conftest.py -> openvaf_py -> openvaf_jax -> jax-spice (root)
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Configure JAX float64 precision directly (avoid full jax_spice import)
import jax

jax.config.update("jax_enable_x64", True)

from typing import Dict, List, Tuple

import numpy as np
import openvaf_py
import pytest

import openvaf_jax

# Path to OpenVAF integration tests (in vendor submodule at project root)
# Path: tests/conftest.py -> openvaf_py -> openvaf_jax -> jax-spice (root) -> vendor
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
INTEGRATION_PATH = PROJECT_ROOT / "vendor" / "OpenVAF" / "integration_tests"

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
    """Wrapper for a compiled Verilog-A model with JAX functions (init + eval)"""

    def __init__(self, module, translator):
        import jax
        import jax.numpy as jnp

        self.module = module
        self.translator = translator

        # Pre-computed default simparams: [analysis_type=DC, mfactor=1.0, gmin=1e-12]
        self._default_simparams = jnp.array([0.0, 1.0, 1e-12])

        # Pre-compile with default params for fast evaluation
        self._default_init_fn, self._default_init_meta = translator.translate_init(
            params={}, temperature=300.15
        )
        self._default_eval_fn, self._default_eval_meta = translator.translate_eval(
            params={}, temperature=300.15, propagate_constants=False
        )
        self._default_init_fn = jax.jit(self._default_init_fn)
        self._default_eval_fn = jax.jit(self._default_eval_fn)

        # Build mapping from user param index to init param index
        # This allows runtime params to be passed to init function
        init_param_names = list(module.init_param_names)
        user_param_names = list(module.param_names)
        user_indices = []  # source indices in user's input array
        init_indices = []  # destination indices in init input array
        for init_idx, init_name in enumerate(init_param_names):
            if init_name in user_param_names:
                user_idx = user_param_names.index(init_name)
                user_indices.append(user_idx)
                init_indices.append(init_idx)
        # Store as JAX arrays for vectorized scatter
        self._user_indices = jnp.array(user_indices, dtype=jnp.int32) if user_indices else None
        self._init_indices = jnp.array(init_indices, dtype=jnp.int32) if init_indices else None
        self._n_init_params = len(init_param_names)

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
        """Build input array with defaults from the parsed Verilog-A model.

        Returns list of floats indexed by module.param_names order.
        Uses the actual defaults computed by the translator from the VA source.
        """
        # Start with the translator's init_inputs which have correct VA defaults
        init_inputs = list(self._default_init_meta['init_inputs'])

        # Build reverse mapping: init_index -> user_index
        # So we can map init defaults back to user param order
        init_to_user = {}
        if self._user_indices is not None:
            for ui, ii in zip(self._user_indices, self._init_indices):
                init_to_user[int(ii)] = int(ui)

        # Build user inputs array with proper defaults
        inputs = []
        for i, (name, kind) in enumerate(zip(self.param_names, self.param_kinds)):
            if kind == 'voltage':
                inputs.append(0.0)
            elif kind == 'current':
                inputs.append(0.0)
            elif kind == 'temperature':
                inputs.append(300.15)  # Operating temperature in Kelvin
            elif kind == 'hidden_state':
                inputs.append(0.0)
            elif kind == 'sysfun':
                inputs.append(1.0 if 'mfactor' in name.lower() else 0.0)
            elif kind == 'param':
                # Find if this param has a mapping to init_inputs
                found = False
                for ii, ui in init_to_user.items():
                    if ui == i and ii < len(init_inputs):
                        inputs.append(init_inputs[ii])
                        found = True
                        break
                if not found:
                    # Fallback for params not in init
                    inputs.append(1.0)
            else:
                inputs.append(0.0)
        return inputs

    def jax_fn(self, inputs: List[float]) -> Tuple[Dict, Dict]:
        """Evaluate the model using init + eval (legacy dict interface).

        This is a compatibility wrapper that mimics the old translate() API.
        The inputs array follows module.param_names order.
        Returns (residuals_dict, jacobian_dict).

        Uses pre-compiled functions (compiled with propagate_constants=False
        so params are read from input arrays at runtime).
        """
        import jax.numpy as jnp

        # Extract mfactor from inputs for simparams
        mfactor = 1.0
        for i, (name, kind, value) in enumerate(zip(self.param_names, self.param_kinds, inputs)):
            if kind == 'sysfun' and name == 'mfactor':
                mfactor = value
                break

        # Use pre-compiled functions (params read from arrays at runtime)
        init_fn = self._default_init_fn
        eval_fn = self._default_eval_fn
        init_meta = self._default_init_meta
        eval_meta = self._default_eval_meta

        # Build init inputs from user inputs (map to init param positions)
        # init_inputs needs params in init function's expected order
        inputs_arr = jnp.asarray(inputs)
        init_inputs = jnp.array(init_meta['init_inputs'])

        # Apply user's param values to init_inputs using vectorized scatter
        # This ensures cache is computed with correct runtime params (e.g., R for resistor)
        if self._user_indices is not None:
            user_values = inputs_arr[self._user_indices]
            init_inputs = init_inputs.at[self._init_indices].set(user_values)

        # Run init to get cache
        cache, _ = init_fn(init_inputs)

        # Build eval inputs
        shared_indices = eval_meta['shared_indices']
        voltage_indices = eval_meta['voltage_indices']

        shared_params = inputs_arr[jnp.array(shared_indices)] if shared_indices else jnp.array([])
        varying_params = inputs_arr[jnp.array(voltage_indices)] if voltage_indices else jnp.array([])

        # Run eval with dynamic simparams
        simparams = jnp.array([0.0, mfactor, 1e-12])  # [analysis_type, mfactor, gmin]
        # Uniform interface: always pass limit_state_in (zeros when not using limits)
        limit_state_in = jnp.zeros(1)  # Minimal dummy array
        limit_funcs = {}  # Empty dict - limit functions not used
        # Cache split: shared_cache is empty, all cache in device_cache
        shared_cache = jnp.array([])
        device_cache = cache
        result = eval_fn(shared_params, varying_params, shared_cache, device_cache, simparams, limit_state_in, limit_funcs)
        res_resist, res_react, jac_resist, jac_react = result[:4]

        # Convert JAX arrays to NumPy first (single device-to-host transfer)
        # Then build dicts from NumPy - much faster than indexing JAX arrays
        import numpy as np
        res_resist_np = np.asarray(res_resist)
        res_react_np = np.asarray(res_react)
        jac_resist_np = np.asarray(jac_resist)
        jac_react_np = np.asarray(jac_react)

        # Convert to dicts
        node_names = eval_meta['node_names']
        jacobian_keys = eval_meta['jacobian_keys']

        residuals = {
            name: {'resist': res_resist_np[i], 'react': res_react_np[i]}
            for i, name in enumerate(node_names)
        }
        jacobian = {
            key: {'resist': jac_resist_np[i], 'react': jac_react_np[i]}
            for i, key in enumerate(jacobian_keys)
        }
        return residuals, jacobian

    def evaluate(self, inputs: List[float]) -> Tuple[Dict, Dict]:
        """Evaluate the JAX function"""
        return self.jax_fn(inputs)

    def run_interpreter(self, params: Dict[str, float]) -> Tuple[List, List]:
        """Run the MIR interpreter"""
        return self.module.run_init_eval(params)


@pytest.fixture(scope="module")
def compile_model():
    """Factory fixture to compile VA models

    Returns a function that compiles a model and returns a CompiledModel.
    The CompiledModel provides a jax_fn method that translates on-the-fly
    with the specific params passed in each call (for test compatibility).
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
            _cache[cache_key] = CompiledModel(module, translator)
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
