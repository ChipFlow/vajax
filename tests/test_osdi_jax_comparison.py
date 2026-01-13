"""OSDI vs openvaf_jax Comparison Tests

This module compares the behavior of openvaf_jax generated models against
OSDI compiled models by sweeping voltages and comparing currents/residuals/Jacobians.

Phase 1: Simple Components (resistor, capacitor, diode)
Phase 2: Complex Transistor Models (PSP103, BSIM4)

The goal is to validate that our JAX translator produces results that match
the reference OSDI implementation.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

# Add openvaf_jax and openvaf_py to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "openvaf_jax" / "openvaf_py"))

from jax_spice import configure_precision, build_simparams, SIMPARAM_DEFAULTS
from jax_spice.debug.jacobian import compare_jacobians
import osdi_py  # noqa: E402 - MUST be imported before openvaf_py/openvaf_jax
import openvaf_py  # noqa: E402
import openvaf_jax  # noqa: E402


# =============================================================================
# VAMS-LRM Tolerances (Section 3.6.1.2)
# =============================================================================
#
# Per VAMS-LRM-2023, each nature defines `abstol` - the maximum negligible value
# for signals associated with the nature. These are used by Newton-Raphson
# convergence checks and signal comparison.
#
# | Nature  | abstol      | Description                    |
# |---------|-------------|--------------------------------|
# | Current | 1e-12 A     | Maximum negligible current     |
# | Voltage | 1e-6 V      | Maximum negligible voltage     |
# | Charge  | 1e-14 C     | Maximum negligible charge      |
# | Flux    | 1e-14 Vs    | Maximum negligible flux        |
#
# Residuals in MNA formulation are currents (KCL equations), so we use
# CURRENT_ABSTOL for residual comparison.

CURRENT_ABSTOL = 1e-12   # 1 pA - from VAMS-LRM Current nature
CURRENT_RTOL = 1e-4      # 0.01% relative tolerance

VOLTAGE_ABSTOL = 1e-6    # 1 µV - from VAMS-LRM Voltage nature
VOLTAGE_RTOL = 1e-4      # 0.01% relative tolerance

# For conductance (Jacobian entries dI/dV), use current-like abstol
CONDUCTANCE_ABSTOL = 1e-12  # S (Siemens)
CONDUCTANCE_RTOL = 1e-4


def compare_with_vams_tolerance(
    osdi_val: float,
    jax_val: float,
    abstol: float,
    rtol: float,
) -> bool:
    """Compare values using VAMS-LRM combined tolerance.

    The combined tolerance formula is: |osdi - jax| < abstol + rtol * max(|osdi|, |jax|)

    This ensures:
    - Small signals (near abstol) are compared with absolute tolerance
    - Large signals are compared with relative tolerance
    - Prevents false positives where 4.79e-11 "passes" against 5e-16

    Args:
        osdi_val: Reference value from OSDI
        jax_val: Value from JAX implementation
        abstol: Absolute tolerance (from VAMS-LRM nature)
        rtol: Relative tolerance

    Returns:
        True if values match within tolerance
    """
    diff = abs(osdi_val - jax_val)
    threshold = abstol + rtol * max(abs(osdi_val), abs(jax_val))
    return diff < threshold


def compare_residuals(
    osdi_residuals: list,
    jax_residuals: list,
    rtol: float = CURRENT_RTOL,
    atol: float = CURRENT_ABSTOL,
) -> tuple:
    """Compare residual arrays using VAMS-LRM current tolerances.

    Residuals in MNA are currents (KCL sum at each node), so we use
    CURRENT_ABSTOL as the maximum negligible value.

    Args:
        osdi_residuals: Reference residuals from OSDI
        jax_residuals: Residuals from JAX implementation
        rtol: Relative tolerance (default: CURRENT_RTOL = 1e-4)
        atol: Absolute tolerance (default: CURRENT_ABSTOL = 1e-12)

    Returns:
        Tuple of (passed, max_abs_diff, max_rel_diff, failed_indices)
    """
    osdi = np.array(osdi_residuals)
    jax = np.array(jax_residuals)

    if osdi.shape != jax.shape:
        return False, float('inf'), float('inf'), []

    failed_indices = []
    max_abs_diff = 0.0
    max_rel_diff = 0.0

    for i, (o, j) in enumerate(zip(osdi, jax)):
        diff = abs(o - j)
        max_abs_diff = max(max_abs_diff, diff)

        # Compute relative diff for reporting
        if abs(o) > atol:
            rel = diff / abs(o)
            max_rel_diff = max(max_rel_diff, rel)

        # Check combined tolerance
        if not compare_with_vams_tolerance(o, j, atol, rtol):
            failed_indices.append(i)

    passed = len(failed_indices) == 0
    return passed, max_abs_diff, max_rel_diff, failed_indices


# Paths
VACASK_DEVICES = project_root / "vendor" / "VACASK" / "devices"
OPENVAF_INTEGRATION = project_root / "vendor" / "OpenVAF" / "integration_tests"
OSDI_CACHE = Path("/tmp/osdi_jax_test_cache")


# =============================================================================
# Helper Functions
# =============================================================================


def find_openvaf_compiler() -> Path | None:
    """Find the OpenVAF compiler binary.

    Returns None if not found (tests will be skipped).
    """
    repo_root = project_root

    # Try workspace target directory first (when built from repo root)
    openvaf = repo_root / "target" / "release" / "openvaf-r"
    if openvaf.exists():
        return openvaf

    openvaf = repo_root / "target" / "debug" / "openvaf-r"
    if openvaf.exists():
        return openvaf

    # Try vendor/OpenVAF target directory (when built from submodule)
    openvaf = repo_root / "vendor" / "OpenVAF" / "target" / "release" / "openvaf-r"
    if openvaf.exists():
        return openvaf

    openvaf = repo_root / "vendor" / "OpenVAF" / "target" / "debug" / "openvaf-r"
    if openvaf.exists():
        return openvaf

    return None


# Check if OpenVAF compiler is available
OPENVAF_COMPILER = find_openvaf_compiler()
OPENVAF_AVAILABLE = OPENVAF_COMPILER is not None


def compile_va_to_osdi(va_path: Path, osdi_path: Path, force: bool = False) -> None:
    """Compile Verilog-A to OSDI using OpenVAF.

    For security, we always recompile from the vendored .va source to ensure
    we're not using potentially tampered .osdi files.

    Args:
        va_path: Path to source Verilog-A file
        osdi_path: Path for output OSDI file
        force: If True, always recompile even if osdi_path exists
    """
    # Security: always recompile from source unless explicitly cached
    # This ensures we're using our vendored .va files, not potentially
    # tampered .osdi files
    if not force and osdi_path.exists():
        # Check if .va is newer than .osdi
        if va_path.stat().st_mtime > osdi_path.stat().st_mtime:
            force = True

    if osdi_path.exists() and not force:
        return

    if OPENVAF_COMPILER is None:
        raise RuntimeError("OpenVAF compiler not available")

    osdi_path.parent.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        [str(OPENVAF_COMPILER), str(va_path), "-o", str(osdi_path)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"OpenVAF compilation failed:\n{result.stderr}")


def get_full_param_info(va_path: Path) -> dict:
    """Get combined parameter info from openvaf_py.

    Combines OSDI descriptor (name, units, description, flags) with
    param defaults to provide complete parameter information.

    Args:
        va_path: Path to Verilog-A file

    Returns:
        Dict with:
            - params: List of param dicts with name, type, default, units, description
            - terminals: Number of terminals
            - nodes: Total node info
    """
    modules = openvaf_py.compile_va(str(va_path))
    module = modules[0]

    osdi_desc = module.get_osdi_descriptor()
    defaults = module.get_param_defaults()

    # Build lowercase lookup for defaults (they use lowercase keys)
    defaults_lower = {k.lower(): v for k, v in defaults.items()}

    params = []
    for p in osdi_desc['params']:
        name = p['name']
        flags = p['flags']

        # Decode type from flags (lower 2 bits)
        type_code = flags & 3
        param_type = {0: 'real', 1: 'int', 2: 'str'}.get(type_code, 'real')

        # Get default value (case-insensitive lookup)
        default = defaults_lower.get(name.lower())

        params.append({
            'name': name,
            'type': param_type,
            'default': default,
            'units': p.get('units', ''),
            'description': p.get('description', ''),
            'is_instance': p.get('is_instance', False),
            'aliases': p.get('aliases', []),
        })

    return {
        'params': params,
        'terminals': osdi_desc.get('num_terminals', 0),
        'nodes': osdi_desc.get('nodes', []),
        'num_nodes': osdi_desc.get('num_nodes', 0),
    }


# =============================================================================
# Model Configurations for Parameterized Tests
# =============================================================================

# Each config defines a model to test with minimal required parameters
MODEL_CONFIGS = [
    {
        "name": "resistor",
        "va_path": VACASK_DEVICES / "resistor.va",
        "params": {"r": 1000.0, "mfactor": 1.0},
        "test_voltages": [[1.0, 0.0], [0.5, -0.5]],
        "skip_jacobian": False,
    },
    {
        "name": "capacitor",
        "va_path": VACASK_DEVICES / "capacitor.va",
        "params": {"c": 1e-12, "mfactor": 1.0},
        "test_voltages": [[1.0, 0.0]],
        "skip_jacobian": False,
    },
    {
        "name": "diode",
        "va_path": VACASK_DEVICES / "diode.va",
        "params": {"Is": 1e-14, "N": 1.0, "Rs": 10.0, "Tnom": 27.0, "mfactor": 1.0},
        "test_voltages": [[0.7, 0.0], [0.0, 0.0]],
        "skip_jacobian": False,
    },
]

# Transistor model configurations for parameterized tests
# Each config specifies how to build voltage vectors and sweep parameters
TRANSISTOR_CONFIGS = [
    {
        "name": "ekv",
        "va_path": OPENVAF_INTEGRATION / "EKV" / "ekv.va",
        "params": {
            "TYPE": 1,       # NMOS
            "W": 1e-6,
            "L": 100e-9,
            "TEMP": 1e21,    # Sentinel: use $temperature
            "TNOM": 1e21,    # Sentinel: use default
            "mfactor": 1.0,
        },
        "terminals": ["d", "g", "s", "b"],  # voltage order
        "vds": 0.5,
        "vgs_for_jacobian": 0.6,
        "xfail": False,
    },
    {
        "name": "mvsg",
        "va_path": OPENVAF_INTEGRATION / "MVSG_CMC" / "mvsg_cmc.va",
        "params": {
            "lg": 100e-9,
            "wg": 1e-6,
            "mfactor": 1.0,
        },
        "terminals": ["d", "g", "s"],  # 3-terminal device
        "vds": 0.5,
        "vgs_for_jacobian": 0.6,
        "xfail": False,
    },
    {
        "name": "psp102",
        "va_path": OPENVAF_INTEGRATION / "PSP102" / "psp102.va",
        "params": {
            "TYPE": 1,       # NMOS
            "W": 1e-6,
            "L": 100e-9,
            "mfactor": 1.0,
        },
        "terminals": ["d", "g", "s", "b"],
        "vds": 0.5,
        "vgs_for_jacobian": 0.6,
        "xfail": True,  # Known PHI node resolution issues
    },
]


def _build_transistor_voltages(terminals: list, vds: float, vgs: float) -> list:
    """Build voltage vector for transistor given terminal order."""
    voltages = []
    for t in terminals:
        if t == "d":
            voltages.append(vds)
        elif t == "g":
            voltages.append(vgs)
        else:  # s, b, or other terminals grounded
            voltages.append(0.0)
    return voltages


# =============================================================================
# Parameterized Tests for Simple Models
# =============================================================================

@pytest.fixture(scope="module")
def compiled_models():
    """Compile all simple models once per test module."""
    compiled = {}
    for config in MODEL_CONFIGS:
        name = config["name"]
        va_path = config["va_path"]
        osdi_path = OSDI_CACHE / f"{name}.osdi"
        compile_va_to_osdi(va_path, osdi_path)
        compiled[name] = {"osdi_path": osdi_path, "va_path": va_path}
    return compiled


class TestSimpleModelsParameterized:
    """Parameterized tests for simple models (resistor, capacitor, diode).

    Uses MODEL_CONFIGS to reduce test boilerplate. Each config specifies:
    - name: Model name for test ID
    - va_path: Path to Verilog-A source
    - params: Default parameters
    - test_voltages: List of voltage vectors to test
    - skip_jacobian: Whether to skip Jacobian tests
    """

    @pytest.mark.parametrize("config", MODEL_CONFIGS, ids=lambda c: c["name"])
    def test_residual_at_test_voltages(self, compiled_models, config):
        """Test residual match at all configured test voltages."""
        name = config["name"]
        paths = compiled_models[name]
        params = config["params"]

        osdi_eval, _, _, _, _, _ = create_osdi_evaluator(paths["osdi_path"], params)
        jax_eval, _, _ = create_jax_evaluator(paths["va_path"], params)

        for voltages in config["test_voltages"]:
            osdi_res = osdi_eval(voltages)
            jax_res = jax_eval(voltages)

            # Compare resistive residuals
            passed, max_abs, max_rel = compare_arrays(
                osdi_res[0], jax_res[0], rtol=1e-10, atol=1e-15
            )
            assert passed, f"{name} resistive mismatch at V={voltages}: abs={max_abs}, rel={max_rel}"

            # Compare reactive residuals
            passed, max_abs, max_rel = compare_arrays(
                osdi_res[1], jax_res[1], rtol=1e-10, atol=1e-15
            )
            assert passed, f"{name} reactive mismatch at V={voltages}: abs={max_abs}, rel={max_rel}"

    @pytest.mark.parametrize("config", MODEL_CONFIGS, ids=lambda c: c["name"])
    def test_jacobian_at_first_voltage(self, compiled_models, config):
        """Test Jacobian match at first configured test voltage."""
        if config.get("skip_jacobian", False):
            pytest.skip(f"Jacobian test skipped for {config['name']}")

        name = config["name"]
        paths = compiled_models[name]
        params = config["params"]
        voltages = config["test_voltages"][0]

        osdi_eval, _, _, _, jacobian_keys, n_nodes = create_osdi_evaluator(paths["osdi_path"], params)
        jax_eval, _, _ = create_jax_evaluator(paths["va_path"], params)

        osdi_res = osdi_eval(voltages)
        jax_res = jax_eval(voltages)

        # Compare resistive Jacobian
        result = compare_jacobians(
            osdi_res[2], jax_res[2], n_nodes, jacobian_keys,
            rtol=1e-10, atol=1e-15
        )
        assert result.passed, f"{name} resistive Jacobian mismatch:\n{result.report}"

        # Compare reactive Jacobian
        result = compare_jacobians(
            osdi_res[3], jax_res[3], n_nodes, jacobian_keys,
            rtol=1e-10, atol=1e-15, reactive=True
        )
        assert result.passed, f"{name} reactive Jacobian mismatch:\n{result.report}"


# =============================================================================
# Parameterized Tests for Transistor Models
# =============================================================================

@pytest.fixture(scope="module")
def compiled_transistors():
    """Compile all transistor models once per test module."""
    compiled = {}
    for config in TRANSISTOR_CONFIGS:
        name = config["name"]
        va_path = config["va_path"]
        if not va_path.exists():
            continue  # Skip missing models
        osdi_path = OSDI_CACHE / f"{name}.osdi"
        compile_va_to_osdi(va_path, osdi_path)
        compiled[name] = {"osdi_path": osdi_path, "va_path": va_path}
    return compiled


def _transistor_xfail(config):
    """Apply xfail marker if config specifies it."""
    if config.get("xfail", False):
        return pytest.param(config, marks=pytest.mark.xfail(
            reason=f"{config['name']} has known PHI node resolution issues"
        ))
    return config


class TestTransistorModelsParameterized:
    """Parameterized tests for transistor models (EKV, MVSG, PSP102).

    Uses TRANSISTOR_CONFIGS to test Vgs sweeps and Jacobians across models.
    """

    @pytest.mark.parametrize("config", [_transistor_xfail(c) for c in TRANSISTOR_CONFIGS],
                             ids=lambda c: c["name"] if isinstance(c, dict) else c.values[0]["name"])
    def test_ids_vs_vgs_sweep(self, compiled_transistors, config):
        """Sweep Vgs at fixed Vds, compare drain current residuals."""
        name = config["name"]
        if name not in compiled_transistors:
            pytest.skip(f"Model {name} not found or failed to compile")
        paths = compiled_transistors[name]
        params = config["params"]
        terminals = config["terminals"]
        vds = config["vds"]

        osdi_eval, _, _, _, _, _ = create_osdi_evaluator(paths["osdi_path"], params)
        jax_eval, _, _ = create_jax_evaluator(paths["va_path"], params)

        all_failed = []
        for vgs in np.linspace(0.0, 1.0, 11):
            voltages = _build_transistor_voltages(terminals, vds, vgs)

            osdi_res = osdi_eval(voltages)
            jax_res = jax_eval(voltages)

            passed, max_abs, max_rel, failed = compare_residuals(
                osdi_res[0], jax_res[0]
            )
            if not passed:
                all_failed.append((vgs, failed, max_abs, max_rel))

        assert len(all_failed) == 0, (
            f"{name} residual comparison failed at {len(all_failed)} points. "
            f"First failure at Vgs={all_failed[0][0]:.2f}V: "
            f"max_abs={all_failed[0][2]:.3e}, max_rel={all_failed[0][3]:.2%}"
        )

    @pytest.mark.parametrize("config", [_transistor_xfail(c) for c in TRANSISTOR_CONFIGS],
                             ids=lambda c: c["name"] if isinstance(c, dict) else c.values[0]["name"])
    def test_jacobian_at_operating_point(self, compiled_transistors, config):
        """Compare Jacobian at typical operating point."""
        name = config["name"]
        if name not in compiled_transistors:
            pytest.skip(f"Model {name} not found or failed to compile")
        paths = compiled_transistors[name]
        params = config["params"]
        terminals = config["terminals"]
        vds = config["vds"]
        vgs = config["vgs_for_jacobian"]

        osdi_eval, _, _, _, jacobian_keys, n_nodes = create_osdi_evaluator(paths["osdi_path"], params)
        jax_eval, _, _ = create_jax_evaluator(paths["va_path"], params)

        voltages = _build_transistor_voltages(terminals, vds, vgs)

        osdi_res = osdi_eval(voltages)
        jax_res = jax_eval(voltages)

        result = compare_jacobians(
            osdi_res[2], jax_res[2], n_nodes, jacobian_keys,
            rtol=CONDUCTANCE_RTOL, atol=CONDUCTANCE_ABSTOL
        )
        assert result.passed, f"{name} Jacobian mismatch:\n{result.report}"


# =============================================================================
# Helper Functions
# =============================================================================

def create_osdi_evaluator(osdi_path: Path, params: dict, temperature: float = 300.0, gmin: float = 0.0):
    """Create an OSDI evaluator function with fixed params.

    Args:
        osdi_path: Path to compiled OSDI file
        params: Dict of parameter name -> value
        temperature: Device temperature in K
        gmin: Minimum conductance for convergence. Default 0.0 for comparison tests.
              Use 1e-12 for actual simulation. JAX doesn't include $simparam("gmin"),
              so we use 0.0 by default to ensure OSDI and JAX compute the same physics.

    Returns:
        Tuple of (evaluate_fn, lib, model, instance, jacobian_keys, n_nodes) where:
            - evaluate_fn: fn(voltages: list) -> (resist_residuals, react_residuals, resist_jac, react_jac)
            - lib: OsdiLibrary object
            - model: OsdiModel object
            - instance: OsdiInstance object
            - jacobian_keys: List of dicts with row, col, has_resist, has_react for Jacobian comparison
            - n_nodes: Number of nodes (terminals + internal)
    """
    print(f"Creating evaluator for {osdi_path}")
    lib = osdi_py.OsdiLibrary(str(osdi_path))
    model = lib.create_model()

    # Set model parameters
    lib_params = lib.get_params()
    for i, p in enumerate(lib_params):
        name = p["name"]
        param_type = p.get("type", "real")
        value = None

        if name in params:
            value = params[name]
        elif name.startswith('$') and name[1:] in params:
            value = params[name[1:]]

        if value is not None:
            if param_type == "int":
                print(f"Setting int param {i} ({name}) to {int(value)}")
                model.set_int_param(i, int(value))
            else:
                print(f"Setting real param {i} ({name}) to {float(value)}")
                model.set_real_param(i, float(value))

    model.process_params()

    # Create instance
    instance = model.create_instance()
    # Map terminal nodes to their indices
    # Note: init_node_mapping sets internal nodes to themselves automatically
    node_indices = list(range(lib.num_terminals))
    print(f"node_indicies: {node_indices}")
    instance.init_node_mapping(node_indices)
    # process_params expects num_terminals (connected ports), not num_nodes
    instance.process_params(temperature, lib.num_terminals)

    n_nodes = lib.num_nodes
    n_terminals = lib.num_terminals

    # Get Jacobian structure for format-aware comparison
    jacobian_keys = lib.get_jacobian_entries()

    def evaluate(voltages):
        """Evaluate device at given voltages (terminals only).

        Internal node voltages are initialized to 0.
        """
        # IMPORTANT: Reference model to keep it alive - instance.model_data is a raw
        # pointer to model.data, so model must not be garbage collected before instance
        _ = model

        # Extend voltages to include internal nodes (set to 0)
        full_voltages = list(voltages) + [0.0] * (n_nodes - n_terminals)

        print(f"OSDI full_voltages = {full_voltages}")
        flags = (
            osdi_py.CALC_RESIST_RESIDUAL
            | osdi_py.CALC_REACT_RESIDUAL
            | osdi_py.CALC_RESIST_JACOBIAN
            | osdi_py.CALC_REACT_JACOBIAN
            | osdi_py.ANALYSIS_DC
        )
        print(f"OSDI flags = {flags}, gmin = {gmin}")
        ret_flags = instance.eval(full_voltages, flags, 0.0, gmin=gmin)
        assert ret_flags == 0, f"OSDI eval returned error flags: {ret_flags}"

        print(f"Fetching residuals for {n_nodes} nodes")
        # Get residuals (pass zeros array to load)
        resist_res = instance.load_residual_resist([0.0] * n_nodes)
        react_res = instance.load_residual_react([0.0] * n_nodes)
        print(f"  resist = {resist_res}")
        print(f"  react  = {react_res}")

        # Get Jacobian
        print(f"Fetching jacobian for {n_nodes} nodes")
        resist_jac = instance.write_jacobian_array_resist()
        react_jac = instance.write_jacobian_array_react()
        print(f"  resist = {resist_jac}")
        print(f"  react  = {react_jac}")

        return resist_res, react_res, resist_jac, react_jac

    return evaluate, lib, model, instance, jacobian_keys, n_nodes


def create_jax_evaluator(va_path: Path, params: dict, temperature: float = 300.0, return_debug_info: bool = False):
    """Create a JAX evaluator function with fixed params.

    Uses proper init + eval flow to handle hidden_state values correctly.
    Complex models (diode, PSP103, BSIM4) have hidden_state values that must
    be computed by the init function, not provided as inputs.

    Args:
        va_path: Path to Verilog-A file
        params: Dict of parameter name -> value
        temperature: Device temperature in K
        return_debug_info: If True, return additional debug info dict

    Returns:
        Tuple of (evaluate_fn, module, metadata) where evaluate_fn has signature:
            fn(voltages: list) -> (resist_residuals, react_residuals, resist_jac, react_jac)
        If return_debug_info=True, also returns debug_info dict with:
            - cache: numpy array of cache values from init
            - collapse_decisions: collapse decision flags
            - shared_inputs: shared parameter values for eval
            - init_inputs: init parameter values
            - init_param_names: names of init parameters
            - eval_param_names: names of eval parameters
            - voltage_indices: indices of voltage parameters
            - non_voltage_indices: indices of non-voltage parameters
    """
    import jax.numpy as jnp

    modules = openvaf_py.compile_va(str(va_path))
    module = modules[0]
    translator = openvaf_jax.OpenVAFToJAX(module)

    # Step 1: Get init function with validated params and compute cache
    init_fn, init_metadata = translator.translate_init(
        params=params,
        temperature=temperature,
    )
    init_inputs = init_metadata['init_inputs']
    init_param_names = init_metadata['param_names']

    # Run init to compute cache (hidden_state values)
    cache, collapse_decisions = init_fn(init_inputs)

    # Step 2: Get eval function with validated params
    eval_fn, eval_metadata = translator.translate_eval(
        params=params,
        temperature=temperature,
    )

    eval_param_names = eval_metadata['param_names']
    eval_param_kinds = eval_metadata['param_kinds']
    voltage_indices = eval_metadata['voltage_indices']
    non_voltage_indices = eval_metadata['shared_indices']
    shared_inputs = eval_metadata['shared_inputs']
    node_names = eval_metadata['node_names']
    n_nodes = len(node_names)

    print(f"JAX voltage_indices = {voltage_indices}")
    print(f"JAX non_voltage_indices = {non_voltage_indices}")

    shared_arr = jnp.array(shared_inputs)
    # Build simparams array using metadata from translate_eval
    # For comparison tests, we use gmin=0 to match OSDI behavior
    simparams_list = build_simparams(eval_metadata, {
        '$analysis_type': 0.0,  # DC analysis
        'gmin': 0.0,            # Match OSDI gmin=0 for comparison
    })
    simparams_arr = jnp.array(simparams_list)

    def evaluate(voltages):
        """Evaluate device at given voltages (terminals only).

        Internal node voltages are initialized to 0.
        """
        # Extend voltages to include internal nodes (set to 0)
        full_voltages = list(voltages) + [0.0] * (n_nodes - len(voltages))

        # Build varying inputs (voltages only)
        varying_inputs = []
        for i in voltage_indices:
            name = eval_param_names[i]
            kind = eval_param_kinds[i]

            if kind == 'implicit_unknown':
                # Implicit equation node - name is like 'inode0', 'inode1'
                # Map to corresponding implicit_equation_* in node_names
                if name.startswith('inode'):
                    idx = int(name[5:])  # Extract index from 'inode0', 'inode1', etc.
                    # Look for implicit_equation_N in node_names
                    implicit_node = f'implicit_equation_{idx}'
                    if implicit_node in node_names:
                        node_idx = node_names.index(implicit_node)
                        varying_inputs.append(full_voltages[node_idx])
                    else:
                        raise Exception(f"Implicit node {implicit_node} not found in node_names: {node_names}")
                else:
                    raise Exception(f"Unknown implicit_unknown format: {name}")
            elif kind == 'voltage':
                # Regular voltage param - format is V(node) or V(node1, node2)
                inner = name[2:-1]  # Remove V( and )
                if ',' in inner:
                    node_pos, node_neg = inner.split(',')
                    node_pos = node_pos.strip()
                    node_neg = node_neg.strip()
                    if node_pos in node_names and node_neg in node_names:
                        idx_pos = node_names.index(node_pos)
                        idx_neg = node_names.index(node_neg)
                        varying_inputs.append(full_voltages[idx_pos] - full_voltages[idx_neg])
                    else:
                        raise Exception(f"Unknown voltage parameters to JAX eval: {node_pos} (POS), {node_neg} (NEG)")
                else:
                    node = inner.strip()
                    if node in node_names:
                        idx = node_names.index(node)
                        varying_inputs.append(full_voltages[idx])
                    else:
                        raise Exception(f"Unknown voltage parameters to JAX eval: {node}")
            else:
                raise Exception(f"Unexpected kind {kind} at voltage index {i}")

        varying_arr = jnp.array(varying_inputs)

        # Run eval with cache and simparams
        result = eval_fn(shared_arr, varying_arr, cache, simparams_arr)
        res_resist, res_react, jac_resist, jac_react = result[:4]

        print(f"Residuals for {n_nodes} nodes")
        print(f"  resist = {res_resist}")
        print(f"  react  = {res_react}")

        print(f"Jacobian for {n_nodes} nodes")
        print(f"  resist = {jac_resist}")
        print(f"  react  = {jac_react}")

        # Convert to lists
        return (
            list(np.array(res_resist)),
            list(np.array(res_react)),
            list(np.array(jac_resist)),
            list(np.array(jac_react)),
        )

    if return_debug_info:
        debug_info = {
            'cache': np.array(cache),
            'collapse_decisions': np.array(collapse_decisions),
            'shared_inputs': np.array(shared_inputs),
            'init_inputs': np.array(init_inputs),
            'init_param_names': init_param_names,
            'eval_param_names': eval_param_names,
            'eval_param_kinds': eval_param_kinds,
            'voltage_indices': voltage_indices,
            'non_voltage_indices': non_voltage_indices,
            'node_names': node_names,
            'init_metadata': init_metadata,
            'eval_metadata': eval_metadata,
        }
        return evaluate, module, eval_metadata, debug_info

    return evaluate, module, eval_metadata


def compare_arrays(
    osdi_arr: list,
    jax_arr: list,
    rtol: float = 1e-10,
    atol: float = 1e-15,
) -> tuple:
    """Compare arrays with relative and absolute tolerance.

    Returns:
        Tuple of (passed, max_abs_diff, max_rel_diff)
    """
    osdi = np.array(osdi_arr)
    jax = np.array(jax_arr)

    if osdi.shape != jax.shape:
        return False, float('inf'), float('inf')

    abs_diff = np.abs(osdi - jax)
    max_abs = np.max(abs_diff) if abs_diff.size > 0 else 0.0

    # Relative difference (avoid division by zero)
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_diff = np.where(
            np.abs(osdi) > atol,
            abs_diff / np.abs(osdi),
            0.0
        )
    max_rel = np.max(rel_diff) if rel_diff.size > 0 else 0.0

    passed = np.allclose(osdi, jax, rtol=rtol, atol=atol)
    return passed, max_abs, max_rel


# =============================================================================
# Phase 1: Simple Component Tests
# =============================================================================


class TestResistorComparison:
    """Additional OSDI vs JAX tests for resistor model.

    Basic residual/Jacobian tests are in TestSimpleModelsParameterized.
    This class contains unique tests like voltage sweep.
    """

    @pytest.fixture(scope="class")
    def osdi_path(self):
        """Compile resistor.va to OSDI once per class."""
        va_path = VACASK_DEVICES / "resistor.va"
        osdi_path = OSDI_CACHE / "resistor.osdi"
        compile_va_to_osdi(va_path, osdi_path)
        return osdi_path

    @pytest.fixture(scope="class")
    def va_path(self):
        """Get resistor.va path."""
        return VACASK_DEVICES / "resistor.va"

    def test_voltage_sweep(self, osdi_path, va_path):
        """Sweep voltage from -1V to +1V, compare currents."""
        params = {"r": 2000.0, "mfactor": 1.0}

        osdi_eval, _, _, _, _, _ = create_osdi_evaluator(osdi_path, params)
        jax_eval, _, _ = create_jax_evaluator(va_path, params)

        max_resist_diff = 0.0

        for v in np.linspace(-1.0, 1.0, 21):
            voltages = [v, 0.0]  # V(A)=v, V(B)=0

            osdi_res = osdi_eval(voltages)
            jax_res = jax_eval(voltages)

            resist_diff = np.max(np.abs(np.array(osdi_res[0]) - np.array(jax_res[0])))
            max_resist_diff = max(max_resist_diff, resist_diff)

        assert max_resist_diff < 1e-10, f"Max resistive diff over sweep: {max_resist_diff}"


class TestCapacitorComparison:
    """Compare OSDI vs JAX for capacitor model."""

    @pytest.fixture(scope="class")
    def osdi_path(self):
        """Compile capacitor.va to OSDI once per class."""
        va_path = VACASK_DEVICES / "capacitor.va"
        osdi_path = OSDI_CACHE / "capacitor.osdi"
        compile_va_to_osdi(va_path, osdi_path)
        return osdi_path

    @pytest.fixture(scope="class")
    def va_path(self):
        """Get capacitor.va path."""
        return VACASK_DEVICES / "capacitor.va"

    def test_reactive_charge(self, osdi_path, va_path):
        """Q = C*V comparison."""
        c_val = 1e-12  # 1pF
        params = {"c": c_val, "mfactor": 1.0}

        osdi_eval, _, _, _, _, _ = create_osdi_evaluator(osdi_path, params)
        jax_eval, _, _ = create_jax_evaluator(va_path, params)

        voltages = [1.0, 0.0]  # V = 1V
        expected_q = c_val * 1.0  # Q = C*V

        osdi_res = osdi_eval(voltages)
        jax_res = jax_eval(voltages)

        # Compare reactive residuals (charge)
        passed, max_abs, max_rel = compare_arrays(
            osdi_res[1], jax_res[1], rtol=1e-10, atol=1e-20
        )
        assert passed, f"Reactive residual mismatch: max_abs={max_abs}, max_rel={max_rel}"

        # Verify absolute value
        assert abs(abs(jax_res[1][0]) - expected_q) < 1e-20, \
            f"Expected Q={expected_q}, got {jax_res[1][0]}"

    def test_reactive_jacobian(self, osdi_path, va_path):
        """dQ/dV = C comparison."""
        c_val = 2e-12  # 2pF
        params = {"c": c_val, "mfactor": 1.0}

        osdi_eval, _, _, _, jacobian_keys, n_nodes = create_osdi_evaluator(osdi_path, params)
        jax_eval, _, _ = create_jax_evaluator(va_path, params)

        voltages = [1.0, 0.0]

        osdi_res = osdi_eval(voltages)
        jax_res = jax_eval(voltages)

        # Compare reactive Jacobian using format-aware comparison
        result = compare_jacobians(
            osdi_res[3], jax_res[3], n_nodes, jacobian_keys,
            rtol=1e-10, atol=1e-20, reactive=True
        )
        assert result.passed, f"Reactive Jacobian mismatch:\n{result.report}"


class TestDiodeComparison:
    """Additional OSDI vs JAX tests for diode model.

    Basic residual/Jacobian tests are in TestSimpleModelsParameterized.
    This class contains unique tests like reverse bias.
    """

    @pytest.fixture(scope="class")
    def osdi_path(self):
        """Compile diode.va to OSDI once per class."""
        va_path = VACASK_DEVICES / "diode.va"
        osdi_path = OSDI_CACHE / "diode.osdi"
        compile_va_to_osdi(va_path, osdi_path)
        return osdi_path

    @pytest.fixture(scope="class")
    def va_path(self):
        """Get diode.va path."""
        return VACASK_DEVICES / "diode.va"

    def test_reverse_bias(self, osdi_path, va_path):
        """Test at V = -1V (reverse bias)."""
        configure_precision()

        params = {
            "Is": 1e-12,  # Saturation current
            "N": 1.0,     # Ideality factor
            "Rs": 10.0,   # Series resistance (non-zero to avoid node collapse)
            "Tnom": 27.0,  # Nominal temperature (Celsius)
            "mfactor": 1.0,
        }

        osdi_eval, _, _, _, _, _ = create_osdi_evaluator(osdi_path, params)
        jax_eval, _, _ = create_jax_evaluator(va_path, params)

        voltages = [-1.0, 0.0]  # V(A)=-1V, V(C)=0 (reverse bias)

        osdi_res = osdi_eval(voltages)
        jax_res = jax_eval(voltages)

        # Compare resistive residuals
        passed, max_abs, max_rel = compare_arrays(
            osdi_res[0], jax_res[0], rtol=1e-8, atol=1e-15        )
        assert passed, f"Resistive residual mismatch: max_abs={max_abs}, max_rel={max_rel}"


# =============================================================================
# Phase 2: Complex Transistor Model Tests (PSP103, BSIM4)
# =============================================================================


def get_psp102_va_path() -> Path:
    """Find PSP102 Verilog-A file."""
    psp102_path = OPENVAF_INTEGRATION / "PSP102" / "psp102.va"
    if psp102_path.exists():
        return psp102_path

    raise FileNotFoundError("PSP102 model not found")


def get_ekv_va_path() -> Path:
    """Find EKV Verilog-A file."""
    ekv_path = OPENVAF_INTEGRATION / "EKV" / "ekv.va"
    if ekv_path.exists():
        return ekv_path

    raise FileNotFoundError("EKV model not found")


def get_mvsg_va_path() -> Path:
    """Find MVSG_CMC Verilog-A file."""
    mvsg_path = OPENVAF_INTEGRATION / "MVSG_CMC" / "mvsg_cmc.va"
    if mvsg_path.exists():
        return mvsg_path

    raise FileNotFoundError("MVSG_CMC model not found")


def get_bsim4_va_path() -> Path:
    """Find BSIM4 Verilog-A file."""
    # Try VACASK devices location
    bsim4_path = VACASK_DEVICES / "bsim4v8.va"
    if bsim4_path.exists():
        return bsim4_path

    raise FileNotFoundError("BSIM4 model not found")


class TestPSP102Comparison:
    """Additional OSDI vs JAX tests for PSP102 MOSFET model.

    Basic Vgs sweep/Jacobian tests are in TestTransistorModelsParameterized.
    This class contains unique tests like Vds sweep and debug helpers.
    """

    @pytest.fixture(scope="class")
    def osdi_path(self):
        """Compile psp102.va to OSDI once per class."""
        va_path = get_psp102_va_path()
        osdi_path = OSDI_CACHE / "psp102.osdi"
        compile_va_to_osdi(va_path, osdi_path)
        return osdi_path

    @pytest.fixture(scope="class")
    def va_path(self):
        """Get psp102.va path."""
        return get_psp102_va_path()

    @pytest.fixture
    def nmos_params(self):
        """Minimal NMOS parameters for PSP102.

        Note: Be careful with units! Check parameter descriptions.
        """
        return {
            "TYPE": 1,       # NMOS (+1) or PMOS (-1)
            "W": 1e-6,       # Width (m)
            "L": 100e-9,     # Length (m)
            "mfactor": 1.0,
        }

    @pytest.mark.xfail(reason="PSP102 has known PHI node resolution issues for NMOS/PMOS branching")
    def test_ids_vs_vds_sweep(self, osdi_path, va_path, nmos_params):
        """Sweep Vds at fixed Vgs, compare Ids (output characteristics).

        Uses VAMS-LRM current tolerances (abstol=1e-12, rtol=1e-4) to catch
        magnitude errors that were previously masked by flat 1e-6 tolerance.
        """
        osdi_eval, _, _, _, _, _ = create_osdi_evaluator(osdi_path, nmos_params)
        jax_eval, _, _ = create_jax_evaluator(va_path, nmos_params)

        all_failed = []
        vgs = 0.6  # Fixed Vgs (above threshold)

        for vds in np.linspace(0.0, 1.0, 11):
            voltages = [vds, vgs, 0.0, 0.0]

            osdi_res = osdi_eval(voltages)
            jax_res = jax_eval(voltages)

            # Compare residuals using VAMS-LRM current tolerance
            passed, max_abs, max_rel, failed = compare_residuals(
                osdi_res[0], jax_res[0]
            )
            if not passed:
                all_failed.append((vds, failed, max_abs, max_rel))

        assert len(all_failed) == 0, (
            f"Residual comparison failed at {len(all_failed)} points. "
            f"First failure at Vds={all_failed[0][0]:.2f}V: "
            f"max_abs={all_failed[0][2]:.3e}, max_rel={all_failed[0][3]:.2%}"
        )

    def test_debug_init_comparison(self, osdi_path, va_path, nmos_params):
        """Debug test: compare init/cache values between OSDI and JAX."""
        # Get OSDI setup
        osdi_eval, osdi_lib, osdi_model, osdi_instance, _, _ = create_osdi_evaluator(osdi_path, nmos_params)

        # Get JAX setup with debug info
        jax_eval, jax_module, jax_metadata, jax_debug = create_jax_evaluator(
            va_path, nmos_params, return_debug_info=True
        )

        print("\n" + "=" * 60)
        print("PSP102 INIT/CACHE COMPARISON")
        print("=" * 60)

        # JAX cache info
        cache = jax_debug['cache']
        print(f"\nJAX cache: {len(cache)} values")
        print(f"  Non-zero: {np.sum(np.abs(cache) > 1e-20)}")
        print(f"  Has inf: {np.sum(np.isinf(cache))}")
        print(f"  Has nan: {np.sum(np.isnan(cache))}")

        # Find cache values that are very large or very small (potential issues)
        print("\nJAX cache values > 1e10:")
        for i, v in enumerate(cache):
            if abs(v) > 1e10:
                print(f"  cache[{i}] = {v:.6e}")

        print("\nJAX cache first 30 non-zero values:")
        count = 0
        for i, v in enumerate(cache):
            if abs(v) > 1e-20:
                print(f"  cache[{i}] = {v:.6e}")
                count += 1
                if count >= 30:
                    break

        # Collapse decisions
        print(f"\nCollapse decisions: {jax_debug['collapse_decisions']}")

        # OSDI parameters info
        osdi_params = osdi_lib.get_params()
        print(f"\nOSDI parameters: {len(osdi_params)}")

        # Compare specific known parameters
        print("\nParameter comparison (JAX init inputs vs expected):")
        init_param_names = jax_debug['init_param_names']
        init_inputs = jax_debug['init_inputs']
        for name in ['TYPE', 'W', 'L', 'mfactor', '$temperature']:
            if name in init_param_names:
                idx = init_param_names.index(name)
                print(f"  {name}: JAX init = {init_inputs[idx]}")

        # Check shared inputs
        shared_inputs = jax_debug['shared_inputs']
        eval_param_names = jax_debug['eval_param_names']
        eval_param_kinds = jax_debug['eval_param_kinds']
        non_voltage_indices = jax_debug['non_voltage_indices']

        print(f"\nJAX shared inputs: {len(shared_inputs)} values")
        print(f"  Non-zero: {np.sum(np.abs(shared_inputs) > 1e-20)}")

        # Show shared inputs that correspond to actual params (not hidden_state)
        print("\nJAX shared inputs (param kind != hidden_state, first 20):")
        count = 0
        for pos, orig_idx in enumerate(non_voltage_indices):
            kind = eval_param_kinds[orig_idx]
            if kind != 'hidden_state' and kind != 'current':
                name = eval_param_names[orig_idx]
                val = shared_inputs[pos]
                if abs(val) > 1e-20:
                    print(f"  [{orig_idx}] {name} ({kind}) = {val}")
                    count += 1
                    if count >= 20:
                        break

        # Node info
        print(f"\nNode names: {jax_debug['node_names']}")
        print(f"Voltage indices: {jax_debug['voltage_indices']}")

        # Now run eval and compare
        voltages = [0.5, 0.6, 0.0, 0.0]
        print(f"\nRunning eval at voltages = {voltages}")

        osdi_res = osdi_eval(voltages)
        jax_res = jax_eval(voltages)

        print(f"\nOSDI residuals: {osdi_res[0]}")
        print(f"JAX residuals:  {jax_res[0]}")

        print(f"\nOSDI Jacobian (first 20): {osdi_res[2][:20]}")
        print(f"JAX Jacobian (first 20):  {jax_res[2][:20]}")

        print(f"\nOSDI Jacobian max abs: {np.max(np.abs(osdi_res[2])):.6e}")
        print(f"JAX Jacobian max abs:  {np.max(np.abs(jax_res[2])):.6e}")

        # This test is for debugging - always passes but prints useful info
        print("\n" + "=" * 60)


@pytest.mark.skip(reason="BSIM4 hangs during compilation - needs deeper debugging")
class TestBSIM4Comparison:
    """Compare OSDI vs JAX for BSIM4 MOSFET model."""

    @pytest.fixture(scope="class")
    def osdi_path(self):
        """Compile bsim4v8.va to OSDI once per class."""
        va_path = get_bsim4_va_path()
        osdi_path = OSDI_CACHE / "bsim4v8.osdi"
        compile_va_to_osdi(va_path, osdi_path)
        return osdi_path

    @pytest.fixture(scope="class")
    def va_path(self):
        """Get bsim4v8.va path."""
        return get_bsim4_va_path()

    @pytest.fixture
    def nmos_params(self):
        """Minimal NMOS parameters for BSIM4."""
        return {
            "L": 100e-9,     # Length
            "W": 1e-6,       # Width
            "NF": 1,         # Number of fingers
            "mfactor": 1.0,
        }

    def test_ids_vs_vgs_sweep(self, osdi_path, va_path, nmos_params):
        """Sweep Vgs at fixed Vds, compare Ids."""
        osdi_eval, _, _, _, _, _ = create_osdi_evaluator(osdi_path, nmos_params)
        jax_eval, _, _ = create_jax_evaluator(va_path, nmos_params)

        max_diff = 0.0
        vds = 0.5

        for vgs in np.linspace(0.0, 1.0, 11):
            voltages = [vds, vgs, 0.0, 0.0]  # D, G, S, B

            osdi_res = osdi_eval(voltages)
            jax_res = jax_eval(voltages)

            resist_diff = np.max(np.abs(np.array(osdi_res[0]) - np.array(jax_res[0])))
            max_diff = max(max_diff, resist_diff)

        assert max_diff < 1e-6, f"Max Ids diff over Vgs sweep: {max_diff}"

    def test_ids_vs_vds_sweep(self, osdi_path, va_path, nmos_params):
        """Sweep Vds at fixed Vgs, compare Ids."""
        osdi_eval, _, _, _, _, _ = create_osdi_evaluator(osdi_path, nmos_params)
        jax_eval, _, _ = create_jax_evaluator(va_path, nmos_params)

        max_diff = 0.0
        vgs = 0.6

        for vds in np.linspace(0.0, 1.0, 11):
            voltages = [vds, vgs, 0.0, 0.0]

            osdi_res = osdi_eval(voltages)
            jax_res = jax_eval(voltages)

            resist_diff = np.max(np.abs(np.array(osdi_res[0]) - np.array(jax_res[0])))
            max_diff = max(max_diff, resist_diff)

        assert max_diff < 1e-6, f"Max Ids diff over Vds sweep: {max_diff}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
