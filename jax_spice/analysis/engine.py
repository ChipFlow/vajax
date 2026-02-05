"""Circuit simulation engine for JAX-SPICE.

Core simulation engine that parses .sim circuit files and runs transient/DC
analysis using JAX-compiled solvers. All devices are compiled from Verilog-A
sources using OpenVAF.

TODO: Split out OpenVAF model compilation and caching into a separate module
(e.g., jax_spice/devices/openvaf_compiler.py) so it can be reused by other
components. The key functionality is:
- _COMPILED_MODEL_CACHE: module-level cache of compiled jitted functions
- _compile_openvaf_models(): compiles VA files to JAX functions with vmap+jit
- Static input preparation and batched evaluation
"""

import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

if TYPE_CHECKING:
    from jax_spice.analysis.ac import ACResult
    from jax_spice.analysis.corners import (
        CornerConfig,
        CornerSweepResult,
        ProcessCorner,
        VoltageCorner,
    )
    from jax_spice.analysis.noise import NoiseResult
    from jax_spice.analysis.transient import AdaptiveConfig
    from jax_spice.analysis.xfer import ACXFResult, DCIncResult, DCXFResult

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

# Suppress scipy's MatrixRankWarning from spsolve - this is expected for circuits
# with floating internal nodes (e.g., PSP103 NOI nodes). The QR solver handles
# near-singular matrices gracefully and produces correct results.
from scipy.sparse.linalg import MatrixRankWarning

warnings.filterwarnings("ignore", category=MatrixRankWarning)

# Note: solver.py contains standalone NR solvers (newton_solve) used by tests
# The engine uses its own nr_solve with analytic jacobians from OpenVAF
from jax_spice import configure_xla_cache, get_float_dtype
from jax_spice._logging import logger
from jax_spice.analysis.dc_operating_point import (
    compute_dc_operating_point as _compute_dc_op_impl,
)
from jax_spice.analysis.homotopy import (
    HomotopyConfig,
    run_homotopy_chain,
)
from jax_spice.analysis.integration import (
    IntegrationMethod,
    get_method_from_options,
)
from jax_spice.analysis.options import SimulationOptions
from jax_spice.analysis.parsing import (
    build_devices as _build_devices_impl,
)
from jax_spice.analysis.parsing import (
    flatten_instances,
    parse_elaborate_directive,
)
from jax_spice.analysis.solver_factories import (
    make_dense_full_mna_solver,
)
from jax_spice.analysis.sources import (
    build_source_fn,
    collect_source_devices_coo,
    get_dc_source_values,
    get_source_fn_for_device,
    get_vdd_value,
    prepare_source_devices_coo,
)
from jax_spice.config import DEFAULT_TEMPERATURE_K
from jax_spice.netlist.parser import VACASKParser
from jax_spice.profiling import ProfileConfig, profile

# Try to import OpenVAF support
# openvaf_jax is at top level, openvaf_py is built from openvaf_jax/openvaf_py
_project_root = Path(__file__).parent.parent.parent
_openvaf_jax_path = _project_root / "openvaf_jax"
_openvaf_py_path = _openvaf_jax_path / "openvaf_py"
if _openvaf_jax_path.exists() and str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
if _openvaf_py_path.exists() and str(_openvaf_py_path) not in sys.path:
    sys.path.insert(0, str(_openvaf_py_path))

try:
    import openvaf_py

    import openvaf_jax

    HAS_OPENVAF = True
except ImportError:
    HAS_OPENVAF = False
    openvaf_py = None
    openvaf_jax = None


# Module-level cache of compiled OpenVAF models
# Keyed by (model_type, va_file_mtime) to detect changes
_COMPILED_MODEL_CACHE: Dict[str, Any] = {}

# Module-level cache for SPICE number parsing (e.g., "1k" -> 1000, "1u" -> 1e-6)
# These are circuit-independent and can be shared across all CircuitEngine instances
_SPICE_NUMBER_CACHE: Dict[str, float] = {}



# OpenVAF model sources (duplicated from CircuitEngine for standalone use)
# Keys are device types, values are (base_path_key, relative_path) tuples
_OPENVAF_MODEL_PATHS = {
    "psp103": ("integration_tests", "PSP103/psp103.va"),
    "resistor": ("vacask", "resistor.va"),
    "capacitor": ("vacask", "capacitor.va"),
    "diode": ("vacask", "diode.va"),
    "sp_diode": ("vacask", "spice/sn/diode.va"),
}


def warmup_models(
    model_types: List[str] | None = None,
    trigger_xla: bool = True,
    log_fn: Callable[[str], None] | None = None,
) -> Dict[str, Any]:
    """Pre-compile OpenVAF models and optionally trigger XLA compilation.

    This function compiles device models ahead of time and caches them,
    reducing startup time for subsequent simulations. When trigger_xla=True,
    it also runs dummy evaluations to trigger XLA compilation, which is
    cached to disk via JAX_COMPILATION_CACHE_DIR.

    Args:
        model_types: List of model types to compile (e.g., ['psp103', 'resistor']).
            If None, compiles all available models.
        trigger_xla: If True, run dummy evaluations to trigger XLA compilation.
            This takes longer but gives fastest subsequent simulation startup.
        log_fn: Optional logging function for progress output.

    Returns:
        Dict mapping model_type to compiled model info dict.

    Example:
        >>> from jax_spice.analysis.engine import warmup_models
        >>> # Warmup specific models
        >>> warmup_models(['psp103', 'resistor'])
        >>> # Warmup all models
        >>> warmup_models()

    Note:
        This function automatically configures the XLA compilation cache
        if not already set.
    """
    global _COMPILED_MODEL_CACHE

    if not HAS_OPENVAF:
        raise ImportError("OpenVAF support required but openvaf_py not available")

    # Configure XLA cache
    configure_xla_cache()

    def log(msg: str):
        if log_fn:
            log_fn(msg)
        else:
            print(msg)

    # Determine which models to compile
    if model_types is None:
        model_types = list(_OPENVAF_MODEL_PATHS.keys())

    # Base paths for VA model sources
    project_root = Path(__file__).parent.parent.parent
    base_paths = {
        "integration_tests": project_root / "vendor" / "OpenVAF" / "integration_tests",
        "vacask": project_root / "vendor" / "VACASK" / "devices",
    }

    log(f"Warming up models: {model_types}")
    results = {}

    for model_type in model_types:
        import pickle
        import time

        # Check if already cached
        if model_type in _COMPILED_MODEL_CACHE:
            cached = _COMPILED_MODEL_CACHE[model_type]
            log(f"  {model_type}: already cached ({len(cached['param_names'])} params)")
            results[model_type] = cached
            continue

        model_info = _OPENVAF_MODEL_PATHS.get(model_type)
        if not model_info:
            log(f"  {model_type}: unknown model type, skipping")
            continue

        base_key, va_path = model_info
        base_path = base_paths.get(base_key)
        if not base_path:
            log(f"  {model_type}: unknown base path key {base_key}, skipping")
            continue

        full_path = base_path / va_path
        if not full_path.exists():
            log(f"  {model_type}: VA file not found at {full_path}, skipping")
            continue

        t0 = time.perf_counter()

        from openvaf_jax.cache import compute_va_hash, get_model_cache_path

        # Try to load from persistent MIR cache
        va_hash = compute_va_hash(full_path)
        cache_path = get_model_cache_path(model_type, va_hash)
        mir_cache_file = cache_path / "mir_data.pkl"

        translator = None

        if mir_cache_file.exists():
            try:
                log(f"  {model_type}: loading from MIR cache...")
                with open(mir_cache_file, "rb") as f:
                    cached_data = pickle.load(f)
                translator = openvaf_jax.OpenVAFToJAX.from_cache(cached_data)
                t1 = time.perf_counter()
                log(f"  {model_type}: loaded from cache in {t1 - t0:.1f}s")
            except Exception as e:
                log(f"  {model_type}: cache load failed ({e}), recompiling...")
                translator = None

        if translator is None:
            # Compile from scratch
            log(f"  {model_type}: compiling VA...")
            modules = openvaf_py.compile_va(str(full_path))
            t1 = time.perf_counter()
            log(f"  {model_type}: VA compiled in {t1 - t0:.1f}s")

            if not modules:
                log(f"  {model_type}: compilation failed, skipping")
                continue

            module = modules[0]
            translator = openvaf_jax.OpenVAFToJAX(module)

            # Save to MIR cache
            try:
                cache_path.mkdir(parents=True, exist_ok=True)
                cache_data = translator.to_cache()
                with open(mir_cache_file, "wb") as f:
                    pickle.dump(cache_data, f)
                log(f"  {model_type}: saved MIR cache")
            except Exception as e:
                log(f"  {model_type}: failed to save cache: {e}")

        # Generate init function
        t2 = time.perf_counter()
        init_fn, init_meta = translator.translate_init_array()
        t3 = time.perf_counter()
        log(f"  {model_type}: init_fn generated in {t3 - t2:.1f}s")

        # Generate eval function with simple split (all shared, voltage varying)
        # This is the same split used by _prepare_transient_setup
        n_eval_params = len(translator.params)
        eval_param_kinds = list(translator.module.param_kinds)
        shared_indices = list(range(n_eval_params))
        varying_indices = [i for i, kind in enumerate(eval_param_kinds) if kind == "voltage"]

        eval_fn, eval_meta = translator.translate_eval_array_with_cache_split(
            shared_indices, varying_indices
        )
        t4 = time.perf_counter()
        log(f"  {model_type}: eval_fn generated in {t4 - t3:.1f}s")

        # Cache the compiled model
        compiled = {
            "translator": translator,
            "init_fn": init_fn,
            "init_meta": init_meta,
            "eval_fn": eval_fn,
            "eval_meta": eval_meta,
            "param_names": init_meta["param_names"],
            "nodes": translator.module.nodes,
        }
        _COMPILED_MODEL_CACHE[model_type] = compiled
        results[model_type] = compiled

        # Trigger XLA compilation with dummy data
        if trigger_xla:
            log(f"  {model_type}: triggering XLA compilation...")
            try:
                n_devices = 1
                n_init_params = len(init_meta["param_names"])
                init_meta["cache_size"]

                # Dummy init call to trigger XLA compilation
                dummy_init_inputs = jnp.zeros((n_devices, n_init_params))
                vmapped_init = jax.jit(jax.vmap(init_fn))
                _ = vmapped_init(dummy_init_inputs)

                t5 = time.perf_counter()
                log(f"  {model_type}: XLA warmup done in {t5 - t4:.1f}s")
            except Exception as e:
                log(f"  {model_type}: XLA warmup failed: {e}")

        t_total = time.perf_counter() - t0
        log(f"  {model_type}: total warmup time {t_total:.1f}s")

    return results


@dataclass
class TransientResult:
    """Result of a transient simulation.

    Attributes:
        times: Array of time points
        voltages: Dict mapping node name (str) to voltage array.
                  Node names come from the netlist (e.g., 'vdd', 'out', 'inp').
        currents: Dict mapping source name (str) to current array.
                  Contains currents through voltage sources (e.g., 'vdd', 'v1').
        stats: Dict with simulation statistics (wall_time, convergence_rate, etc.)
    """

    times: Array
    voltages: Dict[str, Array]
    currents: Dict[str, Array]
    stats: Dict[str, Any]

    @property
    def num_steps(self) -> int:
        """Number of timesteps in the simulation."""
        return len(self.times)

    def voltage(self, node: str) -> Array:
        """Get voltage waveform at a specific node.

        Args:
            node: Node name from the netlist (e.g., 'vdd', 'out')

        Returns:
            Voltage array over time

        Raises:
            KeyError: If node not found
        """
        if node in self.voltages:
            return self.voltages[node]
        # Try case-insensitive lookup
        node_lower = node.lower()
        for key in self.voltages:
            if key.lower() == node_lower:
                return self.voltages[key]
        raise KeyError(f"Node '{node}' not found. Available: {self.node_names}")

    @property
    def node_names(self) -> List[str]:
        """List of node names."""
        return list(self.voltages.keys())

    def current(self, source: str) -> Array:
        """Get current waveform through a voltage source.

        Args:
            source: Source name from the netlist (e.g., 'vdd', 'v1')

        Returns:
            Current array over time (positive = current flows from + to -)

        Raises:
            KeyError: If source not found
        """
        if source in self.currents:
            return self.currents[source]
        # Try case-insensitive lookup
        source_lower = source.lower()
        for key in self.currents:
            if key.lower() == source_lower:
                return self.currents[key]
        raise KeyError(f"Source '{source}' not found. Available: {self.source_names}")

    @property
    def source_names(self) -> List[str]:
        """List of voltage source names with current data."""
        return list(self.currents.keys())


class CircuitEngine:
    """Core circuit simulation engine for JAX-SPICE.

    Parses .sim circuit files and runs transient/DC analysis using JAX-compiled
    solvers. All devices (resistors, capacitors, diodes, MOSFETs) are compiled
    from Verilog-A sources using OpenVAF.

    NODE INDEXING CONVENTIONS
    -------------------------
    The `node_names` dict maps node name strings to integer indices:
        node_names = {'0': 0, '1': 1, '2': 2, 'vdd': 3, 'out': 4, ...}

    - Ground node name comes from the netlist's "ground" statement (e.g., "ground 0")
    - Ground is always index 0
    - Other nodes get indices 1, 2, 3, ... in sorted order

    Two different array layouts are used internally:

    1. TRANSIENT voltage arrays (includes ground):
       - Shape: (n_timesteps, n_external) where n_external = num_nodes
       - V[:, 0] = ground (always 0V)
       - V[:, idx] = voltage for node with index `idx`
       - Access: `V[:, idx]` directly (no offset needed)

    2. AC/XFER/NOISE solution arrays (excludes ground):
       - Shape: (n_freqs, n_unknowns) where n_unknowns = num_nodes - 1
       - Ground is not stored (it's the reference, always 0)
       - X[:, 0] = node with idx=1, X[:, 1] = node with idx=2, etc.
       - Access: `X[:, idx - 1]` (subtract 1 to account for missing ground)

    When building result dicts, use the appropriate indexing:
        # Transient (ground included):
        for name, idx in self.node_names.items():
            if idx > 0:  # skip ground
                voltages[name] = V[:, idx]

        # AC/xfer (ground excluded):
        for name, idx in node_names.items():
            if idx > 0 and idx <= n_unknowns:
                voltages[name] = X[:, idx - 1]
    """

    # Map OSDI module names to device types
    MODULE_TO_DEVICE = {
        "sp_resistor": "resistor",
        "sp_capacitor": "capacitor",
        # Note: sp_diode NOT mapped to diode - uses full SPICE model (spice/sn/diode.va)
        "vsource": "vsource",
        "isource": "isource",
        "psp103va": "psp103",  # PSP103 MOSFET
    }

    # OpenVAF model sources
    # Keys are device types, values are (base_path_key, relative_path) tuples
    # base_path_key: 'integration_tests' or 'vacask'
    OPENVAF_MODELS = {
        "psp103": ("integration_tests", "PSP103/psp103.va"),
        "resistor": ("vacask", "resistor.va"),
        "capacitor": ("vacask", "capacitor.va"),
        "diode": ("vacask", "diode.va"),
        "sp_diode": ("vacask", "spice/sn/diode.va"),  # Full SPICE diode model
    }

    # Default parameter values for OpenVAF models
    # NOTE: Parameter defaults are now extracted from Verilog-A source via openvaf-py's
    # get_param_defaults() method. No manual MODEL_PARAM_DEFAULTS needed.

    # SPICE number suffixes (longer suffixes first for correct matching)
    SUFFIXES = {
        # Standard SPICE
        "t": 1e12,
        "g": 1e9,
        "meg": 1e6,
        "k": 1e3,
        "m": 1e-3,
        "u": 1e-6,
        "n": 1e-9,
        "p": 1e-12,
        "f": 1e-15,
        # Time units (common in PULSE/PWL sources)
        "ms": 1e-3,
        "us": 1e-6,
        "ns": 1e-9,
        "ps": 1e-12,
        "fs": 1e-15,
        # Voltage units
        "mv": 1e-3,
        "uv": 1e-6,
        "nv": 1e-9,
        # Current units (100fa = 100 femtoamps = 1e-13)
        "ma": 1e-3,
        "ua": 1e-6,
        "na": 1e-9,
        "pa": 1e-12,
        "fa": 1e-15,
    }

    def __init__(self, sim_path: Path):
        # Configure XLA compilation cache (uses XDG cache dir by default)
        # This enables caching compiled XLA programs across sessions
        configure_xla_cache()

        self.sim_path = Path(sim_path)
        self.circuit = None
        self.devices = []
        # Node name to index mapping. See NODE INDEXING CONVENTIONS below.
        self.node_names: Dict[str, int] = {}
        self.num_nodes = 0
        self.analysis_params = {}
        self.flat_instances = []

        # OpenVAF compiled models cache
        self._compiled_models: Dict[str, Any] = {}
        self._has_openvaf_devices = False

        # Parsing caches for _build_devices optimization
        self._model_params_cache: Dict[str, Dict[str, float]] = {}
        self._device_type_cache: Dict[str, str] = {}
        # Note: SPICE number parsing uses module-level _SPICE_NUMBER_CACHE

        # Transient setup cache (reused across multiple run_transient calls)
        self._transient_setup_cache: Dict[str, Any] | None = None
        self._transient_setup_key: str | None = None

        # Simulation temperature in Kelvin (default room temperature)
        self._simulation_temperature: float = DEFAULT_TEMPERATURE_K

        # Device-level voltage limiting (pnjlim/fetlim) - Phase 2 of damping implementation
        # When True, generates calls to limit_funcs in device eval instead of passthrough
        self.use_device_limiting: bool = True

        # Unified simulation options (replaces scattered analysis_params)
        self.options = SimulationOptions()

    @property
    def nr_damping(self) -> float:
        """Convenience property for options.nr_damping."""
        return self.options.nr_damping

    @nr_damping.setter
    def nr_damping(self, value: float) -> None:
        self.options.nr_damping = value

    def clear_cache(self):
        """Clear all cached data to free memory.

        Call this to release:
        - Compiled OpenVAF models
        - Cached JIT-compiled NR solvers
        - JAX internal compilation caches
        """
        import gc

        # Clear instance caches
        self._compiled_models.clear()
        self._model_params_cache.clear()
        self._device_type_cache.clear()
        # Note: _SPICE_NUMBER_CACHE is module-level and not cleared here
        if hasattr(self, "_cached_nr_solve"):
            del self._cached_nr_solve
        if hasattr(self, "_cached_solver_key"):
            del self._cached_solver_key

        # Clear transient setup cache
        self._transient_setup_cache = None
        self._transient_setup_key = None

        # Clear JAX compilation caches
        jax.clear_caches()

        # Force garbage collection
        gc.collect()

        logger.info("Cleared caches and freed memory")

    def parse_spice_number(self, s: str | float | int) -> float:
        """Parse SPICE number with suffix (e.g., 1u, 100n, 1.5k)"""
        if not isinstance(s, str):
            return float(s)
        s = s.strip().lower().strip('"')
        if not s:
            return 0.0

        for suffix, multiplier in sorted(self.SUFFIXES.items(), key=lambda x: -len(x[0])):
            if s.endswith(suffix):
                try:
                    return float(s[: -len(suffix)]) * multiplier
                except ValueError:
                    continue
        try:
            return float(s)
        except ValueError:
            return 0.0

    def _parse_spice_number_cached(self, s: str | float | int) -> float:
        """Parse SPICE number with caching for repeated values.

        Uses module-level _SPICE_NUMBER_CACHE to share parsed values
        across all CircuitEngine instances.
        """
        if not isinstance(s, str):
            return float(s)

        if s in _SPICE_NUMBER_CACHE:
            return _SPICE_NUMBER_CACHE[s]

        result = self.parse_spice_number(s)
        _SPICE_NUMBER_CACHE[s] = result
        return result

    def parse(self):
        """Parse the sim file and extract circuit information."""
        import time

        # Clear instance-specific caches when re-parsing (circuit may have changed)
        self._transient_setup_cache = None
        self._transient_setup_key = None
        self._model_params_cache.clear()
        self._device_type_cache.clear()
        # Note: _SPICE_NUMBER_CACHE is module-level and not cleared on re-parse

        logger.info("parse(): starting...")

        t0 = time.perf_counter()
        parser = VACASKParser()
        self.circuit = parser.parse_file(self.sim_path)
        t1 = time.perf_counter()

        logger.info(f"Parsed: {self.circuit.title} ({t1 - t0:.1f}s)")
        logger.debug(f"Models: {list(self.circuit.models.keys())}")
        if self.circuit.subckts:
            logger.debug(f"Subcircuits: {list(self.circuit.subckts.keys())}")

        # Flatten subcircuit instances to leaf devices
        logger.info("Flattening subcircuit instances...")
        self.flat_instances = self._flatten_top_instances()
        t2 = time.perf_counter()

        logger.info(f"Flattened: {len(self.flat_instances)} leaf devices ({t2 - t1:.1f}s)")
        for name, terms, model, params in self.flat_instances[:10]:
            logger.debug(f"  {name}: {model} {terms}")
        if len(self.flat_instances) > 10:
            logger.debug(f"  ... and {len(self.flat_instances) - 10} more")

        # Build node mapping from flattened instances.
        # See NODE INDEXING CONVENTIONS in class docstring for usage.
        # Ground node (from netlist's "ground" statement) is always index 0.
        ground_name = self.circuit.ground or "0"
        node_set = {ground_name}
        for name, terminals, model, params in self.flat_instances:
            for t in terminals:
                node_set.add(t)

        self.node_names = {ground_name: 0}
        for i, name in enumerate(sorted(n for n in node_set if n != ground_name), start=1):
            self.node_names[name] = i
        self.num_nodes = len(self.node_names)
        t3 = time.perf_counter()

        logger.info(f"Node mapping: {self.num_nodes} nodes ({t3 - t2:.1f}s)")

        # Build devices
        self._build_devices()
        t4 = time.perf_counter()

        logger.info(f"Built devices: {len(self.devices)} ({t4 - t3:.1f}s)")

        # Extract analysis parameters
        self._extract_analysis_params()

        return self

    def _get_device_type(self, model_name: str) -> str:
        """Map model name to device type (cached)."""
        if model_name in self._device_type_cache:
            return self._device_type_cache[model_name]

        model = self.circuit.models.get(model_name)
        if model:
            module = model.module.lower()
            result = self.MODULE_TO_DEVICE.get(module, module)
        else:
            # Direct lookup for built-in types
            result = self.MODULE_TO_DEVICE.get(model_name.lower(), model_name.lower())

        self._device_type_cache[model_name] = result
        return result

    def _get_model_params(self, model_name: str) -> Dict[str, float]:
        """Get parsed parameters from a model definition (cached)."""
        if model_name in self._model_params_cache:
            return self._model_params_cache[model_name]

        model = self.circuit.models.get(model_name)
        if not model:
            result = {}
        else:
            result = {k: self.parse_spice_number(v) for k, v in model.params.items()}

        self._model_params_cache[model_name] = result
        return result

    def _parse_elaborate_directive(self) -> Optional[str]:
        """Parse 'elaborate circuit("subckt_name")' directive from control block."""
        text = self.sim_path.read_text()
        return parse_elaborate_directive(text)

    def _flatten_top_instances(self) -> List[Tuple[str, List[str], str, Dict[str, str]]]:
        """Flatten subcircuit instances to leaf devices."""
        elaborate_subckt = self._parse_elaborate_directive()
        return flatten_instances(self.circuit, elaborate_subckt, self.parse_spice_number)

    def _build_devices(self):
        """Build device list from flattened instances."""
        import time

        t_start = time.perf_counter()
        logger.info(f"_build_devices(): starting with {len(self.flat_instances)} instances")

        # Use extracted function for core device building logic
        self.devices, self._has_openvaf_devices = _build_devices_impl(
            flat_instances=self.flat_instances,
            node_names=self.node_names,
            get_device_type=self._get_device_type,
            get_model_params=self._get_model_params,
            parse_number_cached=self._parse_spice_number_cached,
            openvaf_models=self.OPENVAF_MODELS,
        )

        t_loop = time.perf_counter()
        logger.info(f"_build_devices(): loop done in {t_loop - t_start:.1f}s")
        logger.debug(f"Devices: {len(self.devices)}")
        for dev in self.devices[:10]:
            logger.debug(f"  {dev['name']}: {dev['model']} nodes={dev['nodes']}")
        if len(self.devices) > 10:
            logger.debug(f"  ... and {len(self.devices) - 10} more")

        # Compile OpenVAF models if needed
        if self._has_openvaf_devices:
            self._compile_openvaf_models()
            # Compute collapse decisions early using init_fn
            self._compute_early_collapse_decisions()

    def _compile_openvaf_models(self, log_fn=None):
        """Compile OpenVAF models needed by the circuit.

        Uses module-level cache to reuse jitted functions across runner instances.
        Models are only compiled once per process.

        Args:
            log_fn: Optional logging function for progress output
        """
        global _COMPILED_MODEL_CACHE

        if not HAS_OPENVAF:
            raise ImportError("OpenVAF support required but openvaf_py not available")

        def log(msg):
            if log_fn:
                log_fn(msg)
            else:
                logger.info(msg)

        # Find unique OpenVAF model types
        openvaf_types = set()
        for dev in self.devices:
            if dev.get("is_openvaf"):
                openvaf_types.add(dev["model"])

        log(f"Compiling OpenVAF models: {openvaf_types}")

        # Base paths for different VA model sources
        project_root = Path(__file__).parent.parent.parent
        base_paths = {
            "integration_tests": project_root / "vendor" / "OpenVAF" / "integration_tests",
            "vacask": project_root / "vendor" / "VACASK" / "devices",
        }

        for model_type in openvaf_types:
            # Check instance cache first (for this runner)
            if model_type in self._compiled_models:
                continue

            # Check module-level cache (shared across all runners)
            if model_type in _COMPILED_MODEL_CACHE:
                cached = _COMPILED_MODEL_CACHE[model_type]
                log(
                    f"  {model_type}: reusing cached jitted function ({len(cached['param_names'])} params, {len(cached['nodes'])} nodes)"
                )
                self._compiled_models[model_type] = cached
                continue

            model_info = self.OPENVAF_MODELS.get(model_type)
            if not model_info:
                raise ValueError(f"Unknown OpenVAF model type: {model_type}")

            base_key, va_path = model_info
            base_path = base_paths.get(base_key)
            if not base_path:
                raise ValueError(f"Unknown base path key: {base_key}")

            full_path = base_path / va_path
            if not full_path.exists():
                raise FileNotFoundError(f"VA model not found: {full_path}")

            import pickle
            import time

            from openvaf_jax.cache import (
                compute_va_hash,
                get_model_cache_path,
            )

            t0 = time.perf_counter()

            # Try to load from persistent cache
            va_hash = compute_va_hash(full_path)
            cache_path = get_model_cache_path(model_type, va_hash)
            mir_cache_file = cache_path / "mir_data.pkl"

            translator = None
            module = None

            if mir_cache_file.exists():
                try:
                    log(f"  {model_type}: loading from persistent cache...")
                    with open(mir_cache_file, "rb") as f:
                        cached_data = pickle.load(f)
                    translator = openvaf_jax.OpenVAFToJAX.from_cache(cached_data)
                    t1 = time.perf_counter()
                    log(f"  {model_type}: loaded from cache in {t1 - t0:.1f}s")
                except Exception as e:
                    log(f"  {model_type}: cache load failed ({e}), recompiling...")
                    translator = None

            if translator is None:
                # Compile from scratch
                log(f"  {model_type}: compiling VA...")
                modules = openvaf_py.compile_va(str(full_path))
                t1 = time.perf_counter()
                log(f"  {model_type}: VA compiled in {t1 - t0:.1f}s")
                if not modules:
                    raise ValueError(f"Failed to compile {va_path}")

                log(f"  {model_type}: creating translator...")
                module = modules[0]
                translator = openvaf_jax.OpenVAFToJAX(module)
                t2 = time.perf_counter()
                log(f"  {model_type}: translator created in {t2 - t1:.1f}s")

                # Save to persistent cache
                try:
                    cache_path.mkdir(parents=True, exist_ok=True)
                    cache_data = translator.get_cache_data()
                    with open(mir_cache_file, "wb") as f:
                        pickle.dump(cache_data, f)
                    log(f"  {model_type}: saved to persistent cache (hash={va_hash})")
                except Exception as e:
                    log(f"  {model_type}: failed to save cache: {e}")

            # Get model metadata - either from module or cached data
            if module is not None:
                param_names = list(module.param_names)
                param_kinds = list(module.param_kinds)
                nodes = list(module.nodes)
                collapsible_pairs = list(module.collapsible_pairs)
                num_collapsible = module.num_collapsible
            else:
                # Load from cached data (already loaded above)
                param_names = cached_data["param_names"]
                param_kinds = cached_data["param_kinds"]
                nodes = cached_data["nodes"]
                collapsible_pairs = cached_data["collapsible_pairs"]
                num_collapsible = cached_data["num_collapsible"]

            # Get DAE metadata (node names, jacobian keys, etc.) without generating code
            t2 = time.perf_counter()  # Reset timer after cache load
            dae_metadata = translator.get_dae_metadata()
            t3 = time.perf_counter()
            log(f"  {model_type}: DAE metadata extracted in {t3 - t2:.3f}s")

            # Generate init function for cache computation and collapse decisions
            log(f"  {model_type}: generating init function...")
            init_fn, init_meta = translator.translate_init_array()
            vmapped_init = jax.jit(jax.vmap(init_fn))
            t4 = time.perf_counter()
            log(
                f"  {model_type}: init function done in {t4 - t3:.1f}s (cache_size={init_meta['cache_size']})"
            )

            # Build init->eval index mapping for extracting init inputs from eval inputs
            eval_name_to_idx = {n.lower(): i for i, n in enumerate(param_names)}
            init_to_eval_indices = []
            for name in init_meta["param_names"]:
                eval_idx = eval_name_to_idx.get(name.lower(), -1)
                init_to_eval_indices.append(eval_idx)
            init_to_eval_indices = jnp.array(init_to_eval_indices, dtype=jnp.int32)

            # NOTE: MIR data release is deferred to _prepare_static_inputs()
            # so we can generate split eval function after computing constant/varying indices
            # This saves ~28MB for PSP103 after circuit setup is complete

            compiled = {
                "module": module,  # May be None if loaded from cache
                "translator": translator,  # Stored for split function generation
                "dae_metadata": dae_metadata,
                "param_names": param_names,
                "param_kinds": param_kinds,
                "nodes": nodes,
                "collapsible_pairs": collapsible_pairs,
                "num_collapsible": num_collapsible,
                # Init function
                "init_fn": init_fn,
                "vmapped_init": vmapped_init,
                "init_param_names": list(init_meta["param_names"]),
                "init_param_kinds": list(init_meta["param_kinds"]),
                "cache_size": init_meta["cache_size"],
                "cache_mapping": init_meta["cache_mapping"],
                "init_param_defaults": init_meta.get("param_defaults", {}),
                "init_to_eval_indices": init_to_eval_indices,
                # Device-level features (set during init code generation)
                "uses_simparam_gmin": translator.uses_simparam_gmin,
                "uses_analysis": translator.uses_analysis,
                "analysis_type_map": translator.analysis_type_map,
            }

            # Store in both instance and module-level cache
            self._compiled_models[model_type] = compiled
            _COMPILED_MODEL_CACHE[model_type] = compiled

            log(f"  {model_type}: done ({len(param_names)} params, {len(nodes)} nodes)")

    def _compute_early_collapse_decisions(self):
        """Compute collapse decisions for all devices using OpenVAF vmapped_init.

        This is called early (before _setup_internal_nodes) to determine which
        node pairs should collapse for each device. Uses OpenVAF's generic
        collapse mechanism instead of model-specific code.

        OPTIMIZATION: Groups devices by unique parameter combinations and computes
        collapse decisions once per unique combo. For c6288, this reduces from
        ~10k evaluations to just 2 (one for pmos, one for nmos).

        Stores results in self._device_collapse_decisions: Dict[device_name, List[Tuple[int, int]]]
        where each tuple is (node1_idx, node2_idx) that should be collapsed.
        """
        import jax.numpy as jnp
        import numpy as np

        self._device_collapse_decisions: Dict[str, List[Tuple[int, int]]] = {}

        # Group devices by model type
        devices_by_type: Dict[str, List[Dict]] = {}
        for dev in self.devices:
            if dev.get("is_openvaf"):
                model_type = dev["model"]
                devices_by_type.setdefault(model_type, []).append(dev)

        for model_type, devs in devices_by_type.items():
            compiled = self._compiled_models.get(model_type)
            if not compiled:
                continue

            init_fn = compiled.get("init_fn")

            if init_fn is None:
                # No init function - use all collapsible pairs
                collapsible_pairs = compiled.get("collapsible_pairs", [])
                for dev in devs:
                    self._device_collapse_decisions[dev["name"]] = list(collapsible_pairs)
                continue

            # Get init parameters info
            init_param_names = compiled.get("init_param_names", [])
            init_param_defaults = compiled.get("init_param_defaults", {})
            n_init_params = len(init_param_names)
            collapsible_pairs = compiled.get("collapsible_pairs", [])
            n_collapsible = len(collapsible_pairs)

            if n_init_params == 0 or n_collapsible == 0:
                # No init params or no collapsible pairs - collapse decisions are constant
                if init_fn is not None and n_collapsible > 0:
                    try:
                        # Force CPU execution to avoid GPU JIT overhead for tiny computation
                        cpu_device = jax.devices("cpu")[0]
                        with jax.default_device(cpu_device):
                            _, collapse_decisions = init_fn(jnp.array([]))
                        # Convert collapse decisions to pairs (same for all devices)
                        pairs = []
                        for i, (n1, n2) in enumerate(collapsible_pairs):
                            if i < len(collapse_decisions) and float(collapse_decisions[i]) > 0.5:
                                pairs.append((n1, n2))
                        for dev in devs:
                            self._device_collapse_decisions[dev["name"]] = pairs
                        continue
                    except Exception as e:
                        logger.warning(f"Error computing collapse decisions for {model_type}: {e}")
                # Fallback: use all collapsible pairs
                for dev in devs:
                    self._device_collapse_decisions[dev["name"]] = list(collapsible_pairs)
                continue

            # OPTIMIZATION: Group devices by unique parameter combinations
            # For c6288, this reduces 10k evaluations to just 2 (pmos, nmos)
            def get_param_key(dev: Dict) -> Tuple:
                """Build hashable key from device parameters."""
                device_params = dev.get("params", {})
                values = []
                for pname in init_param_names:
                    pname_lower = pname.lower()
                    if pname_lower in device_params:
                        values.append(float(device_params[pname_lower]))
                    elif pname_lower in init_param_defaults:
                        values.append(float(init_param_defaults[pname_lower]))
                    else:
                        values.append(0.0)
                return tuple(values)

            # Group devices by unique parameter combinations
            unique_params: Dict[Tuple, List[Dict]] = {}
            for dev in devs:
                key = get_param_key(dev)
                unique_params.setdefault(key, []).append(dev)

            n_unique = len(unique_params)
            logger.info(
                f"Computing collapse decisions for {model_type}: "
                f"{len(devs)} devices, {n_unique} unique param combos"
            )

            # Compute collapse decisions for each unique parameter combination
            # Force CPU execution to avoid GPU JIT overhead for small computations
            cpu_device = jax.devices("cpu")[0]
            for param_key, param_devs in unique_params.items():
                try:
                    # Single init_fn call for this parameter combination
                    with jax.default_device(cpu_device):
                        init_inputs = jnp.array(param_key, dtype=get_float_dtype())
                        _, collapse_decisions = init_fn(init_inputs)

                    # Convert to pairs
                    pairs = []
                    collapse_np = np.asarray(collapse_decisions)
                    for i, (n1, n2) in enumerate(collapsible_pairs):
                        if i < len(collapse_np) and collapse_np[i] > 0.5:
                            pairs.append((n1, n2))

                    # Apply to all devices with this parameter combination
                    for dev in param_devs:
                        self._device_collapse_decisions[dev["name"]] = pairs

                except Exception as e:
                    logger.warning(
                        f"Error computing collapse for {model_type} params {param_key[:3]}...: {e}"
                    )
                    # Fallback: use all collapsible pairs for these devices
                    for dev in param_devs:
                        self._device_collapse_decisions[dev["name"]] = list(collapsible_pairs)

        logger.debug(
            f"Computed collapse decisions for {len(self._device_collapse_decisions)} devices"
        )

    def _prepare_static_inputs(
        self,
        model_type: str,
        openvaf_devices: List[Dict],
        device_internal_nodes: Dict[str, Dict[str, int]],
        ground: int,
    ) -> Tuple[List[int], List[Dict], jax.Array, jax.Array]:
        """Prepare device inputs and generate split eval function.

        This is called once per simulation. It analyzes parameter constancy across
        devices and generates optimized split eval functions that separate constant
        (shared) params from varying (per-device) params.

        The full parameter array is built in numpy, analyzed, then only the needed
        parts (shared_params, device_params, init_inputs) are converted to JAX.

        Returns:
            (voltage_indices, device_contexts, cache, collapse_decisions) where:
            - voltage_indices is list of param indices that are voltages
            - device_contexts is list of dicts with node_map, voltage_node_pairs for fast update
            - cache is shape (num_devices, cache_size) JAX array with init-computed values
            - collapse_decisions is shape (num_devices, num_collapsible) JAX array with collapse booleans

        Side effects:
            Stores in compiled dict: shared_params, device_params, vmapped_split_eval, etc.
        """
        logger.debug(f"Preparing static inputs for {model_type}")

        compiled = self._compiled_models.get(model_type)
        if not compiled:
            raise ValueError(f"OpenVAF model {model_type} not compiled")

        param_names = compiled["param_names"]
        param_kinds = compiled["param_kinds"]
        model_nodes = compiled["nodes"]

        logger.debug(f"  {len(param_names)} param_names")
        logger.debug(f"  {len(param_kinds)} param_kinds")
        logger.debug(f"  {len(model_nodes)} model_nodes")

        # Find which parameter indices are voltages (always varying)
        voltage_indices = []
        voltage_set = set()
        for i, kind in enumerate(param_kinds):
            if kind == "voltage":
                voltage_indices.append(i)
                voltage_set.add(i)

        n_devices = len(openvaf_devices)
        n_params = len(param_names)
        device_contexts = []

        # === SMART PARAM FILLING: Build only what we need ===
        # Instead of allocating full (n_devices, n_params) array, we:
        # 1. Identify which params vary between devices
        # 2. Build col_values dict with scalar (shared) or array (varying)
        # 3. Construct shared_params, device_params, init_inputs directly
        #
        # Memory savings for c6288: 200MB -> 70MB (4x reduction)

        # col_values: maps col_idx -> scalar value (shared) or 1D array (varying)
        col_values: Dict[int, Any] = {}
        varying_cols_set = set(voltage_set)  # Start with voltage cols as varying

        if n_devices > 0:
            logger.debug("Filling params (smart mode)")
            all_dev_params = [dev["params"] for dev in openvaf_devices]
            model_defaults = compiled.get("init_param_defaults", {})

            # Build param_name -> column index mapping
            param_to_cols = {}
            param_given_to_cols = {}
            limit_param_map: Dict[int, Tuple[str, str]] = {}
            for idx, (name, kind) in enumerate(zip(param_names, param_kinds)):
                name_lower = name.lower()
                if kind == "param":
                    param_to_cols.setdefault(name_lower, []).append(idx)
                elif kind == "param_given":
                    param_given_to_cols.setdefault(name_lower, []).append(idx)
                elif kind == "temperature":
                    col_values[idx] = self._simulation_temperature  # Scalar - same for all devices
                elif kind == "sysfun" and name_lower == "mfactor":
                    col_values[idx] = 1.0  # Scalar
                elif kind in ("prev_state", "enable_lim", "new_state", "enable_integration"):
                    # Limit-related params: handled in codegen, not in shared/device arrays
                    limit_param_map[idx] = (kind, name)

            # Get unique params from devices
            all_unique = set()
            for p in all_dev_params:
                all_unique.update(k.lower() for k in p.keys())

            # Fill param values and identify varying ones
            for pname in all_unique:
                if pname in param_to_cols:
                    vals = np.array(
                        [float(p.get(pname, p.get(pname.upper(), 0.0))) for p in all_dev_params]
                    )
                    # Check if all values are the same
                    if np.all(vals == vals[0]):
                        # Shared (constant) - store scalar
                        for col in param_to_cols[pname]:
                            col_values[col] = float(vals[0])
                    else:
                        # Varying - store array and mark as varying
                        for col in param_to_cols[pname]:
                            col_values[col] = vals
                            varying_cols_set.add(col)
                if pname in param_given_to_cols:
                    for col in param_given_to_cols[pname]:
                        col_values[col] = 1.0  # Scalar

            # Defaults for params not in any device (all shared by definition)
            for pname, cols in param_to_cols.items():
                if pname not in all_unique:
                    if pname in ("tnom", "tref", "tr"):
                        default = 27.0
                    elif pname in ("nf", "mult", "ns", "nd"):
                        default = 1.0
                    else:
                        default = model_defaults.get(pname, 0.0)
                    for col in cols:
                        col_values[col] = default  # Scalar
            logger.debug("Params filled (smart mode)")

        # NOTE: init_param_defaults from openvaf-py contains Verilog-A source defaults
        # These are used in vectorized filling above when no device-level value exists

        for dev_idx, dev in enumerate(openvaf_devices):
            ext_nodes = dev["nodes"]  # [d, g, s, b]
            dev["params"]
            internal_nodes = device_internal_nodes.get(dev["name"], {})

            # Build node map: model node name -> global circuit node index
            # Use actual number of external terminals (not hardcoded 4 for MOSFETs)
            node_map = {}
            n_ext_terminals = len(ext_nodes)
            for i in range(n_ext_terminals):
                model_node = model_nodes[i]
                node_map[model_node] = ext_nodes[i]

            # Internal nodes
            for model_node, global_idx in internal_nodes.items():
                node_map[model_node] = global_idx

            # Map clean VA node names from v2 API (e.g., 'D', 'G', 'S', 'B', 'NOI', ...)
            # These are the names used in metadata['node_names'] and jacobian_keys
            metadata = compiled.get("dae_metadata", {})
            va_terminals = metadata.get("terminals", [])
            va_internal = metadata.get("internal_nodes", [])

            # Map terminal names: D->ext_nodes[0], G->ext_nodes[1], etc.
            for i, va_name in enumerate(va_terminals):
                if i < len(ext_nodes):
                    node_map[va_name] = ext_nodes[i]

            # Map internal names: NOI->internal_nodes['node4'], GP->internal_nodes['node5'], etc.
            # The v2 internal_nodes list order matches node indices starting after terminals
            num_terminals = len(va_terminals)
            for i, va_name in enumerate(va_internal):
                internal_key = f"node{num_terminals + i}"
                if internal_key in internal_nodes:
                    node_map[va_name] = internal_nodes[internal_key]

            # Pre-compute voltage node pairs for fast update
            voltage_node_pairs = []
            for idx in voltage_indices:
                name = param_names[idx]
                node_pair = self._parse_voltage_param(name, node_map, model_nodes, ground)
                voltage_node_pairs.append(node_pair)

            # NOTE: Parameter filling is done by vectorized code above (lines 788-834).
            # The per-device loop that was here has been removed as it was dead code.

            device_contexts.append(
                {
                    "name": dev["name"],
                    "node_map": node_map,
                    "ext_nodes": ext_nodes,
                    "voltage_node_pairs": voltage_node_pairs,
                }
            )

        # Add analysis_type and gmin to col_values if needed
        uses_analysis = compiled.get("uses_analysis", False)
        uses_simparam_gmin = compiled.get("uses_simparam_gmin", False)
        n_params_total = n_params

        if uses_analysis and n_devices > 0:
            logger.debug("Adding analysis_type and gmin to col_values")
            # analysis_type at n_params, gmin at n_params+1
            col_values[n_params] = 0.0  # DC analysis (scalar - same for all)
            col_values[n_params + 1] = 1e-12  # gmin (scalar)
            n_params_total = n_params + 2
        elif uses_simparam_gmin and n_devices > 0:
            col_values[n_params] = 1e-12  # gmin (scalar)
            n_params_total = n_params + 1

        # === Build shared_params and device_params from col_values ===
        # No full (n_devices, n_params) array allocation needed!
        if n_devices >= 1 and n_params_total > 0:
            # Classify columns as shared (scalar value) or varying (array value or voltage)
            # Limit-related params (prev_state, enable_lim, etc.) are excluded - they're
            # handled directly in codegen via limit_param_map
            limit_param_indices = set(limit_param_map.keys())
            shared_indices = []
            varying_indices_list = []
            for col in range(n_params_total):
                if col in limit_param_indices:
                    continue  # Handled separately in codegen
                elif col in varying_cols_set:
                    varying_indices_list.append(col)
                else:
                    shared_indices.append(col)

            n_const = len(shared_indices)
            n_varying = len(varying_indices_list)
            logger.info(
                f"{model_type} parameter analysis: {n_const}/{n_params_total} constant columns, "
                f"{n_varying} varying across {n_devices} devices"
            )

            # Log which parameters vary (for debugging)
            if n_varying > 0 and n_varying <= 30:
                varying_names = [
                    param_names[int(i)] if i < len(param_names) else f"col_{i}"
                    for i in varying_indices_list
                ]
                logger.debug(f"{model_type} varying params: {varying_names}")

            # === Build arrays directly from col_values (no full array allocation) ===

            # Build shared_params from scalar values
            shared_params_list = []
            for col in shared_indices:
                val = col_values.get(col, 0.0)
                if isinstance(val, np.ndarray):
                    # Shouldn't happen for shared cols, but use first value as fallback
                    shared_params_list.append(float(val[0]))
                else:
                    shared_params_list.append(float(val))
            shared_params = jnp.array(shared_params_list, dtype=get_float_dtype())

            # Build device_params from varying columns in col_values
            if n_varying > 0:
                device_params_cols = []
                for col in varying_indices_list:
                    val = col_values.get(col)
                    if val is None:
                        # Voltage column not yet filled - use zeros
                        device_params_cols.append(np.zeros(n_devices, dtype=get_float_dtype()))
                    elif isinstance(val, np.ndarray):
                        device_params_cols.append(val)
                    else:
                        # Scalar that ended up in varying (shouldn't happen often)
                        device_params_cols.append(
                            np.full(n_devices, float(val), dtype=get_float_dtype())
                        )
                device_params = jnp.array(
                    np.column_stack(device_params_cols), dtype=get_float_dtype()
                )
            else:
                device_params = jnp.empty((n_devices, 0), dtype=get_float_dtype())

            # Free col_values - we've extracted shared_params and device_params
            del col_values

            # Generate split functions - translator must be available
            translator = compiled.get("translator")
            if translator is None or translator.dae_data is None:
                raise RuntimeError(
                    f"{model_type}: translator.dae_data not available for split function generation. "
                    f"This indicates a bug - MIR data was released before split funcs could be generated."
                )

            # Compute init cache using split init (avoids large init_inputs array)
            init_to_eval = compiled.get("init_to_eval_indices")
            if init_to_eval is not None:
                logger.info(f"{model_type}: generating split init function...")
                init_to_eval_list = [int(x) for x in init_to_eval]
                split_init_fn, init_split_meta = translator.translate_init_array_split(
                    shared_indices, varying_indices_list, init_to_eval_list
                )
                # vmap with in_axes=(None, 0) - shared broadcasts, device mapped
                # Use cached vmapped+jit to avoid repeated JIT compilation
                code_hash = init_split_meta.get("code_hash", "")
                vmapped_split_init = openvaf_jax.get_vmapped_jit(
                    code_hash, split_init_fn, in_axes=(None, 0)
                )

                # Compute cache using split init (no large init_inputs array needed!)
                logger.info(
                    f"Computing init cache for {model_type} ({n_devices} devices) via split init..."
                )
                cpu_device = jax.devices("cpu")[0]
                with jax.default_device(cpu_device):
                    cache, collapse_decisions = vmapped_split_init(shared_params, device_params)
                logger.info(f"Init cache computed for {model_type}: shape={cache.shape}")
                logger.debug(
                    f"Collapse decisions for {model_type}: shape={collapse_decisions.shape}"
                )

                # Analyze cache constancy - which columns are identical across all devices?
                n_cache_cols = cache.shape[1] if cache.ndim > 1 else 0
                if n_cache_cols > 0 and n_devices > 1:
                    # Move cache to numpy for constancy analysis (faster than JAX comparisons)
                    cache_np = np.asarray(cache)
                    # A column is constant if all values equal the first device's value
                    const_mask = np.all(cache_np == cache_np[0:1, :], axis=0)
                    shared_cache_indices = [int(i) for i in np.where(const_mask)[0]]
                    varying_cache_indices = [int(i) for i in np.where(~const_mask)[0]]

                    n_shared_cache = len(shared_cache_indices)
                    n_varying_cache = len(varying_cache_indices)
                    logger.info(
                        f"{model_type}: cache constancy analysis: "
                        f"{n_shared_cache}/{n_cache_cols} shared, {n_varying_cache} varying"
                    )

                else:
                    shared_cache_indices = []
                    varying_cache_indices = list(range(n_cache_cols))
            else:
                # Fallback for models without init function
                logger.debug("Model has no init function")
                cache = jnp.empty((n_devices, 0), dtype=get_float_dtype())
                collapse_decisions = jnp.empty((n_devices, 0), dtype=jnp.float32)
                shared_cache_indices = []
                varying_cache_indices = []

            # Generate eval function with param split and cache split
            # When use_device_limiting is True, generate calls to pnjlim/fetlim
            use_device_limiting = getattr(self, "use_device_limiting", False)
            logger.info(
                f"{model_type}: generating split eval function (limit_funcs={use_device_limiting})..."
            )

            # Import limit functions - always bound for uniform interface
            from functools import partial

            from jax_spice.analysis.limiting import fetlim, pnjlim

            limit_funcs = {"pnjlim": pnjlim, "fetlim": fetlim}

            # Always use cache split for uniform function signature
            split_fn, split_meta = translator.translate_eval_array_with_cache_split(
                shared_indices,
                varying_indices_list,
                shared_cache_indices,
                varying_cache_indices,
                use_limit_functions=use_device_limiting,
                limit_param_map=limit_param_map,
            )
            # Bind limit_funcs for uniform call signature:
            # (shared, device, shared_cache, device_cache, simparams, limit_state_in)
            split_fn = partial(split_fn, limit_funcs=limit_funcs)
            # in_axes: (None, 0, None, 0, None, 0) - limit_state_in varies per device
            vmapped_split_fn = jax.jit(jax.vmap(split_fn, in_axes=(None, 0, None, 0, None, 0)))

            # Split cache arrays (indexing with empty list gives empty array)
            shared_cache = cache[0, shared_cache_indices]  # (n_shared_cache,) or empty
            device_cache = cache[:, varying_cache_indices]  # (n_devices, n_varying_cache)

            # Build default simparams array
            # simparams[0] = analysis_type (0=DC, 1=AC, 2=transient, 3=noise)
            # simparams[1+] = other simparams as registered by the model (gmin, etc.)
            # For now, use defaults - DC analysis with gmin=1e-12, mfactor=1.0
            default_simparams = jnp.array(
                [0.0, 1.0, 1e-12], dtype=get_float_dtype()
            )  # [analysis_type, mfactor, gmin]

            # Compute voltage positions within device_params (varying indices)
            varying_idx_to_pos = {
                orig_idx: pos for pos, orig_idx in enumerate(varying_indices_list)
            }
            voltage_positions = [
                varying_idx_to_pos[v] for v in voltage_indices if v in varying_idx_to_pos
            ]
            voltage_positions = jnp.array(voltage_positions, dtype=jnp.int32)

            # Store in compiled dict
            compiled["split_eval_fn"] = split_fn
            compiled["vmapped_split_eval"] = vmapped_split_fn
            compiled["shared_indices"] = shared_indices
            compiled["varying_indices"] = varying_indices_list
            compiled["shared_params"] = shared_params
            compiled["device_params"] = device_params
            compiled["voltage_positions_in_varying"] = voltage_positions
            compiled["shared_cache"] = shared_cache
            compiled["device_cache"] = device_cache
            compiled["shared_cache_indices"] = shared_cache_indices
            compiled["varying_cache_indices"] = varying_cache_indices
            compiled["default_simparams"] = default_simparams
            compiled["use_device_limiting"] = use_device_limiting
            compiled["limit_param_map"] = limit_param_map
            if limit_param_map:
                logger.info(
                    f"{model_type}: {len(limit_param_map)} limit params mapped "
                    f"({', '.join(f'{k}:{v[0]}' for k, v in sorted(limit_param_map.items()))})"
                )
            # Store limit metadata for solver to manage limit_state arrays
            if use_device_limiting and split_meta.get("limit_metadata"):
                compiled["limit_metadata"] = split_meta["limit_metadata"]
                compiled["num_limit_states"] = split_meta["limit_metadata"].get("limit_count", 0)
            else:
                compiled["limit_metadata"] = None
                compiled["num_limit_states"] = 0

            split_mem_mb = (
                shared_params.nbytes / 1024 / 1024
                + device_params.nbytes / 1024 / 1024
                + shared_cache.nbytes / 1024 / 1024
                + device_cache.nbytes / 1024 / 1024
            )
            # Compare to theoretical full array size (never allocated)
            theoretical_full_mb = n_devices * n_params_total * 8 / 1024 / 1024
            logger.info(
                f"{model_type}: split eval ready "
                f"(params: shared={len(shared_indices)}, varying={len(varying_indices_list)}; "
                f"cache: shared={len(shared_cache_indices)}, varying={len(varying_cache_indices)}; "
                f"mem={split_mem_mb:.1f}MB vs full={theoretical_full_mb:.1f}MB)"
            )

            # NOTE: Do NOT release MIR data here - different circuits may have different
            # shared/varying splits and need to regenerate split functions. MIR data is
            # small (~MB) compared to circuit data.
        else:
            # This branch should never be reached - all OpenVAF models have devices and parameters
            raise AssertionError(
                f"Cannot prepare inputs for {model_type}: n_devices={n_devices}, n_params={n_params_total}"
            )

        return voltage_indices, device_contexts, cache, collapse_decisions

    def warmup_device_models(self, static_inputs_cache: Dict[str, Tuple]) -> None:
        """Trigger XLA compilation of vmapped device functions.

        This method calls each vmapped_split_eval function with actual inputs
        to trigger XLA compilation. The compiled artifacts are stored in JAX's
        persistent cache (if configured) and can be reused across sessions.

        This separates device model compilation from main loop compilation,
        allowing device models to be pre-compiled during setup while the main
        loop is compiled later.

        Args:
            static_inputs_cache: Dict mapping model_type to cached static inputs:
                (voltage_indices, stamp_indices, voltage_node1, voltage_node2, cache, collapse_decisions)
        """
        import time

        for model_type, compiled in self._compiled_models.items():
            if model_type not in static_inputs_cache:
                continue

            t0 = time.perf_counter()
            logger.info(f"Warming up device model: {model_type}...")

            # Get the vmapped function and inputs
            vmapped_split_eval = compiled["vmapped_split_eval"]
            shared_params = compiled["shared_params"]
            device_params = compiled["device_params"]
            shared_cache = compiled["shared_cache"]
            device_cache = compiled["device_cache"]
            default_simparams = compiled.get("default_simparams", jnp.array([0.0, 1.0, 1e-12]))
            num_limit_states = compiled.get("num_limit_states", 0)

            # Prepare limit_state_in (always needed for uniform interface)
            # Use shape (n_devices, max(1, num_limit_states)) to satisfy vmap
            n_devices = device_params.shape[0]
            n_lim = max(1, num_limit_states)
            limit_state_in = jnp.zeros((n_devices, n_lim), dtype=get_float_dtype())

            # Call the function with actual inputs to trigger XLA compilation
            # Uniform interface: always pass shared_cache, device_cache, limit_state_in
            try:
                _ = vmapped_split_eval(
                    shared_params,
                    device_params,
                    shared_cache,
                    device_cache,
                    default_simparams,
                    limit_state_in,
                )

                # Block until compilation completes
                jax.block_until_ready(_)

                t1 = time.perf_counter()
                logger.info(f"  {model_type}: XLA compilation complete ({t1 - t0:.2f}s)")
            except Exception as e:
                logger.warning(f"  {model_type}: warmup failed: {e}")
            logger.info(f"{model_type} device model ready")

    def _build_stamp_index_mapping(
        self,
        model_type: str,
        device_contexts: List[Dict],
        ground: int,
    ) -> Dict[str, jax.Array]:
        """Pre-compute index mappings for COO-based stamping.

        Called once per model type during setup. Returns arrays that map
        (device_idx, entry_idx) to global matrix indices.

        Args:
            model_type: OpenVAF model type (e.g., 'psp103')
            device_contexts: List of device context dicts with node_map
            ground: Ground node index

        Returns:
            Dict with:
                res_indices: (n_devices, n_residuals) row indices for f vector, -1 for ground
                jac_row_indices: (n_devices, n_jac_entries) row indices for J
                jac_col_indices: (n_devices, n_jac_entries) col indices for J
        """
        compiled = self._compiled_models.get(model_type)
        if not compiled:
            return {}

        metadata = compiled["dae_metadata"]
        node_names = metadata["node_names"]  # Residual node names
        jacobian_keys = metadata["jacobian_keys"]  # (row_name, col_name) pairs

        n_devices = len(device_contexts)
        n_residuals = len(node_names)
        n_jac_entries = len(jacobian_keys)

        # Build residual index array
        res_indices = np.full((n_devices, n_residuals), -1, dtype=np.int32)

        for dev_idx, ctx in enumerate(device_contexts):
            node_map = ctx["node_map"]
            for res_idx, node_name in enumerate(node_names):
                # Map node name to global index
                # V2 API provides clean names like 'D', 'G', 'S', 'B', 'NOI', etc.
                node_idx = node_map.get(node_name, None)

                if node_idx is not None and node_idx != ground and node_idx > 0:
                    res_indices[dev_idx, res_idx] = node_idx - 1  # 0-indexed residual

        # Build Jacobian index arrays
        jac_row_indices = np.full((n_devices, n_jac_entries), -1, dtype=np.int32)
        jac_col_indices = np.full((n_devices, n_jac_entries), -1, dtype=np.int32)

        for dev_idx, ctx in enumerate(device_contexts):
            node_map = ctx["node_map"]
            for jac_idx, (row_name, col_name) in enumerate(jacobian_keys):
                # Map row/col nodes - V2 API provides clean names
                row_idx = node_map.get(row_name, None)
                col_idx = node_map.get(col_name, None)

                if (
                    row_idx is not None
                    and col_idx is not None
                    and row_idx != ground
                    and col_idx != ground
                    and row_idx > 0
                    and col_idx > 0
                ):
                    jac_row_indices[dev_idx, jac_idx] = row_idx - 1
                    jac_col_indices[dev_idx, jac_idx] = col_idx - 1

        return {
            "res_indices": jnp.array(res_indices),
            "jac_row_indices": jnp.array(jac_row_indices),
            "jac_col_indices": jnp.array(jac_col_indices),
        }

    def _precompute_sparse_structure(
        self,
        n_unknowns: int,
        openvaf_by_type: Dict[str, List[Dict]],
        static_inputs_cache: Dict[str, Tuple],
        source_device_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Pre-compute the merged sparse matrix structure for all devices.

        This is called once during setup. The sparsity pattern is fixed for the
        circuit, so we can pre-compute:
        - Merged COO indices for all devices
        - CSR structure (indptr, indices)
        - Mappings from device outputs to value positions

        Each NR iteration only needs to update values, not rebuild indices.

        Returns:
            Dict with pre-computed structure for efficient NR iterations.
        """
        from jax.experimental.sparse import BCOO, BCSR

        # Collect all COO indices (once, during setup)
        all_j_rows = []
        all_j_cols = []

        # Track positions for each device type's contributions
        openvaf_jac_slices = {}  # model_type -> (start, end) in merged arrays

        pos = 0
        for model_type in openvaf_by_type:
            if model_type not in static_inputs_cache:
                continue

            _, stamp_indices, _, _, _, _ = static_inputs_cache[model_type]
            jac_row_idx = stamp_indices["jac_row_indices"]  # (n_devices, n_jac_entries)
            jac_col_idx = stamp_indices["jac_col_indices"]

            # Flatten and filter valid entries
            flat_rows = jac_row_idx.ravel()
            flat_cols = jac_col_idx.ravel()
            valid = (flat_rows >= 0) & (flat_cols >= 0)

            # Get valid indices as numpy for efficient indexing
            valid_rows = np.asarray(flat_rows[valid])
            valid_cols = np.asarray(flat_cols[valid])

            n_valid = len(valid_rows)
            openvaf_jac_slices[model_type] = (pos, pos + n_valid, valid)
            pos += n_valid

            all_j_rows.append(valid_rows)
            all_j_cols.append(valid_cols)

        # Add source device contributions
        source_jac_slices = {}
        for src_type in ("vsource",):  # isource has no Jacobian
            if src_type not in source_device_data:
                continue
            d = source_device_data[src_type]
            j_rows = d["j_rows"].ravel()
            j_cols = d["j_cols"].ravel()
            valid = j_rows >= 0

            valid_rows = np.asarray(j_rows[valid])
            valid_cols = np.asarray(j_cols[valid])

            n_valid = len(valid_rows)
            source_jac_slices[src_type] = (pos, pos + n_valid, valid)
            pos += n_valid

            all_j_rows.append(valid_rows)
            all_j_cols.append(valid_cols)

        # Add diagonal regularization
        diag_start = pos
        diag_rows = np.arange(n_unknowns, dtype=np.int32)
        all_j_rows.append(diag_rows)
        all_j_cols.append(diag_rows)

        # Merge all indices
        if all_j_rows:
            merged_rows = np.concatenate(all_j_rows)
            merged_cols = np.concatenate(all_j_cols)
        else:
            merged_rows = np.array([], dtype=np.int32)
            merged_cols = np.array([], dtype=np.int32)

        n_entries = len(merged_rows)

        # Build BCOO with dummy values to get structure
        dummy_vals = np.ones(n_entries, dtype=get_float_dtype())
        indices = np.stack([merged_rows, merged_cols], axis=1)

        J_bcoo = BCOO((jnp.array(dummy_vals), jnp.array(indices)), shape=(n_unknowns, n_unknowns))
        J_bcoo_summed = J_bcoo.sum_duplicates()

        # Convert to BCSR to get the fixed structure
        J_bcsr = BCSR.from_bcoo(J_bcoo_summed)

        # Pre-compute COO→CSR permutation for fast assembly
        # This avoids sorting on every NR iteration!
        #
        # Algorithm:
        # 1. Compute linear index: linear = row * n_cols + col
        # 2. Sort by linear index to group duplicates
        # 3. Find unique entries and segment IDs for summing duplicates
        # 4. Sort by row (CSR format) to get final data order
        #
        # In NR loop: csr_data = segment_sum(values[coo_to_csr_perm], csr_segment_ids)

        linear_idx = merged_rows * n_unknowns + merged_cols

        # Sort COO by linear index (groups duplicates together)
        coo_sort_perm = np.argsort(linear_idx)
        sorted_linear = linear_idx[coo_sort_perm]
        merged_rows[coo_sort_perm]

        # Find unique entries and inverse mapping for segment_sum
        unique_linear, coo_to_unique = np.unique(sorted_linear, return_inverse=True)
        n_unique = len(unique_linear)

        # Get row indices for unique entries (for CSR row ordering)
        unique_rows = unique_linear // n_unknowns

        # Sort unique entries by row to get CSR order
        unique_to_csr = np.argsort(unique_rows)
        csr_to_unique = np.argsort(unique_to_csr)  # Inverse permutation

        # Final segment IDs: maps sorted COO values to CSR data positions
        csr_segment_ids = csr_to_unique[coo_to_unique]

        # Store the structure
        return {
            "n_entries": n_entries,
            "n_unknowns": n_unknowns,
            "merged_rows": jnp.array(merged_rows),
            "merged_cols": jnp.array(merged_cols),
            "openvaf_jac_slices": openvaf_jac_slices,
            "source_jac_slices": source_jac_slices,
            "diag_start": diag_start,
            # BCSR structure (fixed)
            "bcsr_indices": J_bcsr.indices,
            "bcsr_indptr": J_bcsr.indptr,
            "bcsr_n_data": J_bcsr.data.shape[0],
            # For sum_duplicates mapping
            "bcoo_summed_indices": J_bcoo_summed.indices,
            # Pre-computed COO→CSR mapping (avoids sorting in NR loop!)
            "coo_sort_perm": jnp.array(coo_sort_perm, dtype=jnp.int32),
            "csr_segment_ids": jnp.array(csr_segment_ids, dtype=jnp.int32),
            "csr_n_segments": n_unique,
        }

    def _parse_voltage_param(
        self, name: str, node_map: Dict[str, int], model_nodes: List[str], ground: int
    ) -> Tuple[int, int]:
        """Parse a voltage parameter name and return (node1_idx, node2_idx).

        Handles formats like "V(GP,SI)" or "V(DI)".
        Returns node indices that can be used directly with voltage array.
        """
        import re

        # Node name mapping for different model types
        # Maps Verilog-A node names to generic node indices (node0, node1, ...)
        # PSP103 node order from PSP103_module.include:
        # External: D(node0), G(node1), S(node2), B(node3)
        # Internal: NOI(node4), GP(node5), SI(node6), DI(node7),
        #           BP(node8), BI(node9), BS(node10), BD(node11)
        internal_name_map = {
            # PSP103 MOSFET nodes (corrected order based on PSP103_module.include)
            "NOI": "node4",
            "GP": "node5",
            "SI": "node6",
            "DI": "node7",
            "BP": "node8",
            "BI": "node9",
            "BS": "node10",
            "BD": "node11",
            "G": "node1",
            "D": "node0",
            "S": "node2",
            "B": "node3",
            # Diode nodes - uppercase (simple diode.va)
            "A": "node0",
            "C": "node1",
            "CI": "node2",
            # Diode nodes - lowercase (sp_diode from SPICE models)
            "a": "node0",
            "c": "node1",
            "a_int": "node2",
        }

        # Simple 2-terminal device mapping (resistor, capacitor, inductor)
        # Used as fallback when device has fewer terminals than the mapped node
        simple_2term_map = {
            "A": "node0",
            "B": "node1",  # resistor/capacitor/inductor terminals
        }

        match = re.match(r"V\(([^,)]+)(?:,([^)]+))?\)", name)
        if not match:
            return (ground, ground)

        node1_name = match.group(1).strip()
        node2_name = match.group(2).strip() if match.group(2) else None

        def resolve_node(name_orig: str) -> int:
            """Resolve a Verilog-A node name to a circuit node index."""
            name = name_orig
            # First try internal_name_map
            if name in internal_name_map:
                mapped = internal_name_map[name]
                # Check if the mapped node exists in node_map
                if mapped in node_map:
                    return node_map[mapped]
                # Fallback to simple 2-terminal mapping
                if name in simple_2term_map:
                    simple_mapped = simple_2term_map[name]
                    if simple_mapped in node_map:
                        return node_map[simple_mapped]
            # Try simple 2-terminal map directly
            elif name in simple_2term_map:
                mapped = simple_2term_map[name]
                if mapped in node_map:
                    return node_map[mapped]
            # Try direct lookup
            return node_map.get(name, node_map.get(name.lower(), ground))

        node1_idx = resolve_node(node1_name)
        node2_idx = resolve_node(node2_name) if node2_name else ground

        return (node1_idx, node2_idx)

    def _stamp_batched_results(
        self,
        model_type: str,
        batch_residuals: jax.Array,
        batch_jacobian: jax.Array,
        device_contexts: List[Dict],
        f: jax.Array,
        J: jax.Array,
        ground: int,
    ) -> Tuple[jax.Array, jax.Array]:
        """Stamp batched evaluation results into system matrices using pure JAX.

        Args:
            model_type: The model type (e.g., 'psp103')
            batch_residuals: Shape (num_devices, num_nodes) residuals
            batch_jacobian: Shape (num_devices, num_jac_entries) jacobian values
            device_contexts: List of context dicts from _prepare_batched_inputs
            f: Residual vector to stamp into (JAX array)
            J: Jacobian matrix to stamp into (JAX array)
            ground: Ground node index

        Returns:
            Updated (f, J) tuple
        """
        compiled = self._compiled_models.get(model_type)
        if not compiled:
            return f, J

        metadata = compiled["dae_metadata"]
        node_names = metadata["node_names"]
        jacobian_keys = metadata["jacobian_keys"]

        for dev_idx, ctx in enumerate(device_contexts):
            node_map = ctx["node_map"]

            # Stamp residuals - V2 API provides clean names like 'D', 'G', 'S'
            for res_idx, node_name in enumerate(node_names):
                node_idx = node_map.get(node_name, None)
                if node_idx is None or node_idx == ground:
                    continue

                if node_idx > 0 and node_idx - 1 < f.shape[0]:
                    resist = batch_residuals[dev_idx, res_idx]
                    resist_safe = jnp.where(jnp.isnan(resist), 0.0, resist)
                    f = f.at[node_idx - 1].add(resist_safe)

            # Stamp Jacobian - V2 API provides clean names
            for jac_idx, (row_name, col_name) in enumerate(jacobian_keys):
                row_idx = node_map.get(row_name, None)
                col_idx = node_map.get(col_name, None)

                if row_idx is None or col_idx is None:
                    continue
                if row_idx == ground or col_idx == ground:
                    continue

                ri = row_idx - 1
                ci = col_idx - 1
                if 0 <= ri < f.shape[0] and 0 <= ci < J.shape[1]:
                    resist = batch_jacobian[dev_idx, jac_idx]
                    resist_safe = jnp.where(jnp.isnan(resist), 0.0, resist)
                    J = J.at[ri, ci].add(resist_safe)

        return f, J

    def _extract_analysis_params(self):
        """Extract analysis parameters from parsed control block.

        Uses the structured ControlBlock from Circuit if available,
        falls back to regex parsing for backwards compatibility.

        Options from the netlist are loaded into self.options (SimulationOptions).
        Analysis directive parameters (step, stop, etc.) are also stored in self.options.
        The legacy self.analysis_params dict is kept for backwards compatibility.
        """
        # Reset options to defaults
        self.options = SimulationOptions()

        # Set VACASK default for tran_method
        self.options.tran_method = IntegrationMethod.TRAPEZOIDAL

        # Legacy analysis_params dict for backwards compatibility
        self.analysis_params = {
            "type": "tran",
        }

        # Try to use the parsed control block first
        if self.circuit and self.circuit.control:
            control = self.circuit.control

            # Extract options using unified SimulationOptions
            if control.options:
                opts = control.options.params

                # Handle tran_method specially (needs get_method_from_options)
                self.options.tran_method = get_method_from_options(opts)

                # Update all other options from netlist
                self.options.update_from_netlist(opts, self.parse_spice_number)

                logger.debug(f"Loaded options: {self.options.to_dict()}")

            # Extract analysis parameters from first tran analysis
            for analysis in control.analyses:
                if analysis.analysis_type == "tran":
                    params = analysis.params
                    if "step" in params:
                        self.options.step = self.parse_spice_number(params["step"])
                    if "stop" in params:
                        self.options.stop = self.parse_spice_number(params["stop"])
                    if "maxstep" in params:
                        self.options.maxstep = self.parse_spice_number(params["maxstep"])
                    if "icmode" in params:
                        icmode = params["icmode"]
                        if isinstance(icmode, str):
                            icmode = icmode.strip("\"'")
                        self.options.icmode = icmode
                    break  # Use first tran analysis found

            # Populate legacy analysis_params for backwards compatibility
            self._sync_options_to_analysis_params()
            logger.debug(f"Analysis (from control block): {self.analysis_params}")
            return

        # Fallback: regex parsing for old-style netlists
        text = self.sim_path.read_text()

        # Find tran analysis line
        match = re.search(r"analysis\s+\w+\s+tran\s+([^\n]+)", text)
        if match:
            params_str = match.group(1)
            # Parse individual parameters - they can be in any order
            step_match = re.search(r"step=(\S+)", params_str)
            stop_match = re.search(r"stop=(\S+)", params_str)
            maxstep_match = re.search(r"maxstep=(\S+)", params_str)
            icmode_match = re.search(r'icmode="(\w+)"', params_str)

            if step_match:
                self.options.step = self.parse_spice_number(step_match.group(1))
            if stop_match:
                self.options.stop = self.parse_spice_number(stop_match.group(1))
            if maxstep_match:
                self.options.maxstep = self.parse_spice_number(maxstep_match.group(1))
            if icmode_match:
                self.options.icmode = icmode_match.group(1)

        # Try to extract tran_method from options line
        tran_method_match = re.search(r'tran_method\s*=\s*"?(\w+)"?', text)
        if tran_method_match:
            try:
                self.options.tran_method = IntegrationMethod.from_string(tran_method_match.group(1))
            except ValueError:
                pass  # Keep default

        # Populate legacy analysis_params for backwards compatibility
        self._sync_options_to_analysis_params()
        logger.debug(f"Analysis (from regex): {self.analysis_params}")

    def _sync_options_to_analysis_params(self):
        """Sync SimulationOptions to legacy analysis_params dict for backwards compatibility."""
        opts = self.options

        # Always sync these
        self.analysis_params["tran_method"] = opts.tran_method
        self.analysis_params["icmode"] = opts.icmode

        # Only sync if set (not None)
        if opts.step is not None:
            self.analysis_params["step"] = opts.step
        if opts.stop is not None:
            self.analysis_params["stop"] = opts.stop
        if opts.maxstep is not None:
            self.analysis_params["maxstep"] = opts.maxstep

        # Sync tolerance/control options
        self.analysis_params["tran_lteratio"] = opts.tran_lteratio
        self.analysis_params["tran_redofactor"] = opts.tran_redofactor
        self.analysis_params["nr_convtol"] = opts.nr_convtol
        self.analysis_params["tran_gshunt"] = opts.tran_gshunt
        self.analysis_params["reltol"] = opts.reltol
        self.analysis_params["abstol"] = opts.abstol
        self.analysis_params["tran_fs"] = opts.tran_fs
        self.analysis_params["tran_minpts"] = opts.tran_minpts

    def _build_source_fn(self):
        """Build time-varying source function from device parameters."""
        return build_source_fn(self.devices, self.parse_spice_number)

    @profile
    def run_transient(
        self,
        t_stop: Optional[float] = None,
        dt: Optional[float] = None,
        max_steps: int = 1000000,
        use_sparse: Optional[bool] = None,
        backend: Optional[str] = None,
        use_scan: bool = False,
        use_while_loop: bool = True,
        profile_config: Optional["ProfileConfig"] = None,
        temperature: float = DEFAULT_TEMPERATURE_K,
        adaptive_config: Optional["AdaptiveConfig"] = None,
        checkpoint_interval: Optional[int] = None,
    ) -> TransientResult:
        """Run transient analysis using full Modified Nodal Analysis.

        All computation is JIT-compiled. Automatically uses sparse matrices
        for large circuits (>1000 nodes).

        Uses full MNA with explicit branch currents for voltage sources, providing:
        - Better numerical conditioning (no G=1e12 high-G approximation)
        - More accurate current extraction (branch currents are primary unknowns)
        - Smoother dI/dt transitions

        Args:
            t_stop: Stop time (default: from analysis params or 1ms)
            dt: Time step (default: from analysis params or 1µs). For adaptive mode,
                this is the initial timestep which will be adjusted automatically.
            max_steps: Maximum number of time steps (default: 1M)
            use_sparse: Force sparse (True) or dense (False) solver. If None, auto-detect.
            backend: 'gpu', 'cpu', or None (auto-select based on circuit size).
                     For circuits >500 nodes with GPU available, uses GPU acceleration.
            use_scan: DEPRECATED - ignored
            use_while_loop: DEPRECATED - ignored
            profile_config: If provided, profile just the core simulation (not setup)
            temperature: Simulation temperature in Kelvin (default: 300.15K = 27°C)
            adaptive_config: Configuration for adaptive timestep control. If None,
                             uses default AdaptiveConfig with LTE-based timestep adjustment.
            checkpoint_interval: If set, use GPU memory checkpointing with this many
                steps per buffer. Results are periodically copied to CPU to avoid
                GPU OOM on large circuits. Recommended for circuits with many nodes
                (>1000) and long simulations (>10000 steps). Example: 10000.

        Returns:
            TransientResult with times, voltages, and stats
        """
        # Update simulation temperature if changed (invalidates cached static inputs)
        if temperature != self._simulation_temperature:
            self._simulation_temperature = temperature
            # Sync options.temp (Celsius) from internal Kelvin representation
            self.options.temp = temperature - 273.15
            self._transient_setup_cache = None
            self._transient_setup_key = None
            logger.info(f"Temperature changed to {temperature}K ({temperature - 273.15:.1f}°C)")

        # Emit deprecation warnings for ignored parameters
        if use_scan:
            warnings.warn(
                "use_scan parameter is deprecated and ignored. "
                "Use adaptive_config for timestep control.",
                DeprecationWarning,
                stacklevel=2,
            )
        if use_while_loop is not True:  # Only warn if explicitly set to False
            warnings.warn(
                "use_while_loop parameter is deprecated and ignored. "
                "Use adaptive_config for timestep control.",
                DeprecationWarning,
                stacklevel=2,
            )

        from jax_spice.analysis.gpu_backend import select_backend

        if t_stop is None:
            t_stop = self.analysis_params.get("stop", 1e-3)
        if dt is None:
            dt = self.analysis_params.get("step", 1e-6)

        # Limit number of steps if max_steps is specified
        num_steps = int(t_stop / dt)
        if num_steps > max_steps:
            dt = t_stop / max_steps
            logger.info(f"Limiting to {max_steps} steps, dt={dt:.2e}s")

        # Select backend if not specified
        if backend is None or backend == "auto":
            backend = select_backend(self.num_nodes)

        logger.info(f"Running transient: t_stop={t_stop:.2e}s, dt={dt:.2e}s, backend={backend}")

        # All non-source devices use OpenVAF
        if not self._has_openvaf_devices:
            # Only vsource/isource - trivial circuit
            logger.warning("No OpenVAF devices found - circuit only has sources")

        # Default to dense solver - sparse must be explicitly requested
        if use_sparse is None:
            use_sparse = False

        from jax_spice.analysis.transient import AdaptiveConfig, FullMNAStrategy, extract_results

        # Build config from analysis_params or use provided config
        if adaptive_config is None:
            kwargs = {}

            # LTE options
            if "tran_lteratio" in self.analysis_params:
                kwargs["lte_ratio"] = float(self.analysis_params["tran_lteratio"])
            if "tran_redofactor" in self.analysis_params:
                kwargs["redo_factor"] = float(self.analysis_params["tran_redofactor"])

            # NR options
            if "nr_convtol" in self.analysis_params:
                kwargs["nr_convtol"] = float(self.analysis_params["nr_convtol"])

            # GSHUNT options
            if "tran_gshunt" in self.analysis_params:
                kwargs["gshunt_init"] = float(self.analysis_params["tran_gshunt"])

            # Tolerance options
            if "reltol" in self.analysis_params:
                kwargs["reltol"] = float(self.analysis_params["reltol"])
            if "abstol" in self.analysis_params:
                kwargs["abstol"] = float(self.analysis_params["abstol"])

            # Timestep control options
            if "tran_fs" in self.analysis_params:
                kwargs["tran_fs"] = float(self.analysis_params["tran_fs"])
            if "tran_minpts" in self.analysis_params:
                kwargs["tran_minpts"] = int(self.analysis_params["tran_minpts"])
            if "maxstep" in self.analysis_params:
                kwargs["max_dt"] = float(self.analysis_params["maxstep"])

            # Integration method
            if "tran_method" in self.analysis_params:
                kwargs["integration_method"] = self.analysis_params["tran_method"]

            config = AdaptiveConfig(**kwargs)
        else:
            config = adaptive_config

        # Cache strategy instance for JIT reuse across calls
        # Key includes all parameters that affect strategy construction
        # max_steps is now part of the key since it affects JIT compilation
        cache_key = (
            use_sparse,
            backend,
            max_steps,
            config.lte_ratio,
            config.redo_factor,
            config.reltol,
            config.abstol,
            config.min_dt,
            config.max_dt,
            config.nr_convtol,
            config.gshunt_init,
            config.gshunt_steps,
            config.gshunt_target,
            config.integration_method,
            config.tran_fs,
            config.tran_minpts,
        )

        if not hasattr(self, "_full_mna_strategy_cache"):
            self._full_mna_strategy_cache = {}

        if cache_key not in self._full_mna_strategy_cache:
            # LRU eviction: limit cache size to prevent unbounded growth
            MAX_STRATEGY_CACHE_SIZE = 8
            if len(self._full_mna_strategy_cache) >= MAX_STRATEGY_CACHE_SIZE:
                oldest_key = next(iter(self._full_mna_strategy_cache))
                del self._full_mna_strategy_cache[oldest_key]
                logger.debug("Evicted oldest strategy cache entry")

            logger.info(
                f"Using FullMNAStrategy ({self.num_nodes} nodes, "
                f"{'sparse' if use_sparse else 'dense'}, "
                f"lte_ratio={config.lte_ratio}, redo_factor={config.redo_factor})"
            )
            strategy = FullMNAStrategy(
                self, use_sparse=use_sparse, backend=backend, config=config, max_steps=max_steps
            )
            self._full_mna_strategy_cache[cache_key] = strategy
        else:
            strategy = self._full_mna_strategy_cache[cache_key]
            logger.debug("Reusing cached FullMNAStrategy")

        times_full, V_out, stats = strategy.run(t_stop, dt, checkpoint_interval)

        # Extract sliced numpy results for TransientResult
        times_np, voltages, currents = extract_results(times_full, V_out, stats)

        return TransientResult(
            times=jnp.asarray(times_np),
            voltages={k: jnp.asarray(v) for k, v in voltages.items()},
            currents={k: jnp.asarray(v) for k, v in currents.items()},
            stats=stats,
        )

    # =========================================================================
    # Node Collapse Implementation
    # =========================================================================
    #
    # JAX-SPICE vs VACASK Node Count Comparison:
    #
    # VACASK reports two metrics via 'print stats':
    #   - "Number of nodes" = nodeCount() = all Node objects in nodeMap
    #   - "Number of unknowns" = unknownCount() = system size after collapse
    #
    # Key difference:
    #   - VACASK creates Node objects for ALL internal nodes, even collapsed ones.
    #     After collapse, multiple Node objects share the same unknownIndex.
    #   - JAX-SPICE doesn't create objects for collapsed internal nodes at all.
    #     We directly allocate circuit nodes only for non-collapsed internals.
    #
    # Comparison for c6288 benchmark (PSP103 with all resistance params = 0):
    #   VACASK nodeCount():    ~86,000 (5,123 external + 81k internal Node objects)
    #   VACASK unknownCount(): ~15,234 (actual system matrix size after collapse)
    #   JAX-SPICE total_nodes: ~15,235 (directly matches unknownCount + 1 for ground)
    #
    # JAX-SPICE's approach is more memory-efficient: we don't allocate internal
    # node objects that would just be collapsed anyway. Our total_nodes from
    # _setup_internal_nodes() should match VACASK's unknownCount() + 1.
    #
    # The +1 difference is because VACASK's unknownCount excludes ground (index 0),
    # while JAX-SPICE counts ground as node 0 in the total.
    # =========================================================================

    def _compute_collapse_roots(
        self, collapsible_pairs: List[Tuple[int, int]], n_nodes: int
    ) -> Dict[int, int]:
        """Compute the collapse root for each node using union-find.

        Collapsible pairs (a, b) mean nodes a and b should be the same electrical node.
        We use union-find to compute equivalence classes, preferring external nodes
        (indices 0-3) as roots.

        Args:
            collapsible_pairs: List of (node1, node2) pairs that should collapse
            n_nodes: Total number of model nodes

        Returns:
            Dict mapping each node index to its root (representative) node index
        """
        # Initialize parent array (each node is its own parent)
        parent = list(range(n_nodes))

        def find(x: int) -> int:
            if parent[x] != x:
                parent[x] = find(parent[x])  # Path compression
            return parent[x]

        def union(x: int, y: int) -> None:
            px, py = find(x), find(y)
            if px != py:
                # Prefer external nodes (0-3) as root
                if py < 4:
                    parent[px] = py
                elif px < 4:
                    parent[py] = px
                else:
                    parent[py] = px

        # Apply collapse pairs
        for a, b in collapsible_pairs:
            if b != 4294967295:  # u32::MAX = collapse to ground (handled separately)
                if a < n_nodes and b < n_nodes:
                    union(a, b)

        # Build root mapping for all nodes
        return {i: find(i) for i in range(n_nodes)}

    def _setup_internal_nodes(self) -> Tuple[int, Dict[str, Dict[str, int]]]:
        """Set up internal nodes for OpenVAF devices with node collapse support.

        Node collapse eliminates unnecessary internal nodes when model parameters
        indicate they should be merged (e.g., when resistance parameters are 0).

        Uses precomputed collapse decisions from _compute_early_collapse_decisions()
        which calls OpenVAF's init_fn for each device to determine collapse behavior.

        Returns:
            (total_nodes, device_internal_nodes) where device_internal_nodes maps
            device name to dict of internal node name -> global index
        """
        n_external = self.num_nodes
        next_internal = n_external
        device_internal_nodes = {}

        # Cache collapse roots for devices with identical collapse patterns
        collapse_roots_cache: Dict[Tuple[Tuple[int, int], ...], Dict[int, int]] = {}

        for dev in self.devices:
            if not dev.get("is_openvaf"):
                continue

            model_type = dev["model"]
            compiled = self._compiled_models.get(model_type)
            if not compiled:
                continue

            model_nodes = compiled["nodes"]
            n_model_nodes = len(model_nodes)

            # Get precomputed collapse pairs from _compute_early_collapse_decisions()
            # This uses OpenVAF's generic collapse mechanism
            device_name = dev["name"]
            collapse_pairs = self._device_collapse_decisions.get(device_name, [])

            # Cache collapse roots by pattern (most devices of same type will share)
            pairs_key = tuple(sorted(collapse_pairs))
            if pairs_key not in collapse_roots_cache:
                collapse_roots_cache[pairs_key] = self._compute_collapse_roots(
                    collapse_pairs, n_model_nodes
                )
            collapse_roots = collapse_roots_cache[pairs_key]

            # Include all internal nodes including branch currents
            # Branch currents (names like 'br[Branch(BranchId(N))]') are system unknowns
            # VACASK counts branch currents as unknowns, and we must match for node counts
            n_internal_end = n_model_nodes

            # Map external nodes to device's external circuit nodes
            # Number of external terminals is determined by the device instance
            ext_nodes = dev["nodes"]
            n_ext_terminals = len(ext_nodes)
            ext_node_map = {}
            for i in range(n_ext_terminals):
                ext_node_map[i] = ext_nodes[i]

            # Build node mapping using collapse roots
            # Track which internal root nodes need circuit node allocation
            internal_root_to_circuit: Dict[int, int] = {}
            node_mapping: Dict[int, int] = {}

            # Internal nodes start after external terminals
            for i in range(n_ext_terminals, n_internal_end):
                root = collapse_roots.get(i, i)

                if root < n_ext_terminals:
                    # Root is an external node - use its circuit node
                    node_mapping[i] = ext_node_map[root]
                else:
                    # Root is internal - need to allocate/reuse a circuit node
                    if root not in internal_root_to_circuit:
                        internal_root_to_circuit[root] = next_internal
                        next_internal += 1
                    node_mapping[i] = internal_root_to_circuit[root]

            # Build internal_map: model node name -> circuit node index
            internal_map = {}
            for i in range(n_ext_terminals, n_internal_end):
                node_name = model_nodes[i]
                internal_map[node_name] = node_mapping[i]

            device_internal_nodes[dev["name"]] = internal_map

        if device_internal_nodes:
            n_internal = next_internal - n_external
            logger.info(
                f"Allocated {n_internal} internal nodes for {len(device_internal_nodes)} OpenVAF devices"
            )

        return next_internal, device_internal_nodes

    def _build_transient_setup(self, backend: str = "cpu", use_dense: bool = True) -> Dict:
        """Build and cache transient setup data without creating solver.

        Builds all the setup data needed for transient analysis (device compilation,
        node mapping, source functions) without creating a solver or running simulation.

        Args:
            backend: 'gpu' or 'cpu' for device evaluation
            use_dense: If True, use dense matrices; if False, use sparse

        Returns:
            Dict with setup data: n_total, device_internal_nodes, n_unknowns,
            source_fn, openvaf_by_type, vmapped_fns, static_inputs_cache,
            source_device_data
        """
        from jax_spice.analysis.gpu_backend import get_default_dtype, get_device

        ground = 0
        device = get_device(backend)
        dtype = get_default_dtype(backend)

        # Create transient setup cache key (topology-based)
        setup_cache_key = f"{self.num_nodes}_{len(self.devices)}_{use_dense}_{backend}"

        # Check if we have cached transient setup data
        if self._transient_setup_cache is not None and self._transient_setup_key == setup_cache_key:
            logger.info("Reusing cached transient setup")
            return self._transient_setup_cache

        # Build all setup data (first time or after topology change)
        logger.info("Building transient setup...")

        # Set up internal nodes for OpenVAF devices
        n_total, device_internal_nodes = self._setup_internal_nodes()
        n_unknowns = n_total - 1

        # Build time-varying source function
        source_fn = self._build_source_fn()

        # Group devices by type
        openvaf_by_type: Dict[str, List[Dict]] = {}
        source_devices = []
        for dev in self.devices:
            if dev.get("is_openvaf"):
                model_type = dev["model"]
                if model_type not in openvaf_by_type:
                    openvaf_by_type[model_type] = []
                openvaf_by_type[model_type].append(dev)
            elif dev["model"] in ("vsource", "isource"):
                source_devices.append(dev)

        logger.debug(f"{len(source_devices)} source devices")

        # Prepare static inputs and stamp index mappings
        vmapped_fns: Dict[str, Callable] = {}  # Legacy - kept for API compatibility
        static_inputs_cache: Dict[str, Tuple[Any, List[int], List[Dict], Dict]] = {}

        for model_type in openvaf_by_type:
            compiled = self._compiled_models.get(model_type)
            if compiled and "dae_metadata" in compiled:
                logger.debug(f"{model_type} already compiled")
                voltage_indices, device_contexts, cache, collapse_decisions = (
                    self._prepare_static_inputs(
                        model_type, openvaf_by_type[model_type], device_internal_nodes, ground
                    )
                )
                stamp_indices = self._build_stamp_index_mapping(model_type, device_contexts, ground)
                voltage_node1 = jnp.array(
                    [[n1 for n1, n2 in ctx["voltage_node_pairs"]] for ctx in device_contexts],
                    dtype=jnp.int32,
                )
                voltage_node2 = jnp.array(
                    [[n2 for n1, n2 in ctx["voltage_node_pairs"]] for ctx in device_contexts],
                    dtype=jnp.int32,
                )

                if backend == "gpu":
                    with jax.default_device(device):
                        cache = jnp.array(cache, dtype=dtype)
                else:
                    cache = jnp.array(cache, dtype=get_float_dtype())

                static_inputs_cache[model_type] = (
                    voltage_indices,
                    stamp_indices,
                    voltage_node1,
                    voltage_node2,
                    cache,
                    collapse_decisions,
                )
                n_devs = len(openvaf_by_type[model_type])
                logger.info(f"Prepared {model_type}: {n_devs} devices, cache_size={cache.shape[1]}")

        # Pre-compute source device stamp indices
        source_device_data = self._prepare_source_devices_coo(source_devices, ground, n_unknowns)

        # Cache setup data for reuse
        self._transient_setup_cache = {
            "n_total": n_total,
            "device_internal_nodes": device_internal_nodes,
            "n_unknowns": n_unknowns,
            "source_fn": source_fn,
            "openvaf_by_type": openvaf_by_type,
            "vmapped_fns": vmapped_fns,
            "static_inputs_cache": static_inputs_cache,
            "source_device_data": source_device_data,
        }
        self._transient_setup_key = setup_cache_key
        logger.info("Cached transient setup for reuse")

        # Warm up device models
        self.warmup_device_models(static_inputs_cache)

        return self._transient_setup_cache

    def _get_dc_source_values(
        self, n_vsources: int, n_isources: int
    ) -> Tuple[jax.Array, jax.Array]:
        """Extract DC values from voltage and current sources."""
        return get_dc_source_values(self.devices, n_vsources, n_isources)

    def _get_vdd_value(self) -> float:
        """Find the maximum DC voltage from voltage sources (VDD)."""
        return get_vdd_value(self.devices)

    def _compute_dc_operating_point(
        self,
        n_nodes: int,
        n_vsources: int,
        n_isources: int,
        nr_solve: Callable,
        device_arrays: Dict[str, jax.Array],
        backend: str = "cpu",
        use_dense: bool = True,
        max_iterations: int = 100,
        device_internal_nodes: Optional[Dict[str, Dict[str, int]]] = None,
        source_device_data: Optional[Dict[str, Any]] = None,
        vmapped_fns: Optional[Dict[str, Callable]] = None,
        static_inputs_cache: Optional[Dict[str, Tuple]] = None,
    ) -> jax.Array:
        """Compute DC operating point using VACASK-style homotopy chain.

        Delegates to dc_operating_point.compute_dc_operating_point().
        """
        # Get DC source values
        vsource_dc_vals, isource_dc_vals = self._get_dc_source_values(n_vsources, n_isources)

        return _compute_dc_op_impl(
            n_nodes=n_nodes,
            node_names=self.node_names,
            devices=self.devices,
            nr_solve=nr_solve,
            device_arrays=device_arrays,
            vsource_dc_vals=vsource_dc_vals,
            isource_dc_vals=isource_dc_vals,
            options=self.options,
            vdd_value=self._get_vdd_value(),
            device_internal_nodes=device_internal_nodes,
        )

    def _get_source_fn_for_device(self, dev: Dict):
        """Get the source function for a device, or None if not a source."""
        return get_source_fn_for_device(dev, self.parse_spice_number)

    def _prepare_source_devices_coo(
        self,
        source_devices: List[Dict],
        ground: int,
        n_unknowns: int,
    ) -> Dict[str, Any]:
        """Pre-compute data structures and stamp templates for source devices."""
        return prepare_source_devices_coo(source_devices, ground, n_unknowns)

    def _collect_source_devices_coo(
        self,
        device_data: Dict[str, Any],
        V: jax.Array,
        vsource_vals: jax.Array,
        isource_vals: jax.Array,
        f_indices: List,
        f_values: List,
        j_rows: List,
        j_cols: List,
        j_vals: List,
    ):
        """Collect COO triplets from source devices using fully vectorized operations."""
        collect_source_devices_coo(
            device_data, V, vsource_vals, isource_vals,
            f_indices, f_values, j_rows, j_cols, j_vals
        )

    def _collect_openvaf_coo(
        self,
        batch_residuals: jax.Array,
        batch_jacobian: jax.Array,
        stamp_indices: Dict[str, jax.Array],
        f_indices: List,
        f_values: List,
        j_rows: List,
        j_cols: List,
        j_vals: List,
    ):
        """Collect COO triplets from OpenVAF batched results using pre-computed indices."""
        res_idx = stamp_indices["res_indices"]  # (n_devices, n_residuals)
        jac_row_idx = stamp_indices["jac_row_indices"]  # (n_devices, n_jac_entries)
        jac_col_idx = stamp_indices["jac_col_indices"]

        # Flatten and filter residuals
        flat_res_idx = res_idx.ravel()
        flat_res_val = batch_residuals.ravel()
        valid_res = flat_res_idx >= 0

        # Handle NaN values
        flat_res_val = jnp.where(jnp.isnan(flat_res_val), 0.0, flat_res_val)

        f_indices.append(flat_res_idx[valid_res])
        f_values.append(flat_res_val[valid_res])

        # Flatten and filter Jacobian
        flat_jac_rows = jac_row_idx.ravel()
        flat_jac_cols = jac_col_idx.ravel()
        flat_jac_vals = batch_jacobian.ravel()
        valid_jac = (flat_jac_rows >= 0) & (flat_jac_cols >= 0)

        # Handle NaN values
        flat_jac_vals = jnp.where(jnp.isnan(flat_jac_vals), 0.0, flat_jac_vals)

        j_rows.append(flat_jac_rows[valid_jac])
        j_cols.append(flat_jac_cols[valid_jac])
        j_vals.append(flat_jac_vals[valid_jac])

    def _make_full_mna_build_system_fn(
        self,
        source_device_data: Dict,
        vmapped_fns: Dict,
        static_inputs_cache: Dict,
        n_unknowns: int,
        use_dense: bool = True,
    ) -> Tuple[Callable, Dict, int]:
        """Create GPU-resident build_system function for full MNA formulation.

        Uses true Modified Nodal Analysis with branch currents as explicit unknowns,
        providing more accurate current extraction than the high-G (G=1e12) approximation.

        Full MNA augments the system from n×n to (n+m)×(n+m) where m = number
        of voltage sources. The branch currents become primary unknowns:

            ┌───────────────┐   ┌───┐   ┌───────┐
            │  G + c0*C   B │   │ V │   │ f_node│
            │               │ × │   │ = │       │
            │    B^T      0 │   │ J │   │ E - V │
            └───────────────┘   └───┘   └───────┘

        Where:
        - G = device conductance matrix (n×n)
        - C = device capacitance matrix (n×n)
        - B = incidence matrix mapping currents to nodes (n×m)
        - V = node voltages (n×1)
        - J = branch currents (m×1) - these are the primary unknowns for vsources
        - f_node = device current contributions
        - E = voltage source values (m×1)

        Benefits over high-G approximation:
        - More accurate current extraction (no numerical noise from G=1e12)
        - Smoother dI/dt transitions matching VACASK reference
        - Better conditioned matrices for ill-conditioned circuits

        Args:
            source_device_data: Pre-computed source device stamp templates
            vmapped_fns: Dict of vmapped OpenVAF functions per model type
            static_inputs_cache: Dict of static inputs per model type
            n_unknowns: Number of node voltage unknowns (n_total - 1)
            use_dense: Whether to use dense or sparse matrix assembly

        Returns:
            Tuple of:
            - build_system function with signature:
                build_system(X, vsource_vals, isource_vals, Q_prev, integ_c0, device_arrays,
                             gmin, gshunt, ...) -> (J, f, Q, I_vsource)
              where X = [V; I_branch] is the augmented solution vector
            - device_arrays: Dict[model_type, cache] to pass to build_system
        """
        # Get number of voltage sources for augmentation
        n_vsources = len(source_device_data.get("vsource", {}).get("names", []))
        n_augmented = n_unknowns + n_vsources

        # Capture model types as static list (unrolled at trace time)
        model_types = list(static_inputs_cache.keys())

        # Split cache into metadata (captured) and arrays (passed as argument)
        static_metadata = {}
        device_arrays = {}
        split_eval_info = {}
        for model_type in model_types:
            voltage_indices, stamp_indices, voltage_node1, voltage_node2, cache, _ = (
                static_inputs_cache[model_type]
            )
            static_metadata[model_type] = (
                voltage_indices,
                stamp_indices,
                voltage_node1,
                voltage_node2,
            )

            compiled = self._compiled_models.get(model_type, {})
            n_devices = compiled["device_params"].shape[0]
            num_limit_states = compiled.get("num_limit_states", 0)
            split_eval_info[model_type] = {
                "vmapped_split_eval": compiled["vmapped_split_eval"],
                "shared_params": compiled["shared_params"],
                "device_params": compiled["device_params"],
                "voltage_positions": compiled["voltage_positions_in_varying"],
                "shared_cache": compiled["shared_cache"],
                "default_simparams": compiled.get(
                    "default_simparams", jnp.array([0.0, 1.0, 1e-12])
                ),
                # Device-level limiting info
                "use_device_limiting": compiled.get("use_device_limiting", False),
                "num_limit_states": num_limit_states,
                "n_devices": n_devices,
            }
            device_arrays[model_type] = compiled["device_cache"]

        # Pre-compute vsource node indices as JAX arrays (captured in closure)
        if n_vsources > 0:
            vsource_node_p = jnp.array(source_device_data["vsource"]["node_p"], dtype=jnp.int32)
            vsource_node_n = jnp.array(source_device_data["vsource"]["node_n"], dtype=jnp.int32)
        else:
            vsource_node_p = jnp.zeros(0, dtype=jnp.int32)
            vsource_node_n = jnp.zeros(0, dtype=jnp.int32)

        # Capture options.gmin for diagonal regularization
        # Use at least 1e-6 for dense, 1e-4 for sparse (GPU sparse solvers are more sensitive)
        # Also respect larger netlist GMIN for ill-conditioned circuits
        min_floor = 1e-4 if not use_dense else 1e-6
        min_diag_reg = max(min_floor, self.options.gmin)

        # Compute total limit state size and offsets per model type
        # limit_state is a flat array: [model0_device0_states, model0_device1_states, ..., model1_device0_states, ...]
        total_limit_states = 0
        limit_state_offsets = {}  # model_type -> (start_offset, n_devices, num_limit_states)
        for model_type in model_types:
            if model_type in split_eval_info:
                info = split_eval_info[model_type]
                if info.get("use_device_limiting", False) and info.get("num_limit_states", 0) > 0:
                    n_devices = info["n_devices"]
                    num_limit_states = info["num_limit_states"]
                    limit_state_offsets[model_type] = (
                        total_limit_states,
                        n_devices,
                        num_limit_states,
                    )
                    total_limit_states += n_devices * num_limit_states

        def build_system_full_mna(
            X: jax.Array,  # Augmented solution: [V; I_branch] of size n_total + n_vsources
            vsource_vals: jax.Array,
            isource_vals: jax.Array,
            Q_prev: jax.Array,
            integ_c0: float | jax.Array,
            device_arrays_arg: Dict[str, jax.Array],
            gmin: float | jax.Array = 1e-12,
            gshunt: float | jax.Array = 0.0,
            integ_c1: float | jax.Array = 0.0,
            integ_d1: float | jax.Array = 0.0,
            dQdt_prev: jax.Array | None = None,
            integ_c2: float | jax.Array = 0.0,
            Q_prev2: jax.Array | None = None,
            limit_state_in: jax.Array | None = None,  # Flat array of all limit states
        ) -> Tuple[Any, jax.Array, jax.Array, jax.Array, jax.Array]:
            """Build augmented Jacobian J and residual f for full MNA.

            The solution vector X has structure: [V1, V2, ..., Vn, I_vs1, I_vs2, ..., I_vsm]
            where V are node voltages (ground excluded) and I_vs are branch currents.

            Args:
                X: Augmented solution vector of size n_total + n_vsources
                   X[:n_total] = node voltages (including ground at index 0)
                   X[n_total:] = branch currents for voltage sources
                vsource_vals: Voltage source target values
                isource_vals: Current source values
                Q_prev: Charges from previous timestep
                integ_c0: Integration coefficient for current charges
                device_arrays_arg: Device cache arrays
                gmin, gshunt: Regularization parameters
                integ_c1, integ_d1, dQdt_prev, integ_c2, Q_prev2: Integration history

            Returns:
                J: Augmented Jacobian matrix of size (n_unknowns+n_vsources) × (n_unknowns+n_vsources)
                f: Augmented residual vector of size (n_unknowns+n_vsources)
                Q: Current charges (size n_unknowns)
                I_vsource: Branch currents (extracted directly from X)
            """
            # Extract voltage and current parts from augmented solution
            # X has structure: [V_ground=0, V_1, ..., V_n, I_vs1, ..., I_vsm]
            n_total = n_unknowns + 1  # Total nodes including ground
            V = X[:n_total]
            I_branch = X[n_total:] if n_vsources > 0 else jnp.zeros(0, dtype=get_float_dtype())

            # =====================================================================
            # Device contributions (same as high-G version, but without vsource stamps)
            # =====================================================================
            f_resist_parts = []
            f_react_parts = []
            j_resist_parts = []
            j_react_parts = []
            lim_rhs_resist_parts = []
            lim_rhs_react_parts = []
            # Pre-allocate limit_state_out with static size (filled via slice assignments)
            limit_state_out = jnp.zeros(total_limit_states, dtype=get_float_dtype())

            # Current sources (residual only, no Jacobian)
            if "isource" in source_device_data and isource_vals.size > 0:
                d = source_device_data["isource"]
                f_vals = isource_vals[:, None] * jnp.array([1.0, -1.0])[None, :]
                f_idx = d["f_indices"].ravel()
                f_val = f_vals.ravel()
                f_valid = f_idx >= 0
                f_resist_parts.append(
                    (jnp.where(f_valid, f_idx, 0), jnp.where(f_valid, f_val, 0.0))
                )

            # OpenVAF devices (same as before)
            for model_type in model_types:
                voltage_indices, stamp_indices, voltage_node1, voltage_node2 = static_metadata[
                    model_type
                ]
                cache = device_arrays_arg[model_type]

                voltage_updates = V[voltage_node1] - V[voltage_node2]

                uses_analysis = self._compiled_models.get(model_type, {}).get(
                    "uses_analysis", False
                )
                uses_simparam_gmin = self._compiled_models.get(model_type, {}).get(
                    "uses_simparam_gmin", False
                )

                split_info = split_eval_info[model_type]
                shared_params = split_info["shared_params"]
                device_params = split_info["device_params"]
                voltage_positions = split_info["voltage_positions"]
                shared_cache = split_info["shared_cache"]

                device_params_updated = device_params.at[:, voltage_positions].set(voltage_updates)

                if uses_analysis:
                    analysis_type_val = jnp.where(integ_c0 > 0, 2.0, 0.0)
                    device_params_updated = device_params_updated.at[:, -2].set(analysis_type_val)
                    device_params_updated = device_params_updated.at[:, -1].set(gmin)
                elif uses_simparam_gmin:
                    device_params_updated = device_params_updated.at[:, -1].set(gmin)

                vmapped_split_eval = split_info["vmapped_split_eval"]
                default_simparams = split_info["default_simparams"]
                use_device_limiting = split_info.get("use_device_limiting", False)
                num_limit_states = split_info.get("num_limit_states", 0)

                analysis_type_val = jnp.where(integ_c0 > 0, 2.0, 0.0)
                simparams = default_simparams.at[0].set(analysis_type_val).at[2].set(gmin)

                # Get limit_state slice for this model type (always needed for uniform interface)
                # vmapped function expects limit_state_in argument
                n_dev = split_info["n_devices"]
                n_lim = max(1, num_limit_states)
                if (
                    use_device_limiting
                    and num_limit_states > 0
                    and model_type in limit_state_offsets
                    and limit_state_in is not None
                ):
                    offset, _, n_lim = limit_state_offsets[model_type]
                    # Extract and reshape: flat -> (n_devices, num_limit_states)
                    model_limit_state_in = limit_state_in[offset : offset + n_dev * n_lim].reshape(
                        n_dev, n_lim
                    )
                else:
                    # Use zeros - model has no limits or no valid input
                    model_limit_state_in = jnp.zeros((n_dev, n_lim), dtype=get_float_dtype())

                # Uniform interface: always pass shared_cache, device_cache, limit_state_in
                (
                    batch_res_resist,
                    batch_res_react,
                    batch_jac_resist,
                    batch_jac_react,
                    batch_lim_rhs_resist,
                    batch_lim_rhs_react,
                    _,
                    _,
                    batch_limit_state_out,
                ) = vmapped_split_eval(
                    shared_params,
                    device_params_updated,
                    shared_cache,
                    cache,
                    simparams,
                    model_limit_state_in,
                )

                # Store limit_state_out at pre-computed offset (static slice assignment)
                if use_device_limiting and model_type in limit_state_offsets:
                    offset, _, n_lim = limit_state_offsets[model_type]
                    limit_state_out = limit_state_out.at[offset : offset + n_dev * n_lim].set(
                        batch_limit_state_out.ravel()
                    )

                res_idx = stamp_indices["res_indices"]
                jac_row_idx = stamp_indices["jac_row_indices"]
                jac_col_idx = stamp_indices["jac_col_indices"]

                flat_res_idx = res_idx.ravel()
                flat_res_resist_val = batch_res_resist.ravel()
                valid_res = flat_res_idx >= 0
                flat_res_idx_masked = jnp.where(valid_res, flat_res_idx, 0)
                flat_res_resist_masked = jnp.where(valid_res, flat_res_resist_val, 0.0)
                flat_res_resist_masked = jnp.where(
                    jnp.isnan(flat_res_resist_masked), 0.0, flat_res_resist_masked
                )
                f_resist_parts.append((flat_res_idx_masked, flat_res_resist_masked))

                flat_res_react_val = batch_res_react.ravel()
                flat_res_react_masked = jnp.where(valid_res, flat_res_react_val, 0.0)
                flat_res_react_masked = jnp.where(
                    jnp.isnan(flat_res_react_masked), 0.0, flat_res_react_masked
                )
                f_react_parts.append((flat_res_idx_masked, flat_res_react_masked))

                flat_jac_rows = jac_row_idx.ravel()
                flat_jac_cols = jac_col_idx.ravel()
                flat_jac_resist_vals = batch_jac_resist.ravel()
                valid_jac = (flat_jac_rows >= 0) & (flat_jac_cols >= 0)
                flat_jac_rows_masked = jnp.where(valid_jac, flat_jac_rows, 0)
                flat_jac_cols_masked = jnp.where(valid_jac, flat_jac_cols, 0)
                flat_jac_resist_masked = jnp.where(valid_jac, flat_jac_resist_vals, 0.0)
                flat_jac_resist_masked = jnp.where(
                    jnp.isnan(flat_jac_resist_masked), 0.0, flat_jac_resist_masked
                )
                j_resist_parts.append(
                    (flat_jac_rows_masked, flat_jac_cols_masked, flat_jac_resist_masked)
                )

                flat_jac_react_vals = batch_jac_react.ravel()
                flat_jac_react_masked = jnp.where(valid_jac, flat_jac_react_vals, 0.0)
                flat_jac_react_masked = jnp.where(
                    jnp.isnan(flat_jac_react_masked), 0.0, flat_jac_react_masked
                )
                j_react_parts.append(
                    (flat_jac_rows_masked, flat_jac_cols_masked, flat_jac_react_masked)
                )

                flat_lim_rhs_resist_val = batch_lim_rhs_resist.ravel()
                flat_lim_rhs_resist_masked = jnp.where(valid_res, flat_lim_rhs_resist_val, 0.0)
                flat_lim_rhs_resist_masked = jnp.where(
                    jnp.isnan(flat_lim_rhs_resist_masked), 0.0, flat_lim_rhs_resist_masked
                )
                lim_rhs_resist_parts.append((flat_res_idx_masked, flat_lim_rhs_resist_masked))

                flat_lim_rhs_react_val = batch_lim_rhs_react.ravel()
                flat_lim_rhs_react_masked = jnp.where(valid_res, flat_lim_rhs_react_val, 0.0)
                flat_lim_rhs_react_masked = jnp.where(
                    jnp.isnan(flat_lim_rhs_react_masked), 0.0, flat_lim_rhs_react_masked
                )
                lim_rhs_react_parts.append((flat_res_idx_masked, flat_lim_rhs_react_masked))

            # Build device contribution vectors (size n_unknowns)
            if f_resist_parts:
                all_f_resist_idx = jnp.concatenate([p[0] for p in f_resist_parts])
                all_f_resist_val = jnp.concatenate([p[1] for p in f_resist_parts])
                f_resist = jax.ops.segment_sum(
                    all_f_resist_val, all_f_resist_idx, num_segments=n_unknowns
                )
            else:
                f_resist = jnp.zeros(n_unknowns, dtype=get_float_dtype())

            if f_react_parts:
                all_f_react_idx = jnp.concatenate([p[0] for p in f_react_parts])
                all_f_react_val = jnp.concatenate([p[1] for p in f_react_parts])
                Q = jax.ops.segment_sum(all_f_react_val, all_f_react_idx, num_segments=n_unknowns)
            else:
                Q = jnp.zeros(n_unknowns, dtype=get_float_dtype())

            if lim_rhs_resist_parts:
                all_lim_rhs_resist_idx = jnp.concatenate([p[0] for p in lim_rhs_resist_parts])
                all_lim_rhs_resist_val = jnp.concatenate([p[1] for p in lim_rhs_resist_parts])
                lim_rhs_resist = jax.ops.segment_sum(
                    all_lim_rhs_resist_val, all_lim_rhs_resist_idx, num_segments=n_unknowns
                )
            else:
                lim_rhs_resist = jnp.zeros(n_unknowns, dtype=get_float_dtype())

            if lim_rhs_react_parts:
                all_lim_rhs_react_idx = jnp.concatenate([p[0] for p in lim_rhs_react_parts])
                all_lim_rhs_react_val = jnp.concatenate([p[1] for p in lim_rhs_react_parts])
                lim_rhs_react = jax.ops.segment_sum(
                    all_lim_rhs_react_val, all_lim_rhs_react_idx, num_segments=n_unknowns
                )
            else:
                lim_rhs_react = jnp.zeros(n_unknowns, dtype=get_float_dtype())

            f_resist = f_resist - lim_rhs_resist

            # =====================================================================
            # Compute I_vsource from device residuals via KCL (before adding I_branch)
            # This provides the correct initial current for UIC mode where I_branch=0
            # =====================================================================
            if n_vsources > 0 and "vsource" in source_device_data:
                # Extract device contribution at vsource positive nodes (0-indexed in MNA)
                vsource_node_p_mna = vsource_node_p - 1
                # Handle ground (index 0) - if positive terminal is ground, contribution is 0
                valid_nodes = vsource_node_p > 0
                f_device_at_p = jnp.where(valid_nodes, f_resist[vsource_node_p_mna], 0.0)
                # Vsource current = -device_contribution (by KCL: sum of currents = 0)
                I_vsource_kcl = -f_device_at_p
            else:
                I_vsource_kcl = jnp.zeros(0, dtype=get_float_dtype())

            _dQdt_prev = (
                dQdt_prev
                if dQdt_prev is not None
                else jnp.zeros(n_unknowns, dtype=get_float_dtype())
            )
            _Q_prev2 = (
                Q_prev2 if Q_prev2 is not None else jnp.zeros(n_unknowns, dtype=get_float_dtype())
            )

            # =====================================================================
            # Full MNA: Add branch current contribution to KCL at vsource nodes
            # =====================================================================
            # For each vsource i connecting nodes p and n:
            # - At node p: add +I_branch[i] to residual
            # - At node n: add -I_branch[i] to residual
            if n_vsources > 0:
                # Build B @ I_branch contribution to node residuals
                # B has shape (n_unknowns, n_vsources) with ±1 at (p-1, i) and (n-1, i)
                # We use scatter to add I_branch to the appropriate nodes

                # Positive terminals (add +I_branch)
                p_mna = vsource_node_p - 1  # Convert to 0-indexed MNA
                valid_p = vsource_node_p > 0  # Ground (0) doesn't contribute

                # Negative terminals (add -I_branch)
                n_mna = vsource_node_n - 1
                valid_n = vsource_node_n > 0

                # Create index and value arrays for both terminals
                all_b_idx = jnp.concatenate(
                    [jnp.where(valid_p, p_mna, 0), jnp.where(valid_n, n_mna, 0)]
                )
                all_b_val = jnp.concatenate(
                    [jnp.where(valid_p, I_branch, 0.0), jnp.where(valid_n, -I_branch, 0.0)]
                )

                # Add B @ I_branch to node residuals
                f_branch_contrib = jax.ops.segment_sum(
                    all_b_val, all_b_idx, num_segments=n_unknowns
                )
                f_resist = f_resist + f_branch_contrib

            # Combine for transient
            f_node = (
                f_resist
                + integ_c0 * (Q - lim_rhs_react)
                + integ_c1 * Q_prev
                + integ_d1 * _dQdt_prev
                + integ_c2 * _Q_prev2
            )
            f_node = f_node + gshunt * V[1:]

            # =====================================================================
            # Voltage source equations (rows n_unknowns to n_augmented-1)
            # For each vsource i: V_p - V_n - E_i = 0
            # =====================================================================
            if n_vsources > 0:
                Vp = V[vsource_node_p]
                Vn = V[vsource_node_n]
                f_branch = Vp - Vn - vsource_vals
            else:
                f_branch = jnp.zeros(0, dtype=get_float_dtype())

            # Combine node and branch residuals
            f_augmented = jnp.concatenate([f_node, f_branch])

            # =====================================================================
            # Build augmented Jacobian
            # =====================================================================
            # Device contributions to upper-left G block
            if j_resist_parts:
                all_j_resist_rows = jnp.concatenate([p[0] for p in j_resist_parts])
                all_j_resist_cols = jnp.concatenate([p[1] for p in j_resist_parts])
                all_j_resist_vals = jnp.concatenate([p[2] for p in j_resist_parts])
            else:
                all_j_resist_rows = jnp.zeros(0, dtype=jnp.int32)
                all_j_resist_cols = jnp.zeros(0, dtype=jnp.int32)
                all_j_resist_vals = jnp.zeros(0, dtype=get_float_dtype())

            if j_react_parts:
                all_j_react_rows = jnp.concatenate([p[0] for p in j_react_parts])
                all_j_react_cols = jnp.concatenate([p[1] for p in j_react_parts])
                all_j_react_vals = jnp.concatenate([p[2] for p in j_react_parts])
            else:
                all_j_react_rows = jnp.zeros(0, dtype=jnp.int32)
                all_j_react_cols = jnp.zeros(0, dtype=jnp.int32)
                all_j_react_vals = jnp.zeros(0, dtype=get_float_dtype())

            # Combine G and c0*C contributions
            all_j_rows = jnp.concatenate([all_j_resist_rows, all_j_react_rows])
            all_j_cols = jnp.concatenate([all_j_resist_cols, all_j_react_cols])
            all_j_vals = jnp.concatenate([all_j_resist_vals, integ_c0 * all_j_react_vals])

            if n_vsources > 0:
                # =====================================================================
                # B block: df_node/dI_branch = B (incidence matrix)
                # At node p: df/dI = +1
                # At node n: df/dI = -1
                # =====================================================================
                valid_p = vsource_node_p > 0
                valid_n = vsource_node_n > 0

                branch_indices = jnp.arange(n_vsources, dtype=jnp.int32)

                # B block entries
                b_rows_p = jnp.where(valid_p, vsource_node_p - 1, 0)  # node index (0-indexed)
                b_cols_p = jnp.where(valid_p, n_unknowns + branch_indices, 0)  # branch column
                b_vals_p = jnp.where(valid_p, 1.0, 0.0)

                b_rows_n = jnp.where(valid_n, vsource_node_n - 1, 0)
                b_cols_n = jnp.where(valid_n, n_unknowns + branch_indices, 0)
                b_vals_n = jnp.where(valid_n, -1.0, 0.0)

                # =====================================================================
                # B^T block: df_branch/dV = B^T
                # For vsource i: df_i/dV_p = +1, df_i/dV_n = -1
                # =====================================================================
                bt_rows_p = jnp.where(valid_p, n_unknowns + branch_indices, 0)  # branch row
                bt_cols_p = jnp.where(valid_p, vsource_node_p - 1, 0)  # node column
                bt_vals_p = jnp.where(valid_p, 1.0, 0.0)

                bt_rows_n = jnp.where(valid_n, n_unknowns + branch_indices, 0)
                bt_cols_n = jnp.where(valid_n, vsource_node_n - 1, 0)
                bt_vals_n = jnp.where(valid_n, -1.0, 0.0)

                # Append B and B^T entries
                all_j_rows = jnp.concatenate([all_j_rows, b_rows_p, b_rows_n, bt_rows_p, bt_rows_n])
                all_j_cols = jnp.concatenate([all_j_cols, b_cols_p, b_cols_n, bt_cols_p, bt_cols_n])
                all_j_vals = jnp.concatenate([all_j_vals, b_vals_p, b_vals_n, bt_vals_p, bt_vals_n])

            if use_dense:
                # Dense: COO -> dense matrix via segment_sum
                flat_indices = all_j_rows * n_augmented + all_j_cols
                J_flat = jax.ops.segment_sum(
                    all_j_vals, flat_indices, num_segments=n_augmented * n_augmented
                )
                J = J_flat.reshape((n_augmented, n_augmented))
                # Add regularization to node equations (upper-left block)
                # Note: branch equations should NOT have diagonal regularization (they're exact)
                # min_diag_reg is at least 1e-6, but uses netlist GMIN if larger.
                diag_reg = jnp.concatenate(
                    [jnp.full(n_unknowns, min_diag_reg + gshunt), jnp.zeros(n_vsources)]
                )
                J = J + jnp.diag(diag_reg)
            else:
                # Sparse path - build BCOO sparse matrix
                from jax.experimental.sparse import BCOO

                # Add diagonal regularization for node equations only
                # min_diag_reg is at least 1e-6, but uses netlist GMIN if larger.
                diag_idx = jnp.arange(n_unknowns, dtype=jnp.int32)
                all_j_rows = jnp.concatenate([all_j_rows, diag_idx])
                all_j_cols = jnp.concatenate([all_j_cols, diag_idx])
                all_j_vals = jnp.concatenate(
                    [all_j_vals, jnp.full(n_unknowns, min_diag_reg + gshunt)]
                )

                indices = jnp.stack([all_j_rows, all_j_cols], axis=1)
                J = BCOO((all_j_vals, indices), shape=(n_augmented, n_augmented))

            # Use KCL-computed current (always correct) instead of I_branch from X
            # This is critical for UIC mode where I_branch starts at 0
            I_vsource = I_vsource_kcl

            # limit_state_out was pre-allocated and filled via slice assignments
            return J, f_augmented, Q, I_vsource, limit_state_out

        return build_system_full_mna, device_arrays, total_limit_states

    def _compute_voltage_param(
        self, name: str, V: jax.Array, node_map: Dict[str, int], model_nodes: List[str], ground: int
    ) -> float:
        """Compute a voltage parameter value from node voltages.

        Handles formats like:
        - "V(GP,SI)" - voltage between two nodes
        - "V(DI)" - voltage of a single node (vs ground)
        - "V(node0,node2)" - external terminal voltages
        """
        import re

        # Parse V(node1) or V(node1,node2) format
        match = re.match(r"V\(([^,)]+)(?:,([^)]+))?\)", name)
        if not match:
            return 0.0

        node1_name = match.group(1).strip()
        node2_name = match.group(2).strip() if match.group(2) else None

        # Map node names to indices
        # PSP103 node order from PSP103_module.include:
        # External: D(node0), G(node1), S(node2), B(node3)
        # Internal: NOI(node4), GP(node5), SI(node6), DI(node7),
        #           BP(node8), BI(node9), BS(node10), BD(node11)
        internal_name_map = {
            "NOI": "node4",
            "GP": "node5",
            "SI": "node6",
            "DI": "node7",
            "BP": "node8",
            "BI": "node9",
            "BS": "node10",
            "BD": "node11",
            "G": "node1",
            "D": "node0",
            "S": "node2",
            "B": "node3",
        }

        # Resolve node1
        if node1_name in internal_name_map:
            node1_name = internal_name_map[node1_name]
        node1_idx = node_map.get(node1_name, None)
        if node1_idx is None:
            # Try lowercase
            node1_idx = node_map.get(node1_name.lower(), ground)

        # Resolve node2
        if node2_name:
            if node2_name in internal_name_map:
                node2_name = internal_name_map[node2_name]
            node2_idx = node_map.get(node2_name, None)
            if node2_idx is None:
                node2_idx = node_map.get(node2_name.lower(), ground)
        else:
            node2_idx = ground

        # Compute voltage
        v1 = V[node1_idx] if node1_idx < len(V) else 0.0
        v2 = V[node2_idx] if node2_idx < len(V) else 0.0
        return v1 - v2

    # =========================================================================
    # AC (Small-Signal) Analysis
    # =========================================================================

    def run_ac(
        self,
        freq_start: float = 1.0,
        freq_stop: float = 1e6,
        mode: str = "dec",
        points: int = 10,
        step: Optional[float] = None,
        values: Optional[List[float]] = None,
    ) -> "ACResult":
        """Run AC (small-signal) frequency sweep analysis.

        AC analysis linearizes the circuit around its DC operating point and
        computes the frequency response. The algorithm:
        1. Compute DC operating point
        2. Extract Jr (resistive Jacobian) and Jc (reactive/capacitance Jacobian)
        3. For each frequency f: solve (Jr + j*omega*Jc) * X = U

        Args:
            freq_start: Starting frequency in Hz (default 1.0)
            freq_stop: Ending frequency in Hz (default 1e6)
            mode: Sweep mode - 'lin', 'dec', 'oct', or 'list'
            points: Points per decade/octave (for 'dec'/'oct' modes)
            step: Frequency step for 'lin' mode
            values: Explicit frequency list for 'list' mode

        Returns:
            ACResult with frequencies and complex voltage phasors
        """
        from jax_spice.analysis.ac import ACConfig, run_ac_analysis

        logger.info(f"Running AC analysis: {freq_start:.2e} to {freq_stop:.2e} Hz, mode={mode}")

        # Configure AC analysis
        config = ACConfig(
            freq_start=freq_start,
            freq_stop=freq_stop,
            mode=mode,
            points=points,
            step=step,
            values=values,
        )

        # First, compute DC operating point
        Jr, Jc, V_dc, ac_sources = self._compute_ac_operating_point()

        logger.info(f"DC operating point found, extracting {len(ac_sources)} AC sources")

        # Run AC frequency sweep
        result = run_ac_analysis(
            Jr=Jr,
            Jc=Jc,
            ac_sources=ac_sources,
            config=config,
            node_names=self.node_names,
            dc_voltages=V_dc,
        )

        logger.info(f"AC analysis complete: {len(result.frequencies)} frequency points")
        return result

    def _compute_ac_operating_point(
        self,
    ) -> Tuple[Array, Array, Array, List[Dict]]:
        """Compute DC operating point and extract Jr, Jc for AC analysis.

        Returns:
            Tuple of:
            - Jr: Resistive Jacobian at DC operating point, shape (n, n)
            - Jc: Reactive (capacitance) Jacobian at DC OP, shape (n, n)
            - V_dc: DC operating point voltages, shape (n,)
            - ac_sources: List of AC source specifications
        """
        # Compile OpenVAF models if not already done
        if not self._compiled_models:
            self._compile_openvaf_models()

        # Reuse transient setup infrastructure
        setup = self._build_transient_setup(backend="cpu", use_dense=True)
        n_total = setup["n_total"]
        n_unknowns = setup["n_unknowns"]
        device_internal_nodes = setup["device_internal_nodes"]
        source_device_data = setup["source_device_data"]
        vmapped_fns = setup["vmapped_fns"]
        static_inputs_cache = setup["static_inputs_cache"]

        # Count sources
        n_vsources = len(source_device_data.get("vsource", {}).get("names", []))
        n_isources = len(source_device_data.get("isource", {}).get("names", []))

        # Get DC source values
        vsource_dc_vals, isource_dc_vals = self._get_dc_source_values(n_vsources, n_isources)

        # Create full MNA NR solver for DC operating point
        # Uses true MNA with branch currents as explicit unknowns
        build_system_fn, device_arrays, total_limit_states = self._make_full_mna_build_system_fn(
            source_device_data, vmapped_fns, static_inputs_cache, n_unknowns, use_dense=True
        )
        build_system_jit = jax.jit(build_system_fn)

        # Collect NOI node indices (PSP103 noise correlation internal node)
        noi_indices = []
        if device_internal_nodes:
            for dev_name, internal_nodes in device_internal_nodes.items():
                if "node4" in internal_nodes:  # NOI is node4 in PSP103
                    noi_indices.append(internal_nodes["node4"])
        noi_indices = jnp.array(noi_indices, dtype=jnp.int32) if noi_indices else None

        nr_solve = make_dense_full_mna_solver(
            build_system_jit,
            n_total,
            n_vsources,
            noi_indices=noi_indices,
            max_iterations=self.options.op_itl,
            abstol=self.options.abstol,
            max_step=1.0,
            total_limit_states=total_limit_states,
            options=self.options,
        )

        # Initialize X (augmented: [V, I_branch])
        vdd_value = self._get_vdd_value() or 1.0  # Default to 1.0 if no vsources
        mid_rail = vdd_value / 2.0
        X_init = jnp.zeros(n_total + n_vsources, dtype=get_float_dtype())
        X_init = X_init.at[1:n_total].set(mid_rail)  # Node voltages (skip ground)

        Q_prev = jnp.zeros(n_unknowns, dtype=get_float_dtype())

        # First try direct NR without homotopy
        logger.info("  AC DC: Trying direct NR solver first...")
        X_new, nr_iters, is_converged, max_f, _, _, _, _ = nr_solve(
            X_init,
            vsource_dc_vals,
            isource_dc_vals,
            Q_prev,
            0.0,
            device_arrays,  # integ_c0=0 for DC
        )

        if is_converged:
            V_dc = X_new[:n_total]  # Extract voltage portion
            logger.info(
                f"  AC DC operating point converged via direct NR "
                f"({nr_iters} iters, residual={max_f:.2e})"
            )
        else:
            # Fall back to homotopy chain using the cached NR solver
            logger.info("  AC DC: Direct NR failed, trying homotopy chain...")

            # Configure homotopy from SimulationOptions
            homotopy_config = HomotopyConfig(
                gmin=self.options.gmin,
                gdev_start=self.options.homotopy_startgmin,
                gdev_target=self.options.homotopy_mingmin,
                gmin_factor=self.options.homotopy_gminfactor,
                gmin_factor_min=self.options.homotopy_mingminfactor,
                gmin_factor_max=self.options.homotopy_maxgminfactor,
                gmin_max=self.options.homotopy_maxgmin,
                gmin_max_steps=self.options.homotopy_gminsteps,
                source_step=self.options.homotopy_srcstep,
                source_step_min=self.options.homotopy_minsrcstep,
                source_scale=self.options.homotopy_srcscale,
                source_max_steps=self.options.homotopy_srcsteps,
                chain=self.options.op_homotopy,
                max_iterations=self.options.op_itlcont,
                abstol=self.options.abstol,
                debug=0,
            )

            # Homotopy uses augmented X (includes branch currents)
            result = run_homotopy_chain(
                nr_solve,
                X_init,
                vsource_dc_vals,
                isource_dc_vals,
                Q_prev,
                device_arrays,
                homotopy_config,
            )

            V_dc = result.V[:n_total]  # Extract voltage portion
            logger.info(
                f"  AC DC operating point: {result.method} "
                f"({result.iterations} iterations, converged={result.converged})"
            )

        # Now extract Jr and Jc at the DC operating point
        Jr, Jc = self._extract_ac_jacobians(
            V_dc,
            vmapped_fns,
            static_inputs_cache,
            source_device_data,
            n_unknowns,
            vsource_dc_vals,
            isource_dc_vals,
        )

        # Extract AC source specifications
        ac_sources = self._extract_ac_sources()

        return Jr, Jc, V_dc, ac_sources

    def _extract_ac_jacobians(
        self,
        V: Array,
        vmapped_fns: Dict[str, Callable],
        static_inputs_cache: Dict[str, Tuple],
        source_device_data: Dict[str, Any],
        n_unknowns: int,
        vsource_dc_vals: Array,
        isource_dc_vals: Array,
    ) -> Tuple[Array, Array]:
        """Extract resistive and reactive Jacobians at given operating point.

        Args:
            V: Voltage vector at operating point
            vmapped_fns: Dict of vmapped OpenVAF functions per model type
            static_inputs_cache: Dict of static inputs per model type
            source_device_data: Pre-computed source device stamp templates
            n_unknowns: Number of unknowns
            vsource_dc_vals: DC voltage source values
            isource_dc_vals: DC current source values

        Returns:
            Tuple of (Jr, Jc) dense Jacobian matrices
        """
        model_types = list(static_inputs_cache.keys())

        j_resist_parts = []
        j_react_parts = []

        # === Source device contributions (resistive only) ===
        if "vsource" in source_device_data and vsource_dc_vals.size > 0:
            d = source_device_data["vsource"]
            G = 1e12
            G_arr = jnp.full(d["n"], G)
            j_vals_arr = G_arr[:, None] * d["j_signs"][None, :]
            j_row = d["j_rows"].ravel()
            j_col = d["j_cols"].ravel()
            j_val = j_vals_arr.ravel()
            j_valid = j_row >= 0
            j_resist_parts.append(
                (
                    jnp.where(j_valid, j_row, 0),
                    jnp.where(j_valid, j_col, 0),
                    jnp.where(j_valid, j_val, 0.0),
                )
            )

        # === OpenVAF device contributions ===
        for model_type in model_types:
            voltage_indices, stamp_indices, voltage_node1, voltage_node2, cache, _ = (
                static_inputs_cache[model_type]
            )

            # Compute device voltages
            voltage_updates = V[voltage_node1] - V[voltage_node2]

            # All OpenVAF models use split eval with uniform interface
            compiled = self._compiled_models[model_type]
            uses_analysis = compiled.get("uses_analysis", False)
            uses_simparam_gmin = compiled.get("uses_simparam_gmin", False)

            shared_params = compiled["shared_params"]
            device_params = compiled["device_params"]
            voltage_positions = compiled["voltage_positions_in_varying"]
            vmapped_split_eval = compiled["vmapped_split_eval"]
            shared_cache = compiled["shared_cache"]
            default_simparams = compiled.get("default_simparams", jnp.array([0.0, 1.0, 1e-12]))

            # Update voltage columns in device_params
            device_params_updated = device_params.at[:, voltage_positions].set(voltage_updates)

            # For AC analysis: analysis_type = 1
            if uses_analysis:
                device_params_updated = device_params_updated.at[:, -2].set(1.0)
                device_params_updated = device_params_updated.at[:, -1].set(1e-12)
            elif uses_simparam_gmin:
                device_params_updated = device_params_updated.at[:, -1].set(1e-12)

            # Build simparams array with AC analysis values
            simparams = default_simparams.at[0].set(1.0).at[2].set(1e-12)  # AC=1, gmin=1e-12

            # Prepare limit_state_in (always needed for uniform interface)
            num_limit_states = compiled.get("num_limit_states", 0)
            n_devices = device_params.shape[0]
            n_lim = max(1, num_limit_states)
            model_limit_state_in = jnp.zeros((n_devices, n_lim), dtype=get_float_dtype())

            # Uniform interface: always pass shared_cache, device_cache (cache), limit_state_in
            _, _, batch_jac_resist, batch_jac_react, _, _, _, _, _ = vmapped_split_eval(
                shared_params,
                device_params_updated,
                shared_cache,
                cache,
                simparams,
                model_limit_state_in,
            )

            # Extract Jacobian entries
            jac_row_idx = stamp_indices["jac_row_indices"]
            jac_col_idx = stamp_indices["jac_col_indices"]

            flat_jac_rows = jac_row_idx.ravel()
            flat_jac_cols = jac_col_idx.ravel()
            valid_jac = (flat_jac_rows >= 0) & (flat_jac_cols >= 0)

            # Resistive Jacobian
            flat_jac_resist_vals = batch_jac_resist.ravel()
            flat_jac_resist_masked = jnp.where(valid_jac, flat_jac_resist_vals, 0.0)
            flat_jac_resist_masked = jnp.where(
                jnp.isnan(flat_jac_resist_masked), 0.0, flat_jac_resist_masked
            )
            j_resist_parts.append(
                (
                    jnp.where(valid_jac, flat_jac_rows, 0),
                    jnp.where(valid_jac, flat_jac_cols, 0),
                    flat_jac_resist_masked,
                )
            )

            # Reactive Jacobian
            flat_jac_react_vals = batch_jac_react.ravel()
            flat_jac_react_masked = jnp.where(valid_jac, flat_jac_react_vals, 0.0)
            flat_jac_react_masked = jnp.where(
                jnp.isnan(flat_jac_react_masked), 0.0, flat_jac_react_masked
            )
            j_react_parts.append(
                (
                    jnp.where(valid_jac, flat_jac_rows, 0),
                    jnp.where(valid_jac, flat_jac_cols, 0),
                    flat_jac_react_masked,
                )
            )

        # === Assemble dense Jacobians ===
        # Resistive Jacobian Jr
        if j_resist_parts:
            all_j_resist_rows = jnp.concatenate([p[0] for p in j_resist_parts])
            all_j_resist_cols = jnp.concatenate([p[1] for p in j_resist_parts])
            all_j_resist_vals = jnp.concatenate([p[2] for p in j_resist_parts])
            flat_indices = all_j_resist_rows * n_unknowns + all_j_resist_cols
            Jr_flat = jax.ops.segment_sum(
                all_j_resist_vals, flat_indices, num_segments=n_unknowns * n_unknowns
            )
            Jr = Jr_flat.reshape((n_unknowns, n_unknowns))
        else:
            Jr = jnp.zeros((n_unknowns, n_unknowns), dtype=get_float_dtype())

        # Reactive Jacobian Jc
        if j_react_parts:
            all_j_react_rows = jnp.concatenate([p[0] for p in j_react_parts])
            all_j_react_cols = jnp.concatenate([p[1] for p in j_react_parts])
            all_j_react_vals = jnp.concatenate([p[2] for p in j_react_parts])
            flat_indices = all_j_react_rows * n_unknowns + all_j_react_cols
            Jc_flat = jax.ops.segment_sum(
                all_j_react_vals, flat_indices, num_segments=n_unknowns * n_unknowns
            )
            Jc = Jc_flat.reshape((n_unknowns, n_unknowns))
        else:
            Jc = jnp.zeros((n_unknowns, n_unknowns), dtype=get_float_dtype())

        # Add regularization
        Jr = Jr + 1e-12 * jnp.eye(n_unknowns, dtype=get_float_dtype())

        return Jr, Jc

    def _extract_ac_sources(self) -> List[Dict]:
        """Extract AC source specifications from devices."""
        from jax_spice.analysis.ac import extract_ac_sources

        return extract_ac_sources(self.devices)

    # =========================================================================
    # Transfer Function Analyses (DCINC, DCXF, ACXF)
    # =========================================================================

    def run_dcinc(self) -> "DCIncResult":
        """Run DC incremental (small-signal) analysis.

        DC incremental analysis computes the small-signal response to
        incremental source excitations. Sources with 'mag' parameters
        are used as excitations.

        Algorithm:
        1. Compute DC operating point
        2. Extract resistive Jacobian Jr
        3. Build excitation vector from source 'mag' values
        4. Solve: Jr·dx = du for incremental voltages

        Returns:
            DCIncResult with incremental node voltages
        """
        from jax_spice.analysis.xfer import build_dcinc_excitation, solve_dcinc

        logger.info("Running DCINC analysis")

        # Compute DC operating point and extract Jacobians
        Jr, Jc, V_dc, _ = self._compute_ac_operating_point()

        # Extract sources with mag parameters
        sources = self._extract_all_sources()

        # Build excitation vector
        n_unknowns = Jr.shape[0]
        excitation = build_dcinc_excitation(sources, n_unknowns)

        # Solve
        result = solve_dcinc(
            Jr=Jr,
            excitation=excitation,
            node_names=self.node_names,
            dc_voltages=V_dc,
        )

        logger.info(f"DCINC complete: {len(result.incremental_voltages)} nodes")
        return result

    def run_dcxf(
        self,
        out: Union[str, int] = 1,
    ) -> "DCXFResult":
        """Run DC transfer function analysis.

        For each independent source, computes:
        - tf: Transfer function (Vout / Vsrc)
        - zin: Input impedance seen by source
        - yin: Input admittance (1/zin)

        Args:
            out: Output node (name or index)

        Returns:
            DCXFResult with tf, zin, yin for each source
        """
        from jax_spice.analysis.xfer import solve_dcxf

        logger.info(f"Running DCXF analysis, output node: {out}")

        # Compute DC operating point and extract Jacobians
        Jr, Jc, V_dc, _ = self._compute_ac_operating_point()

        # Extract all sources
        sources = self._extract_all_sources()

        # Solve
        result = solve_dcxf(
            Jr=Jr,
            sources=sources,
            out_node=out,
            node_names=self.node_names,
            dc_voltages=V_dc,
        )

        logger.info(f"DCXF complete: {len(result.tf)} sources analyzed")
        return result

    def run_acxf(
        self,
        out: Union[str, int] = 1,
        freq_start: float = 1.0,
        freq_stop: float = 1e6,
        mode: str = "dec",
        points: int = 10,
        step: Optional[float] = None,
        values: Optional[List[float]] = None,
    ) -> "ACXFResult":
        """Run AC transfer function analysis.

        For each independent source, computes complex-valued:
        - tf: Transfer function over frequency
        - zin: Input impedance over frequency
        - yin: Input admittance over frequency

        Args:
            out: Output node (name or index)
            freq_start: Starting frequency in Hz
            freq_stop: Ending frequency in Hz
            mode: Sweep mode - 'lin', 'dec', 'oct', or 'list'
            points: Points per decade/octave
            step: Frequency step for 'lin' mode
            values: Explicit frequency list for 'list' mode

        Returns:
            ACXFResult with complex tf, zin, yin over frequency
        """
        from jax_spice.analysis.xfer import ACXFConfig, solve_acxf

        logger.info(f"Running ACXF analysis: {freq_start:.2e} to {freq_stop:.2e} Hz, out={out}")

        # Configure ACXF
        config = ACXFConfig(
            out=out,
            freq_start=freq_start,
            freq_stop=freq_stop,
            mode=mode,
            points=points,
            step=step,
            values=values,
        )

        # Compute DC operating point and extract Jacobians
        Jr, Jc, V_dc, _ = self._compute_ac_operating_point()

        # Extract all sources
        sources = self._extract_all_sources()

        # Solve
        result = solve_acxf(
            Jr=Jr,
            Jc=Jc,
            sources=sources,
            config=config,
            node_names=self.node_names,
            dc_voltages=V_dc,
        )

        logger.info(
            f"ACXF complete: {len(result.frequencies)} frequencies, {len(result.tf)} sources"
        )
        return result

    def _extract_all_sources(self) -> List[Dict]:
        """Extract all independent sources for transfer function analysis."""
        from jax_spice.analysis.xfer import extract_all_sources

        return extract_all_sources(self.devices)

    # =========================================================================
    # Noise Analysis
    # =========================================================================

    def run_noise(
        self,
        out: Union[str, int] = 1,
        input_source: str = "",
        freq_start: float = 1.0,
        freq_stop: float = 1e6,
        mode: str = "dec",
        points: int = 10,
        step: Optional[float] = None,
        values: Optional[List[float]] = None,
        temperature: float = DEFAULT_TEMPERATURE_K,
    ) -> "NoiseResult":
        """Run noise analysis.

        Computes output noise power spectral density over frequency sweep.
        For each frequency, sums noise contributions from all devices
        (resistor thermal, diode shot/flicker, etc.) propagated through
        the circuit's small-signal transfer function.

        Args:
            out: Output node (name or index)
            input_source: Name of input source for power gain calculation
            freq_start: Starting frequency in Hz
            freq_stop: Ending frequency in Hz
            mode: Sweep mode - 'lin', 'dec', 'oct', or 'list'
            points: Points per decade/octave
            step: Frequency step for 'lin' mode
            values: Explicit frequency list for 'list' mode
            temperature: Circuit temperature in Kelvin (default 27°C)

        Returns:
            NoiseResult with output noise, power gain, and per-device contributions
        """
        # Update simulation temperature if changed
        if temperature != self._simulation_temperature:
            self._simulation_temperature = temperature
            self.options.temp = temperature - 273.15
            self._transient_setup_cache = None
            self._transient_setup_key = None

        from jax_spice.analysis.noise import (
            NoiseConfig,
            extract_noise_sources,
            run_noise_analysis,
        )

        logger.info(f"Running noise analysis: {freq_start:.2e} to {freq_stop:.2e} Hz, out={out}")

        # Configure noise analysis
        config = NoiseConfig(
            out=out,
            input_source=input_source,
            freq_start=freq_start,
            freq_stop=freq_stop,
            mode=mode,
            points=points,
            step=step,
            values=values,
            temperature=temperature,
        )

        # Compute DC operating point and extract Jacobians
        Jr, Jc, V_dc, _ = self._compute_ac_operating_point()

        # Extract DC currents for noise calculation
        dc_currents = self._extract_dc_currents(V_dc)

        # Extract noise sources from devices
        noise_sources = extract_noise_sources(self.devices, dc_currents, temperature)

        logger.info(f"Found {len(noise_sources)} noise sources")

        # Get input source specification
        input_src = None
        if input_source:
            for src in self._extract_all_sources():
                if src["name"] == input_source:
                    input_src = src
                    break

        # Run noise analysis
        result = run_noise_analysis(
            Jr=Jr,
            Jc=Jc,
            noise_sources=noise_sources,
            input_source=input_src,
            config=config,
            node_names=self.node_names,
            dc_voltages=V_dc,
        )

        logger.info(f"Noise analysis complete: {len(result.frequencies)} frequencies")
        return result

    def _extract_dc_currents(self, V_dc: Array) -> Dict[str, float]:
        """Extract DC currents through devices from operating point."""
        from jax_spice.analysis.noise import extract_dc_currents

        return extract_dc_currents(self.devices, V_dc)

    # =========================================================================
    # Corner Analysis
    # =========================================================================

    def run_corners(
        self, corners: List["CornerConfig"], analysis: str = "transient", **analysis_kwargs
    ) -> "CornerSweepResult":
        """Run simulation across multiple PVT corners.

        Runs the same analysis with different process, voltage, and temperature
        settings. Each corner modifies device parameters before simulation.

        Args:
            corners: List of CornerConfig specifying PVT conditions
            analysis: Analysis type ('transient', 'dc', 'ac')
            **analysis_kwargs: Arguments passed to the analysis method
                (e.g., t_stop, dt for transient)

        Returns:
            CornerSweepResult containing results for all corners

        Example:
            from jax_spice.analysis.corners import create_pvt_corners

            corners = create_pvt_corners(
                processes=['FF', 'TT', 'SS'],
                temperatures=['cold', 'room', 'hot']
            )
            results = engine.run_corners(corners, t_stop=1e-3, dt=1e-6)

            for r in results.converged_results():
                print(f"{r.corner.name}: max voltage = {r.result.voltages['out'].max()}")
        """
        from jax_spice.analysis.corners import CornerResult, CornerSweepResult

        results = []

        for corner in corners:
            # Save original device params
            original_params = self._save_device_params()

            try:
                # Apply corner modifications
                self._apply_process_corner(corner.process)
                self._apply_voltage_corner(corner.voltage)

                # Set temperature (this invalidates cache)
                if corner.temperature != self._simulation_temperature:
                    self._simulation_temperature = corner.temperature
                    self.options.temp = corner.temperature - 273.15
                    self._transient_setup_cache = None
                    self._transient_setup_key = None

                logger.info(
                    f"Running corner: {corner.name} (T={corner.temperature - 273.15:.1f}°C)"
                )

                # Run analysis
                if analysis == "transient":
                    result = self.run_transient(temperature=corner.temperature, **analysis_kwargs)
                elif analysis == "ac":
                    result = self.run_ac(temperature=corner.temperature, **analysis_kwargs)
                elif analysis == "noise":
                    result = self.run_noise(temperature=corner.temperature, **analysis_kwargs)
                else:
                    raise ValueError(f"Unsupported analysis type: {analysis}")

                results.append(
                    CornerResult(
                        corner=corner,
                        result=result,
                        converged=True,
                        stats=getattr(result, "stats", {}),
                    )
                )

            except Exception as e:
                logger.warning(f"Corner {corner.name} failed: {e}")
                results.append(
                    CornerResult(
                        corner=corner, result=None, converged=False, stats={"error": str(e)}
                    )
                )

            finally:
                # Restore original params
                self._restore_device_params(original_params)

        return CornerSweepResult(corners=corners, results=results)

    def _save_device_params(self) -> List[Dict[str, Any]]:
        """Save copy of device parameters for restoration.

        Returns:
            List of device parameter dicts (deep copy)
        """
        return [dict(dev.get("params", {})) for dev in self.devices]

    def _restore_device_params(self, saved: List[Dict[str, Any]]) -> None:
        """Restore device parameters from saved state.

        Args:
            saved: Previously saved parameter dicts
        """
        for dev, params in zip(self.devices, saved):
            dev["params"] = params

    def _apply_process_corner(self, corner: Optional["ProcessCorner"]) -> None:
        """Apply process corner scaling to device parameters.

        Modifies device parameters in place based on corner specification.

        Args:
            corner: Process corner to apply (or None for nominal)
        """
        if corner is None:
            return

        for dev in self.devices:
            if not dev.get("is_openvaf", False):
                continue

            params = dev.get("params", {})
            model = dev.get("model", "")

            # Apply mobility scaling
            mobility_params = ("uo", "mu0", "u0", "betn", "betp", "mue")
            for param in mobility_params:
                if param in params:
                    params[param] = float(params[param]) * corner.mobility_scale

            # Apply Vth shift
            vth_params = ("vth0", "vfb", "delvto", "dvt0")
            for param in vth_params:
                if param in params:
                    params[param] = float(params[param]) + corner.vth_shift

            # Apply Tox scaling
            tox_params = ("tox", "toxe", "toxo", "toxp")
            for param in tox_params:
                if param in params:
                    params[param] = float(params[param]) * corner.tox_scale

            # Apply length delta
            if corner.length_delta != 0:
                if "l" in params:
                    params["l"] = float(params["l"]) + corner.length_delta

            # Apply model-specific overrides
            if model in corner.model_params:
                for param, value in corner.model_params[model].items():
                    params[param] = value

    def _apply_voltage_corner(self, corner: Optional["VoltageCorner"]) -> None:
        """Apply voltage corner scaling to source devices.

        Modifies voltage source DC values based on corner specification.

        Args:
            corner: Voltage corner to apply (or None for nominal)
        """
        if corner is None:
            return

        for dev in self.devices:
            if dev.get("model") != "vsource":
                continue

            name = dev.get("name", "")
            params = dev.get("params", {})

            # Check for explicit source value
            if name in corner.source_values:
                params["dc"] = corner.source_values[name]
            elif "dc" in params:
                # Apply general VDD scaling
                params["dc"] = float(params["dc"]) * corner.vdd_scale
