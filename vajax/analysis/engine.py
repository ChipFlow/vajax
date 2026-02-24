"""Circuit simulation engine for VA-JAX.

Core simulation engine that parses .sim circuit files and runs transient/DC
analysis using JAX-compiled solvers. All devices are compiled from Verilog-A
sources using OpenVAF.

OpenVAF model compilation is handled by vajax.analysis.openvaf_models.
"""

import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

if TYPE_CHECKING:
    from vajax.analysis.ac import ACResult
    from vajax.analysis.corners import (
        CornerConfig,
        CornerSweepResult,
    )
    from vajax.analysis.noise import NoiseResult
    from vajax.analysis.transient import AdaptiveConfig
    from vajax.analysis.xfer import ACXFResult, DCIncResult, DCXFResult

import jax
import jax.numpy as jnp
from jax import Array

# Suppress scipy's MatrixRankWarning from spsolve - this is expected for circuits
# with floating internal nodes (e.g., PSP103 NOI nodes). The QR solver handles
# near-singular matrices gracefully and produces correct results.
from scipy.sparse.linalg import MatrixRankWarning

warnings.filterwarnings("ignore", category=MatrixRankWarning)

# Note: solver.py contains standalone NR solvers (newton_solve) used by tests
# The engine uses its own nr_solve with analytic jacobians from OpenVAF
from vajax import configure_xla_cache, get_float_dtype
from vajax._logging import logger
from vajax.analysis.dc_operating_point import (
    compute_dc_operating_point as _compute_dc_op_impl,
)
from vajax.analysis.homotopy import (
    HomotopyConfig,
    run_homotopy_chain,
)
from vajax.analysis.integration import (
    IntegrationMethod,
    get_method_from_options,
)
from vajax.analysis.mna_builder import build_stamp_index_mapping, make_mna_build_system_fn
from vajax.analysis.node_setup import setup_internal_nodes
from vajax.analysis.openvaf_models import (
    MODEL_PATHS,
    compile_openvaf_models,
    compute_early_collapse_decisions,
    prepare_static_inputs,
    warmup_models,  # noqa: F401 - re-exported for analysis/__init__.py
)
from vajax.analysis.openvaf_models import (
    warmup_device_models as _warmup_device_models_impl,
)
from vajax.analysis.options import SimulationOptions
from vajax.analysis.parsing import (
    build_devices as _build_devices_impl,
)
from vajax.analysis.parsing import (
    flatten_instances,
    parse_elaborate_directive,
)
from vajax.analysis.solver_factories import (
    make_dense_full_mna_solver,
)
from vajax.analysis.sources import (
    build_source_fn,
    collect_source_devices_coo,
    get_dc_source_values,
    get_source_fn_for_device,
    get_vdd_value,
    prepare_source_devices_coo,
)
from vajax.config import DEFAULT_TEMPERATURE_K
from vajax.netlist.parser import VACASKParser
from vajax.profiling import profile

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


# Module-level cache for SPICE number parsing (e.g., "1k" -> 1000, "1u" -> 1e-6)
# These are circuit-independent and can be shared across all CircuitEngine instances
_SPICE_NUMBER_CACHE: Dict[str, float] = {}


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
    """Core circuit simulation engine for VA-JAX.

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

    # OpenVAF model sources - imported from openvaf_models module
    # Keys are device types, values are (base_path_key, relative_path) tuples
    # base_path_key: 'integration_tests' or 'vacask'
    OPENVAF_MODELS = MODEL_PATHS

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

        # Device-level voltage limiting (pnjlim/fetlim).
        # When True, generates calls to limit_funcs in device eval instead of passthrough.
        # The lim_rhs correction (f_corrected = f - J*(V_limited - V_raw)) is computed
        # in the OpenVAF JAX codegen, ensuring NR consistency when limiting is active.
        self.use_device_limiting: bool = True

        # Prepared state for run_transient()
        self._prepared = False

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

        # Clear transient setup cache and prepared state
        self._transient_setup_cache = None
        self._transient_setup_key = None
        self._prepared = False

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
        self._prepared = False
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
            compile_openvaf_models(
                devices=self.devices,
                compiled_models=self._compiled_models,
                model_paths=self.OPENVAF_MODELS,
                log_fn=lambda msg: logger.info(msg),
            )
            # Compute collapse decisions early using init_fn
            self._device_collapse_decisions = compute_early_collapse_decisions(
                devices=self.devices,
                compiled_models=self._compiled_models,
            )

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

    def prepare(
        self,
        *,
        t_stop: Optional[float] = None,
        dt: Optional[float] = None,
        use_sparse: Optional[bool] = None,
        backend: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE_K,
        adaptive_config: Optional["AdaptiveConfig"] = None,
        checkpoint_interval: Optional[int] = None,
    ) -> None:
        """Prepare for transient analysis by configuring all parameters.

        Reads t_stop, dt from netlist analysis params if not overridden.
        Auto-computes max_steps with 10% headroom for LTE timestep reductions.
        Builds and caches the FullMNAStrategy for JIT reuse.

        Call this once before run_transient(). If run_transient() is called
        without prepare(), it will auto-prepare from netlist defaults.

        Args:
            t_stop: Stop time (default: from netlist analysis params)
            dt: Time step (default: from netlist analysis params). For adaptive
                mode, this is the initial timestep.
            use_sparse: Force sparse (True) or dense (False) solver.
                If None, defaults to False (dense).
            backend: 'gpu', 'cpu', or None (auto-select based on circuit size).
            temperature: Simulation temperature in Kelvin (default: 300.15K).
            adaptive_config: Configuration for adaptive timestep control.
                If None, builds from netlist options.
            checkpoint_interval: If set, use GPU memory checkpointing with
                this many steps per buffer.
        """
        from vajax.analysis.gpu_backend import select_backend
        from vajax.analysis.transient import AdaptiveConfig, FullMNAStrategy

        # Update simulation temperature if changed (invalidates cached static inputs)
        if temperature != self._simulation_temperature:
            self._simulation_temperature = temperature
            self.options.temp = temperature - 273.15
            self._transient_setup_cache = None
            self._transient_setup_key = None
            logger.info(f"Temperature changed to {temperature}K ({temperature - 273.15:.1f}°C)")

        # Read from netlist if not overridden
        if t_stop is None:
            t_stop = float(self.analysis_params.get("stop", 1e-3))
        if dt is None:
            dt = float(self.analysis_params.get("step", 1e-6))

        # Auto-compute max_steps with 10% headroom for LTE timestep reductions
        max_steps = int(t_stop / dt * 1.1) + 10

        # Select backend
        if use_sparse is None:
            use_sparse = False
        if backend is None or backend == "auto":
            backend = select_backend(self.num_nodes)

        # Build AdaptiveConfig from netlist options, then apply any explicit overrides
        kwargs = {}

        if "tran_lteratio" in self.analysis_params:
            kwargs["lte_ratio"] = float(self.analysis_params["tran_lteratio"])
        if "tran_redofactor" in self.analysis_params:
            kwargs["redo_factor"] = float(self.analysis_params["tran_redofactor"])
        if "nr_convtol" in self.analysis_params:
            kwargs["nr_convtol"] = float(self.analysis_params["nr_convtol"])
        if "tran_gshunt" in self.analysis_params:
            kwargs["gshunt_init"] = float(self.analysis_params["tran_gshunt"])
        if "reltol" in self.analysis_params:
            kwargs["reltol"] = float(self.analysis_params["reltol"])
        if "abstol" in self.analysis_params:
            kwargs["abstol"] = float(self.analysis_params["abstol"])
        if "tran_fs" in self.analysis_params:
            kwargs["tran_fs"] = float(self.analysis_params["tran_fs"])
        if "tran_minpts" in self.analysis_params:
            kwargs["tran_minpts"] = int(self.analysis_params["tran_minpts"])
        if "maxstep" in self.analysis_params:
            kwargs["max_dt"] = float(self.analysis_params["maxstep"])
        if "tran_method" in self.analysis_params:
            kwargs["integration_method"] = self.analysis_params["tran_method"]

        config = AdaptiveConfig(**kwargs)

        # Apply explicit overrides (e.g. debug_steps=True) on top of netlist config
        if adaptive_config is not None:
            import dataclasses

            # Only override fields that differ from AdaptiveConfig defaults
            defaults = AdaptiveConfig()
            overrides = {}
            for f in dataclasses.fields(adaptive_config):
                val = getattr(adaptive_config, f.name)
                default_val = getattr(defaults, f.name)
                if val != default_val:
                    overrides[f.name] = val
            if overrides:
                config = dataclasses.replace(config, **overrides)
                logger.debug(f"Applied adaptive_config overrides: {overrides}")

        # Cache strategy instance for JIT reuse across calls
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

        # Store prepared state
        self._prepared_t_stop = t_stop
        self._prepared_dt = dt
        self._prepared_strategy = strategy
        self._prepared_config = config
        self._prepared_checkpoint_interval = checkpoint_interval
        self._prepared = True

        logger.info(
            f"Prepared: t_stop={t_stop:.2e}s, dt={dt:.2e}s, "
            f"max_steps={max_steps}, backend={backend}"
        )

        # Warmup: run 1 step to trigger JIT compilation so run_transient() is fast
        strategy.warmup(dt)

    @profile
    def run_transient(self) -> TransientResult:
        """Run transient analysis. Call prepare() first, or will auto-prepare from netlist.

        All computation is JIT-compiled. Uses full MNA with explicit branch
        currents for voltage sources, providing:
        - Better numerical conditioning (no G=1e12 high-G approximation)
        - More accurate current extraction (branch currents are primary unknowns)
        - Smoother dI/dt transitions

        Returns:
            TransientResult with times, voltages, and stats
        """
        if not getattr(self, "_prepared", False):
            self.prepare()

        # All non-source devices use OpenVAF
        if not self._has_openvaf_devices:
            logger.warning("No OpenVAF devices found - circuit only has sources")

        from vajax.analysis.transient import extract_results

        strategy = self._prepared_strategy
        t_stop = self._prepared_t_stop
        dt = self._prepared_dt
        checkpoint_interval = self._prepared_checkpoint_interval

        logger.info(f"Running transient: t_stop={t_stop:.2e}s, dt={dt:.2e}s")

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
    # VA-JAX vs VACASK Node Count Comparison:
    #
    # VACASK reports two metrics via 'print stats':
    #   - "Number of nodes" = nodeCount() = all Node objects in nodeMap
    #   - "Number of unknowns" = unknownCount() = system size after collapse
    #
    # Key difference:
    #   - VACASK creates Node objects for ALL internal nodes, even collapsed ones.
    #     After collapse, multiple Node objects share the same unknownIndex.
    #   - VA-JAX doesn't create objects for collapsed internal nodes at all.
    #     We directly allocate circuit nodes only for non-collapsed internals.
    #
    # Comparison for c6288 benchmark (PSP103 with all resistance params = 0):
    #   VACASK nodeCount():    ~86,000 (5,123 external + 81k internal Node objects)
    #   VACASK unknownCount(): ~15,234 (actual system matrix size after collapse)
    #   VA-JAX total_nodes: ~15,235 (directly matches unknownCount + 1 for ground)
    #
    # VA-JAX's approach is more memory-efficient: we don't allocate internal
    # node objects that would just be collapsed anyway. Our total_nodes from
    # setup_internal_nodes() should match VACASK's unknownCount() + 1.
    #
    # The +1 difference is because VACASK's unknownCount excludes ground (index 0),
    # while VA-JAX counts ground as node 0 in the total.
    # =========================================================================

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
        from vajax.analysis.gpu_backend import get_default_dtype, get_device

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
        n_total, device_internal_nodes = setup_internal_nodes(
            devices=self.devices,
            num_nodes=self.num_nodes,
            compiled_models=self._compiled_models,
            device_collapse_decisions=self._device_collapse_decisions,
        )
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
                voltage_indices, device_contexts, cache, collapse_decisions = prepare_static_inputs(
                    model_type=model_type,
                    openvaf_devices=openvaf_by_type[model_type],
                    device_internal_nodes=device_internal_nodes,
                    compiled_models=self._compiled_models,
                    simulation_temperature=self._simulation_temperature,
                    use_device_limiting=getattr(self, "use_device_limiting", False),
                    parse_voltage_param_fn=self._parse_voltage_param,
                    ground=ground,
                )
                stamp_indices = build_stamp_index_mapping(
                    model_type, device_contexts, ground, self._compiled_models
                )
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
        _warmup_device_models_impl(
            compiled_models=self._compiled_models,
            static_inputs_cache=static_inputs_cache,
        )

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
            device_data, V, vsource_vals, isource_vals, f_indices, f_values, j_rows, j_cols, j_vals
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

    def _make_mna_build_system_fn(
        self,
        source_device_data: Dict,
        vmapped_fns: Dict,
        static_inputs_cache: Dict,
        n_unknowns: int,
        use_dense: bool = True,
    ) -> Tuple[Callable, Dict, int]:
        """Create GPU-resident build_system function for MNA formulation.

        This is a thin wrapper around make_mna_build_system_fn that passes
        the compiled models and options from self.

        See vajax.analysis.mna_builder.make_mna_build_system_fn for full docs.
        """
        return make_mna_build_system_fn(
            source_device_data=source_device_data,
            static_inputs_cache=static_inputs_cache,
            compiled_models=self._compiled_models,
            gmin=self.options.gmin,
            n_unknowns=n_unknowns,
            use_dense=use_dense,
        )

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
        from vajax.analysis.ac import ACConfig, run_ac_analysis

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
            compile_openvaf_models(
                devices=self.devices,
                compiled_models=self._compiled_models,
                model_paths=self.OPENVAF_MODELS,
                log_fn=lambda msg: logger.info(msg),
            )

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
        build_system_fn, device_arrays, total_limit_states = self._make_mna_build_system_fn(
            source_device_data, vmapped_fns, static_inputs_cache, n_unknowns, use_dense=True
        )
        build_system_jit = jax.jit(build_system_fn)

        # Collect NOI node indices and ALL internal device node indices
        noi_indices = []
        all_internal_indices = []
        if device_internal_nodes:
            for dev_name, internal_nodes in device_internal_nodes.items():
                for node_name, node_idx in internal_nodes.items():
                    all_internal_indices.append(node_idx)
                    if node_name == "node4":  # NOI is node4 in PSP103
                        noi_indices.append(node_idx)
        noi_indices = jnp.array(noi_indices, dtype=jnp.int32) if noi_indices else None
        internal_device_indices = (
            jnp.array(sorted(set(all_internal_indices)), dtype=jnp.int32)
            if all_internal_indices
            else None
        )

        nr_solve = make_dense_full_mna_solver(
            build_system_jit,
            n_total,
            n_vsources,
            noi_indices=noi_indices,
            internal_device_indices=internal_device_indices,
            max_iterations=self.options.op_itl,
            abstol=self.options.abstol,
            total_limit_states=total_limit_states,
            options=self.options,
        )
        nr_solve = jax.jit(nr_solve)  # JIT for DC solve from Python context

        # Initialize X (augmented: [V, I_branch])
        vdd_value = self._get_vdd_value() or 1.0  # Default to 1.0 if no vsources
        mid_rail = vdd_value / 2.0
        X_init = jnp.zeros(n_total + n_vsources, dtype=get_float_dtype())
        X_init = X_init.at[1:n_total].set(mid_rail)  # Node voltages (skip ground)

        Q_prev = jnp.zeros(n_unknowns, dtype=get_float_dtype())

        # First try direct NR without homotopy
        logger.info("  AC DC: Trying direct NR solver first...")
        X_new, nr_iters, is_converged, max_f, _, _, _, _, _ = nr_solve(
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
        from vajax.analysis.ac import extract_ac_sources

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
        from vajax.analysis.xfer import build_dcinc_excitation, solve_dcinc

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
        from vajax.analysis.xfer import solve_dcxf

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
        from vajax.analysis.xfer import ACXFConfig, solve_acxf

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
        from vajax.analysis.xfer import extract_all_sources

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

        from vajax.analysis.noise import (
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
        from vajax.analysis.noise import extract_dc_currents

        return extract_dc_currents(self.devices, V_dc)

    # =========================================================================
    # Corner Analysis
    # =========================================================================

    def run_corners(
        self, corners: List["CornerConfig"], analysis: str = "transient", **prepare_kwargs
    ) -> "CornerSweepResult":
        """Run simulation across multiple PVT corners.

        Runs the same analysis with different process, voltage, and temperature
        settings. Each corner modifies device parameters before simulation.

        Args:
            corners: List of CornerConfig specifying PVT conditions
            analysis: Analysis type ('transient', 'dc', 'ac')
            **prepare_kwargs: Arguments passed to prepare() for transient
                (e.g., t_stop, dt, use_sparse)

        Returns:
            CornerSweepResult containing results for all corners

        Example:
            from vajax.analysis.corners import create_pvt_corners

            corners = create_pvt_corners(
                processes=['FF', 'TT', 'SS'],
                temperatures=['cold', 'room', 'hot']
            )
            engine.prepare(t_stop=1e-3, dt=1e-6)
            results = engine.run_corners(corners)

            for r in results.converged_results():
                print(f"{r.corner.name}: max voltage = {r.result.voltages['out'].max()}")
        """
        from vajax.analysis.corners import (
            CornerResult,
            CornerSweepResult,
            apply_process_corner,
            apply_voltage_corner,
        )

        results = []

        for corner in corners:
            # Save original device params
            original_params = self._save_device_params()

            try:
                # Apply corner modifications
                apply_process_corner(self.devices, corner.process)
                apply_voltage_corner(self.devices, corner.voltage)

                logger.info(
                    f"Running corner: {corner.name} (T={corner.temperature - 273.15:.1f}°C)"
                )

                # Run analysis
                if analysis == "transient":
                    self.prepare(temperature=corner.temperature, **prepare_kwargs)
                    result = self.run_transient()
                elif analysis == "ac":
                    self.prepare(temperature=corner.temperature)
                    result = self.run_ac(**prepare_kwargs)
                elif analysis == "noise":
                    result = self.run_noise(temperature=corner.temperature, **prepare_kwargs)
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
