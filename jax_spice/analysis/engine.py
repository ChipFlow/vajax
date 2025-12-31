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
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Any, Optional, Union

import jax
import jax.numpy as jnp
from jax import lax, Array
import numpy as np

from jax_spice.netlist.parser import VACASKParser
from jax_spice.netlist.circuit import Instance
from jax_spice.analysis.mna import DeviceInfo
from jax_spice.analysis.homotopy import (
    HomotopyConfig, HomotopyResult, run_homotopy_chain, gmin_stepping, source_stepping
)
# Note: solver.py contains standalone NR solvers (newton_solve) used by tests
# The engine uses its own nr_solve with analytic jacobians from OpenVAF
from jax_spice.logging import logger
from jax_spice.profiling import profile, profile_section, ProfileConfig

# Try to import OpenVAF support
_openvaf_path = Path(__file__).parent.parent.parent / "openvaf-py"
if str(_openvaf_path) not in sys.path:
    sys.path.insert(0, str(_openvaf_path))

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

# Newton-Raphson solver constants
MAX_NR_ITERATIONS = 100  # Maximum Newton-Raphson iterations per timestep
DEFAULT_ABSTOL = 1e-3  # Absolute tolerance for NR convergence (matches SPICE reltol)


@dataclass
class TransientResult:
    """Result of a transient simulation.

    Attributes:
        times: Array of time points
        voltages: Dict mapping node index to voltage array
        stats: Dict with simulation statistics (wall_time, convergence_rate, etc.)
    """
    times: Array
    voltages: Dict[int, Array]
    stats: Dict[str, Any]

    @property
    def num_steps(self) -> int:
        """Number of timesteps in the simulation."""
        return len(self.times)

    def voltage(self, node: Union[int, str]) -> Array:
        """Get voltage waveform at a specific node.

        Args:
            node: Node index (int) or name (str) - names not yet supported

        Returns:
            Voltage array over time
        """
        if isinstance(node, str):
            raise ValueError(
                f"Node name lookup not yet supported. Use node index. "
                f"Available: {list(self.voltages.keys())}"
            )
        return self.voltages[node]


class CircuitEngine:
    """Core circuit simulation engine for JAX-SPICE.

    Parses .sim circuit files and runs transient/DC analysis using JAX-compiled
    solvers. All devices (resistors, capacitors, diodes, MOSFETs) are compiled
    from Verilog-A sources using OpenVAF.
    """

    # Map OSDI module names to device types
    MODULE_TO_DEVICE = {
        'sp_resistor': 'resistor',
        'sp_capacitor': 'capacitor',
        'sp_diode': 'diode',  # Use simplified diode model (sp_diode model has NaN issues)
        'vsource': 'vsource',
        'isource': 'isource',
        'psp103va': 'psp103',  # PSP103 MOSFET
    }

    # OpenVAF model sources
    # Keys are device types, values are (base_path_key, relative_path) tuples
    # base_path_key: 'integration_tests' or 'vacask'
    OPENVAF_MODELS = {
        'psp103': ('integration_tests', 'PSP103/psp103.va'),
        'resistor': ('vacask', 'resistor.va'),
        'capacitor': ('vacask', 'capacitor.va'),
        'diode': ('vacask', 'diode.va'),
        'sp_diode': ('vacask', 'spice/sn/diode.va'),  # Full SPICE diode model
    }

    # Default parameter values for OpenVAF models
    # NOTE: Parameter defaults are now extracted from Verilog-A source via openvaf-py's
    # get_param_defaults() method. No manual MODEL_PARAM_DEFAULTS needed.

    # SPICE number suffixes
    SUFFIXES = {
        't': 1e12, 'g': 1e9, 'meg': 1e6, 'k': 1e3,
        'm': 1e-3, 'u': 1e-6, 'n': 1e-9, 'p': 1e-12, 'f': 1e-15
    }

    def __init__(self, sim_path: Path):
        self.sim_path = Path(sim_path)
        self.circuit = None
        self.devices = []
        self.node_names = {}
        self.num_nodes = 0
        self.analysis_params = {}
        self.flat_instances = []

        # OpenVAF compiled models cache
        self._compiled_models: Dict[str, Any] = {}
        self._has_openvaf_devices = False

        # Transient setup cache (reused across multiple run_transient calls)
        self._transient_setup_cache: Dict[str, Any] | None = None
        self._transient_setup_key: str | None = None

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
        if hasattr(self, '_cached_nr_solve'):
            del self._cached_nr_solve
        if hasattr(self, '_cached_solver_key'):
            del self._cached_solver_key

        # Clear transient setup cache
        self._transient_setup_cache = None
        self._transient_setup_key = None

        # Clear JAX compilation caches
        jax.clear_caches()

        # Force garbage collection
        gc.collect()

        logger.info("Cleared caches and freed memory")

    def parse_spice_number(self, s: str) -> float:
        """Parse SPICE number with suffix (e.g., 1u, 100n, 1.5k)"""
        if not isinstance(s, str):
            return float(s)
        s = s.strip().lower().strip('"')
        if not s:
            return 0.0

        for suffix, multiplier in sorted(self.SUFFIXES.items(), key=lambda x: -len(x[0])):
            if s.endswith(suffix):
                try:
                    return float(s[:-len(suffix)]) * multiplier
                except ValueError:
                    continue
        try:
            return float(s)
        except ValueError:
            return 0.0

    def parse(self):
        """Parse the sim file and extract circuit information."""
        import time
        import sys

        # Clear transient setup cache when re-parsing (circuit may have changed)
        self._transient_setup_cache = None
        self._transient_setup_key = None

        logger.info("parse(): starting...")

        t0 = time.perf_counter()
        parser = VACASKParser()
        self.circuit = parser.parse_file(self.sim_path)
        t1 = time.perf_counter()

        logger.info(f"Parsed: {self.circuit.title} ({t1-t0:.1f}s)")
        logger.debug(f"Models: {list(self.circuit.models.keys())}")
        if self.circuit.subckts:
            logger.debug(f"Subcircuits: {list(self.circuit.subckts.keys())}")

        # Flatten subcircuit instances to leaf devices
        logger.info("Flattening subcircuit instances...")
        self.flat_instances = self._flatten_top_instances()
        t2 = time.perf_counter()

        logger.info(f"Flattened: {len(self.flat_instances)} leaf devices ({t2-t1:.1f}s)")
        for name, terms, model, params in self.flat_instances[:10]:
            logger.debug(f"  {name}: {model} {terms}")
        if len(self.flat_instances) > 10:
            logger.debug(f"  ... and {len(self.flat_instances) - 10} more")

        # Build node mapping from flattened instances
        node_set = {'0'}
        for name, terminals, model, params in self.flat_instances:
            for t in terminals:
                node_set.add(t)

        self.node_names = {'0': 0}
        for i, name in enumerate(sorted(n for n in node_set if n != '0'), start=1):
            self.node_names[name] = i
        self.num_nodes = len(self.node_names)
        t3 = time.perf_counter()

        logger.info(f"Node mapping: {self.num_nodes} nodes ({t3-t2:.1f}s)")

        # Build devices
        self._build_devices()
        t4 = time.perf_counter()

        logger.info(f"Built devices: {len(self.devices)} ({t4-t3:.1f}s)")

        # Extract analysis parameters
        self._extract_analysis_params()

        return self

    def _get_device_type(self, model_name: str) -> str:
        """Map model name to device type."""
        model = self.circuit.models.get(model_name)
        if model:
            module = model.module.lower()
            return self.MODULE_TO_DEVICE.get(module, module)
        # Direct lookup for built-in types
        return self.MODULE_TO_DEVICE.get(model_name.lower(), model_name.lower())

    def _get_model_params(self, model_name: str) -> Dict[str, float]:
        """Get parsed parameters from a model definition."""
        model = self.circuit.models.get(model_name)
        if not model:
            return {}
        return {k: self.parse_spice_number(v) for k, v in model.params.items()}

    def _parse_elaborate_directive(self) -> Optional[str]:
        """Parse 'elaborate circuit("subckt_name")' directive from control block.

        Returns subcircuit name to elaborate, or None if not found.
        """
        text = self.sim_path.read_text()
        # Match: elaborate circuit("name") or elaborate circuit("name", "other")
        match = re.search(r'elaborate\s+circuit\s*\(\s*"([^"]+)"', text)
        if match:
            return match.group(1)
        return None

    def _flatten_top_instances(self) -> List[Tuple[str, List[str], str, Dict[str, str]]]:
        """Flatten subcircuit instances to leaf devices.

        Returns list of (name, terminals, model, params) tuples for leaf devices.
        Handles 'elaborate circuit("name")' directive if present.
        """
        from jax_spice.netlist.parser import Instance

        flat_instances = []
        ground = self.circuit.ground or '0'

        # Check for elaborate directive
        elaborate_subckt = self._parse_elaborate_directive()
        if elaborate_subckt:
            # Create synthetic top-level instance of the elaborated subcircuit
            subckt = self.circuit.subckts.get(elaborate_subckt)
            if subckt:
                logger.debug(f"Elaborating subcircuit: {elaborate_subckt}")
                # Create synthetic instance with no external ports
                synthetic_inst = Instance(
                    name='top',
                    terminals=subckt.terminals,  # Map to global nodes with same names
                    model=elaborate_subckt,
                    params={}
                )
                self.circuit.top_instances.append(synthetic_inst)

        # Parameters that should be kept as strings (not evaluated as expressions)
        string_params = {'type'}

        def eval_param_expr(key: str, expr: str, param_env: Dict[str, float]):
            """Evaluate a parameter expression like 'w*pfact' or '2*(w+ld)'.

            String parameters (like type="pulse") are preserved as-is.
            """
            if not isinstance(expr, str):
                return float(expr)

            # Check if this is a quoted string value - preserve it
            stripped = expr.strip()
            if (stripped.startswith('"') and stripped.endswith('"')) or \
               (stripped.startswith("'") and stripped.endswith("'")):
                # Return the unquoted string
                return stripped[1:-1]

            # Check if this key should be kept as string
            if key.lower() in string_params:
                return stripped

            # Try direct parse first
            val = self.parse_spice_number(expr)
            if val != 0.0 or stripped in ('0', '0.0'):
                return val

            # Simple expression evaluation with parameter substitution
            try:
                # Replace parameter names with values
                eval_expr = expr
                for name, value in sorted(param_env.items(), key=lambda x: -len(x[0])):
                    eval_expr = eval_expr.replace(name, str(value))
                return float(eval(eval_expr))
            except:
                return 0.0

        def flatten_instance(inst: Instance, prefix: str, port_map: Dict[str, str],
                           param_env: Dict[str, float]):
            """Recursively flatten an instance."""
            model_name = inst.model

            # Check if this is a subcircuit
            subckt = self.circuit.subckts.get(model_name)
            if subckt is None:
                # Leaf device - map terminals and add to list
                mapped_terminals = []
                for t in inst.terminals:
                    if t in port_map:
                        mapped_terminals.append(port_map[t])
                    elif t in self.circuit.globals or t == ground:
                        mapped_terminals.append(t)
                    elif prefix:
                        mapped_terminals.append(f"{prefix}.{t}")
                    else:
                        mapped_terminals.append(t)

                # Evaluate instance parameters with current environment
                inst_params = {}
                for k, v in inst.params.items():
                    inst_params[k] = str(eval_param_expr(k, v, param_env))

                flat_name = f"{prefix}.{inst.name}" if prefix else inst.name
                flat_instances.append((flat_name, mapped_terminals, model_name, inst_params))
            else:
                # Subcircuit - recurse
                # Build new port map
                new_port_map = {}
                for i, term in enumerate(subckt.terminals):
                    if i < len(inst.terminals):
                        inst_term = inst.terminals[i]
                        if inst_term in port_map:
                            new_port_map[term] = port_map[inst_term]
                        elif inst_term in self.circuit.globals or inst_term == ground:
                            new_port_map[term] = inst_term
                        elif prefix:
                            new_port_map[term] = f"{prefix}.{inst_term}"
                        else:
                            new_port_map[term] = inst_term

                # Build new parameter environment
                new_param_env = dict(param_env)
                # Add subcircuit default params
                for k, v in subckt.params.items():
                    new_param_env[k] = eval_param_expr(k, v, new_param_env)
                # Override with instance params
                for k, v in inst.params.items():
                    new_param_env[k] = eval_param_expr(k, v, param_env)

                # New prefix
                new_prefix = f"{prefix}.{inst.name}" if prefix else inst.name

                # Flatten subcircuit instances
                for sub_inst in subckt.instances:
                    flatten_instance(sub_inst, new_prefix, new_port_map, new_param_env)

        # Flatten all top-level instances
        for inst in self.circuit.top_instances:
            # Start with circuit-level parameters
            param_env = {k: self.parse_spice_number(v) for k, v in self.circuit.params.items()}
            flatten_instance(inst, '', {}, param_env)

        return flat_instances

    def _build_devices(self):
        """Build device list from flattened instances."""
        import time
        import sys
        t_start = time.perf_counter()

        logger.info(f"_build_devices(): starting with {len(self.flat_instances)} instances")

        self.devices = []

        # Parameters that should be kept as strings (not parsed as numbers)
        STRING_PARAMS = {'type'}

        for inst_name, inst_terminals, inst_model, inst_params in self.flat_instances:
            model_name = inst_model.lower()
            device_type = self._get_device_type(model_name)
            nodes = [self.node_names[t] for t in inst_terminals]

            # Get model parameters and instance parameters
            model_params = self._get_model_params(model_name)

            # Parse instance params, but keep string params as strings
            parsed_params = {}
            for k, v in inst_params.items():
                if k in STRING_PARAMS:
                    # Keep as string, strip quotes
                    parsed_params[k] = str(v).strip('"').strip("'")
                else:
                    parsed_params[k] = self.parse_spice_number(v)

            # Merge model params with instance params (instance overrides model)
            params = {**model_params, **parsed_params}

            # Track if this is an OpenVAF device
            is_openvaf = device_type in self.OPENVAF_MODELS

            self.devices.append({
                'name': inst_name,
                'model': device_type,
                'nodes': nodes,
                'params': params,
                'original_params': parsed_params,  # Instance params before merge
                'is_openvaf': is_openvaf,
            })

            if is_openvaf:
                self._has_openvaf_devices = True

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
            if dev.get('is_openvaf'):
                openvaf_types.add(dev['model'])

        log(f"Compiling OpenVAF models: {openvaf_types}")

        # Base paths for different VA model sources
        project_root = Path(__file__).parent.parent.parent
        base_paths = {
            'integration_tests': project_root / "openvaf-py" / "vendor" / "OpenVAF" / "integration_tests",
            'vacask': project_root / "vendor" / "VACASK" / "devices",
        }

        for model_type in openvaf_types:
            # Check instance cache first (for this runner)
            if model_type in self._compiled_models:
                continue

            # Check module-level cache (shared across all runners)
            if model_type in _COMPILED_MODEL_CACHE:
                cached = _COMPILED_MODEL_CACHE[model_type]
                log(f"  {model_type}: reusing cached jitted function ({len(cached['param_names'])} params, {len(cached['nodes'])} nodes)")
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

            # Compile from scratch
            import time
            t0 = time.perf_counter()
            log(f"  {model_type}: compiling VA...")
            modules = openvaf_py.compile_va(str(full_path))
            t1 = time.perf_counter()
            log(f"  {model_type}: VA compiled in {t1-t0:.1f}s")
            if not modules:
                raise ValueError(f"Failed to compile {va_path}")

            log(f"  {model_type}: creating translator...")
            module = modules[0]
            translator = openvaf_jax.OpenVAFToJAX(module)
            t2 = time.perf_counter()
            log(f"  {model_type}: translator created in {t2-t1:.1f}s")

            # Generate array function
            log(f"  {model_type}: translate_array() - generating array function...")
            jax_fn_array, array_metadata = translator.translate_array()
            t3 = time.perf_counter()
            log(f"  {model_type}: translate_array() done in {t3-t2:.1f}s")

            # Create JIT-compiled vmapped function for fast batched evaluation
            log(f"  {model_type}: wrapping with vmap+jit...")
            vmapped_fn = jax.jit(jax.vmap(jax_fn_array))
            t4 = time.perf_counter()
            log(f"  {model_type}: vmap+jit wrapped in {t4-t3:.1f}s")

            # Generate separate init and eval_with_cache functions for cleaner hidden_state handling
            log(f"  {model_type}: generating init function...")
            init_fn, init_meta = translator.translate_init_array()
            vmapped_init = jax.jit(jax.vmap(init_fn))
            t5 = time.perf_counter()
            log(f"  {model_type}: init function done in {t5-t4:.1f}s (cache_size={init_meta['cache_size']})")

            log(f"  {model_type}: generating eval_with_cache function...")
            eval_fn_with_cache, eval_cache_meta = translator.translate_eval_array_with_cache()
            vmapped_eval_with_cache = jax.jit(jax.vmap(eval_fn_with_cache))
            t6 = time.perf_counter()
            log(f"  {model_type}: eval_with_cache function done in {t6-t5:.1f}s")

            # Build init->eval index mapping for extracting init inputs from eval inputs
            eval_name_to_idx = {n.lower(): i for i, n in enumerate(module.param_names)}
            init_to_eval_indices = []
            for name in init_meta['param_names']:
                eval_idx = eval_name_to_idx.get(name.lower(), -1)
                init_to_eval_indices.append(eval_idx)
            init_to_eval_indices = jnp.array(init_to_eval_indices, dtype=jnp.int32)

            # NOTE: MIR data release is deferred to _prepare_static_inputs()
            # so we can generate split eval function after computing constant/varying indices
            # This saves ~28MB for PSP103 after circuit setup is complete

            compiled = {
                'module': module,
                'translator': translator,  # Stored for split function generation
                'jax_fn_array': jax_fn_array,
                'vmapped_fn': vmapped_fn,
                'array_metadata': array_metadata,
                'param_names': list(module.param_names),
                'param_kinds': list(module.param_kinds),
                'nodes': list(module.nodes),
                'collapsible_pairs': list(module.collapsible_pairs),
                'num_collapsible': module.num_collapsible,
                # New init/eval_with_cache functions
                'init_fn': init_fn,
                'vmapped_init': vmapped_init,
                'init_param_names': list(init_meta['param_names']),
                'init_param_kinds': list(init_meta['param_kinds']),
                'cache_size': init_meta['cache_size'],
                'cache_mapping': init_meta['cache_mapping'],
                'init_param_defaults': init_meta.get('param_defaults', {}),
                'eval_fn_with_cache': eval_fn_with_cache,
                'vmapped_eval_with_cache': vmapped_eval_with_cache,
                'eval_cache_meta': eval_cache_meta,
                'init_to_eval_indices': init_to_eval_indices,
                # Device-level GMIN via $simparam("gmin")
                'uses_simparam_gmin': eval_cache_meta.get('uses_simparam_gmin', False),
                # Device-level analysis() function
                'uses_analysis': eval_cache_meta.get('uses_analysis', False),
                'analysis_type_map': eval_cache_meta.get('analysis_type_map', {}),
            }

            # Store in both instance and module-level cache
            self._compiled_models[model_type] = compiled
            _COMPILED_MODEL_CACHE[model_type] = compiled

            log(f"  {model_type}: done ({len(module.param_names)} params, {len(module.nodes)} nodes)")

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
            if dev.get('is_openvaf'):
                model_type = dev['model']
                devices_by_type.setdefault(model_type, []).append(dev)

        for model_type, devs in devices_by_type.items():
            compiled = self._compiled_models.get(model_type)
            if not compiled:
                continue

            init_fn = compiled.get('init_fn')

            if init_fn is None:
                # No init function - use all collapsible pairs
                collapsible_pairs = compiled.get('collapsible_pairs', [])
                for dev in devs:
                    self._device_collapse_decisions[dev['name']] = list(collapsible_pairs)
                continue

            # Get init parameters info
            init_param_names = compiled.get('init_param_names', [])
            init_param_defaults = compiled.get('init_param_defaults', {})
            n_init_params = len(init_param_names)
            collapsible_pairs = compiled.get('collapsible_pairs', [])
            n_collapsible = len(collapsible_pairs)

            if n_init_params == 0 or n_collapsible == 0:
                # No init params or no collapsible pairs - collapse decisions are constant
                if init_fn is not None and n_collapsible > 0:
                    try:
                        # Force CPU execution to avoid GPU JIT overhead for tiny computation
                        cpu_device = jax.devices('cpu')[0]
                        with jax.default_device(cpu_device):
                            _, collapse_decisions = init_fn(jnp.array([]))
                        # Convert collapse decisions to pairs (same for all devices)
                        pairs = []
                        for i, (n1, n2) in enumerate(collapsible_pairs):
                            if i < len(collapse_decisions) and float(collapse_decisions[i]) > 0.5:
                                pairs.append((n1, n2))
                        for dev in devs:
                            self._device_collapse_decisions[dev['name']] = pairs
                        continue
                    except Exception as e:
                        logger.warning(f"Error computing collapse decisions for {model_type}: {e}")
                # Fallback: use all collapsible pairs
                for dev in devs:
                    self._device_collapse_decisions[dev['name']] = list(collapsible_pairs)
                continue

            # OPTIMIZATION: Group devices by unique parameter combinations
            # For c6288, this reduces 10k evaluations to just 2 (pmos, nmos)
            def get_param_key(dev: Dict) -> Tuple:
                """Build hashable key from device parameters."""
                device_params = dev.get('params', {})
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
            logger.info(f"Computing collapse decisions for {model_type}: "
                       f"{len(devs)} devices, {n_unique} unique param combos")

            # Compute collapse decisions for each unique parameter combination
            # Force CPU execution to avoid GPU JIT overhead for small computations
            cpu_device = jax.devices('cpu')[0]
            for param_key, param_devs in unique_params.items():
                try:
                    # Single init_fn call for this parameter combination
                    with jax.default_device(cpu_device):
                        init_inputs = jnp.array(param_key, dtype=jnp.float64)
                        _, collapse_decisions = init_fn(init_inputs)

                    # Convert to pairs
                    pairs = []
                    collapse_np = np.asarray(collapse_decisions)
                    for i, (n1, n2) in enumerate(collapsible_pairs):
                        if i < len(collapse_np) and collapse_np[i] > 0.5:
                            pairs.append((n1, n2))

                    # Apply to all devices with this parameter combination
                    for dev in param_devs:
                        self._device_collapse_decisions[dev['name']] = pairs

                except Exception as e:
                    logger.warning(f"Error computing collapse for {model_type} params {param_key[:3]}...: {e}")
                    # Fallback: use all collapsible pairs for these devices
                    for dev in param_devs:
                        self._device_collapse_decisions[dev['name']] = list(collapsible_pairs)

        logger.debug(f"Computed collapse decisions for {len(self._device_collapse_decisions)} devices")

    def _prepare_static_inputs(self, model_type: str, openvaf_devices: List[Dict],
                                device_internal_nodes: Dict[str, Dict[str, int]],
                                ground: int) -> Tuple[jax.Array, List[int], List[Dict], jax.Array, jax.Array]:
        """Prepare static (non-voltage) inputs for all devices once.

        This is called once per simulation and caches the static parameter values.
        Only voltage parameters need to be updated each NR iteration.

        Returns:
            (static_inputs, voltage_indices, device_contexts, cache, collapse_decisions) where:
            - static_inputs is shape (num_devices, num_params) JAX array with static params
            - voltage_indices is list of param indices that are voltages
            - device_contexts is list of dicts with node_map, voltage_node_pairs for fast update
            - cache is shape (num_devices, cache_size) JAX array with init-computed values
            - collapse_decisions is shape (num_devices, num_collapsible) JAX array with collapse booleans
        """
        logger.debug(f"Preparing static inputs for {model_type}")

        compiled = self._compiled_models.get(model_type)
        if not compiled:
            raise ValueError(f"OpenVAF model {model_type} not compiled")

        param_names = compiled['param_names']
        param_kinds = compiled['param_kinds']
        model_nodes = compiled['nodes']

        logger.debug(f"  param_names = {param_names}")
        logger.debug(f"  {len(param_kinds)} param_kinds")
        logger.debug(f"  {len(model_nodes)} model_nodes")

        # Find which parameter indices are voltages
        voltage_indices = []
        for i, kind in enumerate(param_kinds):
            if kind == 'voltage':
                voltage_indices.append(i)

        # Pre-allocate numpy array to avoid 20x memory overhead from jnp.array(nested_list)
        n_devices = len(openvaf_devices)
        n_params = len(param_names)
        all_inputs = np.zeros((n_devices, n_params), dtype=np.float64)
        device_contexts = []

        # === VECTORIZED PARAM FILLING (replaces 26M-iteration inner loop) ===
        # For c6288: reduces 10k × 2.6k = 26M iterations to ~100 vectorized numpy ops
        if n_devices > 0:
            logger.debug("Filling params")
            all_dev_params = [dev['params'] for dev in openvaf_devices]
            # Get defaults from openvaf-py (extracted from Verilog-A source)
            model_defaults = compiled.get('init_param_defaults', {})

            # Build param_name -> column index mapping
            param_to_cols = {}
            param_given_to_cols = {}
            for idx, (name, kind) in enumerate(zip(param_names, param_kinds)):
                name_lower = name.lower()
                if kind == 'param':
                    param_to_cols.setdefault(name_lower, []).append(idx)
                elif kind == 'param_given':
                    param_given_to_cols.setdefault(name_lower, []).append(idx)
                elif kind == 'temperature':
                    all_inputs[:, idx] = 300.15
                elif kind == 'sysfun' and name_lower == 'mfactor':
                    all_inputs[:, idx] = 1.0

            # Get unique params from devices and fill columns
            all_unique = set()
            for p in all_dev_params:
                all_unique.update(k.lower() for k in p.keys())

            for pname in all_unique:
                if pname in param_to_cols:
                    vals = np.array([float(p.get(pname, p.get(pname.upper(), 0.0))) for p in all_dev_params])
                    for col in param_to_cols[pname]:
                        all_inputs[:, col] = vals
                if pname in param_given_to_cols:
                    for col in param_given_to_cols[pname]:
                        all_inputs[:, col] = 1.0

            # Defaults for params not in any device
            for pname, cols in param_to_cols.items():
                if pname not in all_unique:
                    # Special defaults for commonly needed params
                    if pname in ('tnom', 'tref', 'tr'):
                        default = 27.0  # Temperature reference in Celsius
                    elif pname in ('nf', 'mult', 'ns', 'nd'):
                        default = 1.0  # Finger count and multipliers default to 1
                    else:
                        default = model_defaults.get(pname, 0.0)
                    for col in cols:
                        all_inputs[:, col] = default
            logger.debug("Params filled")
            # NOTE: Hidden_state values are now computed by the init function
            # and passed to eval via cache. No manual computation needed here.

        # NOTE: init_param_defaults from openvaf-py contains Verilog-A source defaults
        # These are used in vectorized filling above when no device-level value exists

        for dev_idx, dev in enumerate(openvaf_devices):
            ext_nodes = dev['nodes']  # [d, g, s, b]
            params = dev['params']
            internal_nodes = device_internal_nodes.get(dev['name'], {})

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

            # Also map sim_node names
            for i, model_node in enumerate(model_nodes[:-1]):
                node_map[f'sim_{model_node}'] = node_map.get(model_node, ground)

            # Pre-compute voltage node pairs for fast update
            voltage_node_pairs = []
            for idx in voltage_indices:
                name = param_names[idx]
                node_pair = self._parse_voltage_param(name, node_map, model_nodes, ground)
                voltage_node_pairs.append(node_pair)

            # NOTE: Parameter filling is done by vectorized code above (lines 788-834).
            # The per-device loop that was here has been removed as it was dead code.

            device_contexts.append({
                'name': dev['name'],
                'node_map': node_map,
                'ext_nodes': ext_nodes,
                'voltage_node_pairs': voltage_node_pairs,
            })

        # Append analysis_type and gmin columns if needed
        # Order: [..., analysis_type (inputs[-2]), gmin (inputs[-1])]
        # When uses_analysis is True, we always append both columns to ensure
        # analysis_type is at inputs[-2] and gmin at inputs[-1]
        uses_analysis = compiled.get('uses_analysis', False)
        uses_simparam_gmin = compiled.get('uses_simparam_gmin', False)

        if uses_analysis and n_devices > 0:
            logger.debug("Appending analysis_type and gmin columns") 
            # Analysis type encoding: 0=dc/static, 1=ac, 2=tran, 3=noise
            # Default to DC analysis (type=0)
            analysis_type_column = np.full((n_devices, 1), 0.0, dtype=np.float64)
            all_inputs = np.concatenate([all_inputs, analysis_type_column], axis=1)
            logger.debug(f"{model_type}: appended analysis_type column (uses_analysis=True)")

            # When uses_analysis is True, we must also append gmin column to maintain
            # the fixed offsets: analysis_type at [-2], gmin at [-1]
            default_gmin = 1e-12
            gmin_column = np.full((n_devices, 1), default_gmin, dtype=np.float64)
            all_inputs = np.concatenate([all_inputs, gmin_column], axis=1)
            logger.debug(f"{model_type}: appended gmin column (for uses_analysis layout)")

        elif uses_simparam_gmin and n_devices > 0:
            # Only gmin, at inputs[-1]
            default_gmin = 1e-12
            gmin_column = np.full((n_devices, 1), default_gmin, dtype=np.float64)
            all_inputs = np.concatenate([all_inputs, gmin_column], axis=1)
            logger.debug(f"{model_type}: appended gmin column (uses_simparam_gmin=True)")

        # Compute cache by calling init function
        # Extract init inputs from all_inputs using init_to_eval mapping
        static_inputs = jnp.asarray(all_inputs)
        init_to_eval = compiled.get('init_to_eval_indices')
        vmapped_init = compiled.get('vmapped_init')

        if init_to_eval is not None and vmapped_init is not None:
            logger.debug("Extracting init inputs")
            # Extract init inputs: shape (n_devices, n_init_params)
            init_inputs = static_inputs[:, init_to_eval]
            # Compute cache and collapse decisions
            # init_fn returns (cache, collapse_decisions) tuple
            # Force CPU execution to avoid GPU JIT overhead - this is a one-time setup cost
            # and the computation itself is small compared to the JIT compilation overhead on GPU
            logger.info(f"Computing init cache for {model_type} ({n_devices} devices)...")
            cpu_device = jax.devices('cpu')[0]
            with jax.default_device(cpu_device):
                cache, collapse_decisions = vmapped_init(init_inputs)
            logger.info(f"Init cache computed for {model_type}: shape={cache.shape}")
            logger.debug(f"Collapse decisions for {model_type}: shape={collapse_decisions.shape}")
        else:
            # Fallback for models without init function (e.g., resistor)
            logger.debug("Model has no init function, inputs zeroed")
            cache = jnp.empty((n_devices, 0), dtype=jnp.float64)
            collapse_decisions = jnp.empty((n_devices, 0), dtype=jnp.float32)

        # Analyze parameter constancy across devices and generate split function
        # This optimization reduces HLO slice operations by separating constant from varying params
        if n_devices > 1 and static_inputs.shape[1] > 0:
            # Check which columns have identical values across all devices
            first_row = static_inputs[0]
            const_mask = jnp.all(static_inputs == first_row[None, :], axis=0)

            # Voltage columns vary per NR iteration, so mark them as varying
            voltage_set = set(voltage_indices)
            for v_idx in voltage_indices:
                if v_idx < len(const_mask):
                    const_mask = const_mask.at[v_idx].set(False)

            n_const = int(jnp.sum(const_mask))
            n_varying = static_inputs.shape[1] - n_const
            logger.info(f"{model_type} parameter analysis: {n_const}/{static_inputs.shape[1]} constant columns, "
                       f"{n_varying} varying across {n_devices} devices")

            # Compute shared (constant) and varying indices
            shared_indices = [int(i) for i in jnp.where(const_mask)[0]]
            varying_indices_list = [int(i) for i in jnp.where(~const_mask)[0]]

            # Log which parameters vary (for debugging)
            if n_varying > 0 and n_varying <= 30:
                varying_names = [param_names[int(i)] if i < len(param_names) else f"col_{i}"
                                for i in varying_indices_list]
                logger.debug(f"{model_type} varying params: {varying_names}")

            # Generate split function if translator is available
            translator = compiled.get('translator')
            if translator is not None and translator.dae_data is not None:
                logger.info(f"{model_type}: generating split eval function...")
                try:
                    split_fn, split_meta = translator.translate_eval_array_with_cache_split(
                        shared_indices, varying_indices_list
                    )
                    # Create vmapped split function with in_axes=(None, 0, 0)
                    # shared_params broadcasts, device_params and cache are mapped
                    vmapped_split_fn = jax.jit(jax.vmap(split_fn, in_axes=(None, 0, 0)))

                    # Extract shared_params (constant) and device_params (varying)
                    shared_params = static_inputs[0, shared_indices]  # shape: (n_const,)
                    device_params = static_inputs[:, varying_indices_list]  # shape: (n_devices, n_varying)

                    # Compute voltage positions within device_params (varying indices)
                    # This maps original voltage indices to their positions in device_params
                    varying_idx_to_pos = {orig_idx: pos for pos, orig_idx in enumerate(varying_indices_list)}
                    voltage_positions = [varying_idx_to_pos[v] for v in voltage_indices
                                        if v in varying_idx_to_pos]
                    voltage_positions = jnp.array(voltage_positions, dtype=jnp.int32)

                    # Store in compiled dict
                    compiled['split_eval_fn'] = split_fn
                    compiled['vmapped_split_eval'] = vmapped_split_fn
                    compiled['shared_indices'] = shared_indices
                    compiled['varying_indices'] = varying_indices_list
                    compiled['shared_params'] = shared_params
                    compiled['device_params'] = device_params
                    compiled['voltage_positions_in_varying'] = voltage_positions
                    compiled['use_split_eval'] = True

                    logger.info(f"{model_type}: split eval function ready "
                               f"(shared={len(shared_indices)}, varying={len(varying_indices_list)})")
                except Exception as e:
                    logger.warning(f"{model_type}: failed to generate split function: {e}")
                    compiled['use_split_eval'] = False

                # Release MIR data now that all code generation is complete
                translator.release_mir_data()
            else:
                compiled['use_split_eval'] = False
        else:
            compiled['use_split_eval'] = False

        return static_inputs, voltage_indices, device_contexts, cache, collapse_decisions

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

        metadata = compiled['array_metadata']
        node_names = metadata['node_names']  # Residual node names
        jacobian_keys = metadata['jacobian_keys']  # (row_name, col_name) pairs

        n_devices = len(device_contexts)
        n_residuals = len(node_names)
        n_jac_entries = len(jacobian_keys)

        # Build residual index array
        res_indices = np.full((n_devices, n_residuals), -1, dtype=np.int32)

        for dev_idx, ctx in enumerate(device_contexts):
            node_map = ctx['node_map']
            for res_idx, node_name in enumerate(node_names):
                # Map node name to global index
                if node_name.startswith('sim_'):
                    model_node = node_name[4:]
                else:
                    model_node = node_name
                node_idx = node_map.get(model_node, node_map.get(node_name, None))

                if node_idx is not None and node_idx != ground and node_idx > 0:
                    res_indices[dev_idx, res_idx] = node_idx - 1  # 0-indexed residual

        # Build Jacobian index arrays
        jac_row_indices = np.full((n_devices, n_jac_entries), -1, dtype=np.int32)
        jac_col_indices = np.full((n_devices, n_jac_entries), -1, dtype=np.int32)

        for dev_idx, ctx in enumerate(device_contexts):
            node_map = ctx['node_map']
            for jac_idx, (row_name, col_name) in enumerate(jacobian_keys):
                # Map row node
                row_model = row_name[4:] if row_name.startswith('sim_') else row_name
                row_idx = node_map.get(row_model, node_map.get(row_name, None))

                # Map col node
                col_model = col_name[4:] if col_name.startswith('sim_') else col_name
                col_idx = node_map.get(col_model, node_map.get(col_name, None))

                if (row_idx is not None and col_idx is not None and
                    row_idx != ground and col_idx != ground and
                    row_idx > 0 and col_idx > 0):
                    jac_row_indices[dev_idx, jac_idx] = row_idx - 1
                    jac_col_indices[dev_idx, jac_idx] = col_idx - 1

        return {
            'res_indices': jnp.array(res_indices),
            'jac_row_indices': jnp.array(jac_row_indices),
            'jac_col_indices': jnp.array(jac_col_indices),
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

            _, _, stamp_indices, _, _, _, _ = static_inputs_cache[model_type]
            jac_row_idx = stamp_indices['jac_row_indices']  # (n_devices, n_jac_entries)
            jac_col_idx = stamp_indices['jac_col_indices']

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
        for src_type in ('vsource',):  # isource has no Jacobian
            if src_type not in source_device_data:
                continue
            d = source_device_data[src_type]
            j_rows = d['j_rows'].ravel()
            j_cols = d['j_cols'].ravel()
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
        dummy_vals = np.ones(n_entries, dtype=np.float64)
        indices = np.stack([merged_rows, merged_cols], axis=1)

        J_bcoo = BCOO((jnp.array(dummy_vals), jnp.array(indices)), shape=(n_unknowns, n_unknowns))
        J_bcoo_summed = J_bcoo.sum_duplicates()

        # Convert to BCSR to get the fixed structure
        J_bcsr = BCSR.from_bcoo(J_bcoo_summed)

        # Store the structure
        return {
            'n_entries': n_entries,
            'n_unknowns': n_unknowns,
            'merged_rows': jnp.array(merged_rows),
            'merged_cols': jnp.array(merged_cols),
            'openvaf_jac_slices': openvaf_jac_slices,
            'source_jac_slices': source_jac_slices,
            'diag_start': diag_start,
            # BCSR structure (fixed)
            'bcsr_indices': J_bcsr.indices,
            'bcsr_indptr': J_bcsr.indptr,
            'bcsr_n_data': J_bcsr.data.shape[0],
            # For sum_duplicates mapping
            'bcoo_summed_indices': J_bcoo_summed.indices,
        }

    def _parse_voltage_param(self, name: str, node_map: Dict[str, int],
                              model_nodes: List[str], ground: int) -> Tuple[int, int]:
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
            'NOI': 'node4',
            'GP': 'node5', 'SI': 'node6', 'DI': 'node7', 'BP': 'node8',
            'BI': 'node9', 'BS': 'node10', 'BD': 'node11',
            'G': 'node1', 'D': 'node0', 'S': 'node2', 'B': 'node3',
            # Diode nodes (A=anode, C=cathode, CI=internal cathode)
            'A': 'node0', 'C': 'node1', 'CI': 'node2',
        }

        # Simple 2-terminal device mapping (resistor, capacitor, inductor)
        # Used as fallback when device has fewer terminals than the mapped node
        simple_2term_map = {
            'A': 'node0', 'B': 'node1',  # resistor/capacitor/inductor terminals
        }

        match = re.match(r'V\(([^,)]+)(?:,([^)]+))?\)', name)
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

    def _prepare_batched_inputs(self, model_type: str, openvaf_devices: List[Dict],
                                 V: jax.Array, device_internal_nodes: Dict[str, Dict[str, int]],
                                 ground: int) -> Tuple[jax.Array, List[Dict]]:
        """Prepare batched inputs for all devices of a given OpenVAF model type.

        This is the original method kept for backwards compatibility.
        For better performance, use _prepare_static_inputs + _update_voltage_inputs.
        """
        compiled = self._compiled_models.get(model_type)
        if not compiled:
            raise ValueError(f"OpenVAF model {model_type} not compiled")

        param_names = compiled['param_names']
        param_kinds = compiled['param_kinds']
        model_nodes = compiled['nodes']

        all_inputs = []
        device_contexts = []

        # Get defaults from openvaf-py (extracted from Verilog-A source)
        model_defaults = compiled.get('init_param_defaults', {})

        for dev in openvaf_devices:
            ext_nodes = dev['nodes']  # [d, g, s, b]
            params = dev['params']
            internal_nodes = device_internal_nodes.get(dev['name'], {})

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

            # Also map sim_node names
            for i, model_node in enumerate(model_nodes[:-1]):
                node_map[f'sim_{model_node}'] = node_map.get(model_node, ground)

            # Build input array for this device
            inputs = []
            for name, kind in zip(param_names, param_kinds):
                if kind == 'voltage':
                    voltage_val = self._compute_voltage_param(name, V, node_map, model_nodes, ground)
                    inputs.append(voltage_val)
                elif kind == 'param':
                    param_lower = name.lower()
                    if param_lower in params:
                        inputs.append(float(params[param_lower]))
                    elif name in params:
                        inputs.append(float(params[name]))
                    elif 'temperature' in param_lower or name == '$temperature':
                        inputs.append(300.15)
                    elif param_lower == 'mfactor':
                        inputs.append(params.get('mfactor', 1.0))
                    elif param_lower in model_defaults:
                        # Use model-specific default (handles tnom, etc. correctly)
                        inputs.append(model_defaults[param_lower])
                    elif param_lower in ('tnom', 'tref', 'tr'):
                        # Temperature reference in Celsius (most VA models use 27°C)
                        inputs.append(27.0)
                    else:
                        inputs.append(1.0)
                elif kind == 'hidden_state':
                    inputs.append(0.0)
                elif kind == 'sysfun':
                    # System functions like mfactor
                    # mfactor is the system-level device multiplier and must be 1.0 by default
                    if name.lower() == 'mfactor':
                        inputs.append(params.get('mfactor', 1.0))
                    else:
                        inputs.append(0.0)
                elif kind == 'temperature':
                    inputs.append(300.15)
                else:
                    inputs.append(0.0)

            all_inputs.append(inputs)
            device_contexts.append({
                'name': dev['name'],
                'node_map': node_map,
                'ext_nodes': ext_nodes,
            })

        return jnp.array(all_inputs), device_contexts

    def _stamp_batched_results(self, model_type: str, batch_residuals: jax.Array,
                                batch_jacobian: jax.Array, device_contexts: List[Dict],
                                f: jax.Array, J: jax.Array, ground: int) -> Tuple[jax.Array, jax.Array]:
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

        metadata = compiled['array_metadata']
        node_names = metadata['node_names']
        jacobian_keys = metadata['jacobian_keys']

        for dev_idx, ctx in enumerate(device_contexts):
            node_map = ctx['node_map']

            # Stamp residuals
            for res_idx, node_name in enumerate(node_names):
                if node_name.startswith('sim_'):
                    model_node = node_name[4:]
                else:
                    model_node = node_name

                node_idx = node_map.get(model_node, node_map.get(node_name, None))
                if node_idx is None or node_idx == ground:
                    continue

                if node_idx > 0 and node_idx - 1 < f.shape[0]:
                    resist = batch_residuals[dev_idx, res_idx]
                    resist_safe = jnp.where(jnp.isnan(resist), 0.0, resist)
                    f = f.at[node_idx - 1].add(resist_safe)

            # Stamp Jacobian
            for jac_idx, (row_name, col_name) in enumerate(jacobian_keys):
                row_model = row_name[4:] if row_name.startswith('sim_') else row_name
                col_model = col_name[4:] if col_name.startswith('sim_') else col_name

                row_idx = node_map.get(row_model, node_map.get(row_name, None))
                col_idx = node_map.get(col_model, node_map.get(col_name, None))

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
        """
        from jax_spice.analysis.integration import IntegrationMethod, get_method_from_options

        # Default values
        self.analysis_params = {
            'type': 'tran',
            'step': 1e-6,
            'stop': 1e-3,
            'icmode': 'op',
            'tran_method': IntegrationMethod.BACKWARD_EULER,
        }

        # Try to use the parsed control block first
        if self.circuit and self.circuit.control:
            control = self.circuit.control

            # Extract tran_method from options
            if control.options:
                self.analysis_params['tran_method'] = get_method_from_options(
                    control.options.params
                )
                logger.debug(f"Integration method: {self.analysis_params['tran_method']}")

            # Extract analysis parameters from first tran analysis
            for analysis in control.analyses:
                if analysis.analysis_type == 'tran':
                    params = analysis.params
                    if 'step' in params:
                        self.analysis_params['step'] = self.parse_spice_number(params['step'])
                    if 'stop' in params:
                        self.analysis_params['stop'] = self.parse_spice_number(params['stop'])
                    if 'maxstep' in params:
                        self.analysis_params['maxstep'] = self.parse_spice_number(params['maxstep'])
                    if 'icmode' in params:
                        icmode = params['icmode']
                        if isinstance(icmode, str):
                            icmode = icmode.strip('"\'')
                        self.analysis_params['icmode'] = icmode
                    break  # Use first tran analysis found

            logger.debug(f"Analysis (from control block): {self.analysis_params}")
            return

        # Fallback: regex parsing for old-style netlists
        text = self.sim_path.read_text()

        # Find tran analysis line
        match = re.search(r'analysis\s+\w+\s+tran\s+([^\n]+)', text)
        if match:
            params_str = match.group(1)
            # Parse individual parameters - they can be in any order
            step_match = re.search(r'step=(\S+)', params_str)
            stop_match = re.search(r'stop=(\S+)', params_str)
            maxstep_match = re.search(r'maxstep=(\S+)', params_str)
            icmode_match = re.search(r'icmode="(\w+)"', params_str)

            if step_match:
                self.analysis_params['step'] = self.parse_spice_number(step_match.group(1))
            if stop_match:
                self.analysis_params['stop'] = self.parse_spice_number(stop_match.group(1))
            if icmode_match:
                self.analysis_params['icmode'] = icmode_match.group(1)

        # Try to extract tran_method from options line
        tran_method_match = re.search(r'tran_method\s*=\s*"?(\w+)"?', text)
        if tran_method_match:
            try:
                self.analysis_params['tran_method'] = IntegrationMethod.from_string(
                    tran_method_match.group(1)
                )
            except ValueError:
                pass  # Keep default

        logger.debug(f"Analysis (from regex): {self.analysis_params}")

    def _build_source_fn(self):
        """Build time-varying source function from device parameters."""
        sources = {}

        for dev in self.devices:
            if dev['model'] not in ('vsource', 'isource'):
                continue

            params = dev['params']
            source_type = str(params.get('type', 'dc')).lower()

            logger.debug(f"  Source {dev['name']}: type={source_type}")

            if source_type in ('dc', '0', '0.0', ''):
                # DC source - constant value
                dc_val = params.get('dc', 0)
                sources[dev['name']] = lambda t, v=dc_val: v

            elif source_type == 'pulse':
                # Pulse source - using jnp.where for GPU compatibility
                val0 = params.get('val0', 0)
                val1 = params.get('val1', 1)
                rise = params.get('rise', 1e-9)
                fall = params.get('fall', 1e-9)
                width = params.get('width', 1e-6)
                period = params.get('period', 2e-6)
                delay = params.get('delay', 0)

                def pulse_fn(t, v0=val0, v1=val1, r=rise, f=fall, w=width, p=period, d=delay):
                    # Use jnp.where for GPU-friendly conditionals
                    t_in_period = (t - d) % p
                    # Rising edge
                    rising = v0 + (v1 - v0) * t_in_period / r
                    # Falling edge
                    falling = v1 - (v1 - v0) * (t_in_period - r - w) / f
                    # Select based on phase
                    return jnp.where(
                        t < d, v0,
                        jnp.where(
                            t_in_period < r, rising,
                            jnp.where(
                                t_in_period < r + w, v1,
                                jnp.where(t_in_period < r + w + f, falling, v0)
                            )
                        )
                    )

                sources[dev['name']] = pulse_fn

            elif source_type == 'sine':
                # Sine source
                ampl = params.get('ampl', 1)
                freq = params.get('freq', 1e3)
                sinedc = params.get('sinedc', 0)
                phase = params.get('phase', 0)

                def sine_fn(t, a=ampl, f=freq, dc=sinedc, ph=phase):
                    return dc + a * jnp.sin(2 * jnp.pi * f * t + ph)

                sources[dev['name']] = sine_fn

            else:
                raise ValueError(f"Unknown source type '{source_type}' for device {dev['name']}")

        def source_fn(t):
            return {name: fn(t) for name, fn in sources.items()}

        return source_fn

    @profile
    def run_transient(self, t_stop: Optional[float] = None, dt: Optional[float] = None,
                      max_steps: int = 10000, use_sparse: Optional[bool] = None,
                      backend: Optional[str] = None,
                      use_scan: bool = False,
                      use_while_loop: bool = False,
                      profile_config: Optional['ProfileConfig'] = None) -> TransientResult:
        """Run transient analysis.

        All computation is JIT-compiled. Automatically uses sparse matrices
        for large circuits (>1000 nodes).

        Args:
            t_stop: Stop time (default: from analysis params or 1ms)
            dt: Time step (default: from analysis params or 1µs)
            max_steps: Maximum number of time steps
            use_sparse: Force sparse (True) or dense (False) solver. If None, auto-detect.
            backend: 'gpu', 'cpu', or None (auto-select based on circuit size).
                     For circuits >500 nodes with GPU available, uses GPU acceleration.
            use_scan: If True, use lax.scan (pre-computes all source values)
            use_while_loop: If True, use lax.while_loop (computes sources on-the-fly)
            profile_config: If provided, profile just the core simulation (not setup)

        Returns:
            TransientResult with times, voltages, and stats
        """
        from jax_spice.analysis.gpu_backend import select_backend, is_gpu_available

        if t_stop is None:
            t_stop = self.analysis_params.get('stop', 1e-3)
        if dt is None:
            dt = self.analysis_params.get('step', 1e-6)

        # Limit number of steps
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

        use_dense = not use_sparse

        if use_while_loop:
            # lax.while_loop version - computes sources on-the-fly
            logger.info(f"Using lax.while_loop solver ({self.num_nodes} nodes, "
                       f"{'sparse' if use_sparse else 'dense'})")
            return self._run_transient_while_loop(t_stop, dt, backend=backend, use_dense=use_dense,
                                                   profile_config=profile_config)

        if use_sparse:
            logger.info(f"Using BCOO/BCSR sparse solver ({self.num_nodes} nodes)")
            return self._run_transient_hybrid(t_stop, dt, backend=backend, use_dense=False)
        else:
            logger.info(f"Using dense solver ({self.num_nodes} nodes)")
            return self._run_transient_hybrid(t_stop, dt, backend=backend, use_dense=True)

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

    def _compute_collapse_roots(self, collapsible_pairs: List[Tuple[int, int]], n_nodes: int) -> Dict[int, int]:
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
            if not dev.get('is_openvaf'):
                continue

            model_type = dev['model']
            compiled = self._compiled_models.get(model_type)
            if not compiled:
                continue

            model_nodes = compiled['nodes']
            n_model_nodes = len(model_nodes)

            # Get precomputed collapse pairs from _compute_early_collapse_decisions()
            # This uses OpenVAF's generic collapse mechanism
            device_name = dev['name']
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
            ext_nodes = dev['nodes']
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

            device_internal_nodes[dev['name']] = internal_map

        if device_internal_nodes:
            n_internal = next_internal - n_external
            logger.info(f"Allocated {n_internal} internal nodes for {len(device_internal_nodes)} OpenVAF devices")

        return next_internal, device_internal_nodes

    def _run_transient_hybrid(self, t_stop: float, dt: float,
                               backend: str = "cpu",
                               use_dense: bool = True) -> TransientResult:
        """Run transient analysis with OpenVAF devices.

        This solver handles a mix of simple devices (resistor, capacitor, etc.)
        and OpenVAF-compiled devices (like PSP103 MOSFETs).

        Args:
            t_stop: Simulation stop time
            dt: Time step
            backend: 'gpu' or 'cpu' for device evaluation.
            use_dense: If True, use dense matrices with batched scatter (faster for small circuits).
                      If False, use COO collection + sparse CSR solve (better for large circuits).
        """

        logger.info(f"Importing backend ({backend})")

        from jax_spice.analysis.gpu_backend import get_device, get_default_dtype
        from jax_spice.analysis.sparse import build_csr_arrays, sparse_solve_csr
        from jax.experimental.sparse import BCOO, BCSR

        ground = 0
    
        # Get target device for JAX operations

        logger.info("getting device and dtype")
        device = get_device(backend)
        dtype = get_default_dtype(backend)

        # Create transient setup cache key (topology-based)
        setup_cache_key = f"{self.num_nodes}_{len(self.devices)}_{use_dense}_{backend}"

        # Check if we have cached transient setup data
        if (self._transient_setup_cache is not None and
            self._transient_setup_key == setup_cache_key):
            # Reuse cached setup data
            logger.info("Reusing cached transient setup")
            setup = self._transient_setup_cache
            n_total = setup['n_total']
            device_internal_nodes = setup['device_internal_nodes']
            n_unknowns = setup['n_unknowns']
            source_fn = setup['source_fn']
            openvaf_by_type = setup['openvaf_by_type']
            vmapped_fns = setup['vmapped_fns']
            static_inputs_cache = setup['static_inputs_cache']
            source_device_data = setup['source_device_data']
        else:
            # Build all setup data (first time or after topology change)
            logger.info("Building transient setup (first run)...")

            # Set up internal nodes for OpenVAF devices
            n_total, device_internal_nodes = self._setup_internal_nodes()
            n_unknowns = n_total - 1

            # Build time-varying source function
            source_fn = self._build_source_fn()

            # Group devices by type
            # All non-source devices (resistor, capacitor, diode, etc.) go through OpenVAF
            # Only vsource and isource remain as "source devices" handled separately
            openvaf_by_type: Dict[str, List[Dict]] = {}
            source_devices = []
            for dev in self.devices:
                if dev.get('is_openvaf'):
                    model_type = dev['model']
                    if model_type not in openvaf_by_type:
                        openvaf_by_type[model_type] = []
                    openvaf_by_type[model_type].append(dev)
                elif dev['model'] in ('vsource', 'isource'):
                    source_devices.append(dev)

            logger.debug(f"{len(source_devices)} source devices")
            # Prepare OpenVAF: vmapped functions, static inputs, and stamp index mappings
            vmapped_fns: Dict[str, Callable] = {}
            static_inputs_cache: Dict[str, Tuple[Any, List[int], List[Dict], Dict]] = {}

            for model_type in openvaf_by_type:
                compiled = self._compiled_models.get(model_type)
                if compiled and 'vmapped_fn' in compiled:
                    logger.debug(f"{model_type} already compiled")
                    vmapped_fns[model_type] = compiled['vmapped_fn']
                    # Also store vmapped_eval_with_cache if available (captured at setup, not looked up in traced fn)
                    if 'vmapped_eval_with_cache' in compiled:
                        logger.debug(f"{model_type} has vmapped_eval_with_cache")
                        vmapped_fns[model_type + '_with_cache'] = compiled['vmapped_eval_with_cache']
                    static_inputs, voltage_indices, device_contexts, cache, collapse_decisions = self._prepare_static_inputs(
                        model_type, openvaf_by_type[model_type], device_internal_nodes, ground
                    )
                    # Pre-compute stamp index mapping (once per model type)
                    logger.debug("building stamp_indicies")
                    stamp_indices = self._build_stamp_index_mapping(
                        model_type, device_contexts, ground
                    )
                    # Pre-compute voltage node arrays for vectorized update
                    n_devices = len(device_contexts)
                    # Build arrays directly from list comprehension (setup phase only)
                    logger.debug("building voltage nodes")
                    voltage_node1 = jnp.array([
                        [n1 for n1, n2 in ctx['voltage_node_pairs']]
                        for ctx in device_contexts
                    ], dtype=jnp.int32)
                    voltage_node2 = jnp.array([
                        [n2 for n1, n2 in ctx['voltage_node_pairs']]
                        for ctx in device_contexts
                    ], dtype=jnp.int32)

                    if backend == "gpu":
                        with jax.default_device(device):
                            static_inputs = jnp.array(static_inputs, dtype=dtype)
                            cache = jnp.array(cache, dtype=dtype)
                    else:
                        static_inputs = jnp.array(static_inputs, dtype=jnp.float64)
                        cache = jnp.array(cache, dtype=jnp.float64)
                    static_inputs_cache[model_type] = (
                        static_inputs, voltage_indices, stamp_indices, voltage_node1, voltage_node2, cache, collapse_decisions
                    )
                    n_devs = len(openvaf_by_type[model_type])
                    logger.info(f"Prepared {model_type}: {n_devs} devices, cache_size={cache.shape[1]}")

            # Pre-compute source device stamp indices
            source_device_data = self._prepare_source_devices_coo(source_devices, ground, n_unknowns)

            # Cache setup data for reuse
            self._transient_setup_cache = {
                'n_total': n_total,
                'device_internal_nodes': device_internal_nodes,
                'n_unknowns': n_unknowns,
                'source_fn': source_fn,
                'openvaf_by_type': openvaf_by_type,
                'vmapped_fns': vmapped_fns,
                'static_inputs_cache': static_inputs_cache,
                'source_device_data': source_device_data,
            }
            self._transient_setup_key = setup_cache_key
            logger.info("Cached transient setup for reuse")

        # Variables used outside the cached setup block
        n_external = self.num_nodes
        solver_type = "dense batched scatter" if use_dense else "COO sparse"
        logger.info(f"Total nodes: {n_total} ({n_external} external, {n_total - n_external} internal)")
        logger.info(f"Backend: {backend}, device: {device.platform}")
        logger.info(f"Using {solver_type} solver")

        # Initialize voltages
        V = jnp.zeros(n_total, dtype=jnp.float64)
        V_prev = jnp.zeros(n_total, dtype=jnp.float64)

        # Helper to build source value arrays from dict (called once per timestep)
        def build_source_arrays(source_values: Dict) -> Tuple[jax.Array, jax.Array]:
            """Convert source_values dict to JAX arrays for vectorized GPU access."""
            if 'vsource' in source_device_data:
                d = source_device_data['vsource']
                vsource_vals = jnp.array([
                    source_values.get(name, float(dc))
                    for name, dc in zip(d['names'], d['dc'])
                ])
            else:
                vsource_vals = jnp.array([])

            if 'isource' in source_device_data:
                d = source_device_data['isource']
                isource_vals = jnp.array([
                    source_values.get(name, float(dc))
                    for name, dc in zip(d['names'], d['dc'])
                ])
            else:
                isource_vals = jnp.array([])

            return vsource_vals, isource_vals

        # Time stepping

        logger.info("Running time steps")

        # Determine source counts for cache key
        n_vsources = len(source_device_data.get('vsource', {}).get('names', []))
        n_isources = len(source_device_data.get('isource', {}).get('names', []))
        n_nodes = n_unknowns + 1  # Include ground

        # Create cache key from circuit topology
        cache_key = (n_nodes, n_vsources, n_isources, use_dense)

        # Check if we have a cached AoT-compiled solver for this topology
        if hasattr(self, '_cached_nr_solve') and self._cached_solver_key == cache_key:
            logger.info("Reusing cached AoT-compiled NR solver")
            nr_solve = self._cached_nr_solve
        else:
            # Create GPU-resident build_system function
            # Returns (build_system_fn, device_arrays) - large arrays passed as args to avoid XLA constant folding
            build_system_fn, device_arrays = self._make_gpu_resident_build_system_fn(
                source_device_data, vmapped_fns, static_inputs_cache, n_unknowns, use_dense
            )

            # Store device_arrays for passing through the solver chain
            # These will be passed as traced arguments to avoid XLA constant folding
            self._device_arrays = device_arrays

            # JIT compile build_system_fn - device_arrays will be passed as argument
            build_system_jit = jax.jit(build_system_fn)
            logger.info("Created JIT-wrapped build_system function")

            # Collect NOI node indices (PSP103 noise correlation internal node)
            # These have 1e40 conductance to ground and must be kept at 0V
            noi_indices = []
            if device_internal_nodes:
                for dev_name, internal_nodes in device_internal_nodes.items():
                    if 'node4' in internal_nodes:  # NOI is node4 in PSP103
                        noi_indices.append(internal_nodes['node4'])
            noi_indices = jnp.array(noi_indices, dtype=jnp.int32) if noi_indices else None

            # Create JIT-compiled NR solver
            if use_dense:
                # Dense solver for small/medium circuits
                nr_solve = self._make_jit_compiled_solver(
                    build_system_jit, n_nodes, device_arrays, noi_indices=noi_indices,
                    max_iterations=MAX_NR_ITERATIONS, abstol=DEFAULT_ABSTOL, max_step=1.0
                )
            else:
                # Sparse solver for large circuits
                # Pre-compute nse by running build_system once to get sparsity pattern
                logger.info("Pre-computing sparse matrix nse...")
                V_init = jnp.zeros(n_nodes, dtype=jnp.float64)
                vsource_init = jnp.zeros(n_vsources, dtype=jnp.float64)
                isource_init = jnp.zeros(n_isources, dtype=jnp.float64)
                Q_init = jnp.zeros(n_unknowns, dtype=jnp.float64)
                # Use inv_dt=0 for DC (no reactive terms in probe)
                J_bcoo_probe, _, _ = build_system_fn(V_init, vsource_init, isource_init, Q_init, 0.0, device_arrays)

                # Sum duplicates to get true nse
                unique_indices = jnp.unique(
                    J_bcoo_probe.indices[:, 0] * n_unknowns + J_bcoo_probe.indices[:, 1],
                    size=None
                )
                nse = int(unique_indices.shape[0])
                logger.info(f"Sparse matrix: {J_bcoo_probe.nse} entries -> {nse} unique (nse)")

                # Try Spineax (cuDSS with cached symbolic factorization) on GPU
                use_spineax = False
                if jax.default_backend() == 'gpu':
                    try:
                        from spineax.cudss.solver import CuDSSSolver
                        use_spineax = True
                        logger.info("Spineax available - will use cuDSS with cached symbolic factorization")
                    except Exception as e:
                        logger.info(f"Spineax not available ({type(e).__name__}: {e}) - using JAX spsolve")

                if use_spineax:
                    # Pre-compute BCSR pattern for Spineax
                    from jax.experimental.sparse import BCSR
                    J_bcoo_dedup = J_bcoo_probe.sum_duplicates(nse=nse)
                    J_bcsr_probe = BCSR.from_bcoo(J_bcoo_dedup)
                    logger.info(f"BCSR pattern: indptr={J_bcsr_probe.indptr.shape}, indices={J_bcsr_probe.indices.shape}")

                    nr_solve = self._make_spineax_jit_compiled_solver(
                        build_system_jit, n_nodes, nse,
                        bcsr_indptr=J_bcsr_probe.indptr,
                        bcsr_indices=J_bcsr_probe.indices,
                        device_arrays=device_arrays,
                        noi_indices=noi_indices,
                        max_iterations=MAX_NR_ITERATIONS, abstol=DEFAULT_ABSTOL, max_step=1.0
                    )
                else:
                    # Fallback to JAX spsolve (QR factorization, no caching)
                    nr_solve = self._make_sparse_jit_compiled_solver(
                        build_system_jit, n_nodes, nse, device_arrays,
                        noi_indices=noi_indices,
                        max_iterations=MAX_NR_ITERATIONS, abstol=DEFAULT_ABSTOL, max_step=1.0
                    )

            # Cache JIT-wrapped solver and build_system (JAX handles compilation automatically)
            self._cached_nr_solve = nr_solve
            self._cached_solver_key = cache_key
            self._cached_build_system = build_system_jit
            logger.info(f"Cached {'dense' if use_dense else 'sparse'} NR solver")

            # Note: No explicit warmup needed - JIT compilation happens on first real use
            # (DC operating point solve or first transient step)

        # Early return for setup-only calls (t_stop=0 from _run_transient_while_loop)
        # Setup cache and NR solver are now cached, no need to run DC or transient
        if t_stop <= 0.0:
            return TransientResult(
                times=jnp.array([0.0]),
                voltages={name: jnp.zeros((1,)) for name in self.node_names},
                stats={'setup_only': True},
            )

        # Compute initial condition based on icmode
        icmode = self.analysis_params.get('icmode', 'op')
        if icmode == 'op':
            V = self._compute_dc_operating_point(
                n_nodes=n_nodes,
                n_vsources=n_vsources,
                n_isources=n_isources,
                nr_solve=nr_solve,
                device_arrays=device_arrays,
                backend=backend,
                use_dense=use_dense,
                device_internal_nodes=device_internal_nodes,
                source_device_data=source_device_data,
                vmapped_fns=vmapped_fns,
                static_inputs_cache=static_inputs_cache,
            )
        else:
            # icmode='uic' - initialize VDD nodes to supply voltage
            V = jnp.zeros(n_nodes, dtype=jnp.float64)
            vdd_value = 0.0
            for dev in self.devices:
                if dev['model'] == 'vsource':
                    dc_val = dev['params'].get('dc', 0.0)
                    if dc_val > vdd_value:
                        vdd_value = dc_val
            for name, idx in self.node_names.items():
                if 'vdd' in name.lower() or 'vcc' in name.lower():
                    V = V.at[idx].set(vdd_value)

        times_list = []
        voltage_history = []  # Collect V arrays, convert to dict at end

        total_nr_iters = 0
        non_converged_steps = []  # Track (time, max_residual) for non-converged steps

        # Initialize charge state for reactive (capacitance) tracking
        # Q represents charges at each node, used for backward Euler integration
        # Q has shape (n_unknowns,) = (n_nodes - 1,) since ground is excluded
        n_unknowns = n_nodes - 1
        inv_dt = 1.0 / dt  # Precompute for transient (1/timestep)

        # Initialize Q_prev from the DC operating point to avoid discontinuity at t=0
        if hasattr(self, '_cached_build_system'):
            vsource_dc = source_device_data.get('vsource', {}).get('dc', jnp.array([]))
            isource_dc = source_device_data.get('isource', {}).get('dc', jnp.array([]))
            if vsource_dc.size == 0:
                vsource_dc = jnp.array([])
            if isource_dc.size == 0:
                isource_dc = jnp.array([])
            _, _, Q_prev = self._cached_build_system(V, vsource_dc, isource_dc, jnp.zeros(n_unknowns), 0.0, device_arrays)
            Q_prev.block_until_ready()
        else:
            Q_prev = jnp.zeros(n_unknowns, dtype=jnp.float64)

        # Use integer-based iteration to avoid floating-point comparison issues
        # This ensures Python loop and lax.scan produce the same number of timesteps
        num_timesteps = int(round(t_stop / dt)) + 1

        logger.info(f"Starting NR iteration ({num_timesteps} timesteps, inv_dt={inv_dt:.2e})")
        for step_idx in range(num_timesteps):
            t = step_idx * dt
            source_values = source_fn(t)
            # Build source value arrays once per timestep (Python loop here, not in NR loop)
            vsource_vals, isource_vals = build_source_arrays(source_values)

            # GPU-resident NR solve - JIT compiled, runs on GPU via lax.while_loop
            # Backward Euler: f = f_resist + (Q - Q_prev)/dt, J = J_resist + C/dt
            V_new, iterations, converged, max_f, Q = nr_solve(V, vsource_vals, isource_vals, Q_prev, inv_dt, device_arrays)

            # Transfer results back to Python for logging/tracking (once per timestep)
            nr_iters = int(iterations)
            is_converged = bool(converged)
            residual = float(max_f)

            V = V_new
            Q_prev = Q  # Update charge state for next timestep
            total_nr_iters += nr_iters

            if not is_converged:
                non_converged_steps.append((t, residual))
                if nr_iters >= MAX_NR_ITERATIONS:
                    logger.warning(f"t={t:.2e}s hit max iterations ({MAX_NR_ITERATIONS}), max_f={residual:.2e}")
                else:
                    logger.warning(f"t={t:.2e}s did not converge (max_f={residual:.2e})")

            # Record state (keep as JAX arrays, convert at end)
            times_list.append(t)
            voltage_history.append(V[:n_external])  # Single JAX slice, no float() calls

            V_prev = V

        # Build stats dict
        stats = {
            'total_timesteps': len(times_list),
            'total_nr_iterations': total_nr_iters,
            'non_converged_count': len(non_converged_steps),
            'non_converged_steps': non_converged_steps,
            'convergence_rate': 1.0 - len(non_converged_steps) / max(len(times_list), 1),
        }

        logger.info(f"Completed: {len(times_list)} timesteps, {total_nr_iters} total NR iterations")
        if non_converged_steps:
            logger.info(f"  Non-converged: {len(non_converged_steps)} steps ({100*(1-stats['convergence_rate']):.1f}%)")

        # Convert voltage history to dict format (single stack operation)
        times = jnp.array(times_list)
        if voltage_history:
            V_stacked = jnp.stack(voltage_history)  # Shape: (n_timesteps, n_external)
            voltages = {i: V_stacked[:, i] for i in range(n_external)}
        else:
            voltages = {i: jnp.array([]) for i in range(n_external)}

        return TransientResult(times=times, voltages=voltages, stats=stats)

    def _get_dc_source_values(
        self, n_vsources: int, n_isources: int
    ) -> Tuple[jax.Array, jax.Array]:
        """Extract DC values from voltage and current sources.

        Args:
            n_vsources: Number of voltage sources
            n_isources: Number of current sources

        Returns:
            Tuple of (vsource_dc_vals, isource_dc_vals) as JAX arrays
        """
        vsource_dc_vals = jnp.zeros(n_vsources, dtype=jnp.float64)
        isource_dc_vals = jnp.zeros(n_isources, dtype=jnp.float64)

        vsource_idx = 0
        isource_idx = 0
        for dev in self.devices:
            if dev['model'] == 'vsource':
                dc_val = dev['params'].get('dc', 0.0)
                vsource_dc_vals = vsource_dc_vals.at[vsource_idx].set(float(dc_val))
                vsource_idx += 1
            elif dev['model'] == 'isource':
                source_type = str(dev['params'].get('type', 'dc')).lower()
                dc_val = dev['params'].get('val0' if source_type == 'pulse' else 'dc', 0.0)
                isource_dc_vals = isource_dc_vals.at[isource_idx].set(float(dc_val))
                isource_idx += 1

        return vsource_dc_vals, isource_dc_vals

    def _get_vdd_value(self) -> float:
        """Find the maximum DC voltage from voltage sources (VDD)."""
        vdd_value = 0.0
        for dev in self.devices:
            if dev['model'] == 'vsource':
                dc_val = dev['params'].get('dc', 0.0)
                if dc_val > vdd_value:
                    vdd_value = dc_val
        return vdd_value

    def _compute_dc_operating_point(self, n_nodes: int, n_vsources: int, n_isources: int,
                                     nr_solve: Callable,
                                     device_arrays: Dict[str, Tuple[jax.Array, jax.Array]],
                                     backend: str = "cpu",
                                     use_dense: bool = True,
                                     max_iterations: int = 100,
                                     device_internal_nodes: Optional[Dict[str, Dict[str, int]]] = None,
                                     source_device_data: Optional[Dict[str, Any]] = None,
                                     vmapped_fns: Optional[Dict[str, Callable]] = None,
                                     static_inputs_cache: Optional[Dict[str, Tuple]] = None) -> jax.Array:
        """Compute DC operating point using VACASK-style homotopy chain.

        Uses the homotopy chain (gdev -> gshunt -> src) to find the DC operating
        point even for difficult circuits like ring oscillators where simple
        Newton-Raphson fails due to near-singular Jacobians.

        Args:
            n_nodes: Number of nodes in the system
            n_vsources: Number of voltage sources
            n_isources: Number of current sources
            nr_solve: The cached NR solver function (used for fallback)
            device_arrays: Dict[model_type, (static_inputs, cache)] - passed to nr_solve
            backend: 'gpu' or 'cpu'
            use_dense: Whether using dense solver
            max_iterations: Maximum NR iterations per homotopy step
            device_internal_nodes: Map of device name -> {node_name: circuit_node_idx}
            source_device_data: Pre-computed source device stamp templates
            vmapped_fns: Dict of vmapped OpenVAF functions per model type
            static_inputs_cache: Dict of static inputs per model type

        Returns:
            DC operating point voltages (shape: [n_nodes])
        """
        logger.info("Computing DC operating point...")

        # Find VDD value from voltage sources
        vdd_value = self._get_vdd_value()

        # Initialize V with a good starting point for convergence
        mid_rail = vdd_value / 2.0
        V = jnp.full(n_nodes, mid_rail, dtype=jnp.float64)
        V = V.at[0].set(0.0)  # Ground is always 0

        # Set VDD nodes to full supply voltage
        for name, idx in self.node_names.items():
            name_lower = name.lower()
            if 'vdd' in name_lower or 'vcc' in name_lower:
                V = V.at[idx].set(vdd_value)
                logger.debug(f"  Initialized VDD node '{name}' (idx {idx}) to {vdd_value}V")
            elif name_lower in ('gnd', 'vss', '0'):
                V = V.at[idx].set(0.0)
                logger.debug(f"  Initialized ground node '{name}' (idx {idx}) to 0V")

        # Initialize PSP103 internal nodes
        noi_indices = []
        if device_internal_nodes:
            noi_nodes_initialized = 0
            body_nodes_initialized = 0
            device_external_nodes = {dev['name']: dev.get('nodes', []) for dev in self.devices}

            for dev_name, internal_nodes in device_internal_nodes.items():
                if 'node4' in internal_nodes:
                    noi_idx = internal_nodes['node4']
                    V = V.at[noi_idx].set(0.0)
                    noi_indices.append(noi_idx)
                    noi_nodes_initialized += 1

                ext_nodes = device_external_nodes.get(dev_name, [])
                if len(ext_nodes) >= 4:
                    b_circuit_node = ext_nodes[3]
                    b_voltage = float(V[b_circuit_node])
                    for body_node_name in ['node8', 'node9', 'node10', 'node11']:
                        if body_node_name in internal_nodes:
                            body_idx = internal_nodes[body_node_name]
                            if body_idx > 0 and abs(V[body_idx] - mid_rail) < 0.01:
                                V = V.at[body_idx].set(b_voltage)
                                body_nodes_initialized += 1

            if noi_nodes_initialized > 0:
                logger.debug(f"  Initialized {noi_nodes_initialized} NOI nodes to 0V")
            if body_nodes_initialized > 0:
                logger.debug(f"  Initialized {body_nodes_initialized} body internal nodes")

        noi_indices = jnp.array(noi_indices, dtype=jnp.int32) if noi_indices else None
        logger.debug(f"  Initial V: ground=0V, VDD={vdd_value}V, others={mid_rail}V")

        # Get DC source values
        vsource_dc_vals, isource_dc_vals = self._get_dc_source_values(n_vsources, n_isources)

        n_unknowns = n_nodes - 1
        Q_prev = jnp.zeros(n_unknowns, dtype=jnp.float64)

        # First try direct NR without homotopy (works for well-initialized circuits)
        # Uses analytic jacobians from OpenVAF via the cached nr_solve
        logger.info("  Trying direct NR solver first...")
        V_new, nr_iters, is_converged, max_f, _ = nr_solve(
            V, vsource_dc_vals, isource_dc_vals, Q_prev, 0.0, device_arrays  # inv_dt=0 for DC
        )

        if is_converged:
            V = V_new
            logger.info(f"  DC operating point converged via direct NR "
                       f"({nr_iters} iters, residual={max_f:.2e})")
        else:
            # Fall back to homotopy chain using the cached NR solver
            # This uses analytic jacobians (NOT autodiff)
            logger.info("  Direct NR failed, trying homotopy chain...")

            # Configure homotopy with conservative settings
            homotopy_config = HomotopyConfig(
                gmin=1e-12,
                gdev_start=1e-3,   # Start from moderate GMIN
                gdev_target=1e-13,
                gmin_factor=3.0,   # Conservative stepping factor
                gmin_factor_min=1.1,
                gmin_factor_max=10.0,
                gmin_max=1.0,
                gmin_max_steps=100,
                source_step=0.1,
                source_step_min=0.001,
                source_max_steps=100,
                chain=("gdev", "gshunt", "src"),
                max_iterations=max_iterations,
                abstol=1e-9,
                debug=0,  # Disable debug output for normal runs
            )

            result = run_homotopy_chain(
                nr_solve, V, vsource_dc_vals, isource_dc_vals, Q_prev, device_arrays, homotopy_config
            )

            if result.converged:
                V = result.V
                logger.info(f"  DC operating point converged via {result.method} "
                           f"({result.iterations} total iters, {result.homotopy_steps} homotopy steps)")
            else:
                logger.warning(f"  Homotopy chain did not converge (method={result.method})")
                # For oscillator circuits, accept the best solution we have
                # The DC operating point might be metastable anyway
                V = result.V
                logger.info("  Using best available solution for metastable circuit")

        # Clamp NOI nodes after DC solution
        if noi_indices is not None:
            V = V.at[noi_indices].set(0.0)

        # Log key node voltages
        n_external = self.num_nodes
        logger.info(f"  DC solution: {min(n_external, 5)} node voltages:")
        for i in range(min(n_external, 5)):
            name = next((n for n, idx in self.node_names.items() if idx == i), str(i))
            logger.info(f"    Node {name} (idx {i}): {float(V[i]):.6f}V")

        return V

    def _run_transient_while_loop(self, t_stop: float, dt: float,
                                   backend: str = "cpu",
                                   use_dense: bool = True,
                                   profile_config: Optional['ProfileConfig'] = None) -> TransientResult:
        """Transient analysis using lax.while_loop for the timestep loop.

        This version uses lax.while_loop to eliminate Python loop overhead
        while computing source values on-the-fly (no pre-computation needed).

        Args:
            t_stop: Simulation stop time
            dt: Time step
            backend: 'gpu' or 'cpu' for device evaluation
            use_dense: If True, use dense solver; if False, use sparse solver
            profile_config: If provided, profile just the core simulation loop
        """
        import time as time_module

        # Ensure setup is done
        setup_cache_key = f"{self.num_nodes}_{len(self.devices)}_{use_dense}_{backend}"
        if self._transient_setup_cache is None or self._transient_setup_key != setup_cache_key:
            # Build setup by calling hybrid version with 0 steps
            self._run_transient_hybrid(t_stop=0.0, dt=dt, backend=backend, use_dense=use_dense)

        setup = self._transient_setup_cache
        n_total = setup['n_total']
        n_unknowns = setup['n_unknowns']
        source_device_data = setup['source_device_data']
        device_internal_nodes = setup['device_internal_nodes']
        vmapped_fns = setup.get('vmapped_fns', {})
        static_inputs_cache = setup.get('static_inputs_cache', {})

        n_external = self.num_nodes
        n_nodes = n_unknowns + 1
        n_vsources = len(source_device_data.get('vsource', {}).get('names', []))
        n_isources = len(source_device_data.get('isource', {}).get('names', []))

        # Get or create the NR solver
        cache_key = (n_nodes, n_vsources, n_isources, use_dense)
        if hasattr(self, '_cached_nr_solve') and self._cached_solver_key == cache_key:
            nr_solve = self._cached_nr_solve
        else:
            self._run_transient_hybrid(t_stop=dt, dt=dt, backend=backend, use_dense=use_dense)
            nr_solve = self._cached_nr_solve

        # Build JIT-compatible source evaluation functions
        # These return arrays directly, not dicts
        vsource_fns = []
        vsource_dc = []
        if 'vsource' in source_device_data:
            for name in source_device_data['vsource']['names']:
                dev = next((d for d in self.devices if d['name'] == name), None)
                if dev:
                    fn = self._get_source_fn_for_device(dev)
                    vsource_fns.append(fn if fn else lambda t, v=dev['params'].get('dc', 0): float(v))
                    vsource_dc.append(dev['params'].get('dc', 0))

        isource_fns = []
        isource_dc = []
        if 'isource' in source_device_data:
            for name in source_device_data['isource']['names']:
                dev = next((d for d in self.devices if d['name'] == name), None)
                if dev:
                    fn = self._get_source_fn_for_device(dev)
                    isource_fns.append(fn if fn else lambda t, v=dev['params'].get('dc', 0): float(v))
                    isource_dc.append(dev['params'].get('dc', 0))

        # Pre-allocate output arrays
        # Use round() to avoid floating-point errors in timestep count
        num_timesteps = int(round(t_stop / dt)) + 1
        logger.info(f"While-loop transient: {num_timesteps} timesteps, {n_total} nodes")

        # Generate time array for source pre-computation
        times = jnp.linspace(0.0, t_stop, num_timesteps)

        # Pre-compute source values for ALL timesteps (handles time-varying sources)
        logger.info("Pre-computing source values for all timesteps...")
        t_precompute = time_module.perf_counter()

        # Build vsource values array [num_timesteps, n_vsources]
        if n_vsources > 0:
            vsource_names = source_device_data['vsource']['names']
            all_vsource = []
            for name in vsource_names:
                dev = next((d for d in self.devices if d['name'] == name), None)
                if dev:
                    src_fn = self._get_source_fn_for_device(dev)
                    if src_fn is not None:
                        # Time-varying source - evaluate at all times
                        vals = jax.vmap(src_fn)(times)
                    else:
                        # DC source
                        dc_val = dev['params'].get('dc', 0.0)
                        vals = jnp.full(num_timesteps, float(dc_val))
                    all_vsource.append(vals)
            all_vsource_vals = jnp.stack(all_vsource, axis=1)  # [num_timesteps, n_vsources]
        else:
            all_vsource_vals = jnp.zeros((num_timesteps, 0))

        # Build isource values array [num_timesteps, n_isources]
        if n_isources > 0:
            isource_names = source_device_data['isource']['names']
            all_isource = []
            for name in isource_names:
                dev = next((d for d in self.devices if d['name'] == name), None)
                if dev:
                    src_fn = self._get_source_fn_for_device(dev)
                    if src_fn is not None:
                        vals = jax.vmap(src_fn)(times)
                    else:
                        dc_val = dev['params'].get('dc', 0.0)
                        vals = jnp.full(num_timesteps, float(dc_val))
                    all_isource.append(vals)
            all_isource_vals = jnp.stack(all_isource, axis=1)  # [num_timesteps, n_isources]
        else:
            all_isource_vals = jnp.zeros((num_timesteps, 0))

        logger.info(f"Source pre-computation: {time_module.perf_counter() - t_precompute:.3f}s")

        # Initial state - compute DC operating point if icmode='op'
        icmode = self.analysis_params.get('icmode', 'op')
        if icmode == 'op':
            V0 = self._compute_dc_operating_point(
                n_nodes=n_nodes,
                n_vsources=n_vsources,
                n_isources=n_isources,
                nr_solve=nr_solve,
                device_arrays=self._device_arrays,
                backend=backend,
                use_dense=use_dense,
                device_internal_nodes=device_internal_nodes,
                source_device_data=source_device_data,
                vmapped_fns=vmapped_fns,
                static_inputs_cache=static_inputs_cache,
            )
        else:
            # icmode='uic' - use zeros with VDD nodes initialized
            V0 = jnp.zeros(n_nodes, dtype=jnp.float64)
            # Initialize VDD nodes to supply voltage
            vdd_value = 0.0
            for dev in self.devices:
                if dev['model'] == 'vsource':
                    dc_val = dev['params'].get('dc', 0.0)
                    if dc_val > vdd_value:
                        vdd_value = dc_val
            for name, idx in self.node_names.items():
                if 'vdd' in name.lower() or 'vcc' in name.lower():
                    V0 = V0.at[idx].set(vdd_value)

        # Cache key for the scan function
        # Note: Does NOT include num_timesteps - lax.scan handles variable-length inputs
        # Includes dt in key since inv_dt is captured in the closure
        scan_cache_key = (n_nodes, n_vsources, n_isources, n_external, use_dense, dt)

        if hasattr(self, '_cached_scan_fn') and self._cached_scan_key == scan_cache_key:
            run_simulation_with_outputs = self._cached_scan_fn
            logger.info("Reusing cached lax.scan simulation function")
        else:
            # Create and cache the scan function
            # Source values are passed per-timestep via lax.scan's xs argument
            inv_dt = 1.0 / dt  # Captured in closure for backward Euler

            def make_scan_fn(nr_solve_fn, n_ext, inv_dt_val):
                @jax.jit
                def run_simulation_with_outputs(V_init, Q_init, all_vsource, all_isource, device_arrays_arg):
                    """Run simulation with time-varying sources using lax.scan.

                    Carry includes both V and Q for reactive term tracking.
                    Uses backward Euler: f = f_resist + (Q - Q_prev)/dt
                    device_arrays_arg is passed through to nr_solve to avoid XLA constant folding.
                    """
                    def step_fn(carry, source_vals):
                        V, Q_prev = carry
                        vsource_vals, isource_vals = source_vals
                        V_new, iterations, converged, max_f, Q = nr_solve_fn(
                            V, vsource_vals, isource_vals, Q_prev, inv_dt_val, device_arrays_arg
                        )
                        return (V_new, Q), (V_new[:n_ext], iterations, converged)

                    # Stack source arrays for scan input
                    source_inputs = (all_vsource, all_isource)
                    _, (all_V, all_iters, all_converged) = jax.lax.scan(
                        step_fn, (V_init, Q_init), source_inputs
                    )
                    return all_V, all_iters, all_converged
                return run_simulation_with_outputs

            run_simulation_with_outputs = make_scan_fn(nr_solve, n_external, inv_dt)
            self._cached_scan_fn = run_simulation_with_outputs
            self._cached_scan_key = scan_cache_key
            logger.info(f"Created and cached lax.scan simulation function (inv_dt={inv_dt:.2e})")

        # Initialize charge state for reactive terms by computing Q at the DC operating point
        # This avoids a discontinuity at t=0 (Q - Q_prev = 0 when Q_prev = Q(V0))
        # Q has shape (n_unknowns,) = (n_nodes - 1,) since ground is excluded
        if hasattr(self, '_cached_build_system'):
            # Use DC source values (at t=0)
            vsource_dc = all_vsource_vals[0] if all_vsource_vals.size > 0 else jnp.array([])
            isource_dc = all_isource_vals[0] if all_isource_vals.size > 0 else jnp.array([])
            _, _, Q0 = self._cached_build_system(V0, vsource_dc, isource_dc, jnp.zeros(n_unknowns), 0.0, self._device_arrays)
            logger.debug(f"  Initialized Q0 from DC operating point (max|Q0|={float(jnp.max(jnp.abs(Q0))):.2e})")
        else:
            Q0 = jnp.zeros(n_unknowns, dtype=jnp.float64)

        # Run the simulation (with optional profiling of just the core loop)
        logger.info("Running lax.scan simulation...")
        t0 = time_module.perf_counter()
        if profile_config:
            with profile_section("lax_scan_simulation", profile_config):
                all_V, all_iters, all_converged = run_simulation_with_outputs(V0, Q0, all_vsource_vals, all_isource_vals, self._device_arrays)
                jax.block_until_ready(all_V)
                # Measure time BEFORE profile_section.__exit__ (which saves trace to disk)
                t1 = time_module.perf_counter()
        else:
            all_V, all_iters, all_converged = run_simulation_with_outputs(V0, Q0, all_vsource_vals, all_isource_vals, self._device_arrays)
            jax.block_until_ready(all_V)
            t1 = time_module.perf_counter()
        total_time = t1 - t0

        # Build results
        times = jnp.linspace(0.0, t_stop, num_timesteps)
        total_iters = int(jnp.sum(all_iters))
        non_converged = int(jnp.sum(~all_converged))

        stats = {
            'total_timesteps': num_timesteps,
            'total_nr_iterations': total_iters,
            'non_converged_count': non_converged,
            'non_converged_steps': [],
            'convergence_rate': 1.0 - non_converged / max(num_timesteps, 1),
            'wall_time': total_time,
            'time_per_step_ms': total_time / num_timesteps * 1000,
            'while_loop': True,
        }

        logger.info(f"Completed: {num_timesteps} steps in {total_time:.3f}s "
                   f"({stats['time_per_step_ms']:.2f}ms/step, {total_iters} NR iters)")

        # Convert to dict format
        voltages = {i: all_V[:, i] for i in range(n_external)}

        return TransientResult(times=times, voltages=voltages, stats=stats)

    def _get_source_fn_for_device(self, dev: Dict):
        """Get the source function for a device, or None if not a source."""
        if dev['model'] not in ('vsource', 'isource'):
            return None

        params = dev['params']
        source_type = str(params.get('type', 'dc')).lower()

        if source_type in ('dc', '0', '0.0', ''):
            dc_val = params.get('dc', 0)
            return lambda t, v=dc_val: v

        elif source_type == 'pulse':
            val0 = params.get('val0', 0)
            val1 = params.get('val1', 1)
            rise = params.get('rise', 1e-9)
            fall = params.get('fall', 1e-9)
            width = params.get('width', 1e-6)
            period = params.get('period', 2e-6)
            delay = params.get('delay', 0)

            def pulse_fn(t, v0=val0, v1=val1, r=rise, f=fall, w=width, p=period, d=delay):
                t_in_period = (t - d) % p
                rising = v0 + (v1 - v0) * t_in_period / r
                falling = v1 - (v1 - v0) * (t_in_period - r - w) / f
                return jnp.where(
                    t < d, v0,
                    jnp.where(t_in_period < r, rising,
                        jnp.where(t_in_period < r + w, v1,
                            jnp.where(t_in_period < r + w + f, falling, v0))))
            return pulse_fn

        elif source_type == 'sine':
            sinedc = params.get('sinedc', 0)
            ampl = params.get('ampl', 1)
            freq = params.get('freq', 1e6)
            phase = params.get('phase', 0)

            def sine_fn(t, dc=sinedc, a=ampl, f=freq, ph=phase):
                return dc + a * jnp.sin(2 * jnp.pi * f * t + ph)
            return sine_fn

        return None

    def _prepare_source_devices_coo(
        self,
        source_devices: List[Dict],
        ground: int,
        n_unknowns: int,
    ) -> Dict[str, Any]:
        """Pre-compute data structures and stamp templates for source devices.

        All other devices (resistor, capacitor, diode) are handled via OpenVAF.
        This function only handles vsource and isource.

        Pre-computes static index arrays so runtime collection is fully vectorized
        with no Python loops.

        For 2-terminal devices (p, n), the stamp pattern is:
        - Residual: f[p] += I, f[n] -= I (2 entries, masked by ground)
        - Jacobian: J[p,p] += G, J[p,n] -= G, J[n,p] -= G, J[n,n] += G (4 entries)

        Returns dict with device data and pre-computed stamp templates.
        """
        # Group by model type (only vsource and isource expected)
        logger.debug("Preparing source devices COO")
        by_type: Dict[str, List[Dict]] = {}
        for dev in source_devices:
            model = dev['model']
            if model in ('vsource', 'isource'):
                if model not in by_type:
                    by_type[model] = []
                by_type[model].append(dev)

        result = {}
        for model, devs in by_type.items():
            logger.debug(f"COO for {model}, {devs}")
            n = len(devs)
            # Extract node indices as JAX arrays
            node_p = jnp.array([d['nodes'][0] for d in devs], dtype=jnp.int32)
            node_n = jnp.array([d['nodes'][1] for d in devs], dtype=jnp.int32)
            names = [d['name'] for d in devs]

            # Pre-compute stamp templates for 2-terminal devices
            # Residual indices: [p-1, n-1] for each device, -1 if grounded
            logger.debug("Pre-computing stamp templates")
            f_idx_p = jnp.where(node_p != ground, node_p - 1, -1)
            f_idx_n = jnp.where(node_n != ground, node_n - 1, -1)
            # Stack to shape (n, 2): [[p0, n0], [p1, n1], ...]
            f_indices = jnp.stack([f_idx_p, f_idx_n], axis=1)
            f_signs = jnp.array([1.0, -1.0])  # I at p, -I at n

            # Jacobian indices for 4-entry stamp pattern
            # Entries: (p,p), (p,n), (n,n), (n,p)
            mask_p = node_p != ground
            mask_n = node_n != ground
            mask_both = mask_p & mask_n

            # Row indices: p, p, n, n (or -1 if invalid)
            j_row_pp = jnp.where(mask_p, node_p - 1, -1)
            j_row_pn = jnp.where(mask_both, node_p - 1, -1)
            j_row_nn = jnp.where(mask_n, node_n - 1, -1)
            j_row_np = jnp.where(mask_both, node_n - 1, -1)

            # Col indices: p, n, n, p
            j_col_pp = jnp.where(mask_p, node_p - 1, -1)
            j_col_pn = jnp.where(mask_both, node_n - 1, -1)
            j_col_nn = jnp.where(mask_n, node_n - 1, -1)
            j_col_np = jnp.where(mask_both, node_p - 1, -1)

            # Stack to shape (n, 4)
            j_rows = jnp.stack([j_row_pp, j_row_pn, j_row_nn, j_row_np], axis=1)
            j_cols = jnp.stack([j_col_pp, j_col_pn, j_col_nn, j_col_np], axis=1)
            j_signs = jnp.array([1.0, -1.0, 1.0, -1.0])  # +G, -G, +G, -G

            base_data = {
                'node_p': node_p,
                'node_n': node_n,
                'n': n,
                'f_indices': f_indices,  # (n, 2)
                'f_signs': f_signs,      # (2,)
                'j_rows': j_rows,        # (n, 4)
                'j_cols': j_cols,        # (n, 4)
                'j_signs': j_signs,      # (4,)
            }

            if model == 'vsource':
                dc = jnp.array([d['params'].get('dc', 0.0) for d in devs], dtype=jnp.float64)
                result['vsource'] = {**base_data, 'dc': dc, 'names': names}
            elif model == 'isource':
                dc = jnp.array([d['params'].get('dc', 0.0) for d in devs], dtype=jnp.float64)
                # Current sources have no Jacobian contribution
                result['isource'] = {
                    'node_p': node_p, 'node_n': node_n, 'n': n, 'names': names,
                    'dc': dc,
                    'f_indices': f_indices, 'f_signs': f_signs,
                    # No Jacobian for current sources
                    'j_rows': jnp.zeros((n, 0), dtype=jnp.int32),
                    'j_cols': jnp.zeros((n, 0), dtype=jnp.int32),
                    'j_signs': jnp.array([]),
                }
            logger.debug("Stamp templates complete")
        return result

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
        """Collect COO triplets from source devices using fully vectorized operations.

        All other devices (resistor, capacitor, diode) are handled via OpenVAF.
        This function only handles vsource and isource.

        Uses pre-computed stamp templates from _prepare_source_devices_coo.
        Fully vectorized - no Python loops, all JAX operations.

        Args:
            device_data: Pre-computed stamp templates from _prepare_source_devices_coo
            V: Current voltage vector
            vsource_vals: JAX array of voltage source target values (built once per timestep)
            isource_vals: JAX array of current source values (built once per timestep)
        """

        def _stamp_two_terminal(d: Dict, I: jax.Array, G: jax.Array):
            """Vectorized stamp for 2-terminal devices with current I and conductance G."""
            # Residual: shape (n, 2) -> flatten to (2*n,)
            f_vals = I[:, None] * d['f_signs'][None, :]  # (n, 2)
            f_idx = d['f_indices'].ravel()  # (2*n,)
            f_val = f_vals.ravel()  # (2*n,)

            # Jacobian: shape (n, 4) -> flatten to (4*n,)
            j_vals_arr = G[:, None] * d['j_signs'][None, :]  # (n, 4)
            j_row = d['j_rows'].ravel()  # (4*n,)
            j_col = d['j_cols'].ravel()  # (4*n,)
            j_val = j_vals_arr.ravel()  # (4*n,)

            # Filter valid entries (index >= 0)
            f_valid = f_idx >= 0
            j_valid = j_row >= 0

            f_indices.append(jnp.where(f_valid, f_idx, 0))
            f_values.append(jnp.where(f_valid, f_val, 0.0))
            j_rows.append(jnp.where(j_valid, j_row, 0))
            j_cols.append(jnp.where(j_valid, j_col, 0))
            j_vals.append(jnp.where(j_valid, j_val, 0.0))

        # Voltage sources: I = G * (Vp - Vn - Vtarget), G = 1e12
        if 'vsource' in device_data and vsource_vals.size > 0:
            d = device_data['vsource']
            G = 1e12
            Vp, Vn = V[d['node_p']], V[d['node_n']]
            # vsource_vals is pre-built JAX array - no Python loop here
            I = G * (Vp - Vn - vsource_vals)
            G_arr = jnp.full(d['n'], G)
            _stamp_two_terminal(d, I, G_arr)

        # Current sources (residual only, no Jacobian)
        if 'isource' in device_data and isource_vals.size > 0:
            d = device_data['isource']
            # isource_vals is pre-built JAX array - no Python loop here
            # Residual: +I at p (current leaves), -I at n (current enters)
            # For KCL: f[i] = sum of currents LEAVING node i
            f_vals = isource_vals[:, None] * jnp.array([1.0, -1.0])[None, :]  # (n, 2)
            f_idx = d['f_indices'].ravel()
            f_val = f_vals.ravel()
            f_valid = f_idx >= 0
            f_indices.append(jnp.where(f_valid, f_idx, 0))
            f_values.append(jnp.where(f_valid, f_val, 0.0))

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
        res_idx = stamp_indices['res_indices']  # (n_devices, n_residuals)
        jac_row_idx = stamp_indices['jac_row_indices']  # (n_devices, n_jac_entries)
        jac_col_idx = stamp_indices['jac_col_indices']

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

    def _make_gpu_resident_build_system_fn(
        self,
        source_device_data: Dict[str, Any],
        vmapped_fns: Dict[str, Callable],
        static_inputs_cache: Dict[str, Tuple],
        n_unknowns: int,
        use_dense: bool,
    ) -> Tuple[Callable, Dict[str, Tuple[jax.Array, jax.Array]]]:
        """Create a JIT-compilable function that builds J and f from V.

        This closure captures only small metadata (indices, vmapped functions).
        Large arrays (static_inputs, cache) are passed as arguments to avoid
        XLA constant folding overhead during compilation.

        Args:
            source_device_data: Pre-computed source device stamp templates
            vmapped_fns: Dict of vmapped OpenVAF functions per model type
            static_inputs_cache: Dict of (static_inputs, voltage_indices, stamp_indices,
                                          voltage_node1, voltage_node2, cache, ...) per model type
            n_unknowns: Number of unknowns (total nodes - 1 for ground)
            use_dense: Whether to use dense or sparse matrix assembly

        Returns:
            Tuple of:
            - build_system function with signature:
                build_system(V, vsource_vals, isource_vals, Q_prev, inv_dt, device_arrays) -> (J, f, Q)
            - device_arrays: Dict[model_type, (static_inputs, cache)] to pass to build_system

            For DC analysis: pass inv_dt=0.0 and Q_prev=zeros (reactive terms ignored)
            For transient: pass inv_dt=1/dt and Q_prev from previous timestep
        """
        from jax.experimental.sparse import BCOO, BCSR
        from jax_spice.analysis.sparse import sparse_solve_csr

        # Capture model types as static list (unrolled at trace time)
        model_types = list(static_inputs_cache.keys())

        # Split cache into metadata (captured) and arrays (passed as argument)
        # This avoids XLA constant folding the large static_inputs arrays
        static_metadata = {}
        device_arrays = {}
        split_eval_info = {}  # Store split eval info for models that support it
        for model_type in model_types:
            static_inputs, voltage_indices, stamp_indices, voltage_node1, voltage_node2, cache, _ = \
                static_inputs_cache[model_type]
            # Metadata: small arrays and indices - captured in closure
            static_metadata[model_type] = (voltage_indices, stamp_indices, voltage_node1, voltage_node2)
            # Arrays: large arrays - passed as argument
            device_arrays[model_type] = (static_inputs, cache)

            # Check if this model has split eval support
            compiled = self._compiled_models.get(model_type, {})
            if compiled.get('use_split_eval', False):
                split_eval_info[model_type] = {
                    'vmapped_split_eval': compiled['vmapped_split_eval'],
                    'shared_params': compiled['shared_params'],
                    'device_params': compiled['device_params'],
                    'voltage_positions': compiled['voltage_positions_in_varying'],
                }

        def build_system(V: jax.Array, vsource_vals: jax.Array, isource_vals: jax.Array,
                        Q_prev: jax.Array, inv_dt: float | jax.Array,
                        device_arrays_arg: Dict[str, Tuple[jax.Array, jax.Array]],
                        gmin: float | jax.Array = 1e-12, gshunt: float | jax.Array = 0.0
                        ) -> Tuple[Any, jax.Array, jax.Array]:
            """Build Jacobian J and residual f from current voltages.

            Fully JAX-traceable - no Python lists or dynamic allocation.
            All device contributions are concatenated into fixed-size arrays.

            For transient analysis with backward Euler:
                f = f_resist + (Q - Q_prev) / dt
                J = J_resist + C / dt

            Args:
                V: Current voltage vector
                vsource_vals: Voltage source values
                isource_vals: Current source values
                Q_prev: Charges from previous timestep (shape: n_unknowns)
                inv_dt: 1/dt for transient, 0 for DC analysis
                device_arrays_arg: Dict[model_type, (static_inputs, cache)] - large arrays passed as args
                gmin: GMIN conductance for device models (default 1e-12)
                gshunt: Shunt conductance to ground for homotopy (default 0)

            Returns:
                J: Jacobian matrix (dense or BCOO)
                f: Residual vector
                Q: Current charges (for tracking across timesteps)
            """
            f_resist_parts = []
            f_react_parts = []  # Charge contributions
            j_resist_parts = []
            j_react_parts = []  # Capacitance contributions

            # === Source devices contribution ===
            # Voltage sources: I = G * (Vp - Vn - Vtarget), G = 1e12
            # Sources are resistive only (no reactive component)
            if 'vsource' in source_device_data and vsource_vals.size > 0:
                d = source_device_data['vsource']
                G = 1e12
                Vp, Vn = V[d['node_p']], V[d['node_n']]
                I = G * (Vp - Vn - vsource_vals)
                G_arr = jnp.full(d['n'], G)

                # Residual contribution (resistive only)
                f_vals = I[:, None] * d['f_signs'][None, :]  # (n, 2)
                f_idx = d['f_indices'].ravel()
                f_val = f_vals.ravel()
                f_valid = f_idx >= 0
                f_resist_parts.append((jnp.where(f_valid, f_idx, 0), jnp.where(f_valid, f_val, 0.0)))

                # Jacobian contribution (resistive only)
                j_vals_arr = G_arr[:, None] * d['j_signs'][None, :]  # (n, 4)
                j_row = d['j_rows'].ravel()
                j_col = d['j_cols'].ravel()
                j_val = j_vals_arr.ravel()
                j_valid = j_row >= 0
                j_resist_parts.append((
                    jnp.where(j_valid, j_row, 0),
                    jnp.where(j_valid, j_col, 0),
                    jnp.where(j_valid, j_val, 0.0)
                ))

            # Current sources (resistive residual only)
            # Residual: +I at p (current leaves), -I at n (current enters)
            if 'isource' in source_device_data and isource_vals.size > 0:
                d = source_device_data['isource']
                f_vals = isource_vals[:, None] * jnp.array([1.0, -1.0])[None, :]
                f_idx = d['f_indices'].ravel()
                f_val = f_vals.ravel()
                f_valid = f_idx >= 0
                f_resist_parts.append((jnp.where(f_valid, f_idx, 0), jnp.where(f_valid, f_val, 0.0)))

            # === OpenVAF devices contribution (unrolled at trace time) ===
            # Devices return 4 arrays: (res_resist, res_react, jac_resist, jac_react)
            for model_type in model_types:
                # Get metadata from closure (small) and arrays from argument (large)
                voltage_indices, stamp_indices, voltage_node1, voltage_node2 = static_metadata[model_type]
                static_inputs, cache = device_arrays_arg[model_type]
                vmapped_fn = vmapped_fns[model_type]

                # Vectorized voltage update
                voltage_updates = V[voltage_node1] - V[voltage_node2]
                batch_inputs = static_inputs.at[:, jnp.array(voltage_indices)].set(voltage_updates)

                # Update analysis_type and gmin columns if this model uses them
                # inv_dt > 0 means transient, inv_dt = 0 means DC
                # Analysis type encoding: 0=dc/static, 1=ac, 2=tran, 3=noise
                uses_analysis = self._compiled_models.get(model_type, {}).get('uses_analysis', False)
                uses_simparam_gmin = self._compiled_models.get(model_type, {}).get('uses_simparam_gmin', False)
                if uses_analysis:
                    # Determine analysis type: DC (0) vs transient (2)
                    analysis_type_val = jnp.where(inv_dt > 0, 2.0, 0.0)  # tran=2, dc=0
                    batch_inputs = batch_inputs.at[:, -2].set(analysis_type_val)
                    batch_inputs = batch_inputs.at[:, -1].set(gmin)
                elif uses_simparam_gmin:
                    batch_inputs = batch_inputs.at[:, -1].set(gmin)

                # Batched device evaluation with cache - returns 4 arrays
                # Use split eval if available (reduces HLO slice operations by ~99%)
                split_info = split_eval_info.get(model_type)
                if split_info is not None and cache.size > 0:
                    # Split eval path: shared_params broadcast, device_params per-device
                    shared_params = split_info['shared_params']
                    device_params = split_info['device_params']
                    voltage_positions = split_info['voltage_positions']

                    # Update voltage columns in device_params
                    device_params_updated = device_params.at[:, voltage_positions].set(voltage_updates)

                    # Handle analysis_type and gmin in device_params
                    # These are the last 1-2 columns if the model uses them
                    if uses_analysis:
                        analysis_type_val = jnp.where(inv_dt > 0, 2.0, 0.0)
                        device_params_updated = device_params_updated.at[:, -2].set(analysis_type_val)
                        device_params_updated = device_params_updated.at[:, -1].set(gmin)
                    elif uses_simparam_gmin:
                        device_params_updated = device_params_updated.at[:, -1].set(gmin)

                    vmapped_split_eval = split_info['vmapped_split_eval']
                    batch_res_resist, batch_res_react, batch_jac_resist, batch_jac_react = \
                        vmapped_split_eval(shared_params, device_params_updated, cache)
                elif vmapped_fns.get(model_type + '_with_cache') is not None and cache.size > 0:
                    # Standard eval_with_cache path
                    vmapped_eval_with_cache = vmapped_fns[model_type + '_with_cache']
                    batch_res_resist, batch_res_react, batch_jac_resist, batch_jac_react = \
                        vmapped_eval_with_cache(batch_inputs, cache)
                else:
                    # Fallback to original vmapped function
                    batch_res_resist, batch_res_react, batch_jac_resist, batch_jac_react = vmapped_fn(batch_inputs)

                # NOTE: Huge value masking removed - NOI constraint enforcement in NR solver
                # (lines 5601-5606) now handles 1e40 conductance by zeroing NOI rows/cols in J and f.
                # The value masking was causing 4 extra jnp.where ops per NR iteration.

                # Collect COO triplets using pre-computed indices
                res_idx = stamp_indices['res_indices']
                jac_row_idx = stamp_indices['jac_row_indices']
                jac_col_idx = stamp_indices['jac_col_indices']

                # === Resistive residuals ===
                flat_res_idx = res_idx.ravel()
                flat_res_resist_val = batch_res_resist.ravel()
                valid_res = flat_res_idx >= 0
                flat_res_idx_masked = jnp.where(valid_res, flat_res_idx, 0)
                flat_res_resist_masked = jnp.where(valid_res, flat_res_resist_val, 0.0)
                flat_res_resist_masked = jnp.where(jnp.isnan(flat_res_resist_masked), 0.0, flat_res_resist_masked)
                f_resist_parts.append((flat_res_idx_masked, flat_res_resist_masked))

                # === Reactive residuals (charges) ===
                flat_res_react_val = batch_res_react.ravel()
                flat_res_react_masked = jnp.where(valid_res, flat_res_react_val, 0.0)
                flat_res_react_masked = jnp.where(jnp.isnan(flat_res_react_masked), 0.0, flat_res_react_masked)
                f_react_parts.append((flat_res_idx_masked, flat_res_react_masked))

                # === Resistive Jacobian (conductances) ===
                flat_jac_rows = jac_row_idx.ravel()
                flat_jac_cols = jac_col_idx.ravel()
                flat_jac_resist_vals = batch_jac_resist.ravel()
                valid_jac = (flat_jac_rows >= 0) & (flat_jac_cols >= 0)
                flat_jac_rows_masked = jnp.where(valid_jac, flat_jac_rows, 0)
                flat_jac_cols_masked = jnp.where(valid_jac, flat_jac_cols, 0)
                flat_jac_resist_masked = jnp.where(valid_jac, flat_jac_resist_vals, 0.0)
                flat_jac_resist_masked = jnp.where(jnp.isnan(flat_jac_resist_masked), 0.0, flat_jac_resist_masked)
                j_resist_parts.append((
                    flat_jac_rows_masked,
                    flat_jac_cols_masked,
                    flat_jac_resist_masked
                ))

                # === Reactive Jacobian (capacitances) ===
                flat_jac_react_vals = batch_jac_react.ravel()
                flat_jac_react_masked = jnp.where(valid_jac, flat_jac_react_vals, 0.0)
                flat_jac_react_masked = jnp.where(jnp.isnan(flat_jac_react_masked), 0.0, flat_jac_react_masked)
                j_react_parts.append((
                    flat_jac_rows_masked,
                    flat_jac_cols_masked,
                    flat_jac_react_masked
                ))

            # === Build resistive residual vector f_resist ===
            if f_resist_parts:
                all_f_resist_idx = jnp.concatenate([p[0] for p in f_resist_parts])
                all_f_resist_val = jnp.concatenate([p[1] for p in f_resist_parts])
                f_resist = jax.ops.segment_sum(all_f_resist_val, all_f_resist_idx, num_segments=n_unknowns)
            else:
                f_resist = jnp.zeros(n_unknowns, dtype=jnp.float64)

            # === Build reactive residual vector Q (charges) ===
            if f_react_parts:
                all_f_react_idx = jnp.concatenate([p[0] for p in f_react_parts])
                all_f_react_val = jnp.concatenate([p[1] for p in f_react_parts])
                Q = jax.ops.segment_sum(all_f_react_val, all_f_react_idx, num_segments=n_unknowns)
            else:
                Q = jnp.zeros(n_unknowns, dtype=jnp.float64)

            # === Combine for transient: f = f_resist + (Q - Q_prev) / dt ===
            # For DC (inv_dt=0): f = f_resist
            # For transient: f = f_resist + inv_dt * (Q - Q_prev)
            f = f_resist + inv_dt * (Q - Q_prev)

            # === Add GSHUNT contribution to residual ===
            # GSHUNT is a shunt conductance to ground: I = gshunt * V
            # V[1:] are the non-ground node voltages (ground at index 0 is excluded)
            V_nonground = V[1:]
            f = f + gshunt * V_nonground

            # === Build resistive Jacobian J_resist ===
            if j_resist_parts:
                all_j_resist_rows = jnp.concatenate([p[0] for p in j_resist_parts])
                all_j_resist_cols = jnp.concatenate([p[1] for p in j_resist_parts])
                all_j_resist_vals = jnp.concatenate([p[2] for p in j_resist_parts])
            else:
                all_j_resist_rows = jnp.zeros(0, dtype=jnp.int32)
                all_j_resist_cols = jnp.zeros(0, dtype=jnp.int32)
                all_j_resist_vals = jnp.zeros(0, dtype=jnp.float64)

            # === Build reactive Jacobian C (capacitances) ===
            if j_react_parts:
                all_j_react_rows = jnp.concatenate([p[0] for p in j_react_parts])
                all_j_react_cols = jnp.concatenate([p[1] for p in j_react_parts])
                all_j_react_vals = jnp.concatenate([p[2] for p in j_react_parts])
            else:
                all_j_react_rows = jnp.zeros(0, dtype=jnp.int32)
                all_j_react_cols = jnp.zeros(0, dtype=jnp.int32)
                all_j_react_vals = jnp.zeros(0, dtype=jnp.float64)

            # === Combine Jacobians: J = J_resist + C / dt ===
            # Concatenate COO triplets and scale reactive by inv_dt
            all_j_rows = jnp.concatenate([all_j_resist_rows, all_j_react_rows])
            all_j_cols = jnp.concatenate([all_j_resist_cols, all_j_react_cols])
            all_j_vals = jnp.concatenate([all_j_resist_vals, inv_dt * all_j_react_vals])

            if use_dense:
                # Dense: COO -> dense matrix via segment_sum
                flat_indices = all_j_rows * n_unknowns + all_j_cols
                J_flat = jax.ops.segment_sum(
                    all_j_vals, flat_indices, num_segments=n_unknowns * n_unknowns
                )
                J = J_flat.reshape((n_unknowns, n_unknowns))
                # Add regularization (1e-9) + gshunt to diagonal
                J = J + (1e-9 + gshunt) * jnp.eye(n_unknowns, dtype=jnp.float64)
            else:
                # Sparse path - build BCOO sparse matrix
                from jax.experimental.sparse import BCOO

                # Add diagonal entries: regularization (1e-3) + gshunt
                diag_idx = jnp.arange(n_unknowns, dtype=jnp.int32)
                all_j_rows = jnp.concatenate([all_j_rows, diag_idx])
                all_j_cols = jnp.concatenate([all_j_cols, diag_idx])
                # Large regularization needed for GPU spsolve (stricter than scipy)
                all_j_vals = jnp.concatenate([all_j_vals, jnp.full(n_unknowns, 1e-3 + gshunt)])

                # Build BCOO with duplicates (BCSR.from_bcoo handles them)
                indices = jnp.stack([all_j_rows, all_j_cols], axis=1)
                J = BCOO((all_j_vals, indices), shape=(n_unknowns, n_unknowns))

            return J, f, Q

        return build_system, device_arrays

    def _make_jit_compiled_solver(
        self,
        build_system_jit: Callable,
        n_nodes: int,
        device_arrays: Dict[str, Tuple[jax.Array, jax.Array]],
        noi_indices: Optional[jax.Array] = None,
        max_iterations: int = MAX_NR_ITERATIONS,
        abstol: float = 1e-6,
        max_step: float = 1.0,
    ) -> Callable:
        """Create a JIT-compiled NR solver with nested JIT for build_system.

        Uses layered JIT compilation for efficient compilation of large circuits:
        - build_system_jit is already JIT-wrapped (via jax.jit)
        - When the outer JIT traces nr_solve, the inner JIT becomes a call boundary
        - This prevents inlining of all device evaluations into the outer trace

        Args:
            build_system_jit: JIT-wrapped function (V, vsource_vals, isource_vals, Q_prev, inv_dt, device_arrays) -> (J, f, Q)
            n_nodes: Total node count including ground (V.shape[0])
            device_arrays: Dict[model_type, (static_inputs, cache)] - passed as traced arg to avoid XLA constant folding
            noi_indices: Optional array of NOI node indices to constrain to 0V
            max_iterations: Maximum NR iterations
            abstol: Absolute tolerance for convergence
            max_step: Maximum voltage step per iteration

        Returns:
            JIT-compiled function: (V, vsource_vals, isource_vals, Q_prev, inv_dt, device_arrays, gmin, gshunt) -> (V, iters, converged, max_f, Q)
            gmin and gshunt have defaults (1e-12, 0.0) for backward compatibility with transient calls.
        """
        # Pre-compute NOI indices for O(n) masking (instead of O(n²) boolean matrix ops)
        # NOI nodes have indices in the full V vector, but residuals use 0-indexed (ground excluded)
        # So NOI residual index = NOI node index - 1
        if noi_indices is not None and len(noi_indices) > 0:
            n_unknowns = n_nodes - 1
            noi_res_idx = noi_indices - 1  # Convert to residual indices (pre-computed for JIT)
            # Create mask: True for nodes to include in convergence check, False for NOI
            residual_mask = jnp.ones(n_unknowns, dtype=jnp.bool_)
            residual_mask = residual_mask.at[noi_res_idx].set(False)
        else:
            noi_res_idx = None
            residual_mask = None

        def nr_solve(V_init: jax.Array, vsource_vals: jax.Array, isource_vals: jax.Array,
                    Q_prev: jax.Array, inv_dt: float | jax.Array,
                    device_arrays_arg: Dict[str, Tuple[jax.Array, jax.Array]],
                    gmin: float | jax.Array = 1e-12, gshunt: float | jax.Array = 0.0):
            # State: (V, iteration, converged, max_f, max_delta, Q)
            # Q is tracked to return the final charges
            # Q has shape (n_unknowns,) = (n_nodes - 1,) since ground is excluded
            init_Q = jnp.zeros(n_nodes - 1, dtype=jnp.float64)
            init_state = (
                V_init,
                jnp.array(0, dtype=jnp.int32),
                jnp.array(False),
                jnp.array(jnp.inf),
                jnp.array(jnp.inf),
                init_Q,
            )

            def cond_fn(state):
                V, iteration, converged, max_f, max_delta, Q = state
                return jnp.logical_and(~converged, iteration < max_iterations)

            def body_fn(state):
                V, iteration, _, _, _, _ = state

                # Build system (J, f, Q) - calls JIT'd function
                # Q_prev is fixed for all NR iterations within this timestep
                # gmin/gshunt passed through for homotopy support
                # device_arrays passed as traced arg to avoid XLA constant folding
                J, f, Q = build_system_jit(V, vsource_vals, isource_vals, Q_prev, inv_dt, device_arrays_arg, gmin, gshunt)

                # Check residual convergence
                # Note: f already excludes ground (has shape n_unknowns = n_nodes - 1)
                # f[0] = node 1's residual, f[1] = node 2's residual, etc.
                # Mask out NOI residuals (they have 1e40 conductance and would dominate max_f)
                if residual_mask is not None:
                    f_masked = jnp.where(residual_mask, f, 0.0)
                    max_f = jnp.max(jnp.abs(f_masked))
                else:
                    max_f = jnp.max(jnp.abs(f))
                residual_converged = max_f < abstol

                # Enforce NOI node constraints BEFORE linear solve
                # NOI nodes have 1e40 conductance to ground which causes numerical instability.
                # We enforce delta[noi] = 0 by modifying J and f:
                # - Set J[noi, :] = 0 and J[:, noi] = 0 (decouple from other nodes)
                # - Set J[noi, noi] = 1.0 (make it solvable)
                # - Set f[noi] = 0.0 (so delta[noi] = 0)
                # Using O(n) index-based operations instead of O(n²) boolean matrix ops
                if noi_res_idx is not None:
                    # Zero NOI rows and columns, set diagonal to 1.0
                    J = J.at[noi_res_idx, :].set(0.0)           # Zero NOI rows
                    J = J.at[:, noi_res_idx].set(0.0)           # Zero NOI columns
                    J = J.at[noi_res_idx, noi_res_idx].set(1.0) # Diagonal = 1 for solvability
                    f = f.at[noi_res_idx].set(0.0)              # Zero NOI residuals

                # Solve: J @ delta = -f (only updating non-ground nodes)
                delta = jax.scipy.linalg.solve(J, -f)

                # Step limiting
                max_delta = jnp.max(jnp.abs(delta))
                scale = jnp.where(max_delta > max_step, max_step / max_delta, 1.0)
                delta = delta * scale

                # Update V (ground at index 0 stays fixed)
                V_new = V.at[1:].add(delta)

                # Clamp NOI nodes to 0V (they should always be 0V)
                if noi_indices is not None and len(noi_indices) > 0:
                    V_new = V_new.at[noi_indices].set(0.0)

                # Check delta-based convergence
                delta_converged = max_delta < 1e-12

                converged = jnp.logical_or(residual_converged, delta_converged)

                return (V_new, iteration + 1, converged, max_f, max_delta, Q)

            # Run NR loop on GPU
            V_final, iterations, converged, max_f, max_delta, _ = lax.while_loop(
                cond_fn, body_fn, init_state
            )

            # Recompute Q from the converged voltage
            # The Q from body_fn was computed from V before the update, not V_new
            _, _, Q_final = build_system_jit(V_final, vsource_vals, isource_vals, Q_prev, inv_dt, device_arrays_arg, gmin, gshunt)

            return V_final, iterations, converged, max_f, Q_final

        # Return JIT-wrapped function - compilation happens lazily on first call
        logger.info(f"Creating JIT-compiled NR solver: V({n_nodes}), NOI constrained: {noi_indices is not None}")
        return jax.jit(nr_solve)

    def _make_sparse_jit_compiled_solver(
        self,
        build_system_jit: Callable,
        n_nodes: int,
        nse: int,
        device_arrays: Dict[str, Tuple[jax.Array, jax.Array]],
        noi_indices: Optional[jax.Array] = None,
        max_iterations: int = MAX_NR_ITERATIONS,
        abstol: float = 1e-6,
        max_step: float = 1.0,
    ) -> Callable:
        """Create a JIT-compiled sparse NR solver using spsolve.

        Uses JAX's sparse direct solver (QR factorization) for large circuits
        where dense linear algebra would OOM.

        Args:
            build_system_jit: JIT-wrapped function returning (J_bcoo, f, Q)
            n_nodes: Total node count including ground
            nse: Number of stored elements after summing duplicates
            device_arrays: Dict[model_type, (static_inputs, cache)] - passed as traced arg
            noi_indices: Optional array of NOI node indices to constrain to 0V
            max_iterations: Maximum NR iterations
            abstol: Absolute tolerance for convergence
            max_step: Maximum voltage step per iteration

        Returns:
            JIT-compiled sparse solver function: (V, vsrc, isrc, Q_prev, inv_dt, device_arrays) -> (V, iters, converged, max_f, Q)
        """
        from jax.experimental.sparse import BCSR
        from jax.experimental.sparse.linalg import spsolve

        # Pre-compute residual mask if we have NOI nodes
        if noi_indices is not None and len(noi_indices) > 0:
            n_unknowns = n_nodes - 1
            residual_mask = jnp.ones(n_unknowns, dtype=jnp.bool_)
            noi_residual_indices = noi_indices - 1
            residual_mask = residual_mask.at[noi_residual_indices].set(False)
        else:
            residual_mask = None

        def nr_solve(V_init: jax.Array, vsource_vals: jax.Array, isource_vals: jax.Array,
                    Q_prev: jax.Array, inv_dt: float | jax.Array,
                    device_arrays_arg: Dict[str, Tuple[jax.Array, jax.Array]],
                    gmin: float | jax.Array = 1e-12, gshunt: float | jax.Array = 0.0):
            # Q has shape (n_unknowns,) = (n_nodes - 1,) since ground is excluded
            init_Q = jnp.zeros(n_nodes - 1, dtype=jnp.float64)
            init_state = (
                V_init,
                jnp.array(0, dtype=jnp.int32),
                jnp.array(False),
                jnp.array(jnp.inf),
                jnp.array(jnp.inf),
                init_Q,
            )

            def cond_fn(state):
                V, iteration, converged, max_f, max_delta, Q = state
                return jnp.logical_and(~converged, iteration < max_iterations)

            def body_fn(state):
                V, iteration, _, _, _, _ = state

                # Build sparse system (J_bcoo, f, Q) with homotopy parameters
                J_bcoo, f, Q = build_system_jit(V, vsource_vals, isource_vals, Q_prev, inv_dt, device_arrays_arg, gmin, gshunt)

                # Check residual convergence
                # Mask out NOI residuals (they have 1e40 conductance and would dominate max_f)
                if residual_mask is not None:
                    f_masked = jnp.where(residual_mask, f, 0.0)
                    max_f = jnp.max(jnp.abs(f_masked))
                else:
                    max_f = jnp.max(jnp.abs(f))
                residual_converged = max_f < abstol

                # Sum duplicates before converting to BCSR (required for scipy fallback)
                J_bcoo_dedup = J_bcoo.sum_duplicates(nse=nse)

                # Convert BCOO to BCSR for spsolve
                J_bcsr = BCSR.from_bcoo(J_bcoo_dedup)

                # Sparse solve: J @ delta = -f using QR factorization
                # J and f are n_unknowns sized (ground already excluded)
                delta = spsolve(
                    J_bcsr.data, J_bcsr.indices, J_bcsr.indptr, -f, tol=1e-6
                )

                # Step limiting
                max_delta = jnp.max(jnp.abs(delta))
                scale = jnp.where(max_delta > max_step, max_step / max_delta, 1.0)
                delta = delta * scale

                # Update V (ground at index 0 stays fixed, delta is for nodes 1:n_nodes)
                V_new = V.at[1:].add(delta)

                # Clamp NOI nodes to 0V
                if noi_indices is not None and len(noi_indices) > 0:
                    V_new = V_new.at[noi_indices].set(0.0)

                # Check delta-based convergence
                delta_converged = max_delta < 1e-12
                converged = jnp.logical_or(residual_converged, delta_converged)

                return (V_new, iteration + 1, converged, max_f, max_delta, Q)

            V_final, iterations, converged, max_f, max_delta, _ = lax.while_loop(
                cond_fn, body_fn, init_state
            )

            # Recompute Q from the converged voltage
            # The Q from body_fn was computed from V before the update, not V_new
            _, _, Q_final = build_system_jit(V_final, vsource_vals, isource_vals, Q_prev, inv_dt, device_arrays_arg, gmin, gshunt)

            return V_final, iterations, converged, max_f, Q_final

        logger.info(f"Creating sparse JIT-compiled NR solver: V({n_nodes}), NOI constrained: {noi_indices is not None}")
        return jax.jit(nr_solve)

    def _make_spineax_jit_compiled_solver(
        self,
        build_system_jit: Callable,
        n_nodes: int,
        nse: int,
        bcsr_indptr: jax.Array,
        bcsr_indices: jax.Array,
        device_arrays: Dict[str, Tuple[jax.Array, jax.Array]],
        noi_indices: Optional[jax.Array] = None,
        max_iterations: int = MAX_NR_ITERATIONS,
        abstol: float = 1e-6,
        max_step: float = 1.0,
    ) -> Callable:
        """Create a JIT-compiled sparse NR solver using Spineax/cuDSS.

        Uses Spineax's cuDSS wrapper with cached symbolic factorization.
        The symbolic analysis (METIS reordering, fill-in pattern) is done once
        when the solver is created, and reused for all subsequent solves.

        Args:
            build_system_jit: JIT-wrapped function returning (J_bcoo, f, Q)
            n_nodes: Total node count including ground
            nse: Number of stored elements after summing duplicates
            bcsr_indptr: Pre-computed BCSR row pointers
            bcsr_indices: Pre-computed BCSR column indices
            device_arrays: Dict[model_type, (static_inputs, cache)] - passed as traced arg
            noi_indices: Optional array of NOI node indices to constrain to 0V
            max_iterations: Maximum NR iterations
            abstol: Absolute tolerance for convergence
            max_step: Maximum voltage step per iteration

        Returns:
            JIT-compiled sparse solver function: (V, vsrc, isrc, Q_prev, inv_dt, device_arrays) -> (V, iters, converged, max_f, Q)
        """
        from jax.experimental.sparse import BCSR
        from spineax.cudss.solver import CuDSSSolver

        n_unknowns = n_nodes - 1  # Exclude ground

        # Pre-compute residual mask if we have NOI nodes
        if noi_indices is not None and len(noi_indices) > 0:
            residual_mask = jnp.ones(n_unknowns, dtype=jnp.bool_)
            noi_residual_indices = noi_indices - 1
            residual_mask = residual_mask.at[noi_residual_indices].set(False)
        else:
            residual_mask = None

        # Create Spineax solver with pre-computed sparsity pattern
        # This does METIS reordering and symbolic analysis ONCE
        spineax_solver = CuDSSSolver(
            bcsr_indptr,
            bcsr_indices,
            device_id=0,
            mtype_id=1,  # General matrix
            mview_id=0,  # Full matrix
        )
        logger.info(f"Created Spineax solver with cached symbolic factorization")

        def nr_solve(V_init: jax.Array, vsource_vals: jax.Array, isource_vals: jax.Array,
                    Q_prev: jax.Array, inv_dt: float | jax.Array,
                    device_arrays_arg: Dict[str, Tuple[jax.Array, jax.Array]],
                    gmin: float | jax.Array = 1e-12, gshunt: float | jax.Array = 0.0):
            # State: (V, iteration, converged, max_f, max_delta, Q)
            # Q is tracked to return the final charges
            init_Q = jnp.zeros(n_unknowns, dtype=jnp.float64)
            init_state = (
                V_init,
                jnp.array(0, dtype=jnp.int32),
                jnp.array(False),
                jnp.array(jnp.inf),
                jnp.array(jnp.inf),
                init_Q,
            )

            def cond_fn(state):
                V, iteration, converged, max_f, max_delta, Q = state
                return jnp.logical_and(~converged, iteration < max_iterations)

            def body_fn(state):
                V, iteration, _, _, _, _ = state

                # Build sparse system (J_bcoo, f, Q)
                J_bcoo, f, Q = build_system_jit(V, vsource_vals, isource_vals, Q_prev, inv_dt, device_arrays_arg, gmin, gshunt)

                # Check residual convergence
                # Mask out NOI residuals (they have 1e40 conductance and would dominate max_f)
                if residual_mask is not None:
                    f_masked = jnp.where(residual_mask, f, 0.0)
                    max_f = jnp.max(jnp.abs(f_masked))
                else:
                    max_f = jnp.max(jnp.abs(f))
                residual_converged = max_f < abstol

                # Sum duplicates before converting to BCSR
                J_bcoo_dedup = J_bcoo.sum_duplicates(nse=nse)

                # Convert BCOO to BCSR to get data in correct order
                J_bcsr = BCSR.from_bcoo(J_bcoo_dedup)

                # Solve using Spineax (cached symbolic factorization)
                # Only numeric refactorization + triangular solve happens here
                delta, _info = spineax_solver(-f, J_bcsr.data)

                # Step limiting
                max_delta = jnp.max(jnp.abs(delta))
                scale = jnp.where(max_delta > max_step, max_step / max_delta, 1.0)
                delta = delta * scale

                # Update V (ground at index 0 stays fixed)
                V_new = V.at[1:].add(delta)

                # Clamp NOI nodes to 0V
                if noi_indices is not None and len(noi_indices) > 0:
                    V_new = V_new.at[noi_indices].set(0.0)

                # Check delta-based convergence
                delta_converged = max_delta < 1e-12
                converged = jnp.logical_or(residual_converged, delta_converged)

                return (V_new, iteration + 1, converged, max_f, max_delta, Q)

            V_final, iterations, converged, max_f, max_delta, _ = lax.while_loop(
                cond_fn, body_fn, init_state
            )

            # Recompute Q from the converged voltage
            # The Q from body_fn was computed from V before the update, not V_new
            _, _, Q_final = build_system_jit(V_final, vsource_vals, isource_vals, Q_prev, inv_dt, device_arrays_arg, gmin, gshunt)

            return V_final, iterations, converged, max_f, Q_final

        logger.info(f"Creating Spineax JIT-compiled NR solver: V({n_nodes})")
        return jax.jit(nr_solve)

    def _compute_voltage_param(self, name: str, V: jax.Array,
                                node_map: Dict[str, int],
                                model_nodes: List[str],
                                ground: int) -> float:
        """Compute a voltage parameter value from node voltages.

        Handles formats like:
        - "V(GP,SI)" - voltage between two nodes
        - "V(DI)" - voltage of a single node (vs ground)
        - "V(node0,node2)" - external terminal voltages
        """
        import re

        # Parse V(node1) or V(node1,node2) format
        match = re.match(r'V\(([^,)]+)(?:,([^)]+))?\)', name)
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
            'NOI': 'node4',
            'GP': 'node5', 'SI': 'node6', 'DI': 'node7', 'BP': 'node8',
            'BI': 'node9', 'BS': 'node10', 'BD': 'node11',
            'G': 'node1', 'D': 'node0', 'S': 'node2', 'B': 'node3',
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
        mode: str = 'dec',
        points: int = 10,
        step: Optional[float] = None,
        values: Optional[List[float]] = None,
    ) -> 'ACResult':
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
        from jax_spice.analysis.ac import ACConfig, ACResult, run_ac_analysis

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

        # Prepare device batching (same logic as transient setup)
        ground = 0  # Ground is always node 0

        # Set up internal nodes using the same method as transient
        n_total, device_internal_nodes = self._setup_internal_nodes()
        n_unknowns = n_total - 1  # Ground excluded

        # Group devices by model type
        openvaf_by_type: Dict[str, List[Dict]] = {}
        for dev in self.devices:
            model = dev['model']
            if model in ('vsource', 'isource'):
                continue
            openvaf_by_type.setdefault(model, []).append(dev)

        # Prepare vmapped functions and static inputs
        vmapped_fns: Dict[str, Callable] = {}
        static_inputs_cache: Dict[str, Tuple] = {}

        for model_type in openvaf_by_type:
            compiled = self._compiled_models.get(model_type)
            if compiled and 'vmapped_fn' in compiled:
                vmapped_fns[model_type] = compiled['vmapped_fn']
                # Also store vmapped_eval_with_cache if available (captured at setup, not looked up in traced fn)
                if 'vmapped_eval_with_cache' in compiled:
                    vmapped_fns[model_type + '_with_cache'] = compiled['vmapped_eval_with_cache']
                static_inputs, voltage_indices, device_contexts, cache, collapse_decisions = self._prepare_static_inputs(
                    model_type, openvaf_by_type[model_type], device_internal_nodes, ground
                )
                stamp_indices = self._build_stamp_index_mapping(
                    model_type, device_contexts, ground
                )
                voltage_node1 = jnp.array([
                    [n1 for n1, n2 in ctx['voltage_node_pairs']]
                    for ctx in device_contexts
                ], dtype=jnp.int32)
                voltage_node2 = jnp.array([
                    [n2 for n1, n2 in ctx['voltage_node_pairs']]
                    for ctx in device_contexts
                ], dtype=jnp.int32)
                static_inputs = jnp.array(static_inputs, dtype=jnp.float64)
                cache = jnp.array(cache, dtype=jnp.float64)
                static_inputs_cache[model_type] = (
                    static_inputs, voltage_indices, stamp_indices, voltage_node1, voltage_node2, cache, collapse_decisions
                )

        # Count sources
        n_vsources = sum(1 for d in self.devices if d['model'] == 'vsource')
        n_isources = sum(1 for d in self.devices if d['model'] == 'isource')

        # Prepare source devices
        source_devices = [d for d in self.devices if d['model'] in ('vsource', 'isource')]
        source_device_data = self._prepare_source_devices_coo(source_devices, ground, n_unknowns)

        # Get DC source values
        vsource_dc_vals, isource_dc_vals = self._get_dc_source_values(n_vsources, n_isources)

        # Create NR solver for AC DC operating point calculation
        # This uses analytic jacobians from OpenVAF (no autodiff)
        # AC analysis uses dense solver (simpler circuits typically)
        build_system_fn, device_arrays = self._make_gpu_resident_build_system_fn(
            source_device_data, vmapped_fns, static_inputs_cache, n_unknowns, use_dense=True
        )
        build_system_jit = jax.jit(build_system_fn)

        # Collect NOI node indices (PSP103 noise correlation internal node)
        noi_indices = []
        if device_internal_nodes:
            for dev_name, internal_nodes in device_internal_nodes.items():
                if 'node4' in internal_nodes:  # NOI is node4 in PSP103
                    noi_indices.append(internal_nodes['node4'])
        noi_indices = jnp.array(noi_indices, dtype=jnp.int32) if noi_indices else None

        nr_solve = self._make_jit_compiled_solver(
            build_system_jit, n_total, device_arrays, noi_indices=noi_indices,
            max_iterations=MAX_NR_ITERATIONS, abstol=DEFAULT_ABSTOL, max_step=1.0
        )

        # Initialize V (including internal nodes)
        vdd_value = self._get_vdd_value() or 1.0  # Default to 1.0 if no vsources
        mid_rail = vdd_value / 2.0
        V_dc = jnp.full(n_total, mid_rail, dtype=jnp.float64)
        V_dc = V_dc.at[0].set(0.0)

        Q_prev = jnp.zeros(n_unknowns, dtype=jnp.float64)

        # First try direct NR without homotopy
        logger.info("  AC DC: Trying direct NR solver first...")
        V_new, nr_iters, is_converged, max_f, _ = nr_solve(
            V_dc, vsource_dc_vals, isource_dc_vals, Q_prev, 0.0, device_arrays  # inv_dt=0 for DC
        )

        if is_converged:
            V_dc = V_new
            logger.info(f"  AC DC operating point converged via direct NR "
                       f"({nr_iters} iters, residual={max_f:.2e})")
        else:
            # Fall back to homotopy chain using the cached NR solver
            logger.info("  AC DC: Direct NR failed, trying homotopy chain...")

            homotopy_config = HomotopyConfig(
                gmin=1e-12,
                gdev_start=1e-3,
                gdev_target=1e-13,
                gmin_factor=3.0,
                gmin_factor_min=1.1,
                gmin_factor_max=10.0,
                gmin_max=1.0,
                gmin_max_steps=100,
                source_step=0.1,
                source_step_min=0.001,
                source_max_steps=100,
                chain=("gdev", "gshunt", "src"),
                max_iterations=100,
                abstol=1e-9,
                debug=0,
            )

            result = run_homotopy_chain(
                nr_solve, V_dc, vsource_dc_vals, isource_dc_vals, Q_prev, device_arrays, homotopy_config
            )

            V_dc = result.V
            logger.info(f"  AC DC operating point: {result.method} "
                       f"({result.iterations} iterations, converged={result.converged})")

        # Now extract Jr and Jc at the DC operating point
        Jr, Jc = self._extract_ac_jacobians(
            V_dc, vmapped_fns, static_inputs_cache, source_device_data,
            n_unknowns, vsource_dc_vals, isource_dc_vals
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
        if 'vsource' in source_device_data and vsource_dc_vals.size > 0:
            d = source_device_data['vsource']
            G = 1e12
            G_arr = jnp.full(d['n'], G)
            j_vals_arr = G_arr[:, None] * d['j_signs'][None, :]
            j_row = d['j_rows'].ravel()
            j_col = d['j_cols'].ravel()
            j_val = j_vals_arr.ravel()
            j_valid = j_row >= 0
            j_resist_parts.append((
                jnp.where(j_valid, j_row, 0),
                jnp.where(j_valid, j_col, 0),
                jnp.where(j_valid, j_val, 0.0)
            ))

        # === OpenVAF device contributions ===
        for model_type in model_types:
            static_inputs, voltage_indices, stamp_indices, voltage_node1, voltage_node2, cache, _ = \
                static_inputs_cache[model_type]
            vmapped_fn = vmapped_fns[model_type]

            # Compute device voltages
            voltage_updates = V[voltage_node1] - V[voltage_node2]
            batch_inputs = static_inputs.at[:, jnp.array(voltage_indices)].set(voltage_updates)

            # Update analysis_type and gmin columns if this model uses them
            # For AC analysis: analysis_type = 1
            uses_analysis = self._compiled_models.get(model_type, {}).get('uses_analysis', False)
            uses_simparam_gmin = self._compiled_models.get(model_type, {}).get('uses_simparam_gmin', False)
            if uses_analysis:
                batch_inputs = batch_inputs.at[:, -2].set(1.0)  # AC analysis
                batch_inputs = batch_inputs.at[:, -1].set(1e-12)  # Default gmin
            elif uses_simparam_gmin:
                batch_inputs = batch_inputs.at[:, -1].set(1e-12)  # Default gmin

            # Evaluate devices
            vmapped_eval_with_cache = vmapped_fns.get(model_type + '_with_cache')
            if vmapped_eval_with_cache is not None and cache.size > 0:
                _, _, batch_jac_resist, batch_jac_react = vmapped_eval_with_cache(batch_inputs, cache)
            else:
                _, _, batch_jac_resist, batch_jac_react = vmapped_fn(batch_inputs)

            # Extract Jacobian entries
            jac_row_idx = stamp_indices['jac_row_indices']
            jac_col_idx = stamp_indices['jac_col_indices']

            flat_jac_rows = jac_row_idx.ravel()
            flat_jac_cols = jac_col_idx.ravel()
            valid_jac = (flat_jac_rows >= 0) & (flat_jac_cols >= 0)

            # Resistive Jacobian
            flat_jac_resist_vals = batch_jac_resist.ravel()
            flat_jac_resist_masked = jnp.where(valid_jac, flat_jac_resist_vals, 0.0)
            flat_jac_resist_masked = jnp.where(jnp.isnan(flat_jac_resist_masked), 0.0, flat_jac_resist_masked)
            j_resist_parts.append((
                jnp.where(valid_jac, flat_jac_rows, 0),
                jnp.where(valid_jac, flat_jac_cols, 0),
                flat_jac_resist_masked
            ))

            # Reactive Jacobian
            flat_jac_react_vals = batch_jac_react.ravel()
            flat_jac_react_masked = jnp.where(valid_jac, flat_jac_react_vals, 0.0)
            flat_jac_react_masked = jnp.where(jnp.isnan(flat_jac_react_masked), 0.0, flat_jac_react_masked)
            j_react_parts.append((
                jnp.where(valid_jac, flat_jac_rows, 0),
                jnp.where(valid_jac, flat_jac_cols, 0),
                flat_jac_react_masked
            ))

        # === Assemble dense Jacobians ===
        # Resistive Jacobian Jr
        if j_resist_parts:
            all_j_resist_rows = jnp.concatenate([p[0] for p in j_resist_parts])
            all_j_resist_cols = jnp.concatenate([p[1] for p in j_resist_parts])
            all_j_resist_vals = jnp.concatenate([p[2] for p in j_resist_parts])
            flat_indices = all_j_resist_rows * n_unknowns + all_j_resist_cols
            Jr_flat = jax.ops.segment_sum(all_j_resist_vals, flat_indices,
                                          num_segments=n_unknowns * n_unknowns)
            Jr = Jr_flat.reshape((n_unknowns, n_unknowns))
        else:
            Jr = jnp.zeros((n_unknowns, n_unknowns), dtype=jnp.float64)

        # Reactive Jacobian Jc
        if j_react_parts:
            all_j_react_rows = jnp.concatenate([p[0] for p in j_react_parts])
            all_j_react_cols = jnp.concatenate([p[1] for p in j_react_parts])
            all_j_react_vals = jnp.concatenate([p[2] for p in j_react_parts])
            flat_indices = all_j_react_rows * n_unknowns + all_j_react_cols
            Jc_flat = jax.ops.segment_sum(all_j_react_vals, flat_indices,
                                          num_segments=n_unknowns * n_unknowns)
            Jc = Jc_flat.reshape((n_unknowns, n_unknowns))
        else:
            Jc = jnp.zeros((n_unknowns, n_unknowns), dtype=jnp.float64)

        # Add regularization
        Jr = Jr + 1e-12 * jnp.eye(n_unknowns, dtype=jnp.float64)

        return Jr, Jc

    def _extract_ac_sources(self) -> List[Dict]:
        """Extract AC source specifications from devices.

        Returns:
            List of AC source specifications with mag, phase, node indices
        """
        ac_sources = []

        for dev in self.devices:
            if dev['model'] == 'vsource':
                params = dev['params']
                mag = params.get('mag', 0.0)
                phase = params.get('phase', 0.0)

                # Only include sources with non-zero AC magnitude
                if mag != 0.0:
                    nodes = dev.get('nodes', [0, 0])
                    ac_sources.append({
                        'name': dev['name'],
                        'pos_node': nodes[0] if len(nodes) > 0 else 0,
                        'neg_node': nodes[1] if len(nodes) > 1 else 0,
                        'mag': float(mag),
                        'phase': float(phase),
                    })

        return ac_sources

    # =========================================================================
    # Transfer Function Analyses (DCINC, DCXF, ACXF)
    # =========================================================================

    def run_dcinc(self) -> 'DCIncResult':
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
        from jax_spice.analysis.xfer import DCIncResult, solve_dcinc, build_dcinc_excitation

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
    ) -> 'DCXFResult':
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
        from jax_spice.analysis.xfer import DCXFResult, solve_dcxf

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
        mode: str = 'dec',
        points: int = 10,
        step: Optional[float] = None,
        values: Optional[List[float]] = None,
    ) -> 'ACXFResult':
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
        from jax_spice.analysis.xfer import ACXFConfig, ACXFResult, solve_acxf

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

        logger.info(f"ACXF complete: {len(result.frequencies)} frequencies, "
                    f"{len(result.tf)} sources")
        return result

    def _extract_all_sources(self) -> List[Dict]:
        """Extract all independent sources for transfer function analysis.

        Returns:
            List of source specifications with type, name, nodes, mag
        """
        sources = []

        for dev in self.devices:
            if dev['model'] in ('vsource', 'isource'):
                params = dev['params']
                nodes = dev.get('nodes', [0, 0])

                sources.append({
                    'name': dev['name'],
                    'type': dev['model'],
                    'pos_node': nodes[0] if len(nodes) > 0 else 0,
                    'neg_node': nodes[1] if len(nodes) > 1 else 0,
                    'mag': float(params.get('mag', 0.0)),
                    'dc': float(params.get('dc', 0.0)),
                })

        return sources

    # =========================================================================
    # Noise Analysis
    # =========================================================================

    def run_noise(
        self,
        out: Union[str, int] = 1,
        input_source: str = '',
        freq_start: float = 1.0,
        freq_stop: float = 1e6,
        mode: str = 'dec',
        points: int = 10,
        step: Optional[float] = None,
        values: Optional[List[float]] = None,
        temperature: float = 300.15,
    ) -> 'NoiseResult':
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
        from jax_spice.analysis.noise import (
            NoiseConfig, NoiseResult, run_noise_analysis,
            extract_noise_sources,
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
        noise_sources = extract_noise_sources(
            self.devices, dc_currents, temperature
        )

        logger.info(f"Found {len(noise_sources)} noise sources")

        # Get input source specification
        input_src = None
        if input_source:
            for src in self._extract_all_sources():
                if src['name'] == input_source:
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
        """Extract DC currents through devices from operating point.

        Args:
            V_dc: DC operating point voltages

        Returns:
            Dict mapping device name to DC current
        """
        dc_currents = {}

        for dev in self.devices:
            name = dev.get('name', '')
            model = dev.get('model', '')
            params = dev.get('params', {})
            nodes = dev.get('nodes', [0, 0])

            pos_node = nodes[0] if len(nodes) > 0 else 0
            neg_node = nodes[1] if len(nodes) > 1 else 0

            # Get node voltages
            v_pos = float(V_dc[pos_node - 1]) if pos_node > 0 and pos_node <= len(V_dc) else 0.0
            v_neg = float(V_dc[neg_node - 1]) if neg_node > 0 and neg_node <= len(V_dc) else 0.0
            v_diff = v_pos - v_neg

            if model == 'resistor':
                r = float(params.get('r', 1000.0))
                if r > 0:
                    dc_currents[name] = v_diff / r

            elif model in ('diode', 'd'):
                # Diode current: I = Is * (exp(V/nVT) - 1)
                Is = float(params.get('is', 1e-14))
                n = float(params.get('n', 1.0))
                vt = 0.0259  # Thermal voltage at 300K
                if v_diff > -5 * n * vt:  # Avoid overflow
                    dc_currents[name] = Is * (jnp.exp(v_diff / (n * vt)) - 1)
                else:
                    dc_currents[name] = -Is

        return dc_currents

