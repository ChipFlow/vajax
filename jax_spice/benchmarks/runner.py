"""VACASK Benchmark Runner

Generic runner for VACASK benchmark circuits. Parses benchmark .sim files
and runs transient analysis using the production JAX-based solver.

For circuits with OpenVAF-compiled devices (like PSP103 MOSFETs), uses a
hybrid solver that combines the JIT-compiled solver for simple devices
with Python-based Newton-Raphson for complex Verilog-A models.

TODO: Split out OpenVAF model compilation and caching into a separate module
(e.g., jax_spice/devices/openvaf_compiler.py) so it can be reused by other
components. The key functionality is:
- _COMPILED_MODEL_CACHE: module-level cache of compiled jitted functions
- _compile_openvaf_models(): compiles VA files to JAX functions with vmap+jit
- Static input preparation and batched evaluation
"""

import re
import sys
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Any, Optional

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np

from jax_spice.netlist.parser import VACASKParser
from jax_spice.netlist.circuit import Instance
from jax_spice.analysis.mna import MNASystem, DeviceInfo
from jax_spice.analysis.transient import transient_analysis_jit
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


class VACASKBenchmarkRunner:
    """Generic runner for VACASK benchmark circuits.

    Parses a benchmark .sim file and runs transient analysis using our solver.
    Handles resistors, capacitors, diodes, and voltage sources automatically.
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
    # These match the defaults in the Verilog-A model definitions
    # Used when circuit doesn't specify a parameter value
    MODEL_PARAM_DEFAULTS = {
        'diode': {
            'is': 1e-14, 'n': 1.0, 'rs': 0.0, 'bv': 1e20, 'ibv': 1e-10,
            'xti': 3.0, 'eg': 1.12, 'tnom': 27.0, 'cjo': 0.0, 'vj': 1.0,
            'm': 0.5, 'fc': 0.5, 'tt': 0.0, 'area': 1.0,
        },
        'sp_diode': {
            # Full SPICE diode model defaults from spice/sn/diode.va
            'is': 1e-14, 'jsw': 0.0, 'tnom': 0.0, 'rs': 0.0,
            'n': 1.0, 'ns': 1.0, 'tt': 0.0, 'cjo': 0.0, 'vj': 1.0,
            'm': 0.5, 'bv': 0.0, 'ibv': 1e-3, 'area': 0.0, 'pj': 0.0,
        },
        'resistor': {
            'r': 1000.0, 'zeta': 0.0, 'tnom': 300.0,
        },
        'capacitor': {
            'c': 1e-12,
        },
    }

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
        """Clear all cached data to free memory between benchmarks.

        This should be called after completing a benchmark run to release:
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

            compiled = {
                'module': module,
                'translator': translator,
                'jax_fn_array': jax_fn_array,
                'vmapped_fn': vmapped_fn,
                'array_metadata': array_metadata,
                'param_names': list(module.param_names),
                'param_kinds': list(module.param_kinds),
                'nodes': list(module.nodes),
                'collapsible_pairs': list(module.collapsible_pairs),
                'num_collapsible': module.num_collapsible,
            }

            # Store in both instance and module-level cache
            self._compiled_models[model_type] = compiled
            _COMPILED_MODEL_CACHE[model_type] = compiled

            log(f"  {model_type}: done ({len(module.param_names)} params, {len(module.nodes)} nodes)")

    def _prepare_static_inputs(self, model_type: str, openvaf_devices: List[Dict],
                                device_internal_nodes: Dict[str, Dict[str, int]],
                                ground: int) -> Tuple[jax.Array, List[int], List[Dict]]:
        """Prepare static (non-voltage) inputs for all devices once.

        This is called once per simulation and caches the static parameter values.
        Only voltage parameters need to be updated each NR iteration.

        Returns:
            (static_inputs, voltage_indices, device_contexts) where:
            - static_inputs is shape (num_devices, num_params) numpy array with static params
            - voltage_indices is list of param indices that are voltages
            - device_contexts is list of dicts with node_map, voltage_node_pairs for fast update
        """
        compiled = self._compiled_models.get(model_type)
        if not compiled:
            raise ValueError(f"OpenVAF model {model_type} not compiled")

        param_names = compiled['param_names']
        param_kinds = compiled['param_kinds']
        model_nodes = compiled['nodes']

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

            # Get model-specific defaults
            model_defaults = self.MODEL_PARAM_DEFAULTS.get(model_type, {})

            # Build a set of provided parameter names (lowercase) for param_given lookup
            provided_params = set(k.lower() for k in params.keys())

            # Fill input array for this device (static params only, voltages stay 0)
            for param_idx, (name, kind) in enumerate(zip(param_names, param_kinds)):
                if kind == 'voltage':
                    pass  # Already 0 from np.zeros
                elif kind == 'temperature':
                    # Temperature parameters (e.g., $temperature)
                    all_inputs[dev_idx, param_idx] = 300.15  # ~27°C in Kelvin
                elif kind == 'sysfun':
                    # System functions like mfactor
                    # mfactor is the system-level device multiplier and must be 1.0 by default
                    # (it's independent of the MULT model parameter)
                    if name.lower() == 'mfactor':
                        all_inputs[dev_idx, param_idx] = params.get('mfactor', 1.0)
                elif kind == 'param_given':
                    # param_given flags indicate whether a parameter was explicitly provided
                    # Set to 1.0 if the parameter is in the params dict
                    param_lower = name.lower()
                    if param_lower in provided_params or name in params:
                        all_inputs[dev_idx, param_idx] = 1.0
                    # Otherwise leave at 0.0 (not provided)
                elif kind == 'param':
                    param_lower = name.lower()
                    if param_lower in params:
                        all_inputs[dev_idx, param_idx] = float(params[param_lower])
                    elif name in params:
                        all_inputs[dev_idx, param_idx] = float(params[name])
                    elif 'temperature' in param_lower or name == '$temperature':
                        all_inputs[dev_idx, param_idx] = 300.15
                    elif param_lower == 'mfactor':
                        all_inputs[dev_idx, param_idx] = params.get('mfactor', 1.0)
                    elif param_lower in model_defaults:
                        # Use model-specific default (handles tnom, etc. correctly)
                        all_inputs[dev_idx, param_idx] = model_defaults[param_lower]
                    elif param_lower in ('tnom', 'tref', 'tr'):
                        # Temperature reference in Celsius (most VA models use 27°C)
                        all_inputs[dev_idx, param_idx] = 27.0
                    elif param_lower == 'cox':
                        # COX = gate oxide capacitance per unit area = eps_r * eps_0 / t_ox
                        # Must be computed from TOXO and EPSROXO if not provided directly
                        toxo = float(params.get('toxo', params.get('TOXO', 2e-9)))
                        epsroxo = float(params.get('epsroxo', params.get('EPSROXO', 3.9)))
                        eps0 = 8.854187817e-12  # vacuum permittivity F/m
                        all_inputs[dev_idx, param_idx] = epsroxo * eps0 / toxo
                    else:
                        all_inputs[dev_idx, param_idx] = 1.0
                # hidden_state, current stay 0 from np.zeros
                # EXCEPT for PSP103 L_i and W_i which must be computed from L and W
                # The OpenVAF code generator doesn't correctly handle these phi nodes
                elif kind == 'hidden_state':
                    if model_type == 'psp103':
                        name_lower = name.lower()
                        # PSP103 hidden_state params computed via CLIP_LOW from base params
                        # Reference: PSP103_module.include lines 551-713

                        # Basic device geometry
                        if name_lower == 'l_i':
                            l_val = params.get('l', params.get('L', 1e-6))
                            all_inputs[dev_idx, param_idx] = max(float(l_val), 1e-9)
                        elif name_lower == 'w_i':
                            w_val = params.get('w', params.get('W', 1e-5))
                            nf_val = params.get('nf', params.get('NF', 1.0))
                            all_inputs[dev_idx, param_idx] = max(float(w_val) / float(nf_val), 1e-9)
                        elif name_lower == 'nf_i':
                            nf_val = params.get('nf', params.get('NF', 1.0))
                            all_inputs[dev_idx, param_idx] = max(float(nf_val), 1.0)
                        elif name_lower == 'chnl_type':
                            type_val = params.get('type', params.get('TYPE', 1))
                            all_inputs[dev_idx, param_idx] = 1 if float(type_val) > 0 else -1

                        # Computed geometry-dependent values
                        # lcinv2 = 1/L^2 (inverse length squared, used in channel current)
                        elif name_lower == 'lcinv2':
                            l_val = max(float(params.get('l', params.get('L', 1e-6))), 1e-9)
                            all_inputs[dev_idx, param_idx] = 1.0 / (l_val * l_val)

                        # Switch parameters (pass through)
                        elif name_lower == 'swgeo_i':
                            all_inputs[dev_idx, param_idx] = float(params.get('swgeo', params.get('SWGEO', 1)))
                        elif name_lower == 'swigate_i':
                            all_inputs[dev_idx, param_idx] = float(params.get('swigate', params.get('SWIGATE', 0)))
                        elif name_lower == 'swimpact_i':
                            all_inputs[dev_idx, param_idx] = float(params.get('swimpact', params.get('SWIMPACT', 0)))
                        elif name_lower == 'swgidl_i':
                            all_inputs[dev_idx, param_idx] = float(params.get('swgidl', params.get('SWGIDL', 0)))
                        elif name_lower == 'swjuncap_i':
                            all_inputs[dev_idx, param_idx] = float(params.get('swjuncap', params.get('SWJUNCAP', 0)))
                        elif name_lower == 'swjunasym_i':
                            all_inputs[dev_idx, param_idx] = float(params.get('swjunasym', params.get('SWJUNASYM', 0)))
                        elif name_lower == 'swnud_i':
                            all_inputs[dev_idx, param_idx] = float(params.get('swnud', params.get('SWNUD', 0)))
                        elif name_lower == 'swedge_i':
                            all_inputs[dev_idx, param_idx] = float(params.get('swedge', params.get('SWEDGE', 0)))
                        elif name_lower == 'swdelvtac_i':
                            all_inputs[dev_idx, param_idx] = float(params.get('swdelvtac', params.get('SWDELVTAC', 0)))
                        elif name_lower == 'swqsat_i':
                            all_inputs[dev_idx, param_idx] = float(params.get('swqsat', params.get('SWQSAT', 0)))
                        elif name_lower == 'swqpart_i':
                            all_inputs[dev_idx, param_idx] = float(params.get('swqpart', params.get('SWQPART', 0)))
                        elif name_lower == 'swign_i':
                            # SWIGN defaults to 1.0 in param kind (fallback), so hidden_state should too
                            # If SWIGN_i = 0, mig = 1e-40 which creates 1e40 conductance (breaks NR)
                            all_inputs[dev_idx, param_idx] = float(params.get('swign', params.get('SWIGN', 1)))

                        # Noise parameters - FNT is thermal noise coefficient, defaults to 1.0
                        # If FNT_i = 0, nt = 0, and mig = 1e-40 (breaks NR with 1e40 conductance)
                        elif name_lower == 'fnt_i':
                            all_inputs[dev_idx, param_idx] = max(float(params.get('fnt', params.get('FNT', 1.0))), 0.0)
                        elif name_lower == 'fntexc_i':
                            all_inputs[dev_idx, param_idx] = max(float(params.get('fntexc', params.get('FNTEXC', 0.0))), 0.0)
                        elif name_lower == 'fntedge_i':
                            all_inputs[dev_idx, param_idx] = max(float(params.get('fntedge', params.get('FNTEDGE', 1.0))), 0.0)

                        # Critical oxide/material parameters with CLIP_LOW
                        elif name_lower == 'qmc_i':
                            all_inputs[dev_idx, param_idx] = max(float(params.get('qmc', params.get('QMC', 1.0))), 0.0)
                        elif name_lower == 'toxo_i':
                            all_inputs[dev_idx, param_idx] = max(float(params.get('toxo', params.get('TOXO', 2e-9))), 1e-10)
                        elif name_lower == 'epsroxo_i':
                            all_inputs[dev_idx, param_idx] = max(float(params.get('epsroxo', params.get('EPSROXO', 3.9))), 1.0)
                        elif name_lower == 'nsubo_i':
                            all_inputs[dev_idx, param_idx] = max(float(params.get('nsubo', params.get('NSUBO', 3e23))), 1e20)

                        # Geometry parameters
                        elif name_lower == 'wseg_i':
                            all_inputs[dev_idx, param_idx] = max(float(params.get('wseg', params.get('WSEG', 1.5e-10))), 1e-10)
                        elif name_lower == 'npck_i':
                            all_inputs[dev_idx, param_idx] = max(float(params.get('npck', params.get('NPCK', 1e24))), 0.0)
                        elif name_lower == 'wsegp_i':
                            all_inputs[dev_idx, param_idx] = max(float(params.get('wsegp', params.get('WSEGP', 0.9e-8))), 1e-10)
                        elif name_lower == 'lpck_i':
                            all_inputs[dev_idx, param_idx] = max(float(params.get('lpck', params.get('LPCK', 5.5e-8))), 1e-10)

                        # Overlap oxide parameters
                        elif name_lower == 'toxovo_i':
                            all_inputs[dev_idx, param_idx] = max(float(params.get('toxovo', params.get('TOXOVO', 1.5e-9))), 1e-10)
                        elif name_lower == 'toxovdo_i':
                            all_inputs[dev_idx, param_idx] = max(float(params.get('toxovdo', params.get('TOXOVDO', 2e-9))), 1e-10)
                        elif name_lower == 'lov_i':
                            all_inputs[dev_idx, param_idx] = max(float(params.get('lov', params.get('LOV', 10e-9))), 0.0)
                        elif name_lower == 'lovd_i':
                            all_inputs[dev_idx, param_idx] = max(float(params.get('lovd', params.get('LOVD', 0))), 0.0)

                        # Mobility and beta parameters
                        elif name_lower == 'lp1_i':
                            all_inputs[dev_idx, param_idx] = max(float(params.get('lp1', params.get('LP1', 1.5e-7))), 1e-10)
                        elif name_lower == 'lp2_i':
                            all_inputs[dev_idx, param_idx] = max(float(params.get('lp2', params.get('LP2', 8.5e-10))), 1e-10)
                        elif name_lower == 'wbet_i':
                            all_inputs[dev_idx, param_idx] = max(float(params.get('wbet', params.get('WBET', 5e-10))), 1e-10)

                        # Short channel effect parameters
                        elif name_lower == 'axl_i':
                            all_inputs[dev_idx, param_idx] = max(float(params.get('axl', params.get('AXL', 0.2))), 0.0)
                        elif name_lower == 'alp1l2_i':
                            all_inputs[dev_idx, param_idx] = max(float(params.get('alp1l2', params.get('ALP1L2', 0.1))), 0.0)
                        elif name_lower == 'alp2l2_i':
                            all_inputs[dev_idx, param_idx] = max(float(params.get('alp2l2', params.get('ALP2L2', 0.5))), 0.0)

                        # Reference parameters
                        elif name_lower == 'saref_i':
                            all_inputs[dev_idx, param_idx] = max(float(params.get('saref', params.get('SAREF', 1e-9))), 1e-9)
                        elif name_lower == 'sbref_i':
                            all_inputs[dev_idx, param_idx] = max(float(params.get('sbref', params.get('SBREF', 1e-9))), 1e-9)
                        elif name_lower == 'scref_i':
                            all_inputs[dev_idx, param_idx] = max(float(params.get('scref', params.get('SCREF', 0))), 0.0)

                        # Velocity saturation params (CLIP_BOTH)
                        elif name_lower == 'kvsat_i':
                            val = float(params.get('kvsat', params.get('KVSAT', 0)))
                            all_inputs[dev_idx, param_idx] = min(max(val, -1.0), 1.0)
                        elif name_lower == 'kvsatac_i':
                            val = float(params.get('kvsatac', params.get('KVSATAC', 0)))
                            all_inputs[dev_idx, param_idx] = min(max(val, -1.0), 1.0)
                        elif name_lower == 'web_i':
                            all_inputs[dev_idx, param_idx] = float(params.get('web', params.get('WEB', 0)))
                        elif name_lower == 'wec_i':
                            all_inputs[dev_idx, param_idx] = float(params.get('wec', params.get('WEC', 0)))

                        # Geometry direct copies (from instance params)
                        elif name_lower == 'sa_i':
                            all_inputs[dev_idx, param_idx] = float(params.get('sa', params.get('SA', 0)))
                        elif name_lower == 'sb_i':
                            all_inputs[dev_idx, param_idx] = float(params.get('sb', params.get('SB', 0)))
                        elif name_lower == 'sd_i':
                            all_inputs[dev_idx, param_idx] = float(params.get('sd', params.get('SD', 0)))
                        elif name_lower == 'sc_i':
                            all_inputs[dev_idx, param_idx] = float(params.get('sc', params.get('SC', 0)))
                        elif name_lower == 'xgw_i':
                            all_inputs[dev_idx, param_idx] = float(params.get('xgw', params.get('XGW', 0)))
                        elif name_lower == 'absource_i':
                            all_inputs[dev_idx, param_idx] = float(params.get('absource', params.get('ABSOURCE', 0)))
                        elif name_lower == 'lssource_i':
                            all_inputs[dev_idx, param_idx] = float(params.get('lssource', params.get('LSSOURCE', 0)))
                        elif name_lower == 'lgsource_i':
                            all_inputs[dev_idx, param_idx] = float(params.get('lgsource', params.get('LGSOURCE', 0)))
                        elif name_lower == 'abdrain_i':
                            all_inputs[dev_idx, param_idx] = float(params.get('abdrain', params.get('ABDRAIN', 0)))
                        elif name_lower == 'lsdrain_i':
                            all_inputs[dev_idx, param_idx] = float(params.get('lsdrain', params.get('LSDRAIN', 0)))
                        elif name_lower == 'lgdrain_i':
                            all_inputs[dev_idx, param_idx] = float(params.get('lgdrain', params.get('LGDRAIN', 0)))
                        elif name_lower == 'as_i':
                            all_inputs[dev_idx, param_idx] = float(params.get('as', params.get('AS', 0)))
                        elif name_lower == 'ps_i':
                            all_inputs[dev_idx, param_idx] = float(params.get('ps', params.get('PS', 0)))
                        elif name_lower == 'ad_i':
                            all_inputs[dev_idx, param_idx] = float(params.get('ad', params.get('AD', 0)))
                        elif name_lower == 'pd_i':
                            all_inputs[dev_idx, param_idx] = float(params.get('pd', params.get('PD', 0)))
                        elif name_lower == 'jw_i':
                            all_inputs[dev_idx, param_idx] = float(params.get('jw', params.get('JW', 0)))
                        elif name_lower == 'scc_i':
                            all_inputs[dev_idx, param_idx] = max(float(params.get('scc', params.get('SCC', 0))), 0.0)
                        elif name_lower == 'ngcon_i':
                            ngcon = float(params.get('ngcon', params.get('NGCON', 1)))
                            all_inputs[dev_idx, param_idx] = 1.0 if ngcon < 1.5 else 2.0

                        # Resistance parameters
                        # NOTE: If RSH/RSHD are not in model card, they default to 1.0 in 'param' kind
                        # (via line 675 fallback). So hidden_state must also use 1.0 default.
                        elif name_lower == 'rshg_i':
                            all_inputs[dev_idx, param_idx] = max(float(params.get('rshg', params.get('RSHG', 0))), 0.0)
                        elif name_lower == 'rsh_i':
                            # RSH defaults to 1.0 in param kind if not in model card
                            all_inputs[dev_idx, param_idx] = max(float(params.get('rsh', params.get('RSH', 1.0))), 0.0)
                        elif name_lower == 'rshd_i':
                            # RSHD defaults to 1.0 in param kind if not in model card
                            all_inputs[dev_idx, param_idx] = max(float(params.get('rshd', params.get('RSHD', 1.0))), 0.0)
                        elif name_lower == 'rint_i':
                            all_inputs[dev_idx, param_idx] = max(float(params.get('rint', params.get('RINT', 0))), 0.0)
                        elif name_lower == 'rvpoly_i':
                            all_inputs[dev_idx, param_idx] = max(float(params.get('rvpoly', params.get('RVPOLY', 0))), 0.0)

                        # Edge parameters
                        elif name_lower == 'nsubedgeo_i':
                            all_inputs[dev_idx, param_idx] = max(float(params.get('nsubedgeo', params.get('NSUBEDGEO', 1e20))), 1e20)
                        elif name_lower == 'lpedge_i':
                            all_inputs[dev_idx, param_idx] = max(float(params.get('lpedge', params.get('LPEDGE', 1e-10))), 1e-10)

                        # Temperature
                        elif name_lower == 'tr_i':
                            all_inputs[dev_idx, param_idx] = max(float(params.get('tr', params.get('TR', 27.0))), -273.0)

                        # LOD parameters
                        elif name_lower == 'llodkuo_i':
                            all_inputs[dev_idx, param_idx] = max(float(params.get('llodkuo', params.get('LLODKUO', 0))), 0.0)
                        elif name_lower == 'wlodkuo_i':
                            all_inputs[dev_idx, param_idx] = max(float(params.get('wlodkuo', params.get('WLODKUO', 0))), 0.0)
                        elif name_lower == 'llodvth_i':
                            all_inputs[dev_idx, param_idx] = max(float(params.get('llodvth', params.get('LLODVTH', 0))), 0.0)
                        elif name_lower == 'wlodvth_i':
                            all_inputs[dev_idx, param_idx] = max(float(params.get('wlodvth', params.get('WLODVTH', 0))), 0.0)
                        elif name_lower == 'lodetao_i':
                            all_inputs[dev_idx, param_idx] = max(float(params.get('lodetao', params.get('LODETAO', 0))), 0.0)

                        # Stress parameters
                        elif name_lower == 'sca_i':
                            all_inputs[dev_idx, param_idx] = max(float(params.get('sca', params.get('SCA', 0))), 0.0)
                        elif name_lower == 'scb_i':
                            all_inputs[dev_idx, param_idx] = max(float(params.get('scb', params.get('SCB', 0))), 0.0)

                        # MULT_i = MULT (device multiplier)
                        elif name_lower == 'mult_i':
                            all_inputs[dev_idx, param_idx] = float(params.get('mult', params.get('MULT', 1.0)))

                        # EPSSI = 11.7 (silicon dielectric constant - fixed)
                        elif name_lower == 'epssi':
                            all_inputs[dev_idx, param_idx] = 11.7

                        # ============================================================
                        # Critical intermediate computed params
                        # These are computed from model params and used in device eval
                        # ============================================================
                        elif name_lower == 'epsox':
                            # Oxide permittivity = EPSO * EPSROX
                            epsrox = float(params.get('epsroxo', params.get('EPSROXO', 3.9)))
                            all_inputs[dev_idx, param_idx] = 8.85418782e-12 * epsrox

                        elif name_lower == 'coxprime':
                            # Gate oxide capacitance/area = EPSOX / TOX
                            epsrox = float(params.get('epsroxo', params.get('EPSROXO', 3.9)))
                            tox = max(float(params.get('toxo', params.get('TOXO', 2e-9))), 1e-10)
                            all_inputs[dev_idx, param_idx] = 8.85418782e-12 * epsrox / tox

                        elif name_lower == 'tox_sq':
                            # TOX squared
                            tox = max(float(params.get('toxo', params.get('TOXO', 2e-9))), 1e-10)
                            all_inputs[dev_idx, param_idx] = tox * tox

                        elif name_lower == 'cox_over_q':
                            # Cox / q for charge calculations
                            epsrox = float(params.get('epsroxo', params.get('EPSROXO', 3.9)))
                            tox = max(float(params.get('toxo', params.get('TOXO', 2e-9))), 1e-10)
                            qele = 1.602176634e-19
                            all_inputs[dev_idx, param_idx] = 8.85418782e-12 * epsrox / tox / qele

                        elif name_lower == 'nsub':
                            # Substrate doping - from NSUBO + length/width variations
                            nsubo = max(float(params.get('nsubo', params.get('NSUBO', 3e23))), 1e20)
                            all_inputs[dev_idx, param_idx] = nsubo

                        elif name_lower == 'nsub0e':
                            # Edge substrate doping
                            nsubo = max(float(params.get('nsubo', params.get('NSUBO', 3e23))), 1e20)
                            all_inputs[dev_idx, param_idx] = nsubo

                        elif name_lower == 'npcke':
                            # Edge NPCK
                            npck = max(float(params.get('npck', params.get('NPCK', 1e24))), 0.0)
                            all_inputs[dev_idx, param_idx] = npck

                        elif name_lower == 'lpcke':
                            # Edge LPCK
                            lpck = max(float(params.get('lpck', params.get('LPCK', 5.5e-8))), 1e-10)
                            all_inputs[dev_idx, param_idx] = lpck

                        # Effective channel geometry (LE, WE, GPE, GWE)
                        elif name_lower in ('le', 'gpe', 'lnewle'):
                            # Effective channel length
                            l_val = max(float(params.get('l', params.get('L', 1e-6))), 1e-9)
                            dlq = float(params.get('dlq', params.get('DLQ', 0)))
                            all_inputs[dev_idx, param_idx] = l_val + dlq

                        elif name_lower in ('we', 'gwe', 'lnewwe'):
                            # Effective channel width
                            w_val = max(float(params.get('w', params.get('W', 1e-5))), 1e-9)
                            nf = max(float(params.get('nf', params.get('NF', 1))), 1)
                            dwq = float(params.get('dwq', params.get('DWQ', 0)))
                            all_inputs[dev_idx, param_idx] = w_val / nf + dwq

                        elif name_lower == 'ile':
                            # 1 / LE
                            l_val = max(float(params.get('l', params.get('L', 1e-6))), 1e-9)
                            all_inputs[dev_idx, param_idx] = 1.0 / l_val

                        elif name_lower == 'iwe':
                            # 1 / WE
                            w_val = max(float(params.get('w', params.get('W', 1e-5))), 1e-9)
                            nf = max(float(params.get('nf', params.get('NF', 1))), 1)
                            all_inputs[dev_idx, param_idx] = 1.0 / (w_val / nf)

                        elif name_lower == 'ilewe':
                            # 1 / (LE * WE)
                            l_val = max(float(params.get('l', params.get('L', 1e-6))), 1e-9)
                            w_val = max(float(params.get('w', params.get('W', 1e-5))), 1e-9)
                            nf = max(float(params.get('nf', params.get('NF', 1))), 1)
                            all_inputs[dev_idx, param_idx] = 1.0 / (l_val * w_val / nf)

                        elif name_lower in ('iile', 'iilecv'):
                            # 1 / LE^2
                            l_val = max(float(params.get('l', params.get('L', 1e-6))), 1e-9)
                            all_inputs[dev_idx, param_idx] = 1.0 / (l_val * l_val)

                        elif name_lower in ('iiwe', 'iiwecv'):
                            # 1 / WE^2
                            w_val = max(float(params.get('w', params.get('W', 1e-5))), 1e-9)
                            nf = max(float(params.get('nf', params.get('NF', 1))), 1)
                            we = w_val / nf
                            all_inputs[dev_idx, param_idx] = 1.0 / (we * we)

                        elif name_lower in ('iilewe', 'iilewecv'):
                            # 1 / (LE^2 * WE)
                            l_val = max(float(params.get('l', params.get('L', 1e-6))), 1e-9)
                            w_val = max(float(params.get('w', params.get('W', 1e-5))), 1e-9)
                            nf = max(float(params.get('nf', params.get('NF', 1))), 1)
                            all_inputs[dev_idx, param_idx] = 1.0 / (l_val * l_val * w_val / nf)

                        elif name_lower in ('iiilewe', 'iiilewecv'):
                            # 1 / (LE^3 * WE)
                            l_val = max(float(params.get('l', params.get('L', 1e-6))), 1e-9)
                            w_val = max(float(params.get('w', params.get('W', 1e-5))), 1e-9)
                            nf = max(float(params.get('nf', params.get('NF', 1))), 1)
                            all_inputs[dev_idx, param_idx] = 1.0 / (l_val ** 3 * w_val / nf)

                        elif name_lower in ('ilecv', 'iil'):
                            # 1 / L (for capacitance)
                            l_val = max(float(params.get('l', params.get('L', 1e-6))), 1e-9)
                            all_inputs[dev_idx, param_idx] = 1.0 / l_val

                        elif name_lower in ('iilcv', 'iiwcv'):
                            # 1 / L^2 or 1 / W^2 (for capacitance)
                            l_val = max(float(params.get('l', params.get('L', 1e-6))), 1e-9)
                            all_inputs[dev_idx, param_idx] = 1.0 / (l_val * l_val)

                        elif name_lower in ('iilwcv', 'iiilwcv'):
                            # 1 / (L^2 * W) or 1 / (L^3 * W)
                            l_val = max(float(params.get('l', params.get('L', 1e-6))), 1e-9)
                            w_val = max(float(params.get('w', params.get('W', 1e-5))), 1e-9)
                            nf = max(float(params.get('nf', params.get('NF', 1))), 1)
                            if 'iii' in name_lower:
                                all_inputs[dev_idx, param_idx] = 1.0 / (l_val ** 3 * w_val / nf)
                            else:
                                all_inputs[dev_idx, param_idx] = 1.0 / (l_val ** 2 * w_val / nf)

                        # Edge geometry
                        elif name_lower == 'we_edge':
                            # Edge effective width
                            w_val = max(float(params.get('w', params.get('W', 1e-5))), 1e-9)
                            nf = max(float(params.get('nf', params.get('NF', 1))), 1)
                            all_inputs[dev_idx, param_idx] = w_val / nf

                        elif name_lower == 'iwe_edge':
                            # 1 / WE_edge
                            w_val = max(float(params.get('w', params.get('W', 1e-5))), 1e-9)
                            nf = max(float(params.get('nf', params.get('NF', 1))), 1)
                            all_inputs[dev_idx, param_idx] = 1.0 / (w_val / nf)

                        elif name_lower == 'gpe_edge':
                            # Edge effective length
                            l_val = max(float(params.get('l', params.get('L', 1e-6))), 1e-9)
                            all_inputs[dev_idx, param_idx] = l_val

                        # Binning/interpolation factors (AA, BB)
                        elif name_lower == 'aa':
                            # Binning interpolation factor
                            all_inputs[dev_idx, param_idx] = 1.0

                        elif name_lower == 'bb':
                            # Binning interpolation factor
                            all_inputs[dev_idx, param_idx] = 0.0

                        # Temperature factors
                        elif name_lower == 'tmpx':
                            # Temperature-dependent mobility factor
                            all_inputs[dev_idx, param_idx] = 1.0

                        elif name_lower == 'temp0':
                            # Temperature factor
                            all_inputs[dev_idx, param_idx] = 1.0

                        elif name_lower == 'temp00':
                            # Temperature factor for stress
                            all_inputs[dev_idx, param_idx] = 1.0

                        # Beta factors
                        elif name_lower == 'fbet1e':
                            fbet1 = float(params.get('fbet1', params.get('FBET1', -0.3)))
                            all_inputs[dev_idx, param_idx] = fbet1

                        # LP1 effective
                        elif name_lower == 'lp1e':
                            lp1 = max(float(params.get('lp1', params.get('LP1', 1.5e-7))), 1e-10)
                            all_inputs[dev_idx, param_idx] = lp1

                        # Noise params
                        elif name_lower == 'lnoi':
                            l_val = max(float(params.get('l', params.get('L', 1e-6))), 1e-9)
                            all_inputs[dev_idx, param_idx] = l_val

                        elif name_lower == 'lred':
                            # Reduced channel length
                            l_val = max(float(params.get('l', params.get('L', 1e-6))), 1e-9)
                            all_inputs[dev_idx, param_idx] = l_val

                        # NEFFAC_i - effective doping with accumulation factor
                        elif name_lower == 'neffac_i':
                            nsubo = max(float(params.get('nsubo', params.get('NSUBO', 3e23))), 1e20)
                            facneffac = float(params.get('facneffaco', params.get('FACNEFFACO', 0.8)))
                            all_inputs[dev_idx, param_idx] = nsubo * facneffac

                        # ============================================================
                        # Temperature-dependent hidden_state params
                        # Reference: PSP103_macrodefs.include TempInitialize macro
                        # ============================================================
                        # Physical constants
                        elif name_lower == 'kbol_over_qele':
                            # Boltzmann/charge = kT/q factor
                            all_inputs[dev_idx, param_idx] = 8.617333262e-5  # eV/K

                        # Temperature variables (assuming T = 27°C = 300.15K)
                        elif name_lower == 'tkr':
                            # Reference temp in Kelvin: TR + 273.15
                            tr_val = float(params.get('tr', params.get('TR', 27.0)))
                            all_inputs[dev_idx, param_idx] = tr_val + 273.15
                        elif name_lower == 'tka':
                            # Ambient temp in Kelvin (no self-heating: TKA = $temperature + DTA)
                            all_inputs[dev_idx, param_idx] = 300.15
                        elif name_lower == 'tkd':
                            # Device temp in Kelvin (no self-heating: TKD = TKA)
                            all_inputs[dev_idx, param_idx] = 300.15
                        elif name_lower == 'tkd_sq':
                            # TKD squared
                            all_inputs[dev_idx, param_idx] = 300.15 * 300.15
                        elif name_lower == 'rta':
                            # TKA / TKR ratio
                            all_inputs[dev_idx, param_idx] = 1.0
                        elif name_lower == 'delta':
                            # TKA - TKR
                            all_inputs[dev_idx, param_idx] = 0.0
                        elif name_lower == 'delt':
                            # TKD - TKR
                            all_inputs[dev_idx, param_idx] = 0.0
                        elif name_lower == 'rtn':
                            # TKR / TKD ratio
                            all_inputs[dev_idx, param_idx] = 1.0
                        elif name_lower == 'ln_rtn':
                            # ln(rTn) = 0 when T = Tref
                            all_inputs[dev_idx, param_idx] = 0.0

                        # Thermal voltage and related
                        elif name_lower == 'phita':
                            # TKA * KBOL / QELE
                            all_inputs[dev_idx, param_idx] = 300.15 * 8.617333262e-5
                        elif name_lower == 'inv_phita':
                            all_inputs[dev_idx, param_idx] = 1.0 / (300.15 * 8.617333262e-5)
                        elif name_lower == 'phit':
                            # Thermal voltage = kT/q = TKD * KBOL / QELE
                            all_inputs[dev_idx, param_idx] = 300.15 * 8.617333262e-5  # ~0.02586V
                        elif name_lower == 'inv_phit':
                            all_inputs[dev_idx, param_idx] = 1.0 / (300.15 * 8.617333262e-5)

                        # Bandgap energy: Eg = 1.179 - 9.025e-5 * TKD - 3.05e-7 * TKD^2
                        elif name_lower == 'eg':
                            tkd = 300.15
                            all_inputs[dev_idx, param_idx] = 1.179 - 9.025e-5 * tkd - 3.05e-7 * tkd * tkd

                        # phibFac = (1.045 + 4.5e-4*TKD) * (0.523 + 1.4e-3*TKD - 1.48e-6*TKD^2) * TKD^2 / 9.0e4
                        elif name_lower == 'phibfac':
                            tkd = 300.15
                            tkd_sq = tkd * tkd
                            phibfac = (1.045 + 4.5e-4 * tkd) * (0.523 + 1.4e-3 * tkd - 1.48e-6 * tkd_sq) * tkd_sq / 9.0e4
                            all_inputs[dev_idx, param_idx] = max(phibfac, 1.0e-3)

                        # Noise thermal factor: nt0 = 4 * KBOL * TKD
                        elif name_lower == 'nt0':
                            all_inputs[dev_idx, param_idx] = 4.0 * 1.38066e-23 * 300.15

                        # JUNCAP temperature params (phitr/phitd are the key ones)
                        elif name_lower == 'phitr':
                            all_inputs[dev_idx, param_idx] = 8.617333262e-5 * 300.15
                        elif name_lower == 'phitrinv':
                            all_inputs[dev_idx, param_idx] = 1.0 / (8.617333262e-5 * 300.15)
                        elif name_lower == 'phitd':
                            all_inputs[dev_idx, param_idx] = 8.617333262e-5 * 300.15
                        elif name_lower == 'phitdinv':
                            all_inputs[dev_idx, param_idx] = 1.0 / (8.617333262e-5 * 300.15)
                        elif name_lower == 'auxt':
                            # auxillary temp variable = TKD
                            all_inputs[dev_idx, param_idx] = 300.15

                        # ============================================================
                        # JUNCAP200 Junction Model Parameters
                        # Reference: JUNCAP200_InitModel.include, JUNCAP200_parlist.include
                        # ============================================================
                        # Constants for JUNCAP initialization
                        elif name_lower == 'frev_i':
                            val = float(params.get('frev', params.get('FREV', 1000.0)))
                            all_inputs[dev_idx, param_idx] = max(min(val, 1e40), 1.0)
                        elif name_lower == 'ifactor_i':
                            all_inputs[dev_idx, param_idx] = max(float(params.get('ifactor', params.get('IFACTOR', 1.0))), 0.0)
                        elif name_lower == 'cfactor_i':
                            all_inputs[dev_idx, param_idx] = max(float(params.get('cfactor', params.get('CFACTOR', 1.0))), 0.0)

                        # Junction parameters with default clipping (source-side)
                        elif name_lower == 'cjorbot_i':
                            all_inputs[dev_idx, param_idx] = max(float(params.get('cjorbot', params.get('CJORBOT', 1e-3))), 1e-12)
                        elif name_lower == 'cjorsti_i':
                            all_inputs[dev_idx, param_idx] = max(float(params.get('cjorsti', params.get('CJORSTI', 1e-9))), 1e-18)
                        elif name_lower == 'cjorgat_i':
                            all_inputs[dev_idx, param_idx] = max(float(params.get('cjorgat', params.get('CJORGAT', 1e-9))), 1e-18)
                        elif name_lower == 'vbirbot_i':
                            all_inputs[dev_idx, param_idx] = max(float(params.get('vbirbot', params.get('VBIRBOT', 1.0))), 0.05)
                        elif name_lower == 'vbirsti_i':
                            all_inputs[dev_idx, param_idx] = max(float(params.get('vbirsti', params.get('VBIRSTI', 1.0))), 0.05)
                        elif name_lower == 'vbirgat_i':
                            all_inputs[dev_idx, param_idx] = max(float(params.get('vbirgat', params.get('VBIRGAT', 1.0))), 0.05)
                        elif name_lower == 'pbot_i':
                            val = float(params.get('pbot', params.get('PBOT', 0.5)))
                            all_inputs[dev_idx, param_idx] = max(min(val, 0.95), 0.05)
                        elif name_lower == 'psti_i':
                            val = float(params.get('psti', params.get('PSTI', 0.5)))
                            all_inputs[dev_idx, param_idx] = max(min(val, 0.95), 0.05)
                        elif name_lower == 'pgat_i':
                            val = float(params.get('pgat', params.get('PGAT', 0.5)))
                            all_inputs[dev_idx, param_idx] = max(min(val, 0.95), 0.05)
                        elif name_lower == 'phigbot_i':
                            all_inputs[dev_idx, param_idx] = float(params.get('phigbot', params.get('PHIGBOT', 1.16)))
                        elif name_lower == 'phigsti_i':
                            all_inputs[dev_idx, param_idx] = float(params.get('phigsti', params.get('PHIGSTI', 1.16)))
                        elif name_lower == 'phiggat_i':
                            all_inputs[dev_idx, param_idx] = float(params.get('phiggat', params.get('PHIGGAT', 1.16)))
                        elif name_lower == 'idsatrbot_i':
                            all_inputs[dev_idx, param_idx] = max(float(params.get('idsatrbot', params.get('IDSATRBOT', 1e-12))), 1e-30)
                        elif name_lower == 'idsatrsti_i':
                            all_inputs[dev_idx, param_idx] = max(float(params.get('idsatrsti', params.get('IDSATRSTI', 1e-18))), 1e-30)
                        elif name_lower == 'idsatrgat_i':
                            all_inputs[dev_idx, param_idx] = max(float(params.get('idsatrgat', params.get('IDSATRGAT', 1e-18))), 1e-30)
                        elif name_lower == 'xjunsti_i':
                            all_inputs[dev_idx, param_idx] = max(float(params.get('xjunsti', params.get('XJUNSTI', 1e-7))), 1e-9)
                        elif name_lower == 'xjungat_i':
                            all_inputs[dev_idx, param_idx] = max(float(params.get('xjungat', params.get('XJUNGAT', 1e-7))), 1e-9)
                        elif name_lower == 'mefftatbot_i':
                            all_inputs[dev_idx, param_idx] = max(float(params.get('mefftatbot', params.get('MEFFTATBOT', 0.25))), 0.001)
                        elif name_lower == 'mefftatsti_i':
                            all_inputs[dev_idx, param_idx] = max(float(params.get('mefftatsti', params.get('MEFFTATSTI', 0.25))), 0.001)
                        elif name_lower == 'mefftatgat_i':
                            all_inputs[dev_idx, param_idx] = max(float(params.get('mefftatgat', params.get('MEFFTATGAT', 0.25))), 0.001)
                        elif name_lower == 'vbrbot_i':
                            all_inputs[dev_idx, param_idx] = max(float(params.get('vbrbot', params.get('VBRBOT', 10.0))), 0.1)
                        elif name_lower == 'vbrsti_i':
                            all_inputs[dev_idx, param_idx] = max(float(params.get('vbrsti', params.get('VBRSTI', 10.0))), 0.1)
                        elif name_lower == 'vbrgat_i':
                            all_inputs[dev_idx, param_idx] = max(float(params.get('vbrgat', params.get('VBRGAT', 10.0))), 0.1)
                        elif name_lower == 'pbrbot_i':
                            all_inputs[dev_idx, param_idx] = max(float(params.get('pbrbot', params.get('PBRBOT', 4.0))), 0.1)
                        elif name_lower == 'pbrsti_i':
                            all_inputs[dev_idx, param_idx] = max(float(params.get('pbrsti', params.get('PBRSTI', 4.0))), 0.1)
                        elif name_lower == 'pbrgat_i':
                            all_inputs[dev_idx, param_idx] = max(float(params.get('pbrgat', params.get('PBRGAT', 4.0))), 0.1)

                        # Drain-side junction params (same defaults, D suffix in param names)
                        elif name_lower == 'cjorbotd_i':
                            all_inputs[dev_idx, param_idx] = max(float(params.get('cjorbotd', params.get('CJORBOTD', 1e-3))), 1e-12)
                        elif name_lower == 'cjorstid_i':
                            all_inputs[dev_idx, param_idx] = max(float(params.get('cjorstid', params.get('CJORSTID', 1e-9))), 1e-18)
                        elif name_lower == 'cjorgatd_i':
                            all_inputs[dev_idx, param_idx] = max(float(params.get('cjorgatd', params.get('CJORGATD', 1e-9))), 1e-18)
                        elif name_lower == 'vbirbotd_i':
                            all_inputs[dev_idx, param_idx] = max(float(params.get('vbirbotd', params.get('VBIRBOTD', 1.0))), 0.05)
                        elif name_lower == 'vbirstid_i':
                            all_inputs[dev_idx, param_idx] = max(float(params.get('vbirstid', params.get('VBIRSTID', 1.0))), 0.05)
                        elif name_lower == 'vbirgatd_i':
                            all_inputs[dev_idx, param_idx] = max(float(params.get('vbirgatd', params.get('VBIRGATD', 1.0))), 0.05)
                        elif name_lower == 'pbotd_i':
                            val = float(params.get('pbotd', params.get('PBOTD', 0.5)))
                            all_inputs[dev_idx, param_idx] = max(min(val, 0.95), 0.05)
                        elif name_lower == 'pstid_i':
                            val = float(params.get('pstid', params.get('PSTID', 0.5)))
                            all_inputs[dev_idx, param_idx] = max(min(val, 0.95), 0.05)
                        elif name_lower == 'pgatd_i':
                            val = float(params.get('pgatd', params.get('PGATD', 0.5)))
                            all_inputs[dev_idx, param_idx] = max(min(val, 0.95), 0.05)
                        elif name_lower == 'phigbotd_i':
                            all_inputs[dev_idx, param_idx] = float(params.get('phigbotd', params.get('PHIGBOTD', 1.16)))
                        elif name_lower == 'phigstid_i':
                            all_inputs[dev_idx, param_idx] = float(params.get('phigstid', params.get('PHIGSTID', 1.16)))
                        elif name_lower == 'phiggatd_i':
                            all_inputs[dev_idx, param_idx] = float(params.get('phiggatd', params.get('PHIGGATD', 1.16)))

                        # ============================================================
                        # JUNCAP200 Computed Junction Params (temperature-dependent)
                        # Reference: JUNCAP200_InitModel.include lines 183-312
                        # ============================================================
                        # Pre-compute key temperature values for JUNCAP
                        # TRJ defaults to 21°C, but we use 27°C for consistency
                        elif name_lower in ('deltaphigr', 'deltaphigd'):
                            # Bandgap temperature corrections
                            # deltaphigr = -(7.02e-4 * tkr^2) / (1108 + tkr)
                            # deltaphigd = -(7.02e-4 * tkd^2) / (1108 + tkd)
                            tkr = 273.15 + float(params.get('trj', params.get('TRJ', 21.0)))
                            tkd = 300.15  # Device temp = 27°C
                            if name_lower == 'deltaphigr':
                                all_inputs[dev_idx, param_idx] = -(7.02e-4 * tkr * tkr) / (1108.0 + tkr)
                            else:
                                all_inputs[dev_idx, param_idx] = -(7.02e-4 * tkd * tkd) / (1108.0 + tkd)

                        elif name_lower in ('phigrbot', 'phigrsti', 'phigrgat'):
                            # Bandgap at reference temp: phigr = PHIG + deltaphigr
                            tkr = 273.15 + float(params.get('trj', params.get('TRJ', 21.0)))
                            deltaphigr = -(7.02e-4 * tkr * tkr) / (1108.0 + tkr)
                            if name_lower == 'phigrbot':
                                phig = float(params.get('phigbot', params.get('PHIGBOT', 1.16)))
                            elif name_lower == 'phigrsti':
                                phig = float(params.get('phigsti', params.get('PHIGSTI', 1.16)))
                            else:
                                phig = float(params.get('phiggat', params.get('PHIGGAT', 1.16)))
                            all_inputs[dev_idx, param_idx] = phig + deltaphigr

                        elif name_lower in ('phigdbot', 'phigdsti', 'phigdgat'):
                            # Bandgap at device temp: phigd = PHIG + deltaphigd
                            tkd = 300.15
                            deltaphigd = -(7.02e-4 * tkd * tkd) / (1108.0 + tkd)
                            if name_lower == 'phigdbot':
                                phig = float(params.get('phigbot', params.get('PHIGBOT', 1.16)))
                            elif name_lower == 'phigdsti':
                                phig = float(params.get('phigsti', params.get('PHIGSTI', 1.16)))
                            else:
                                phig = float(params.get('phiggat', params.get('PHIGGAT', 1.16)))
                            all_inputs[dev_idx, param_idx] = phig + deltaphigd

                        elif name_lower in ('ftdbot', 'ftdsti', 'ftdgat'):
                            # Temperature factor: ftd = auxt^1.5 * exp(0.5*(phigr/phitr - phigd/phitd))
                            import math
                            tkr = 273.15 + float(params.get('trj', params.get('TRJ', 21.0)))
                            tkd = 300.15
                            auxt = tkd / tkr
                            phitr = 8.617333262e-5 * tkr
                            phitd = 8.617333262e-5 * tkd
                            deltaphigr = -(7.02e-4 * tkr * tkr) / (1108.0 + tkr)
                            deltaphigd = -(7.02e-4 * tkd * tkd) / (1108.0 + tkd)
                            if name_lower == 'ftdbot':
                                phig = float(params.get('phigbot', params.get('PHIGBOT', 1.16)))
                            elif name_lower == 'ftdsti':
                                phig = float(params.get('phigsti', params.get('PHIGSTI', 1.16)))
                            else:
                                phig = float(params.get('phiggat', params.get('PHIGGAT', 1.16)))
                            phigr = phig + deltaphigr
                            phigd = phig + deltaphigd
                            ftd = (auxt ** 1.5) * math.exp(0.5 * (phigr / phitr - phigd / phitd))
                            all_inputs[dev_idx, param_idx] = ftd

                        elif name_lower in ('idsatbot', 'idsatsti', 'idsatgat'):
                            # Saturation current: idsat = IDSATR * ftd^2
                            import math
                            tkr = 273.15 + float(params.get('trj', params.get('TRJ', 21.0)))
                            tkd = 300.15
                            auxt = tkd / tkr
                            phitr = 8.617333262e-5 * tkr
                            phitd = 8.617333262e-5 * tkd
                            deltaphigr = -(7.02e-4 * tkr * tkr) / (1108.0 + tkr)
                            deltaphigd = -(7.02e-4 * tkd * tkd) / (1108.0 + tkd)
                            if name_lower == 'idsatbot':
                                phig = float(params.get('phigbot', params.get('PHIGBOT', 1.16)))
                                idsatr = float(params.get('idsatrbot', params.get('IDSATRBOT', 1e-12)))
                            elif name_lower == 'idsatsti':
                                phig = float(params.get('phigsti', params.get('PHIGSTI', 1.16)))
                                idsatr = float(params.get('idsatrsti', params.get('IDSATRSTI', 1e-18)))
                            else:
                                phig = float(params.get('phiggat', params.get('PHIGGAT', 1.16)))
                                idsatr = float(params.get('idsatrgat', params.get('IDSATRGAT', 1e-18)))
                            phigr = phig + deltaphigr
                            phigd = phig + deltaphigd
                            ftd = (auxt ** 1.5) * math.exp(0.5 * (phigr / phitr - phigd / phitd))
                            all_inputs[dev_idx, param_idx] = idsatr * ftd * ftd

                        elif name_lower in ('ubibot', 'ubisti', 'ubigat'):
                            # Built-in voltage before limiting: ubi = VBIR*auxt - 2*phitd*ln(ftd)
                            import math
                            tkr = 273.15 + float(params.get('trj', params.get('TRJ', 21.0)))
                            tkd = 300.15
                            auxt = tkd / tkr
                            phitr = 8.617333262e-5 * tkr
                            phitd = 8.617333262e-5 * tkd
                            deltaphigr = -(7.02e-4 * tkr * tkr) / (1108.0 + tkr)
                            deltaphigd = -(7.02e-4 * tkd * tkd) / (1108.0 + tkd)
                            if name_lower == 'ubibot':
                                phig = float(params.get('phigbot', params.get('PHIGBOT', 1.16)))
                                vbir = float(params.get('vbirbot', params.get('VBIRBOT', 1.0)))
                            elif name_lower == 'ubisti':
                                phig = float(params.get('phigsti', params.get('PHIGSTI', 1.16)))
                                vbir = float(params.get('vbirsti', params.get('VBIRSTI', 1.0)))
                            else:
                                phig = float(params.get('phiggat', params.get('PHIGGAT', 1.16)))
                                vbir = float(params.get('vbirgat', params.get('VBIRGAT', 1.0)))
                            phigr = phig + deltaphigr
                            phigd = phig + deltaphigd
                            ftd = (auxt ** 1.5) * math.exp(0.5 * (phigr / phitr - phigd / phitd))
                            ubi = vbir * auxt - 2.0 * phitd * math.log(ftd)
                            all_inputs[dev_idx, param_idx] = ubi

                        elif name_lower in ('vbibot', 'vbisti', 'vbigat'):
                            # Built-in voltage limited: vbi = ubi + phitd*ln(1+exp((vbilow-ubi)/phitd))
                            import math
                            tkr = 273.15 + float(params.get('trj', params.get('TRJ', 21.0)))
                            tkd = 300.15
                            auxt = tkd / tkr
                            phitr = 8.617333262e-5 * tkr
                            phitd = 8.617333262e-5 * tkd
                            vbilow = 0.05
                            deltaphigr = -(7.02e-4 * tkr * tkr) / (1108.0 + tkr)
                            deltaphigd = -(7.02e-4 * tkd * tkd) / (1108.0 + tkd)
                            if name_lower == 'vbibot':
                                phig = float(params.get('phigbot', params.get('PHIGBOT', 1.16)))
                                vbir = float(params.get('vbirbot', params.get('VBIRBOT', 1.0)))
                            elif name_lower == 'vbisti':
                                phig = float(params.get('phigsti', params.get('PHIGSTI', 1.16)))
                                vbir = float(params.get('vbirsti', params.get('VBIRSTI', 1.0)))
                            else:
                                phig = float(params.get('phiggat', params.get('PHIGGAT', 1.16)))
                                vbir = float(params.get('vbirgat', params.get('VBIRGAT', 1.0)))
                            phigr = phig + deltaphigr
                            phigd = phig + deltaphigd
                            ftd = (auxt ** 1.5) * math.exp(0.5 * (phigr / phitr - phigd / phitd))
                            ubi = vbir * auxt - 2.0 * phitd * math.log(ftd)
                            vbi = ubi + phitd * math.log(1.0 + math.exp((vbilow - ubi) / phitd))
                            all_inputs[dev_idx, param_idx] = vbi

                        elif name_lower in ('vbiinvbot', 'vbiinvsti', 'vbiinvgat'):
                            # Inverse built-in voltage: vbiinv = 1/vbi
                            import math
                            tkr = 273.15 + float(params.get('trj', params.get('TRJ', 21.0)))
                            tkd = 300.15
                            auxt = tkd / tkr
                            phitr = 8.617333262e-5 * tkr
                            phitd = 8.617333262e-5 * tkd
                            vbilow = 0.05
                            deltaphigr = -(7.02e-4 * tkr * tkr) / (1108.0 + tkr)
                            deltaphigd = -(7.02e-4 * tkd * tkd) / (1108.0 + tkd)
                            if name_lower == 'vbiinvbot':
                                phig = float(params.get('phigbot', params.get('PHIGBOT', 1.16)))
                                vbir = float(params.get('vbirbot', params.get('VBIRBOT', 1.0)))
                            elif name_lower == 'vbiinvsti':
                                phig = float(params.get('phigsti', params.get('PHIGSTI', 1.16)))
                                vbir = float(params.get('vbirsti', params.get('VBIRSTI', 1.0)))
                            else:
                                phig = float(params.get('phiggat', params.get('PHIGGAT', 1.16)))
                                vbir = float(params.get('vbirgat', params.get('VBIRGAT', 1.0)))
                            phigr = phig + deltaphigr
                            phigd = phig + deltaphigd
                            ftd = (auxt ** 1.5) * math.exp(0.5 * (phigr / phitr - phigd / phitd))
                            ubi = vbir * auxt - 2.0 * phitd * math.log(ftd)
                            vbi = ubi + phitd * math.log(1.0 + math.exp((vbilow - ubi) / phitd))
                            all_inputs[dev_idx, param_idx] = 1.0 / vbi

                        elif name_lower in ('one_minus_pbot', 'one_minus_psti', 'one_minus_pgat'):
                            # Grading: one_minus_P = 1 - P
                            if name_lower == 'one_minus_pbot':
                                p = max(min(float(params.get('pbot', params.get('PBOT', 0.5))), 0.95), 0.05)
                            elif name_lower == 'one_minus_psti':
                                p = max(min(float(params.get('psti', params.get('PSTI', 0.5))), 0.95), 0.05)
                            else:
                                p = max(min(float(params.get('pgat', params.get('PGAT', 0.5))), 0.95), 0.05)
                            all_inputs[dev_idx, param_idx] = 1.0 - p

                        elif name_lower in ('one_over_one_minus_pbot', 'one_over_one_minus_psti', 'one_over_one_minus_pgat'):
                            if name_lower == 'one_over_one_minus_pbot':
                                p = max(min(float(params.get('pbot', params.get('PBOT', 0.5))), 0.95), 0.05)
                            elif name_lower == 'one_over_one_minus_psti':
                                p = max(min(float(params.get('psti', params.get('PSTI', 0.5))), 0.95), 0.05)
                            else:
                                p = max(min(float(params.get('pgat', params.get('PGAT', 0.5))), 0.95), 0.05)
                            all_inputs[dev_idx, param_idx] = 1.0 / (1.0 - p)

                        elif name_lower in ('cjobot', 'cjosti', 'cjogat'):
                            # Temperature-scaled zero-bias capacitance
                            # cjo = CJOR * (VBIR/vbi)^P
                            import math
                            tkr = 273.15 + float(params.get('trj', params.get('TRJ', 21.0)))
                            tkd = 300.15
                            auxt = tkd / tkr
                            phitr = 8.617333262e-5 * tkr
                            phitd = 8.617333262e-5 * tkd
                            vbilow = 0.05
                            deltaphigr = -(7.02e-4 * tkr * tkr) / (1108.0 + tkr)
                            deltaphigd = -(7.02e-4 * tkd * tkd) / (1108.0 + tkd)
                            if name_lower == 'cjobot':
                                phig = float(params.get('phigbot', params.get('PHIGBOT', 1.16)))
                                vbir = max(float(params.get('vbirbot', params.get('VBIRBOT', 1.0))), 0.05)
                                cjor = max(float(params.get('cjorbot', params.get('CJORBOT', 1e-3))), 1e-12)
                                p = max(min(float(params.get('pbot', params.get('PBOT', 0.5))), 0.95), 0.05)
                            elif name_lower == 'cjosti':
                                phig = float(params.get('phigsti', params.get('PHIGSTI', 1.16)))
                                vbir = max(float(params.get('vbirsti', params.get('VBIRSTI', 1.0))), 0.05)
                                cjor = max(float(params.get('cjorsti', params.get('CJORSTI', 1e-9))), 1e-18)
                                p = max(min(float(params.get('psti', params.get('PSTI', 0.5))), 0.95), 0.05)
                            else:
                                phig = float(params.get('phiggat', params.get('PHIGGAT', 1.16)))
                                vbir = max(float(params.get('vbirgat', params.get('VBIRGAT', 1.0))), 0.05)
                                cjor = max(float(params.get('cjorgat', params.get('CJORGAT', 1e-9))), 1e-18)
                                p = max(min(float(params.get('pgat', params.get('PGAT', 0.5))), 0.95), 0.05)
                            phigr = phig + deltaphigr
                            phigd = phig + deltaphigd
                            ftd = (auxt ** 1.5) * math.exp(0.5 * (phigr / phitr - phigd / phitd))
                            ubi = vbir * auxt - 2.0 * phitd * math.log(ftd)
                            vbi = ubi + phitd * math.log(1.0 + math.exp((vbilow - ubi) / phitd))
                            cjo = cjor * ((vbir / vbi) ** p)
                            all_inputs[dev_idx, param_idx] = cjo

                        elif name_lower in ('qprefbot', 'qprefsti', 'qprefgat'):
                            # Charge prefactor: qpref = cjo * vbi / (1 - P)
                            import math
                            tkr = 273.15 + float(params.get('trj', params.get('TRJ', 21.0)))
                            tkd = 300.15
                            auxt = tkd / tkr
                            phitr = 8.617333262e-5 * tkr
                            phitd = 8.617333262e-5 * tkd
                            vbilow = 0.05
                            deltaphigr = -(7.02e-4 * tkr * tkr) / (1108.0 + tkr)
                            deltaphigd = -(7.02e-4 * tkd * tkd) / (1108.0 + tkd)
                            if name_lower == 'qprefbot':
                                phig = float(params.get('phigbot', params.get('PHIGBOT', 1.16)))
                                vbir = max(float(params.get('vbirbot', params.get('VBIRBOT', 1.0))), 0.05)
                                cjor = max(float(params.get('cjorbot', params.get('CJORBOT', 1e-3))), 1e-12)
                                p = max(min(float(params.get('pbot', params.get('PBOT', 0.5))), 0.95), 0.05)
                            elif name_lower == 'qprefsti':
                                phig = float(params.get('phigsti', params.get('PHIGSTI', 1.16)))
                                vbir = max(float(params.get('vbirsti', params.get('VBIRSTI', 1.0))), 0.05)
                                cjor = max(float(params.get('cjorsti', params.get('CJORSTI', 1e-9))), 1e-18)
                                p = max(min(float(params.get('psti', params.get('PSTI', 0.5))), 0.95), 0.05)
                            else:
                                phig = float(params.get('phiggat', params.get('PHIGGAT', 1.16)))
                                vbir = max(float(params.get('vbirgat', params.get('VBIRGAT', 1.0))), 0.05)
                                cjor = max(float(params.get('cjorgat', params.get('CJORGAT', 1e-9))), 1e-18)
                                p = max(min(float(params.get('pgat', params.get('PGAT', 0.5))), 0.95), 0.05)
                            phigr = phig + deltaphigr
                            phigd = phig + deltaphigd
                            ftd = (auxt ** 1.5) * math.exp(0.5 * (phigr / phitr - phigd / phitd))
                            ubi = vbir * auxt - 2.0 * phitd * math.log(ftd)
                            vbi = ubi + phitd * math.log(1.0 + math.exp((vbilow - ubi) / phitd))
                            cjo = cjor * ((vbir / vbi) ** p)
                            qpref = cjo * vbi / (1.0 - p)
                            all_inputs[dev_idx, param_idx] = qpref

                        elif name_lower in ('qpref2bot', 'qpref2sti', 'qpref2gat'):
                            # qpref2 = a * cjo, where a = 0.0025
                            import math
                            a_const = 0.0025
                            tkr = 273.15 + float(params.get('trj', params.get('TRJ', 21.0)))
                            tkd = 300.15
                            auxt = tkd / tkr
                            phitr = 8.617333262e-5 * tkr
                            phitd = 8.617333262e-5 * tkd
                            vbilow = 0.05
                            deltaphigr = -(7.02e-4 * tkr * tkr) / (1108.0 + tkr)
                            deltaphigd = -(7.02e-4 * tkd * tkd) / (1108.0 + tkd)
                            if name_lower == 'qpref2bot':
                                phig = float(params.get('phigbot', params.get('PHIGBOT', 1.16)))
                                vbir = max(float(params.get('vbirbot', params.get('VBIRBOT', 1.0))), 0.05)
                                cjor = max(float(params.get('cjorbot', params.get('CJORBOT', 1e-3))), 1e-12)
                                p = max(min(float(params.get('pbot', params.get('PBOT', 0.5))), 0.95), 0.05)
                            elif name_lower == 'qpref2sti':
                                phig = float(params.get('phigsti', params.get('PHIGSTI', 1.16)))
                                vbir = max(float(params.get('vbirsti', params.get('VBIRSTI', 1.0))), 0.05)
                                cjor = max(float(params.get('cjorsti', params.get('CJORSTI', 1e-9))), 1e-18)
                                p = max(min(float(params.get('psti', params.get('PSTI', 0.5))), 0.95), 0.05)
                            else:
                                phig = float(params.get('phiggat', params.get('PHIGGAT', 1.16)))
                                vbir = max(float(params.get('vbirgat', params.get('VBIRGAT', 1.0))), 0.05)
                                cjor = max(float(params.get('cjorgat', params.get('CJORGAT', 1e-9))), 1e-18)
                                p = max(min(float(params.get('pgat', params.get('PGAT', 0.5))), 0.95), 0.05)
                            phigr = phig + deltaphigr
                            phigd = phig + deltaphigd
                            ftd = (auxt ** 1.5) * math.exp(0.5 * (phigr / phitr - phigd / phitd))
                            ubi = vbir * auxt - 2.0 * phitd * math.log(ftd)
                            vbi = ubi + phitd * math.log(1.0 + math.exp((vbilow - ubi) / phitd))
                            cjo = cjor * ((vbir / vbi) ** p)
                            all_inputs[dev_idx, param_idx] = a_const * cjo

                        elif name_lower in ('wdepnulrbot', 'wdepnulrsti', 'wdepnulrgat'):
                            # Zero-bias depletion width: wdepnulr = EPSSI / CJOR (for bot)
                            # wdepnulrsti = XJUNSTI * EPSSI / CJORSTI
                            EPSSI = 1.035e-10  # Si permittivity = 11.7 * eps0
                            if name_lower == 'wdepnulrbot':
                                cjor = max(float(params.get('cjorbot', params.get('CJORBOT', 1e-3))), 1e-12)
                                all_inputs[dev_idx, param_idx] = EPSSI / cjor
                            elif name_lower == 'wdepnulrsti':
                                cjor = max(float(params.get('cjorsti', params.get('CJORSTI', 1e-9))), 1e-18)
                                xjun = max(float(params.get('xjunsti', params.get('XJUNSTI', 1e-7))), 1e-9)
                                all_inputs[dev_idx, param_idx] = xjun * EPSSI / cjor
                            else:
                                cjor = max(float(params.get('cjorgat', params.get('CJORGAT', 1e-9))), 1e-18)
                                xjun = max(float(params.get('xjungat', params.get('XJUNGAT', 1e-7))), 1e-9)
                                all_inputs[dev_idx, param_idx] = xjun * EPSSI / cjor

                        elif name_lower in ('wdepnulrinvbot', 'wdepnulrinvsti', 'wdepnulrinvgat'):
                            EPSSI = 1.035e-10
                            if name_lower == 'wdepnulrinvbot':
                                cjor = max(float(params.get('cjorbot', params.get('CJORBOT', 1e-3))), 1e-12)
                                wdep = EPSSI / cjor
                            elif name_lower == 'wdepnulrinvsti':
                                cjor = max(float(params.get('cjorsti', params.get('CJORSTI', 1e-9))), 1e-18)
                                xjun = max(float(params.get('xjunsti', params.get('XJUNSTI', 1e-7))), 1e-9)
                                wdep = xjun * EPSSI / cjor
                            else:
                                cjor = max(float(params.get('cjorgat', params.get('CJORGAT', 1e-9))), 1e-18)
                                xjun = max(float(params.get('xjungat', params.get('XJUNGAT', 1e-7))), 1e-9)
                                wdep = xjun * EPSSI / cjor
                            all_inputs[dev_idx, param_idx] = 1.0 / wdep

                        elif name_lower in ('vbirbotinv', 'vbirstiinv', 'vbirgatinv'):
                            if name_lower == 'vbirbotinv':
                                vbir = max(float(params.get('vbirbot', params.get('VBIRBOT', 1.0))), 0.05)
                            elif name_lower == 'vbirstiinv':
                                vbir = max(float(params.get('vbirsti', params.get('VBIRSTI', 1.0))), 0.05)
                            else:
                                vbir = max(float(params.get('vbirgat', params.get('VBIRGAT', 1.0))), 0.05)
                            all_inputs[dev_idx, param_idx] = 1.0 / vbir

                        elif name_lower == 'perfc':
                            # erfc approximation constant: perfc = sqrt(pi) * aerfc
                            import math
                            aerfc = 0.707106781186548  # 1/sqrt(2)
                            all_inputs[dev_idx, param_idx] = math.sqrt(math.pi) * aerfc

                        elif name_lower == 'berfc':
                            import math
                            aerfc = 0.707106781186548
                            perfc = math.sqrt(math.pi) * aerfc
                            all_inputs[dev_idx, param_idx] = (-5.0 * aerfc + 6.0 - perfc**(-2.0)) / 3.0

                        elif name_lower == 'cerfc':
                            import math
                            aerfc = 0.707106781186548
                            perfc = math.sqrt(math.pi) * aerfc
                            berfc = (-5.0 * aerfc + 6.0 - perfc**(-2.0)) / 3.0
                            all_inputs[dev_idx, param_idx] = 1.0 - aerfc - berfc

                        elif name_lower in ('deltaebot', 'deltaesti', 'deltaegat'):
                            # Half bandgap, limited to phitd: deltaE = max(0.5*phigd, phitd)
                            tkd = 300.15
                            phitd = 8.617333262e-5 * tkd
                            deltaphigd = -(7.02e-4 * tkd * tkd) / (1108.0 + tkd)
                            if name_lower == 'deltaebot':
                                phig = float(params.get('phigbot', params.get('PHIGBOT', 1.16)))
                            elif name_lower == 'deltaesti':
                                phig = float(params.get('phigsti', params.get('PHIGSTI', 1.16)))
                            else:
                                phig = float(params.get('phiggat', params.get('PHIGGAT', 1.16)))
                            phigd = phig + deltaphigd
                            all_inputs[dev_idx, param_idx] = max(0.5 * phigd, phitd)

                        elif name_lower == 'alphaav':
                            # Avalanche: alphaav = 1 - 1/FREV
                            frev = max(min(float(params.get('frev', params.get('FREV', 1000.0))), 1e40), 1.0)
                            all_inputs[dev_idx, param_idx] = 1.0 - 1.0 / frev

                        elif name_lower in ('vbrinvbot', 'vbrinvsti', 'vbrinvgat'):
                            if name_lower == 'vbrinvbot':
                                vbr = max(float(params.get('vbrbot', params.get('VBRBOT', 10.0))), 0.1)
                            elif name_lower == 'vbrinvsti':
                                vbr = max(float(params.get('vbrsti', params.get('VBRSTI', 10.0))), 0.1)
                            else:
                                vbr = max(float(params.get('vbrgat', params.get('VBRGAT', 10.0))), 0.1)
                            all_inputs[dev_idx, param_idx] = 1.0 / vbr

                        elif name_lower in ('fstopbot', 'fstopsti', 'fstopgat'):
                            # fstop = 1 / (1 - alphaav^PBR)
                            import math
                            frev = max(min(float(params.get('frev', params.get('FREV', 1000.0))), 1e40), 1.0)
                            alphaav = 1.0 - 1.0 / frev
                            if name_lower == 'fstopbot':
                                pbr = max(float(params.get('pbrbot', params.get('PBRBOT', 4.0))), 0.1)
                            elif name_lower == 'fstopsti':
                                pbr = max(float(params.get('pbrsti', params.get('PBRSTI', 4.0))), 0.1)
                            else:
                                pbr = max(float(params.get('pbrgat', params.get('PBRGAT', 4.0))), 0.1)
                            all_inputs[dev_idx, param_idx] = 1.0 / (1.0 - alphaav ** pbr)

                        elif name_lower in ('slopebot', 'slopesti', 'slopegat'):
                            # slope = -(fstop^2 * alphaav^(PBR-1)) * PBR * VBRinv
                            import math
                            frev = max(min(float(params.get('frev', params.get('FREV', 1000.0))), 1e40), 1.0)
                            alphaav = 1.0 - 1.0 / frev
                            if name_lower == 'slopebot':
                                pbr = max(float(params.get('pbrbot', params.get('PBRBOT', 4.0))), 0.1)
                                vbr = max(float(params.get('vbrbot', params.get('VBRBOT', 10.0))), 0.1)
                            elif name_lower == 'slopesti':
                                pbr = max(float(params.get('pbrsti', params.get('PBRSTI', 4.0))), 0.1)
                                vbr = max(float(params.get('vbrsti', params.get('VBRSTI', 10.0))), 0.1)
                            else:
                                pbr = max(float(params.get('pbrgat', params.get('PBRGAT', 4.0))), 0.1)
                                vbr = max(float(params.get('vbrgat', params.get('VBRGAT', 10.0))), 0.1)
                            fstop = 1.0 / (1.0 - alphaav ** pbr)
                            all_inputs[dev_idx, param_idx] = -(fstop * fstop * (alphaav ** (pbr - 1.0))) * pbr / vbr

                        elif name_lower in ('atatbot', 'atatsti', 'atatgat'):
                            # atat = deltaE / phitd
                            tkd = 300.15
                            phitd = 8.617333262e-5 * tkd
                            deltaphigd = -(7.02e-4 * tkd * tkd) / (1108.0 + tkd)
                            if name_lower == 'atatbot':
                                phig = float(params.get('phigbot', params.get('PHIGBOT', 1.16)))
                            elif name_lower == 'atatsti':
                                phig = float(params.get('phigsti', params.get('PHIGSTI', 1.16)))
                            else:
                                phig = float(params.get('phiggat', params.get('PHIGGAT', 1.16)))
                            phigd = phig + deltaphigd
                            deltaE = max(0.5 * phigd, phitd)
                            all_inputs[dev_idx, param_idx] = deltaE / phitd

                        elif name_lower in ('btatpartbot', 'btatpartsti', 'btatpartgat'):
                            # btatpart = sqrt(32*MEFFTAT*MELE*QELE*deltaE^3) / (3*HBAR)
                            import math
                            MELE = 9.109e-31
                            QELE = 1.602e-19
                            HBAR = 1.0546e-34
                            tkd = 300.15
                            phitd = 8.617333262e-5 * tkd
                            deltaphigd = -(7.02e-4 * tkd * tkd) / (1108.0 + tkd)
                            if name_lower == 'btatpartbot':
                                phig = float(params.get('phigbot', params.get('PHIGBOT', 1.16)))
                                meff = max(float(params.get('mefftatbot', params.get('MEFFTATBOT', 0.25))), 0.001)
                            elif name_lower == 'btatpartsti':
                                phig = float(params.get('phigsti', params.get('PHIGSTI', 1.16)))
                                meff = max(float(params.get('mefftatsti', params.get('MEFFTATSTI', 0.25))), 0.001)
                            else:
                                phig = float(params.get('phiggat', params.get('PHIGGAT', 1.16)))
                                meff = max(float(params.get('mefftatgat', params.get('MEFFTATGAT', 0.25))), 0.001)
                            phigd = phig + deltaphigd
                            deltaE = max(0.5 * phigd, phitd)
                            all_inputs[dev_idx, param_idx] = math.sqrt(32.0 * meff * MELE * QELE * deltaE**3) / (3.0 * HBAR)

                        elif name_lower in ('fbbtbot', 'fbbtsti', 'fbbtgat'):
                            # fbbt = FBBTR * (1 + STFBBT*(tkd-tkr)), clipped to >= 0
                            tkr = 273.15 + float(params.get('trj', params.get('TRJ', 21.0)))
                            tkd = 300.15
                            if name_lower == 'fbbtbot':
                                fbbtr = float(params.get('fbbtrbot', params.get('FBBTRBOT', 1e9)))
                                stfbbt = float(params.get('stfbbtbot', params.get('STFBBTBOT', -1e-3)))
                            elif name_lower == 'fbbtsti':
                                fbbtr = float(params.get('fbbtrsti', params.get('FBBTRSTI', 1e9)))
                                stfbbt = float(params.get('stfbbtsti', params.get('STFBBTSTI', -1e-3)))
                            else:
                                fbbtr = float(params.get('fbbtrgat', params.get('FBBTRGAT', 1e9)))
                                stfbbt = float(params.get('stfbbtgat', params.get('STFBBTGAT', -1e-3)))
                            fbbt = fbbtr * (1.0 + stfbbt * (tkd - tkr))
                            all_inputs[dev_idx, param_idx] = max(fbbt, 0.0)

                        # ============================================================
                        # Drain-side junction params (_d suffix) - same formulas
                        # ============================================================
                        elif name_lower in ('phigrbot_d', 'phigrsti_d', 'phigrgat_d'):
                            tkr = 273.15 + float(params.get('trj', params.get('TRJ', 21.0)))
                            deltaphigr = -(7.02e-4 * tkr * tkr) / (1108.0 + tkr)
                            if name_lower == 'phigrbot_d':
                                phig = float(params.get('phigbotd', params.get('PHIGBOTD', 1.16)))
                            elif name_lower == 'phigrsti_d':
                                phig = float(params.get('phigstid', params.get('PHIGSTID', 1.16)))
                            else:
                                phig = float(params.get('phiggatd', params.get('PHIGGATD', 1.16)))
                            all_inputs[dev_idx, param_idx] = phig + deltaphigr

                        elif name_lower in ('phigdbot_d', 'phigdsti_d', 'phigdgat_d'):
                            tkd = 300.15
                            deltaphigd = -(7.02e-4 * tkd * tkd) / (1108.0 + tkd)
                            if name_lower == 'phigdbot_d':
                                phig = float(params.get('phigbotd', params.get('PHIGBOTD', 1.16)))
                            elif name_lower == 'phigdsti_d':
                                phig = float(params.get('phigstid', params.get('PHIGSTID', 1.16)))
                            else:
                                phig = float(params.get('phiggatd', params.get('PHIGGATD', 1.16)))
                            all_inputs[dev_idx, param_idx] = phig + deltaphigd

                        elif name_lower in ('ftdbot_d', 'ftdsti_d', 'ftdgat_d'):
                            import math
                            tkr = 273.15 + float(params.get('trj', params.get('TRJ', 21.0)))
                            tkd = 300.15
                            auxt = tkd / tkr
                            phitr = 8.617333262e-5 * tkr
                            phitd = 8.617333262e-5 * tkd
                            deltaphigr = -(7.02e-4 * tkr * tkr) / (1108.0 + tkr)
                            deltaphigd = -(7.02e-4 * tkd * tkd) / (1108.0 + tkd)
                            if name_lower == 'ftdbot_d':
                                phig = float(params.get('phigbotd', params.get('PHIGBOTD', 1.16)))
                            elif name_lower == 'ftdsti_d':
                                phig = float(params.get('phigstid', params.get('PHIGSTID', 1.16)))
                            else:
                                phig = float(params.get('phiggatd', params.get('PHIGGATD', 1.16)))
                            phigr = phig + deltaphigr
                            phigd = phig + deltaphigd
                            ftd = (auxt ** 1.5) * math.exp(0.5 * (phigr / phitr - phigd / phitd))
                            all_inputs[dev_idx, param_idx] = ftd

                        elif name_lower in ('idsatbot_d', 'idsatsti_d', 'idsatgat_d'):
                            import math
                            tkr = 273.15 + float(params.get('trj', params.get('TRJ', 21.0)))
                            tkd = 300.15
                            auxt = tkd / tkr
                            phitr = 8.617333262e-5 * tkr
                            phitd = 8.617333262e-5 * tkd
                            deltaphigr = -(7.02e-4 * tkr * tkr) / (1108.0 + tkr)
                            deltaphigd = -(7.02e-4 * tkd * tkd) / (1108.0 + tkd)
                            if name_lower == 'idsatbot_d':
                                phig = float(params.get('phigbotd', params.get('PHIGBOTD', 1.16)))
                                idsatr = float(params.get('idsatrbotd', params.get('IDSATRBOTD', 1e-12)))
                            elif name_lower == 'idsatsti_d':
                                phig = float(params.get('phigstid', params.get('PHIGSTID', 1.16)))
                                idsatr = float(params.get('idsatrstid', params.get('IDSATRSTID', 1e-18)))
                            else:
                                phig = float(params.get('phiggatd', params.get('PHIGGATD', 1.16)))
                                idsatr = float(params.get('idsatrgatd', params.get('IDSATRGATD', 1e-18)))
                            phigr = phig + deltaphigr
                            phigd = phig + deltaphigd
                            ftd = (auxt ** 1.5) * math.exp(0.5 * (phigr / phitr - phigd / phitd))
                            all_inputs[dev_idx, param_idx] = idsatr * ftd * ftd

                        elif name_lower in ('ubibot_d', 'ubisti_d', 'ubigat_d'):
                            import math
                            tkr = 273.15 + float(params.get('trj', params.get('TRJ', 21.0)))
                            tkd = 300.15
                            auxt = tkd / tkr
                            phitr = 8.617333262e-5 * tkr
                            phitd = 8.617333262e-5 * tkd
                            deltaphigr = -(7.02e-4 * tkr * tkr) / (1108.0 + tkr)
                            deltaphigd = -(7.02e-4 * tkd * tkd) / (1108.0 + tkd)
                            if name_lower == 'ubibot_d':
                                phig = float(params.get('phigbotd', params.get('PHIGBOTD', 1.16)))
                                vbir = float(params.get('vbirbotd', params.get('VBIRBOTD', 1.0)))
                            elif name_lower == 'ubisti_d':
                                phig = float(params.get('phigstid', params.get('PHIGSTID', 1.16)))
                                vbir = float(params.get('vbirstid', params.get('VBIRSTID', 1.0)))
                            else:
                                phig = float(params.get('phiggatd', params.get('PHIGGATD', 1.16)))
                                vbir = float(params.get('vbirgatd', params.get('VBIRGATD', 1.0)))
                            phigr = phig + deltaphigr
                            phigd = phig + deltaphigd
                            ftd = (auxt ** 1.5) * math.exp(0.5 * (phigr / phitr - phigd / phitd))
                            ubi = vbir * auxt - 2.0 * phitd * math.log(ftd)
                            all_inputs[dev_idx, param_idx] = ubi

                        elif name_lower in ('vbibot_d', 'vbisti_d', 'vbigat_d'):
                            import math
                            tkr = 273.15 + float(params.get('trj', params.get('TRJ', 21.0)))
                            tkd = 300.15
                            auxt = tkd / tkr
                            phitr = 8.617333262e-5 * tkr
                            phitd = 8.617333262e-5 * tkd
                            vbilow = 0.05
                            deltaphigr = -(7.02e-4 * tkr * tkr) / (1108.0 + tkr)
                            deltaphigd = -(7.02e-4 * tkd * tkd) / (1108.0 + tkd)
                            if name_lower == 'vbibot_d':
                                phig = float(params.get('phigbotd', params.get('PHIGBOTD', 1.16)))
                                vbir = float(params.get('vbirbotd', params.get('VBIRBOTD', 1.0)))
                            elif name_lower == 'vbisti_d':
                                phig = float(params.get('phigstid', params.get('PHIGSTID', 1.16)))
                                vbir = float(params.get('vbirstid', params.get('VBIRSTID', 1.0)))
                            else:
                                phig = float(params.get('phiggatd', params.get('PHIGGATD', 1.16)))
                                vbir = float(params.get('vbirgatd', params.get('VBIRGATD', 1.0)))
                            phigr = phig + deltaphigr
                            phigd = phig + deltaphigd
                            ftd = (auxt ** 1.5) * math.exp(0.5 * (phigr / phitr - phigd / phitd))
                            ubi = vbir * auxt - 2.0 * phitd * math.log(ftd)
                            vbi = ubi + phitd * math.log(1.0 + math.exp((vbilow - ubi) / phitd))
                            all_inputs[dev_idx, param_idx] = vbi

                        elif name_lower in ('vbiinvbot_d', 'vbiinvsti_d', 'vbiinvgat_d'):
                            import math
                            tkr = 273.15 + float(params.get('trj', params.get('TRJ', 21.0)))
                            tkd = 300.15
                            auxt = tkd / tkr
                            phitr = 8.617333262e-5 * tkr
                            phitd = 8.617333262e-5 * tkd
                            vbilow = 0.05
                            deltaphigr = -(7.02e-4 * tkr * tkr) / (1108.0 + tkr)
                            deltaphigd = -(7.02e-4 * tkd * tkd) / (1108.0 + tkd)
                            if name_lower == 'vbiinvbot_d':
                                phig = float(params.get('phigbotd', params.get('PHIGBOTD', 1.16)))
                                vbir = float(params.get('vbirbotd', params.get('VBIRBOTD', 1.0)))
                            elif name_lower == 'vbiinvsti_d':
                                phig = float(params.get('phigstid', params.get('PHIGSTID', 1.16)))
                                vbir = float(params.get('vbirstid', params.get('VBIRSTID', 1.0)))
                            else:
                                phig = float(params.get('phiggatd', params.get('PHIGGATD', 1.16)))
                                vbir = float(params.get('vbirgatd', params.get('VBIRGATD', 1.0)))
                            phigr = phig + deltaphigr
                            phigd = phig + deltaphigd
                            ftd = (auxt ** 1.5) * math.exp(0.5 * (phigr / phitr - phigd / phitd))
                            ubi = vbir * auxt - 2.0 * phitd * math.log(ftd)
                            vbi = ubi + phitd * math.log(1.0 + math.exp((vbilow - ubi) / phitd))
                            all_inputs[dev_idx, param_idx] = 1.0 / vbi

                        elif name_lower in ('one_minus_pbot_d', 'one_minus_psti_d', 'one_minus_pgat_d'):
                            if name_lower == 'one_minus_pbot_d':
                                p = max(min(float(params.get('pbotd', params.get('PBOTD', 0.5))), 0.95), 0.05)
                            elif name_lower == 'one_minus_psti_d':
                                p = max(min(float(params.get('pstid', params.get('PSTID', 0.5))), 0.95), 0.05)
                            else:
                                p = max(min(float(params.get('pgatd', params.get('PGATD', 0.5))), 0.95), 0.05)
                            all_inputs[dev_idx, param_idx] = 1.0 - p

                        elif name_lower in ('one_over_one_minus_pbot_d', 'one_over_one_minus_psti_d', 'one_over_one_minus_pgat_d'):
                            if name_lower == 'one_over_one_minus_pbot_d':
                                p = max(min(float(params.get('pbotd', params.get('PBOTD', 0.5))), 0.95), 0.05)
                            elif name_lower == 'one_over_one_minus_psti_d':
                                p = max(min(float(params.get('pstid', params.get('PSTID', 0.5))), 0.95), 0.05)
                            else:
                                p = max(min(float(params.get('pgatd', params.get('PGATD', 0.5))), 0.95), 0.05)
                            all_inputs[dev_idx, param_idx] = 1.0 / (1.0 - p)

                        elif name_lower in ('cjobot_d', 'cjosti_d', 'cjogat_d'):
                            import math
                            tkr = 273.15 + float(params.get('trj', params.get('TRJ', 21.0)))
                            tkd = 300.15
                            auxt = tkd / tkr
                            phitr = 8.617333262e-5 * tkr
                            phitd = 8.617333262e-5 * tkd
                            vbilow = 0.05
                            deltaphigr = -(7.02e-4 * tkr * tkr) / (1108.0 + tkr)
                            deltaphigd = -(7.02e-4 * tkd * tkd) / (1108.0 + tkd)
                            if name_lower == 'cjobot_d':
                                phig = float(params.get('phigbotd', params.get('PHIGBOTD', 1.16)))
                                vbir = max(float(params.get('vbirbotd', params.get('VBIRBOTD', 1.0))), 0.05)
                                cjor = max(float(params.get('cjorbotd', params.get('CJORBOTD', 1e-3))), 1e-12)
                                p = max(min(float(params.get('pbotd', params.get('PBOTD', 0.5))), 0.95), 0.05)
                            elif name_lower == 'cjosti_d':
                                phig = float(params.get('phigstid', params.get('PHIGSTID', 1.16)))
                                vbir = max(float(params.get('vbirstid', params.get('VBIRSTID', 1.0))), 0.05)
                                cjor = max(float(params.get('cjorstid', params.get('CJORSTID', 1e-9))), 1e-18)
                                p = max(min(float(params.get('pstid', params.get('PSTID', 0.5))), 0.95), 0.05)
                            else:
                                phig = float(params.get('phiggatd', params.get('PHIGGATD', 1.16)))
                                vbir = max(float(params.get('vbirgatd', params.get('VBIRGATD', 1.0))), 0.05)
                                cjor = max(float(params.get('cjorgatd', params.get('CJORGATD', 1e-9))), 1e-18)
                                p = max(min(float(params.get('pgatd', params.get('PGATD', 0.5))), 0.95), 0.05)
                            phigr = phig + deltaphigr
                            phigd = phig + deltaphigd
                            ftd = (auxt ** 1.5) * math.exp(0.5 * (phigr / phitr - phigd / phitd))
                            ubi = vbir * auxt - 2.0 * phitd * math.log(ftd)
                            vbi = ubi + phitd * math.log(1.0 + math.exp((vbilow - ubi) / phitd))
                            cjo = cjor * ((vbir / vbi) ** p)
                            all_inputs[dev_idx, param_idx] = cjo

                        elif name_lower in ('qprefbot_d', 'qprefsti_d', 'qprefgat_d'):
                            import math
                            tkr = 273.15 + float(params.get('trj', params.get('TRJ', 21.0)))
                            tkd = 300.15
                            auxt = tkd / tkr
                            phitr = 8.617333262e-5 * tkr
                            phitd = 8.617333262e-5 * tkd
                            vbilow = 0.05
                            deltaphigr = -(7.02e-4 * tkr * tkr) / (1108.0 + tkr)
                            deltaphigd = -(7.02e-4 * tkd * tkd) / (1108.0 + tkd)
                            if name_lower == 'qprefbot_d':
                                phig = float(params.get('phigbotd', params.get('PHIGBOTD', 1.16)))
                                vbir = max(float(params.get('vbirbotd', params.get('VBIRBOTD', 1.0))), 0.05)
                                cjor = max(float(params.get('cjorbotd', params.get('CJORBOTD', 1e-3))), 1e-12)
                                p = max(min(float(params.get('pbotd', params.get('PBOTD', 0.5))), 0.95), 0.05)
                            elif name_lower == 'qprefsti_d':
                                phig = float(params.get('phigstid', params.get('PHIGSTID', 1.16)))
                                vbir = max(float(params.get('vbirstid', params.get('VBIRSTID', 1.0))), 0.05)
                                cjor = max(float(params.get('cjorstid', params.get('CJORSTID', 1e-9))), 1e-18)
                                p = max(min(float(params.get('pstid', params.get('PSTID', 0.5))), 0.95), 0.05)
                            else:
                                phig = float(params.get('phiggatd', params.get('PHIGGATD', 1.16)))
                                vbir = max(float(params.get('vbirgatd', params.get('VBIRGATD', 1.0))), 0.05)
                                cjor = max(float(params.get('cjorgatd', params.get('CJORGATD', 1e-9))), 1e-18)
                                p = max(min(float(params.get('pgatd', params.get('PGATD', 0.5))), 0.95), 0.05)
                            phigr = phig + deltaphigr
                            phigd = phig + deltaphigd
                            ftd = (auxt ** 1.5) * math.exp(0.5 * (phigr / phitr - phigd / phitd))
                            ubi = vbir * auxt - 2.0 * phitd * math.log(ftd)
                            vbi = ubi + phitd * math.log(1.0 + math.exp((vbilow - ubi) / phitd))
                            cjo = cjor * ((vbir / vbi) ** p)
                            qpref = cjo * vbi / (1.0 - p)
                            all_inputs[dev_idx, param_idx] = qpref

                        elif name_lower in ('qpref2bot_d', 'qpref2sti_d', 'qpref2gat_d'):
                            import math
                            a_const = 0.0025
                            tkr = 273.15 + float(params.get('trj', params.get('TRJ', 21.0)))
                            tkd = 300.15
                            auxt = tkd / tkr
                            phitr = 8.617333262e-5 * tkr
                            phitd = 8.617333262e-5 * tkd
                            vbilow = 0.05
                            deltaphigr = -(7.02e-4 * tkr * tkr) / (1108.0 + tkr)
                            deltaphigd = -(7.02e-4 * tkd * tkd) / (1108.0 + tkd)
                            if name_lower == 'qpref2bot_d':
                                phig = float(params.get('phigbotd', params.get('PHIGBOTD', 1.16)))
                                vbir = max(float(params.get('vbirbotd', params.get('VBIRBOTD', 1.0))), 0.05)
                                cjor = max(float(params.get('cjorbotd', params.get('CJORBOTD', 1e-3))), 1e-12)
                                p = max(min(float(params.get('pbotd', params.get('PBOTD', 0.5))), 0.95), 0.05)
                            elif name_lower == 'qpref2sti_d':
                                phig = float(params.get('phigstid', params.get('PHIGSTID', 1.16)))
                                vbir = max(float(params.get('vbirstid', params.get('VBIRSTID', 1.0))), 0.05)
                                cjor = max(float(params.get('cjorstid', params.get('CJORSTID', 1e-9))), 1e-18)
                                p = max(min(float(params.get('pstid', params.get('PSTID', 0.5))), 0.95), 0.05)
                            else:
                                phig = float(params.get('phiggatd', params.get('PHIGGATD', 1.16)))
                                vbir = max(float(params.get('vbirgatd', params.get('VBIRGATD', 1.0))), 0.05)
                                cjor = max(float(params.get('cjorgatd', params.get('CJORGATD', 1e-9))), 1e-18)
                                p = max(min(float(params.get('pgatd', params.get('PGATD', 0.5))), 0.95), 0.05)
                            phigr = phig + deltaphigr
                            phigd = phig + deltaphigd
                            ftd = (auxt ** 1.5) * math.exp(0.5 * (phigr / phitr - phigd / phitd))
                            ubi = vbir * auxt - 2.0 * phitd * math.log(ftd)
                            vbi = ubi + phitd * math.log(1.0 + math.exp((vbilow - ubi) / phitd))
                            cjo = cjor * ((vbir / vbi) ** p)
                            all_inputs[dev_idx, param_idx] = a_const * cjo

                        elif name_lower in ('wdepnulrbot_d', 'wdepnulrsti_d', 'wdepnulrgat_d'):
                            EPSSI = 1.035e-10
                            if name_lower == 'wdepnulrbot_d':
                                cjor = max(float(params.get('cjorbotd', params.get('CJORBOTD', 1e-3))), 1e-12)
                                all_inputs[dev_idx, param_idx] = EPSSI / cjor
                            elif name_lower == 'wdepnulrsti_d':
                                cjor = max(float(params.get('cjorstid', params.get('CJORSTID', 1e-9))), 1e-18)
                                xjun = max(float(params.get('xjunstid', params.get('XJUNSTID', 1e-7))), 1e-9)
                                all_inputs[dev_idx, param_idx] = xjun * EPSSI / cjor
                            else:
                                cjor = max(float(params.get('cjorgatd', params.get('CJORGATD', 1e-9))), 1e-18)
                                xjun = max(float(params.get('xjungatd', params.get('XJUNGATD', 1e-7))), 1e-9)
                                all_inputs[dev_idx, param_idx] = xjun * EPSSI / cjor

                        elif name_lower in ('wdepnulrinvbot_d', 'wdepnulrinvsti_d', 'wdepnulrinvgat_d'):
                            EPSSI = 1.035e-10
                            if name_lower == 'wdepnulrinvbot_d':
                                cjor = max(float(params.get('cjorbotd', params.get('CJORBOTD', 1e-3))), 1e-12)
                                wdep = EPSSI / cjor
                            elif name_lower == 'wdepnulrinvsti_d':
                                cjor = max(float(params.get('cjorstid', params.get('CJORSTID', 1e-9))), 1e-18)
                                xjun = max(float(params.get('xjunstid', params.get('XJUNSTID', 1e-7))), 1e-9)
                                wdep = xjun * EPSSI / cjor
                            else:
                                cjor = max(float(params.get('cjorgatd', params.get('CJORGATD', 1e-9))), 1e-18)
                                xjun = max(float(params.get('xjungatd', params.get('XJUNGATD', 1e-7))), 1e-9)
                                wdep = xjun * EPSSI / cjor
                            all_inputs[dev_idx, param_idx] = 1.0 / wdep

                        elif name_lower in ('vbirbotinv_d', 'vbirstiinv_d', 'vbirgatinv_d'):
                            if name_lower == 'vbirbotinv_d':
                                vbir = max(float(params.get('vbirbotd', params.get('VBIRBOTD', 1.0))), 0.05)
                            elif name_lower == 'vbirstiinv_d':
                                vbir = max(float(params.get('vbirstid', params.get('VBIRSTID', 1.0))), 0.05)
                            else:
                                vbir = max(float(params.get('vbirgatd', params.get('VBIRGATD', 1.0))), 0.05)
                            all_inputs[dev_idx, param_idx] = 1.0 / vbir

                        elif name_lower in ('deltaebot_d', 'deltaesti_d', 'deltaegat_d'):
                            tkd = 300.15
                            phitd = 8.617333262e-5 * tkd
                            deltaphigd = -(7.02e-4 * tkd * tkd) / (1108.0 + tkd)
                            if name_lower == 'deltaebot_d':
                                phig = float(params.get('phigbotd', params.get('PHIGBOTD', 1.16)))
                            elif name_lower == 'deltaesti_d':
                                phig = float(params.get('phigstid', params.get('PHIGSTID', 1.16)))
                            else:
                                phig = float(params.get('phiggatd', params.get('PHIGGATD', 1.16)))
                            phigd = phig + deltaphigd
                            all_inputs[dev_idx, param_idx] = max(0.5 * phigd, phitd)

                        elif name_lower in ('atatbot_d', 'atatsti_d', 'atatgat_d'):
                            tkd = 300.15
                            phitd = 8.617333262e-5 * tkd
                            deltaphigd = -(7.02e-4 * tkd * tkd) / (1108.0 + tkd)
                            if name_lower == 'atatbot_d':
                                phig = float(params.get('phigbotd', params.get('PHIGBOTD', 1.16)))
                            elif name_lower == 'atatsti_d':
                                phig = float(params.get('phigstid', params.get('PHIGSTID', 1.16)))
                            else:
                                phig = float(params.get('phiggatd', params.get('PHIGGATD', 1.16)))
                            phigd = phig + deltaphigd
                            deltaE = max(0.5 * phigd, phitd)
                            all_inputs[dev_idx, param_idx] = deltaE / phitd

                        elif name_lower in ('btatpartbot_d', 'btatpartsti_d', 'btatpartgat_d'):
                            import math
                            MELE = 9.109e-31
                            QELE = 1.602e-19
                            HBAR = 1.0546e-34
                            tkd = 300.15
                            phitd = 8.617333262e-5 * tkd
                            deltaphigd = -(7.02e-4 * tkd * tkd) / (1108.0 + tkd)
                            if name_lower == 'btatpartbot_d':
                                phig = float(params.get('phigbotd', params.get('PHIGBOTD', 1.16)))
                                meff = max(float(params.get('mefftatbotd', params.get('MEFFTATBOTD', 0.25))), 0.001)
                            elif name_lower == 'btatpartsti_d':
                                phig = float(params.get('phigstid', params.get('PHIGSTID', 1.16)))
                                meff = max(float(params.get('mefftatstid', params.get('MEFFTATSTID', 0.25))), 0.001)
                            else:
                                phig = float(params.get('phiggatd', params.get('PHIGGATD', 1.16)))
                                meff = max(float(params.get('mefftatgatd', params.get('MEFFTATGATD', 0.25))), 0.001)
                            phigd = phig + deltaphigd
                            deltaE = max(0.5 * phigd, phitd)
                            all_inputs[dev_idx, param_idx] = math.sqrt(32.0 * meff * MELE * QELE * deltaE**3) / (3.0 * HBAR)

                        elif name_lower in ('fbbtbot_d', 'fbbtsti_d', 'fbbtgat_d'):
                            tkr = 273.15 + float(params.get('trj', params.get('TRJ', 21.0)))
                            tkd = 300.15
                            if name_lower == 'fbbtbot_d':
                                fbbtr = float(params.get('fbbtrbotd', params.get('FBBTRBOTD', 1e9)))
                                stfbbt = float(params.get('stfbbtbotd', params.get('STFBBTBOTD', -1e-3)))
                            elif name_lower == 'fbbtsti_d':
                                fbbtr = float(params.get('fbbtrstid', params.get('FBBTRSTID', 1e9)))
                                stfbbt = float(params.get('stfbbtstid', params.get('STFBBTSTID', -1e-3)))
                            else:
                                fbbtr = float(params.get('fbbtrgatd', params.get('FBBTRGATD', 1e9)))
                                stfbbt = float(params.get('stfbbtgatd', params.get('STFBBTGATD', -1e-3)))
                            fbbt = fbbtr * (1.0 + stfbbt * (tkd - tkr))
                            all_inputs[dev_idx, param_idx] = max(fbbt, 0.0)

                        elif name_lower in ('fstopbot_d', 'fstopsti_d', 'fstopgat_d'):
                            import math
                            frev = max(min(float(params.get('frev', params.get('FREV', 1000.0))), 1e40), 1.0)
                            alphaav = 1.0 - 1.0 / frev
                            if name_lower == 'fstopbot_d':
                                pbr = max(float(params.get('pbrbotd', params.get('PBRBOTD', 4.0))), 0.1)
                            elif name_lower == 'fstopsti_d':
                                pbr = max(float(params.get('pbrstid', params.get('PBRSTID', 4.0))), 0.1)
                            else:
                                pbr = max(float(params.get('pbrgatd', params.get('PBRGATD', 4.0))), 0.1)
                            all_inputs[dev_idx, param_idx] = 1.0 / (1.0 - alphaav ** pbr)

                        elif name_lower in ('slopebot_d', 'slopesti_d', 'slopegat_d'):
                            import math
                            frev = max(min(float(params.get('frev', params.get('FREV', 1000.0))), 1e40), 1.0)
                            alphaav = 1.0 - 1.0 / frev
                            if name_lower == 'slopebot_d':
                                pbr = max(float(params.get('pbrbotd', params.get('PBRBOTD', 4.0))), 0.1)
                                vbr = max(float(params.get('vbrbotd', params.get('VBRBOTD', 10.0))), 0.1)
                            elif name_lower == 'slopesti_d':
                                pbr = max(float(params.get('pbrstid', params.get('PBRSTID', 4.0))), 0.1)
                                vbr = max(float(params.get('vbrstid', params.get('VBRSTID', 10.0))), 0.1)
                            else:
                                pbr = max(float(params.get('pbrgatd', params.get('PBRGATD', 4.0))), 0.1)
                                vbr = max(float(params.get('vbrgatd', params.get('VBRGATD', 10.0))), 0.1)
                            fstop = 1.0 / (1.0 - alphaav ** pbr)
                            all_inputs[dev_idx, param_idx] = -(fstop * fstop * (alphaav ** (pbr - 1.0))) * pbr / vbr

                        elif name_lower in ('vbrinvbot_d', 'vbrinvsti_d', 'vbrinvgat_d'):
                            if name_lower == 'vbrinvbot_d':
                                vbr = max(float(params.get('vbrbotd', params.get('VBRBOTD', 10.0))), 0.1)
                            elif name_lower == 'vbrinvsti_d':
                                vbr = max(float(params.get('vbrstid', params.get('VBRSTID', 10.0))), 0.1)
                            else:
                                vbr = max(float(params.get('vbrgatd', params.get('VBRGATD', 10.0))), 0.1)
                            all_inputs[dev_idx, param_idx] = 1.0 / vbr

                        # ============================================================
                        # PSP103 Geometry/Stress Hidden State Params
                        # ============================================================
                        elif name_lower == 'invnf':
                            # invNF = 1.0 / NF (inverse of finger count)
                            nf = float(params.get('nf', params.get('NF', 1.0)))
                            all_inputs[dev_idx, param_idx] = 1.0 / max(nf, 1.0)
                        elif name_lower == 'le':
                            # LE = effective length (usually = L * NF)
                            L = float(params.get('l', params.get('L', 1e-6)))
                            nf = float(params.get('nf', params.get('NF', 1.0)))
                            all_inputs[dev_idx, param_idx] = L * max(nf, 1.0)
                        elif name_lower == 'we':
                            # WE = effective width
                            W = float(params.get('w', params.get('W', 1e-6)))
                            all_inputs[dev_idx, param_idx] = W
                        elif name_lower == 'sa_i':
                            all_inputs[dev_idx, param_idx] = max(float(params.get('sa', params.get('SA', 0.0))), 0.0)
                        elif name_lower == 'sb_i':
                            all_inputs[dev_idx, param_idx] = max(float(params.get('sb', params.get('SB', 0.0))), 0.0)

                        # ============================================================
                        # Temperature in Kelvin
                        # ============================================================
                        elif name_lower == 'tk':
                            # Device temperature in Kelvin (default 300.15K = 27°C)
                            all_inputs[dev_idx, param_idx] = 300.15

                        # ============================================================
                        # Junction Geometry Params
                        # ============================================================
                        elif name_lower == 'absource_i':
                            all_inputs[dev_idx, param_idx] = max(float(params.get('absource', params.get('ABSOURCE', 1.0e-12))), 1e-18)
                        elif name_lower == 'lssource_i':
                            all_inputs[dev_idx, param_idx] = max(float(params.get('lssource', params.get('LSSOURCE', 1.0e-6))), 1e-12)
                        elif name_lower == 'lgsource_i':
                            all_inputs[dev_idx, param_idx] = max(float(params.get('lgsource', params.get('LGSOURCE', 1.0e-6))), 1e-12)
                        elif name_lower == 'abdrain_i':
                            all_inputs[dev_idx, param_idx] = max(float(params.get('abdrain', params.get('ABDRAIN', 1.0e-12))), 1e-18)
                        elif name_lower == 'lsdrain_i':
                            all_inputs[dev_idx, param_idx] = max(float(params.get('lsdrain', params.get('LSDRAIN', 1.0e-6))), 1e-12)
                        elif name_lower == 'lgdrain_i':
                            all_inputs[dev_idx, param_idx] = max(float(params.get('lgdrain', params.get('LGDRAIN', 1.0e-6))), 1e-12)

                        # ============================================================
                        # Scaled Junction Capacitances (source-side)
                        # cjosbot = MULT_i * ABSOURCE_i * cjobot
                        # ============================================================
                        elif name_lower == 'cjosbot':
                            import math
                            mult = float(params.get('mult', params.get('MULT', 1.0)))
                            absource = max(float(params.get('absource', params.get('ABSOURCE', 1.0e-12))), 1e-18)
                            # Calculate cjobot using JUNCAP200 formulas
                            tkr = 273.15 + float(params.get('trj', params.get('TRJ', 21.0)))
                            tkd = 300.15
                            auxt = tkd / tkr
                            phitr = 8.617333262e-5 * tkr
                            phitd = 8.617333262e-5 * tkd
                            vbilow = 0.05
                            deltaphigr = -(7.02e-4 * tkr * tkr) / (1108.0 + tkr)
                            deltaphigd = -(7.02e-4 * tkd * tkd) / (1108.0 + tkd)
                            phig = float(params.get('phigbot', params.get('PHIGBOT', 1.16)))
                            vbir = max(float(params.get('vbirbot', params.get('VBIRBOT', 1.0))), 0.05)
                            cjor = max(float(params.get('cjorbot', params.get('CJORBOT', 1e-3))), 1e-12)
                            p = max(min(float(params.get('pbot', params.get('PBOT', 0.5))), 0.95), 0.05)
                            phigr = phig + deltaphigr
                            phigd = phig + deltaphigd
                            ftd = (auxt ** 1.5) * math.exp(0.5 * (phigr / phitr - phigd / phitd))
                            ubi = vbir * auxt - 2.0 * phitd * math.log(ftd)
                            vbi = ubi + phitd * math.log(1.0 + math.exp((vbilow - ubi) / phitd))
                            cjo = cjor * ((vbir / vbi) ** p)
                            all_inputs[dev_idx, param_idx] = mult * absource * cjo

                        elif name_lower == 'cjossti':
                            import math
                            mult = float(params.get('mult', params.get('MULT', 1.0)))
                            lssource = max(float(params.get('lssource', params.get('LSSOURCE', 1.0e-6))), 1e-12)
                            tkr = 273.15 + float(params.get('trj', params.get('TRJ', 21.0)))
                            tkd = 300.15
                            auxt = tkd / tkr
                            phitr = 8.617333262e-5 * tkr
                            phitd = 8.617333262e-5 * tkd
                            vbilow = 0.05
                            deltaphigr = -(7.02e-4 * tkr * tkr) / (1108.0 + tkr)
                            deltaphigd = -(7.02e-4 * tkd * tkd) / (1108.0 + tkd)
                            phig = float(params.get('phigsti', params.get('PHIGSTI', 1.16)))
                            vbir = max(float(params.get('vbirsti', params.get('VBIRSTI', 1.0))), 0.05)
                            cjor = max(float(params.get('cjorsti', params.get('CJORSTI', 1e-9))), 1e-18)
                            p = max(min(float(params.get('psti', params.get('PSTI', 0.5))), 0.95), 0.05)
                            phigr = phig + deltaphigr
                            phigd = phig + deltaphigd
                            ftd = (auxt ** 1.5) * math.exp(0.5 * (phigr / phitr - phigd / phitd))
                            ubi = vbir * auxt - 2.0 * phitd * math.log(ftd)
                            vbi = ubi + phitd * math.log(1.0 + math.exp((vbilow - ubi) / phitd))
                            cjo = cjor * ((vbir / vbi) ** p)
                            all_inputs[dev_idx, param_idx] = mult * lssource * cjo

                        elif name_lower == 'cjosgat':
                            import math
                            mult = float(params.get('mult', params.get('MULT', 1.0)))
                            lgsource = max(float(params.get('lgsource', params.get('LGSOURCE', 1.0e-6))), 1e-12)
                            tkr = 273.15 + float(params.get('trj', params.get('TRJ', 21.0)))
                            tkd = 300.15
                            auxt = tkd / tkr
                            phitr = 8.617333262e-5 * tkr
                            phitd = 8.617333262e-5 * tkd
                            vbilow = 0.05
                            deltaphigr = -(7.02e-4 * tkr * tkr) / (1108.0 + tkr)
                            deltaphigd = -(7.02e-4 * tkd * tkd) / (1108.0 + tkd)
                            phig = float(params.get('phiggat', params.get('PHIGGAT', 1.16)))
                            vbir = max(float(params.get('vbirgat', params.get('VBIRGAT', 1.0))), 0.05)
                            cjor = max(float(params.get('cjorgat', params.get('CJORGAT', 1e-9))), 1e-18)
                            p = max(min(float(params.get('pgat', params.get('PGAT', 0.5))), 0.95), 0.05)
                            phigr = phig + deltaphigr
                            phigd = phig + deltaphigd
                            ftd = (auxt ** 1.5) * math.exp(0.5 * (phigr / phitr - phigd / phitd))
                            ubi = vbir * auxt - 2.0 * phitd * math.log(ftd)
                            vbi = ubi + phitd * math.log(1.0 + math.exp((vbilow - ubi) / phitd))
                            cjo = cjor * ((vbir / vbi) ** p)
                            all_inputs[dev_idx, param_idx] = mult * lgsource * cjo

                        # ============================================================
                        # Scaled Built-in Voltages (source-side)
                        # vbisbot = vbibot (just a copy)
                        # ============================================================
                        elif name_lower == 'vbisbot':
                            import math
                            tkr = 273.15 + float(params.get('trj', params.get('TRJ', 21.0)))
                            tkd = 300.15
                            auxt = tkd / tkr
                            phitr = 8.617333262e-5 * tkr
                            phitd = 8.617333262e-5 * tkd
                            vbilow = 0.05
                            deltaphigr = -(7.02e-4 * tkr * tkr) / (1108.0 + tkr)
                            deltaphigd = -(7.02e-4 * tkd * tkd) / (1108.0 + tkd)
                            phig = float(params.get('phigbot', params.get('PHIGBOT', 1.16)))
                            vbir = max(float(params.get('vbirbot', params.get('VBIRBOT', 1.0))), 0.05)
                            phigr = phig + deltaphigr
                            phigd = phig + deltaphigd
                            ftd = (auxt ** 1.5) * math.exp(0.5 * (phigr / phitr - phigd / phitd))
                            ubi = vbir * auxt - 2.0 * phitd * math.log(ftd)
                            vbi = ubi + phitd * math.log(1.0 + math.exp((vbilow - ubi) / phitd))
                            all_inputs[dev_idx, param_idx] = vbi

                        elif name_lower == 'vbissti':
                            import math
                            tkr = 273.15 + float(params.get('trj', params.get('TRJ', 21.0)))
                            tkd = 300.15
                            auxt = tkd / tkr
                            phitr = 8.617333262e-5 * tkr
                            phitd = 8.617333262e-5 * tkd
                            vbilow = 0.05
                            deltaphigr = -(7.02e-4 * tkr * tkr) / (1108.0 + tkr)
                            deltaphigd = -(7.02e-4 * tkd * tkd) / (1108.0 + tkd)
                            phig = float(params.get('phigsti', params.get('PHIGSTI', 1.16)))
                            vbir = max(float(params.get('vbirsti', params.get('VBIRSTI', 1.0))), 0.05)
                            phigr = phig + deltaphigr
                            phigd = phig + deltaphigd
                            ftd = (auxt ** 1.5) * math.exp(0.5 * (phigr / phitr - phigd / phitd))
                            ubi = vbir * auxt - 2.0 * phitd * math.log(ftd)
                            vbi = ubi + phitd * math.log(1.0 + math.exp((vbilow - ubi) / phitd))
                            all_inputs[dev_idx, param_idx] = vbi

                        elif name_lower == 'vbisgat':
                            import math
                            tkr = 273.15 + float(params.get('trj', params.get('TRJ', 21.0)))
                            tkd = 300.15
                            auxt = tkd / tkr
                            phitr = 8.617333262e-5 * tkr
                            phitd = 8.617333262e-5 * tkd
                            vbilow = 0.05
                            deltaphigr = -(7.02e-4 * tkr * tkr) / (1108.0 + tkr)
                            deltaphigd = -(7.02e-4 * tkd * tkd) / (1108.0 + tkd)
                            phig = float(params.get('phiggat', params.get('PHIGGAT', 1.16)))
                            vbir = max(float(params.get('vbirgat', params.get('VBIRGAT', 1.0))), 0.05)
                            phigr = phig + deltaphigr
                            phigd = phig + deltaphigd
                            ftd = (auxt ** 1.5) * math.exp(0.5 * (phigr / phitr - phigd / phitd))
                            ubi = vbir * auxt - 2.0 * phitd * math.log(ftd)
                            vbi = ubi + phitd * math.log(1.0 + math.exp((vbilow - ubi) / phitd))
                            all_inputs[dev_idx, param_idx] = vbi

                        # ============================================================
                        # Scaled Junction Capacitances (drain-side)
                        # ============================================================
                        elif name_lower == 'cjosbotd':
                            import math
                            mult = float(params.get('mult', params.get('MULT', 1.0)))
                            abdrain = max(float(params.get('abdrain', params.get('ABDRAIN', 1.0e-12))), 1e-18)
                            tkr = 273.15 + float(params.get('trj', params.get('TRJ', 21.0)))
                            tkd = 300.15
                            auxt = tkd / tkr
                            phitr = 8.617333262e-5 * tkr
                            phitd = 8.617333262e-5 * tkd
                            vbilow = 0.05
                            deltaphigr = -(7.02e-4 * tkr * tkr) / (1108.0 + tkr)
                            deltaphigd = -(7.02e-4 * tkd * tkd) / (1108.0 + tkd)
                            phig = float(params.get('phigbotd', params.get('PHIGBOTD', 1.16)))
                            vbir = max(float(params.get('vbirbotd', params.get('VBIRBOTD', 1.0))), 0.05)
                            cjor = max(float(params.get('cjorbotd', params.get('CJORBOTD', 1e-3))), 1e-12)
                            p = max(min(float(params.get('pbotd', params.get('PBOTD', 0.5))), 0.95), 0.05)
                            phigr = phig + deltaphigr
                            phigd = phig + deltaphigd
                            ftd = (auxt ** 1.5) * math.exp(0.5 * (phigr / phitr - phigd / phitd))
                            ubi = vbir * auxt - 2.0 * phitd * math.log(ftd)
                            vbi = ubi + phitd * math.log(1.0 + math.exp((vbilow - ubi) / phitd))
                            cjo = cjor * ((vbir / vbi) ** p)
                            all_inputs[dev_idx, param_idx] = mult * abdrain * cjo

                        elif name_lower == 'cjosstid':
                            import math
                            mult = float(params.get('mult', params.get('MULT', 1.0)))
                            lsdrain = max(float(params.get('lsdrain', params.get('LSDRAIN', 1.0e-6))), 1e-12)
                            tkr = 273.15 + float(params.get('trj', params.get('TRJ', 21.0)))
                            tkd = 300.15
                            auxt = tkd / tkr
                            phitr = 8.617333262e-5 * tkr
                            phitd = 8.617333262e-5 * tkd
                            vbilow = 0.05
                            deltaphigr = -(7.02e-4 * tkr * tkr) / (1108.0 + tkr)
                            deltaphigd = -(7.02e-4 * tkd * tkd) / (1108.0 + tkd)
                            phig = float(params.get('phigstid', params.get('PHIGSTID', 1.16)))
                            vbir = max(float(params.get('vbirstid', params.get('VBIRSTID', 1.0))), 0.05)
                            cjor = max(float(params.get('cjorstid', params.get('CJORSTID', 1e-9))), 1e-18)
                            p = max(min(float(params.get('pstid', params.get('PSTID', 0.5))), 0.95), 0.05)
                            phigr = phig + deltaphigr
                            phigd = phig + deltaphigd
                            ftd = (auxt ** 1.5) * math.exp(0.5 * (phigr / phitr - phigd / phitd))
                            ubi = vbir * auxt - 2.0 * phitd * math.log(ftd)
                            vbi = ubi + phitd * math.log(1.0 + math.exp((vbilow - ubi) / phitd))
                            cjo = cjor * ((vbir / vbi) ** p)
                            all_inputs[dev_idx, param_idx] = mult * lsdrain * cjo

                        elif name_lower == 'cjosgatd':
                            import math
                            mult = float(params.get('mult', params.get('MULT', 1.0)))
                            lgdrain = max(float(params.get('lgdrain', params.get('LGDRAIN', 1.0e-6))), 1e-12)
                            tkr = 273.15 + float(params.get('trj', params.get('TRJ', 21.0)))
                            tkd = 300.15
                            auxt = tkd / tkr
                            phitr = 8.617333262e-5 * tkr
                            phitd = 8.617333262e-5 * tkd
                            vbilow = 0.05
                            deltaphigr = -(7.02e-4 * tkr * tkr) / (1108.0 + tkr)
                            deltaphigd = -(7.02e-4 * tkd * tkd) / (1108.0 + tkd)
                            phig = float(params.get('phiggatd', params.get('PHIGGATD', 1.16)))
                            vbir = max(float(params.get('vbirgatd', params.get('VBIRGATD', 1.0))), 0.05)
                            cjor = max(float(params.get('cjorgatd', params.get('CJORGATD', 1e-9))), 1e-18)
                            p = max(min(float(params.get('pgatd', params.get('PGATD', 0.5))), 0.95), 0.05)
                            phigr = phig + deltaphigr
                            phigd = phig + deltaphigd
                            ftd = (auxt ** 1.5) * math.exp(0.5 * (phigr / phitr - phigd / phitd))
                            ubi = vbir * auxt - 2.0 * phitd * math.log(ftd)
                            vbi = ubi + phitd * math.log(1.0 + math.exp((vbilow - ubi) / phitd))
                            cjo = cjor * ((vbir / vbi) ** p)
                            all_inputs[dev_idx, param_idx] = mult * lgdrain * cjo

                        # ============================================================
                        # Scaled Built-in Voltages (drain-side)
                        # ============================================================
                        elif name_lower == 'vbisbotd':
                            import math
                            tkr = 273.15 + float(params.get('trj', params.get('TRJ', 21.0)))
                            tkd = 300.15
                            auxt = tkd / tkr
                            phitr = 8.617333262e-5 * tkr
                            phitd = 8.617333262e-5 * tkd
                            vbilow = 0.05
                            deltaphigr = -(7.02e-4 * tkr * tkr) / (1108.0 + tkr)
                            deltaphigd = -(7.02e-4 * tkd * tkd) / (1108.0 + tkd)
                            phig = float(params.get('phigbotd', params.get('PHIGBOTD', 1.16)))
                            vbir = max(float(params.get('vbirbotd', params.get('VBIRBOTD', 1.0))), 0.05)
                            phigr = phig + deltaphigr
                            phigd = phig + deltaphigd
                            ftd = (auxt ** 1.5) * math.exp(0.5 * (phigr / phitr - phigd / phitd))
                            ubi = vbir * auxt - 2.0 * phitd * math.log(ftd)
                            vbi = ubi + phitd * math.log(1.0 + math.exp((vbilow - ubi) / phitd))
                            all_inputs[dev_idx, param_idx] = vbi

                        elif name_lower == 'vbisstid':
                            import math
                            tkr = 273.15 + float(params.get('trj', params.get('TRJ', 21.0)))
                            tkd = 300.15
                            auxt = tkd / tkr
                            phitr = 8.617333262e-5 * tkr
                            phitd = 8.617333262e-5 * tkd
                            vbilow = 0.05
                            deltaphigr = -(7.02e-4 * tkr * tkr) / (1108.0 + tkr)
                            deltaphigd = -(7.02e-4 * tkd * tkd) / (1108.0 + tkd)
                            phig = float(params.get('phigstid', params.get('PHIGSTID', 1.16)))
                            vbir = max(float(params.get('vbirstid', params.get('VBIRSTID', 1.0))), 0.05)
                            phigr = phig + deltaphigr
                            phigd = phig + deltaphigd
                            ftd = (auxt ** 1.5) * math.exp(0.5 * (phigr / phitr - phigd / phitd))
                            ubi = vbir * auxt - 2.0 * phitd * math.log(ftd)
                            vbi = ubi + phitd * math.log(1.0 + math.exp((vbilow - ubi) / phitd))
                            all_inputs[dev_idx, param_idx] = vbi

                        elif name_lower == 'vbisgatd':
                            import math
                            tkr = 273.15 + float(params.get('trj', params.get('TRJ', 21.0)))
                            tkd = 300.15
                            auxt = tkd / tkr
                            phitr = 8.617333262e-5 * tkr
                            phitd = 8.617333262e-5 * tkd
                            vbilow = 0.05
                            deltaphigr = -(7.02e-4 * tkr * tkr) / (1108.0 + tkr)
                            deltaphigd = -(7.02e-4 * tkd * tkd) / (1108.0 + tkd)
                            phig = float(params.get('phiggatd', params.get('PHIGGATD', 1.16)))
                            vbir = max(float(params.get('vbirgatd', params.get('VBIRGATD', 1.0))), 0.05)
                            phigr = phig + deltaphigr
                            phigd = phig + deltaphigd
                            ftd = (auxt ** 1.5) * math.exp(0.5 * (phigr / phitr - phigd / phitd))
                            ubi = vbir * auxt - 2.0 * phitd * math.log(ftd)
                            vbi = ubi + phitd * math.log(1.0 + math.exp((vbilow - ubi) / phitd))
                            all_inputs[dev_idx, param_idx] = vbi

                        # ============================================================
                        # PSP103 Geometry Computed Params
                        # ============================================================
                        elif name_lower == 'il':
                            # iL = 1/L (inverse length)
                            L = float(params.get('l', params.get('L', 1e-6)))
                            all_inputs[dev_idx, param_idx] = 1.0 / max(L, 1e-12)
                        elif name_lower == 'iw':
                            # iW = 1/W (inverse width)
                            W = float(params.get('w', params.get('W', 1e-6)))
                            all_inputs[dev_idx, param_idx] = 1.0 / max(W, 1e-12)
                        elif name_lower == 'l_f':
                            # L_f = effective L
                            L = float(params.get('l', params.get('L', 1e-6)))
                            all_inputs[dev_idx, param_idx] = L
                        elif name_lower == 'w_f':
                            # W_f = effective W
                            W = float(params.get('w', params.get('W', 1e-6)))
                            all_inputs[dev_idx, param_idx] = W
                        elif name_lower == 'ile':
                            # iLE = 1/LE (inverse effective length)
                            L = float(params.get('l', params.get('L', 1e-6)))
                            nf = float(params.get('nf', params.get('NF', 1.0)))
                            LE = L * max(nf, 1.0)
                            all_inputs[dev_idx, param_idx] = 1.0 / max(LE, 1e-12)
                        elif name_lower == 'iwe':
                            # iWE = 1/WE (inverse effective width)
                            W = float(params.get('w', params.get('W', 1e-6)))
                            all_inputs[dev_idx, param_idx] = 1.0 / max(W, 1e-12)
                        elif name_lower in ('lecv', 'lcv'):
                            # LEcv, Lcv = CV effective length (= L for now)
                            L = float(params.get('l', params.get('L', 1e-6)))
                            all_inputs[dev_idx, param_idx] = L
                        elif name_lower in ('wecv', 'wcv'):
                            # WEcv, Wcv = CV effective width (= W for now)
                            W = float(params.get('w', params.get('W', 1e-6)))
                            all_inputs[dev_idx, param_idx] = W
                        elif name_lower == 'l_slif':
                            # L_slif = L for SLIF model
                            L = float(params.get('l', params.get('L', 1e-6)))
                            all_inputs[dev_idx, param_idx] = L
                        elif name_lower == 'xgwe':
                            # XGWE = XGW * WE (gate width extension * effective width)
                            xgw = float(params.get('xgw', params.get('XGW', 0.0)))
                            W = float(params.get('w', params.get('W', 1e-6)))
                            all_inputs[dev_idx, param_idx] = xgw * W
                        elif name_lower in ('dellps', 'delwod'):
                            # Length/width offsets - default to 0
                            all_inputs[dev_idx, param_idx] = 0.0

                        # ============================================================
                        # PSP103 Instance Params that default to 0
                        # ============================================================
                        elif name_lower in ('sd_i', 'sc_i', 'xgw_i', 'jw_i', 'scc_i', 'ngcon_i'):
                            # Layout-related instance params - default to 0 or 1
                            if name_lower == 'ngcon_i':
                                all_inputs[dev_idx, param_idx] = 1.0  # Number of gate contacts
                            else:
                                all_inputs[dev_idx, param_idx] = 0.0

                        # ============================================================
                        # PSP103 AC scaling params (default to 0)
                        # ============================================================
                        elif name_lower.endswith('_i') and any(x in name_lower for x in ['kvsat', 'web', 'wec', 'cfac', 'thesat', 'axac', 'alpac', 'pocfac', 'plcfac', 'pwcfac', 'pothesatac', 'plthesatac', 'pwthesatac', 'poaxac', 'plaxac', 'pwaxac', 'poalpac', 'plalpac', 'pwalpac', 'kvsatac']):
                            # AC velocity saturation and scaling params - default to 0
                            all_inputs[dev_idx, param_idx] = 0.0

                        # ============================================================
                        # PSP103 _p Process Params (copy from base param or default)
                        # ============================================================
                        elif name_lower.endswith('_p'):
                            # _p params are binning-adjusted values - use base param or default
                            base_name = name_lower[:-2]
                            base_val = params.get(base_name, params.get(base_name.upper(), None))
                            if base_val is not None:
                                all_inputs[dev_idx, param_idx] = float(base_val)
                            else:
                                # Provide sensible defaults for critical _p params
                                p_defaults = {
                                    'tox': 2e-9,      # Oxide thickness
                                    'epsrox': 3.9,    # Oxide permittivity
                                    'neff': 5e23,     # Effective doping
                                    'vfb': -1.0,      # Flatband voltage
                                    'betn': 30e-3,    # Mobility factor
                                    'mue': 500.0,     # Low-field mobility
                                    'themu': 1.5,     # Mobility reduction
                                    'cs': 0.0,        # Carrier scattering
                                    'xcor': 0.0,      # Coulomb scattering
                                    'feta': 1.0,      # DIBL factor
                                    'rs': 0.0,        # Series resistance
                                    'thesat': 1.0,    # Velocity saturation
                                    'ax': 3.0,        # Velocity saturation exponent
                                    'alp': 0.01,      # Channel length modulation
                                    'vp': 0.05,       # Pinch-off voltage
                                    'cf': 0.0,        # Fringing cap
                                    'ct': 0.0,        # DIBL param
                                    'toxov': 2e-9,    # Overlap oxide thickness
                                    'nov': 5e23,      # Overlap doping
                                    'iginv': 0.0,     # Gate current
                                    'igov': 0.0,      # Overlap gate current
                                    'gc2': 0.0,       # Gate current
                                    'gc3': 0.0,       # Gate current
                                    'chib': 3.1,      # Barrier height
                                    'agidl': 0.0,     # GIDL current
                                    'bgidl': 0.0,     # GIDL current
                                    'cgidl': 0.0,     # GIDL current
                                }
                                default_val = p_defaults.get(base_name, 0.0)
                                all_inputs[dev_idx, param_idx] = default_val

                        # ============================================================
                        # PSP103 lp_ Local Params (copy from _p param or base)
                        # ============================================================
                        elif name_lower.startswith('lp_'):
                            # lp_* params are local copies of _p params
                            base_name = name_lower[3:]  # Strip lp_ prefix
                            base_val = params.get(base_name, params.get(base_name.upper(), None))
                            if base_val is not None:
                                all_inputs[dev_idx, param_idx] = float(base_val)
                            else:
                                # Use same defaults as _p params
                                p_defaults = {
                                    'tox': 2e-9,
                                    'epsrox': 3.9,
                                    'neff': 5e23,
                                    'vfb': -1.0,
                                    'betn': 30e-3,
                                    'mue': 500.0,
                                    'themu': 1.5,
                                    'cs': 0.0,
                                    'xcor': 0.0,
                                    'feta': 1.0,
                                    'rs': 0.0,
                                    'rsb': 0.0,
                                    'rsg': 0.0,
                                    'thesat': 1.0,
                                    'ax': 3.0,
                                    'alp': 0.01,
                                    'alp1': 0.0,
                                    'alp2': 0.0,
                                    'vp': 0.05,
                                    'a1': 0.0,
                                    'a2': 0.0,
                                    'a3': 0.0,
                                    'a4': 0.0,
                                    'gco': 0.0,
                                    'cf': 0.0,
                                    'cfd': 0.0,
                                    'cfb': 0.0,
                                    'ct': 0.0,
                                    'ctg': 0.0,
                                    'ctb': 0.0,
                                    'stct': 0.0,
                                    'psce': 0.0,
                                    'psced': 0.0,
                                    'psceb': 0.0,
                                    'cox': 0.0,
                                    'cgov': 0.0,
                                    'cgovd': 0.0,
                                    'cgbov': 0.0,
                                    'cfr': 0.0,
                                    'cfrd': 0.0,
                                    'iginv': 0.0,
                                    'igov': 0.0,
                                    'igovd': 0.0,
                                    'stig': 0.0,
                                    'gc2': 0.0,
                                    'gc3': 0.0,
                                    'gc2ov': 0.0,
                                    'gc3ov': 0.0,
                                    'chib': 3.1,
                                    'agidl': 0.0,
                                    'agidld': 0.0,
                                    'bgidl': 0.0,
                                    'bgidld': 0.0,
                                    'cgidl': 0.0,
                                    'cgidld': 0.0,
                                    'fnt': 0.0,
                                    'fntexc': 0.0,
                                    'nfa': 0.0,
                                    'nfb': 0.0,
                                    'nfc': 0.0,
                                    'ef': 0.0,
                                    'rg': 0.0,
                                    'rse': 0.0,
                                    'rde': 0.0,
                                    'rbulk': 0.0,
                                    'rwell': 0.0,
                                    'rjuns': 1e12,  # Very high default for junction R
                                    'rjund': 1e12,
                                }
                                default_val = p_defaults.get(base_name, 0.0)
                                all_inputs[dev_idx, param_idx] = default_val

                        # For other hidden_state params, try to find matching base param
                        else:
                            handled = False
                            # Strip _i suffix and look for base param
                            if name_lower.endswith('_i'):
                                base_name = name_lower[:-2]
                                base_val = params.get(base_name, params.get(base_name.upper(), None))
                                if base_val is None:
                                    # Try with 'o' suffix (common PSP103 pattern: toxo -> tox_i)
                                    base_name_o = base_name + 'o'
                                    base_val = params.get(base_name_o, params.get(base_name_o.upper(), None))
                                if base_val is not None:
                                    all_inputs[dev_idx, param_idx] = float(base_val)
                                    handled = True
                                else:
                                    # Use sensible defaults for critical _i params
                                    i_defaults = {
                                        'tox': 2e-9,           # Oxide thickness (TOXO)
                                        'epsrox': 3.9,         # Oxide permittivity (EPSROXO)
                                        'neff': 5e23,          # Effective doping (from NSUBO)
                                        'vfb': -1.0,           # Flatband voltage (VFBO)
                                        'betn': 30e-3,         # Mobility factor (UO*W/L)
                                        'mue': 0.5,            # Low-field mobility (MUEO)
                                        'themu': 1.5,          # Mobility reduction (THEMUO)
                                        'cs': 0.0,             # Carrier scattering (CSO)
                                        'thecs': 0.0,          # CS temperature coeff
                                        'xcor': 0.15,          # Coulomb scattering (XCORO)
                                        'feta': 1.0,           # DIBL factor (FETAO)
                                        'rs': 0.0,             # Series resistance
                                        'rsb': 0.0,            # Body resistance
                                        'rsg': 0.0,            # Gate resistance
                                        'thesat': 1e-6,        # Velocity saturation (THESATO)
                                        'ax': 20.0,            # Velocity saturation exponent (AXO)
                                        'alp': 0.01,           # Channel length modulation
                                        'alp1': 0.0,           # CLM param
                                        'alp2': 0.0,           # CLM param
                                        'vp': 0.25,            # Pinch-off voltage (VPO)
                                        'a1': 1.0,             # Impact ionization (A1O)
                                        'a2': 10.0,            # Impact ionization (A2O)
                                        'a3': 1.0,             # Impact ionization (A3O)
                                        'a4': 0.0,             # Impact ionization (A4O)
                                        'gco': 5.0,            # Gate current (GCOO)
                                        'cf': 0.0,             # Fringing cap
                                        'cfd': 0.0,            # Drain fringing cap
                                        'cfb': 0.3,            # Bulk fringing cap (CFBO)
                                        'ct': 0.0,             # DIBL param
                                        'ctg': 0.0,            # DIBL param
                                        'ctb': 0.0,            # DIBL param
                                        'stct': 0.0,           # CT temperature coeff
                                        'psce': 0.0,           # Short channel param
                                        'psced': 0.0,          # Short channel param
                                        'psceb': 0.0,          # Short channel param
                                        'toxov': 2e-9,         # Overlap oxide thickness (TOXOVO)
                                        'toxovd': 2e-9,        # Drain overlap oxide (TOXOVDO)
                                        'nov': 5e25,           # Overlap doping (NOVO)
                                        'novd': 5e25,          # Drain overlap doping (NOVDO)
                                        'iginv': 0.0,          # Gate current (inversion)
                                        'igov': 0.0,           # Overlap gate current
                                        'igovd': 0.0,          # Drain overlap gate current
                                        'stig': 1.5,           # Gate current temp coeff (STIGO)
                                        'gc2': 1.0,            # Gate current param (GC2O)
                                        'gc3': -1.0,           # Gate current param (GC3O)
                                        'gc2ov': 0.0,          # Overlap gate current
                                        'gc3ov': 0.0,          # Overlap gate current
                                        'chib': 3.1,           # Barrier height (CHIBO)
                                        'agidl': 0.0,          # GIDL current
                                        'agidld': 0.0,         # Drain GIDL current
                                        'bgidl': 35.0,         # GIDL current (BGIDLO)
                                        'bgidld': 41.0,        # Drain GIDL (BGIDLDO)
                                        'stbgidl': 0.0,        # GIDL temp coeff
                                        'stbgidld': 0.0,       # Drain GIDL temp coeff
                                        'cgidl': 0.15,         # GIDL current (CGIDLO)
                                        'cgidld': 0.0,         # Drain GIDL (CGIDLDO)
                                        'cox': 0.0,            # Oxide capacitance (computed)
                                        'cgov': 0.0,           # Overlap cap
                                        'cgovd': 0.0,          # Drain overlap cap
                                        'cgbov': 0.0,          # Bulk overlap cap (CGBOVL)
                                        'cfr': 0.0,            # Fringing resistance
                                        'cfrd': 0.0,           # Drain fringing resistance
                                        'fnt': 1.0,            # Noise factor (FNTO)
                                        'fntexc': 0.0,         # Excess noise
                                        'nfa': 0.0,            # Noise param
                                        'nfb': 0.0,            # Noise param
                                        'nfc': 0.0,            # Noise param
                                        'ef': 0.0,             # Flicker noise exponent
                                        # Edge device params
                                        'vfbedge': -1.0,       # Edge flatband
                                        'stvfbedge': 0.0,      # Edge FB temp coeff
                                        'dphibedge': 0.0,      # Edge band bending
                                        'neffedge': 5e23,      # Edge doping
                                        # Misc
                                        'np': 1.5e26,          # Gate poly doping (NPO)
                                        'facneffac': 0.8,      # NUD factor (FACNEFFACO)
                                        'gfacnud': 0.1,        # NUD factor (GFACNUDO)
                                        'vsbnud': 0.0,         # NUD voltage (VSBNUDO)
                                        'dvsbnud': 1.0,        # NUD voltage (DVSBNUDO)
                                        'vnsub': 0.0,          # Substrate voltage (VNSUBO)
                                        'nslp': 0.05,          # Slope factor (NSLPO)
                                        'dnsub': 0.0,          # Doping variation (DNSUBO)
                                        'dphib': 0.0,          # Band bending (DPHIBO)
                                        'delvtac': 0.0,        # Threshold adjust (DELVTACO)
                                        'stbet': 1.75,         # Beta temp coeff (STBETO)
                                        'stmue': 0.5,          # Mobility temp coeff (STMUEO)
                                        'stthemu': -0.1,       # TheMU temp coeff
                                        'stcs': -5.0,          # CS temp coeff (STCSO)
                                        'stthecs': 0.0,        # TheCS temp coeff
                                        'stxcor': 1.25,        # XCOR temp coeff (STXCORO)
                                        'strs': -2.0,          # RS temp coeff (STRSO)
                                        'sta2': -0.5,          # A2 temp coeff (STA2O)
                                    }
                                    default_val = i_defaults.get(base_name, None)
                                    if default_val is not None:
                                        all_inputs[dev_idx, param_idx] = default_val
                                        handled = True
                            # ============================================================
                            # Computed geometry-dependent hidden_state (Ring benchmark defaults)
                            # These values depend on L_i, W_i and other geometry params
                            # Using simplified defaults for basic operation
                            # ============================================================
                            if not handled:
                                # Get device geometry for scaling
                                l_i = max(float(params.get('l', params.get('L', 1e-6))), 1e-9)
                                w_i = max(float(params.get('w', params.get('W', 1e-5))), 1e-9)
                                nsubo = float(params.get('nsubo', params.get('NSUBO', 3e23)))
                                npck = float(params.get('npck', params.get('NPCK', 1e24)))
                                lpck = float(params.get('lpck', params.get('LPCK', 5.5e-8)))

                                geom_computed = {
                                    # Effective geometry values (LE, WE with scaling)
                                    'le': l_i,
                                    'we': w_i,
                                    'lecv': l_i,
                                    'wecv': w_i,
                                    'lcv': l_i,
                                    'wcv': w_i,
                                    'ile': 1.0 / l_i if l_i > 0 else 1e9,
                                    'iwe': 1.0 / w_i if w_i > 0 else 1e9,
                                    'il': 1.0 / l_i if l_i > 0 else 1e9,
                                    'iw': 1.0 / w_i if w_i > 0 else 1e9,
                                    'invnf': 1.0,  # 1/NF
                                    'l_f': l_i,
                                    'l_slif': l_i,
                                    'w_f': w_i,

                                    # Doping geometry scaling
                                    'nsub0e': nsubo,  # NSUB0 effective
                                    'npcke': npck,    # NPCK effective
                                    'lpcke': lpck,    # LPCK effective
                                    'aa': 0.1,        # sqrt(NSUB0e) scaling
                                    'bb': 0.05,       # sqrt(NSUB0e + 0.5*NPCKe) - sqrt(NSUB0e)
                                    'nsub': nsubo,    # Final substrate doping

                                    # Mobility scaling
                                    'fbet1e': 1.0,    # Beta L scaling factor
                                    'lp1e': float(params.get('lp1', params.get('LP1', 1.5e-7))),

                                    # Geometry polynomial exponents (GPE, GWE)
                                    'gpe': 2.0,       # L polynomial exponent
                                    'gwe': 2.0,       # W polynomial exponent

                                    # Temp variables
                                    'tmpx': 0.0,
                                    'temp0': 0.0,
                                    'temp00': 0.0,
                                    'lnoi': l_i,      # Noise length
                                    'lred': 0.0,      # Length reduction

                                    # Edge device params
                                    'we_edge': w_i,
                                    'iwe_edge': 1.0 / w_i if w_i > 0 else 1e9,
                                    'gpe_edge': 2.0,
                                    'xgwe': 0.0,      # Edge gate width

                                    # LOD scaling
                                    'kvthowe': 1.0,
                                    'kuowe': 1.0,
                                    'ilewe': 1.0 / (l_i * w_i) if l_i * w_i > 0 else 1e18,
                                    'iile': 1.0 / l_i if l_i > 0 else 1e9,
                                    'iiwe': 1.0 / w_i if w_i > 0 else 1e9,
                                    'iilewe': 1.0 / (l_i * w_i) if l_i * w_i > 0 else 1e18,
                                    'iiilewe': 1.0 / (l_i * l_i * w_i) if l_i * w_i > 0 else 1e27,

                                    # CV scaling (same as above)
                                    'ilecv': 1.0 / l_i if l_i > 0 else 1e9,
                                    'iilewecv': 1.0 / (l_i * w_i) if l_i * w_i > 0 else 1e18,
                                    'iiilewecv': 1.0 / (l_i * l_i * w_i) if l_i * w_i > 0 else 1e27,
                                    'iilcv': 1.0 / l_i if l_i > 0 else 1e9,
                                    'iilwcv': 1.0 / (l_i * w_i) if l_i * w_i > 0 else 1e18,
                                    'iiilwcv': 1.0 / (l_i * l_i * w_i) if l_i * w_i > 0 else 1e27,

                                    # Stress/LOD params
                                    'tmpa': 0.0,
                                    'tmpb': 0.0,
                                    'loop': 0.0,
                                    'invsa': 0.0,
                                    'invsb': 0.0,
                                    'invsaref': 0.0,
                                    'invsbref': 0.0,
                                    'lx': l_i,
                                    'wx': w_i,
                                    'templ': 0.0,
                                    'tempw': 0.0,
                                    'kstressu0': 1.0,
                                    'rhobeta': 1.0,
                                    'rhobetaref': 1.0,
                                    'kstressvth0': 1.0,
                                    'dellps': 0.0,    # Length delta from stress
                                    'delwod': 0.0,    # Width delta from LOD

                                    # More geometry scaling (case variations)
                                    'iilecv': 1.0 / l_i if l_i > 0 else 1e9,
                                    'iiwecv': 1.0 / w_i if w_i > 0 else 1e9,
                                    'iiwcv': 1.0 / w_i if w_i > 0 else 1e9,
                                }

                                default_val = geom_computed.get(name_lower, None)
                                if default_val is not None:
                                    all_inputs[dev_idx, param_idx] = default_val
                                    handled = True

                            # ============================================================
                            # Oxide and mobility computed params
                            # These are critical for device current calculation
                            # ============================================================
                            if not handled:
                                # Get key model params
                                toxo = float(params.get('toxo', params.get('TOXO', 2e-9)))
                                epsroxo = float(params.get('epsroxo', params.get('EPSROXO', 3.9)))
                                nsubo = float(params.get('nsubo', params.get('NSUBO', 3e23)))
                                ax = float(params.get('axo', params.get('AXO', 20.0)))
                                vp = float(params.get('vpo', params.get('VPO', 0.25)))
                                l_i = max(float(params.get('l', params.get('L', 1e-6))), 1e-9)
                                w_i = max(float(params.get('w', params.get('W', 1e-5))), 1e-9)
                                ld = float(params.get('ld', params.get('LD', l_i * 0.5)))  # Diffusion length ~half gate length

                                # Physical constants
                                eps0 = 8.854e-12  # F/m
                                epssi = 11.7 * eps0
                                epsox = epsroxo * eps0
                                q = 1.602e-19     # C

                                # Oxide capacitance params
                                cox_prime = epsox / max(toxo, 1e-10)  # F/m²

                                oxide_computed = {
                                    'epsox': epsox,
                                    'coxprime': cox_prime,
                                    'tox_sq': toxo * toxo,
                                    'cox_over_q': cox_prime / q,
                                    'neffac_i': nsubo * 0.8,  # ~NEFF with FACNEFFAC
                                    'qq': 0.0,  # Surface charge (computed)
                                    'e_eff0': 0.0,  # Effective field (computed)
                                    'eta_mu': 0.5,  # Mobility reduction
                                    'eta_mu1': 0.5,

                                    # Inverse params (critical - used as divisors)
                                    'inv_ax': 1.0 / max(ax, 0.01),
                                    'inv_vp': 1.0 / max(vp, 0.01),

                                    # Overlap capacitance
                                    'coxovprime': epsox / max(float(params.get('toxovo', params.get('TOXOVO', 1.5e-9))), 1e-10),
                                    'coxovprime_d': epsox / max(float(params.get('toxovdo', params.get('TOXOVDO', 2e-9))), 1e-10),

                                    # Gate overlap conductance
                                    'gov_s': 1e-12,  # Small default
                                    'gov_d': 1e-12,
                                    'gov2_s': 0.0,
                                    'gov2_d': 0.0,
                                    'inv_gov': 1e12,  # Large inverse

                                    # Surface potential params
                                    'sp_ov_eps': 1e-6,
                                    'sp_ov_eps2_s': 1e-12,
                                    'sp_ov_delta': 0.01,
                                    'sp_ov_a_s': 1.0,
                                    'sp_ov_delta1_s': 0.01,
                                    'sp_ov_eps2_d': 1e-12,
                                    'sp_ov_a_d': 1.0,
                                    'sp_ov_delta1_d': 0.01,

                                    # Edge device params (from base params + EDGE suffix)
                                    'st2vfb_i': 0.0,  # 2nd order VFB temp coeff
                                    'ctedge_i': 0.0,
                                    'betnedge_i': float(params.get('betnedge', params.get('BETNEDGE', 0.03))),
                                    'stbetedge_i': float(params.get('stbetedge', params.get('STBETEDGE', 1.75))),
                                    'psceedge_i': 0.0,
                                    'pscebedge_i': 0.0,
                                    'pscededge_i': 0.0,
                                    'cfedge_i': 0.0,
                                    'cfdedge_i': 0.0,
                                    'cfbedge_i': 0.3,
                                    'fntedge_i': 0.0,
                                    'nfaedge_i': 0.0,
                                    'nfbedge_i': 0.0,
                                    'nfcedge_i': 0.0,
                                    'efedge_i': 0.0,

                                    # Series resistance
                                    'rse_i': 0.0,  # Source series R
                                    'rde_i': 0.0,  # Drain series R

                                    # LOD scaling factors
                                    'factuo_i': 1.0,
                                    'delvto_i': 0.0,
                                    'factuoedge_i': 1.0,
                                    'delvtoedge_i': 0.0,
                                }

                                default_val = oxide_computed.get(name_lower, None)
                                if default_val is not None:
                                    all_inputs[dev_idx, param_idx] = default_val
                                    handled = True

                            # ============================================================
                            # Surface potential and temperature computed params
                            # ============================================================
                            if not handled:
                                # Get temperature and doping
                                tkd = 300.15  # Device temp in Kelvin
                                phit = tkd * 8.617333262e-5  # Thermal voltage ~0.026V
                                nsubo = float(params.get('nsubo', params.get('NSUBO', 3e23)))
                                vfbo = float(params.get('vfbo', params.get('VFBO', -1.0)))
                                l_i = max(float(params.get('l', params.get('L', 1e-6))), 1e-9)

                                # Surface potential DC params
                                phib_dc = 0.9  # Bulk potential (~2*phit*ln(NSUB/ni))
                                g0_dc = 1e-6   # Small transconductance

                                sp_computed = {
                                    'ilcv': 1.0 / l_i if l_i > 0 else 1e9,

                                    # DC surface potential params
                                    'phib_dc': phib_dc,
                                    'g_0_dc': g0_dc,
                                    'kp': 1.5,  # Short channel factor
                                    'np': 1.0,  # Non-ideality
                                    'arg2max': 10.0,  # Limiting arg
                                    'qlim2': 0.01,  # Charge limit
                                    'qb0': 0.0,  # Bulk charge
                                    'dphibq': 0.0,  # Potential correction
                                    'sqrt_phib_dc': phib_dc ** 0.5,
                                    'phix_dc': phib_dc,  # Initial surface pot
                                    'aphi_dc': 1.0,
                                    'bphi_dc': 1.0,
                                    'phix2': phib_dc,
                                    'phix1_dc': phib_dc,
                                    'alpha_b': 0.1,  # Body effect
                                    'us1': 0.5,  # Velocity saturation
                                    'us21': 0.25,

                                    # AC surface potential (same defaults)
                                    'phib_ac': phib_dc,
                                    'g_0_ac': g0_dc,
                                    'phix_ac': phib_dc,
                                    'aphi_ac': 1.0,
                                    'bphi_ac': 1.0,
                                    'phix1_ac': phib_dc,

                                    # Temperature-scaled params (at T=27C, tf factors = 1)
                                    'vfb_t': vfbo,
                                    'tf_ct': 1.0,
                                    'ct_t': 0.0,
                                    'ctg_t': 0.0,
                                    'tf_bet': 1.0,
                                    'betn_t': float(params.get('betn', params.get('BETNO', 0.03))),
                                    'bet_i': float(params.get('betn', params.get('BETNO', 0.03))),
                                    'themu_t': float(params.get('themu', params.get('THEMUO', 1.5))),
                                    'tf_mue': 1.0,
                                    'mue_t': float(params.get('mue', params.get('MUEO', 0.5))),
                                    'thecs_t': float(params.get('thecs', params.get('THECSO', 0))),
                                    'tf_cs': 1.0,
                                    'cs_t': 0.0,
                                    'tf_xcor': 1.0,
                                    'xcor_t': float(params.get('xcor', params.get('XCORO', 0.15))),
                                    'tf_ther': 1.0,
                                    'rs_t': 0.0,
                                    'ther_i': 0.0,
                                    'tf_thesat': 1.0,
                                    'thesat_t': float(params.get('thesat', params.get('THESATO', 1e-6))),
                                    'thesatac_t': 0.0,
                                    'a2_t': float(params.get('a2', params.get('A2O', 10.0))),

                                    # Noise
                                    'nt': 4.0 * 1.38066e-23 * tkd * float(params.get('fnt', params.get('FNTO', 0))),
                                    'sfl_prefac': 0.0,

                                    # Edge params at temperature
                                    'vfbedge_t': vfbo,
                                    'tf_betedge': 1.0,
                                    'betnedge_t': float(params.get('betnedge', params.get('BETNEDGE', 0.03))),
                                    'betedge_i': float(params.get('betnedge', params.get('BETNEDGE', 0.03))),
                                    'phit0edge': phib_dc,
                                    'phibedge': phib_dc,
                                    'gfedge': 1e-6,
                                    'gfedge2': 1e-12,
                                    'lngfedge2': -27.6,  # ln(1e-12)
                                    'phixedge': phib_dc,
                                    'aphiedge': 1.0,
                                    'bphiedge': 1.0,
                                    'phix2edge': phib_dc,
                                    'phix1edge': phib_dc,
                                    'sfl_prefac_edge': 0.0,
                                    'ntedge': 0.0,

                                    # Gate tunneling
                                    'inv_chib': 1.0 / 3.1,  # Si/SiO2 barrier ~3.1eV
                                    'b_fact': 1.0,
                                    'bch': 1.0,
                                    'bov': 1.0,
                                    'bov_d': 1.0,
                                    'gcq': 1.0,
                                    'gcqov': 1.0,
                                    'tf_ig': 1.0,
                                    'agidls': 0.0,
                                    'agidlds': 0.0,
                                    'bgidl_t': 0.0,
                                    'bgidls': 0.0,

                                    # GIDL temperature params
                                    'bgidld_t': 0.0,
                                    'bgidlds': 0.0,
                                    'fac_exc': 0.0,

                                    # Initial conductance values (all should start at small positive values)
                                    'ggate': 1e-12,
                                    'gsource': 1e-12,
                                    'gdrain': 1e-12,
                                    'gbulk': 1e-12,
                                    'gjuns': 1e-12,
                                    'gjund': 1e-12,
                                    'gwell': 1e-12,

                                    # Junction area/perimeter params (compute from W/L if available)
                                    'abs_i': w_i * ld,  # Source bottom junction area
                                    'lss_i': 2.0 * (w_i + ld),  # Source sidewall perimeter
                                    'lgs_i': w_i,  # Source gate-edge perimeter
                                    'abd_i': w_i * ld,  # Drain bottom junction area
                                    'lsd_i': 2.0 * (w_i + ld),  # Drain sidewall perimeter
                                    'lgd_i': w_i,  # Drain gate-edge perimeter

                                    # Junction well correction params
                                    'jwcorr': 1.0,
                                    'jww': 1.0,

                                    # Junction voltage limits (source side)
                                    'vbimin_s': -0.9,
                                    'vfmin_s': -0.3,
                                    'vch_s': 0.4,
                                    'vbbtlim_s': -0.5,
                                    'vmax_s': 0.6,
                                    'exp_vmax_over_phitd_s': 1.0,

                                    # Junction voltage limits (drain side)
                                    'vbimin_d': -0.9,
                                    'vfmin_d': -0.3,
                                    'vch_d': 0.4,
                                    'vbbtlim_d': -0.5,
                                    'vmax_d': 0.6,
                                    'exp_vmax_over_phitd_d': 1.0,

                                    # Junction saturation current (source side)
                                    'isatfor1_s': 1e-15,
                                    'mfor1_s': 1.0,
                                    'isatfor2_s': 1e-18,
                                    'mfor2_s': 2.0,
                                    'isatrev_s': 1e-15,
                                    'mrev_s': 1.0,
                                    'm0flag_s': 0.0,
                                    'xhighf1_s': 10.0,
                                    'expxhf1_s': 2.2e4,  # exp(10)
                                    'xhighf2_s': 20.0,
                                    'expxhf2_s': 4.85e8,  # exp(20)
                                    'xhighr_s': 10.0,
                                    'expxhr_s': 2.2e4,

                                    # Junction saturation current (drain side)
                                    'isatfor1_d': 1e-15,
                                    'mfor1_d': 1.0,
                                    'isatfor2_d': 1e-18,
                                    'mfor2_d': 2.0,
                                    'isatrev_d': 1e-15,
                                    'mrev_d': 1.0,
                                    'm0flag_d': 0.0,
                                    'xhighf1_d': 10.0,
                                    'expxhf1_d': 2.2e4,
                                    'xhighf2_d': 20.0,
                                    'expxhf2_d': 4.85e8,
                                    'xhighr_d': 10.0,
                                    'expxhr_d': 2.2e4,

                                    # Thermal resistance params
                                    'rth_m': 0.0,  # Thermal resistance
                                    'cth_m': 0.0,  # Thermal capacitance
                                    'rth_i': 0.0,
                                    'cth_i': 0.0,
                                }

                                default_val = sp_computed.get(name_lower, None)
                                if default_val is not None:
                                    all_inputs[dev_idx, param_idx] = default_val
                                    handled = True

                            # Catch-all: set safe defaults based on parameter name pattern
                            # Many hidden_state params are multipliers, flags, or intermediate values
                            # Setting them to safe non-zero values avoids division-by-zero
                            if not handled:
                                safe_val = None
                                name_lower = name.lower()

                                # Flags and indicators (typically 0 or 1)
                                if any(x in name_lower for x in ['flag', 'm0flag', 'zflag', 'swflag']):
                                    safe_val = 0.0  # Default flag off

                                # Multipliers and correction factors (default to 1)
                                elif any(x in name_lower for x in ['mult', 'fact', 'fac_', 'cor', 'tf_', '_t']):
                                    safe_val = 1.0

                                # Saturation currents and small values (avoid zero)
                                elif any(x in name_lower for x in ['isat', 'isatfor', 'isatrev', '_i']):
                                    safe_val = 1e-15

                                # Exponents and high injection params
                                elif any(x in name_lower for x in ['exp', 'xhigh']):
                                    safe_val = 1.0

                                # Ideality factors (m values, typically 1-2)
                                elif name_lower.startswith('m') and any(c.isdigit() for c in name_lower):
                                    safe_val = 1.0
                                elif name_lower.startswith('mfor') or name_lower.startswith('mrev'):
                                    safe_val = 1.0

                                # Voltage limits (use reasonable default)
                                elif any(x in name_lower for x in ['vmax', 'vbi', 'vmin', 'vlim', 'vch', 'vbbt']):
                                    safe_val = 0.5

                                # Potentials and built-in voltages
                                elif any(x in name_lower for x in ['phi', 'psi', 'pot']):
                                    safe_val = 0.8

                                # Areas and lengths (small positive)
                                elif any(x in name_lower for x in ['area', 'len', 'perim', 'abs_', 'lss_', 'lgs_', 'abd_', 'lsd_', 'lgd_']):
                                    safe_val = 1e-12

                                # Junction params with _s or _d suffix (symmetric defaults)
                                elif name_lower.endswith('_s') or name_lower.endswith('_d'):
                                    safe_val = 1e-15

                                # Intermediate calculations (h1-h5, tt0-tt2, etc.)
                                elif name_lower in ['h1', 'h2', 'h2d', 'h3', 'h4', 'h5']:
                                    safe_val = 1.0
                                elif name_lower in ['tt0', 'tt1', 'tt2']:
                                    safe_val = 1e-12
                                elif name_lower in ['vj', 'vjlim', 'vjsrh', 'vbbt', 'vav']:
                                    safe_val = 0.0
                                elif name_lower in ['z', 'zinv', 'zfrac']:
                                    safe_val = 1.0
                                elif name_lower == 'idmult':
                                    safe_val = 1.0
                                elif name_lower == 'tmp':
                                    safe_val = 300.15  # Temperature

                                # Conductance values
                                elif 'cond' in name_lower or name_lower.startswith('g'):
                                    safe_val = 1e-12

                                # Capacitance values
                                elif 'cap' in name_lower or name_lower.startswith('c'):
                                    safe_val = 1e-15

                                # Anything else with 'two_' prefix
                                elif name_lower.startswith('two_'):
                                    safe_val = 2.0 * 0.8  # two_psistar ~ 2*psi

                                # Junction params (pmax, ysq, terfc, etc.)
                                elif name_lower in ['pmax', 'ysq', 'terfc', 'erfcpos']:
                                    safe_val = 1.0
                                elif name_lower in ['alphaje']:
                                    safe_val = 1.0
                                elif name_lower in ['vmaxbot', 'vmaxsti', 'vmaxgat']:
                                    safe_val = 0.6
                                elif name_lower in ['vbibot2', 'vbisti2', 'vbigat2', 'vbibot2r', 'vbisti2r', 'vbigat2r']:
                                    safe_val = 0.8
                                elif name_lower in ['pbot2', 'psti2', 'pgat2']:
                                    safe_val = 0.5
                                elif name_lower in ['i1_cor', 'i2_cor', 'i3_cor', 'i4_cor', 'i5_cor']:
                                    safe_val = 1.0
                                elif name_lower in ['m0_rev', 'mcor_rev']:
                                    safe_val = 1.0

                                # Fall back to 1.0 for any remaining hidden_state
                                if safe_val is None:
                                    safe_val = 1.0

                                all_inputs[dev_idx, param_idx] = safe_val
                                handled = True
                                if dev_idx == 0:
                                    logger.debug(f"PSP103 hidden_state {name} (idx={param_idx}) set to safe default: {safe_val}")

                            # Debug: log truly unhandled hidden_state params (only for first device)
                            if not handled and dev_idx == 0:
                                logger.warning(f"Unhandled PSP103 hidden_state: {name} (idx={param_idx})")
                    # Other hidden_state stay at 0

            device_contexts.append({
                'name': dev['name'],
                'node_map': node_map,
                'ext_nodes': ext_nodes,
                'voltage_node_pairs': voltage_node_pairs,
            })

        return jnp.asarray(all_inputs), voltage_indices, device_contexts

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

            _, _, stamp_indices, _, _ = static_inputs_cache[model_type]
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
                    model_defaults = self.MODEL_PARAM_DEFAULTS.get(model_type, {})
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
        """Extract analysis parameters from control block."""
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

            step = self.parse_spice_number(step_match.group(1)) if step_match else 1e-6
            stop = self.parse_spice_number(stop_match.group(1)) if stop_match else 1e-3

            self.analysis_params = {
                'type': 'tran',
                'step': step,
                'stop': stop,
                'icmode': icmode_match.group(1) if icmode_match else 'op',
            }
        else:
            # Default values
            self.analysis_params = {
                'type': 'tran',
                'step': 1e-6,
                'stop': 1e-3,
                'icmode': 'op',
            }

        logger.debug(f"Analysis: {self.analysis_params}")

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

    def to_mna_system(self) -> MNASystem:
        """Convert parsed devices to MNASystem for production analysis.

        Returns:
            MNASystem ready for simulation
        """
        # Invert node_names to get index->name mapping
        index_to_name = {v: k for k, v in self.node_names.items()}

        system = MNASystem(
            num_nodes=self.num_nodes,
            node_names=self.node_names,
            ground_node=0
        )

        for dev in self.devices:
            # Get terminal names from node indices
            terminals = [index_to_name.get(n, str(n)) for n in dev['nodes']]

            device_info = DeviceInfo(
                name=dev['name'],
                model_name=dev['model'],
                terminals=terminals,
                node_indices=dev['nodes'],
                params=dev['params']
            )
            system.devices.append(device_info)

        return system

    @profile
    def run_transient(self, t_stop: Optional[float] = None, dt: Optional[float] = None,
                      max_steps: int = 10000, use_sparse: Optional[bool] = None,
                      backend: Optional[str] = None,
                      use_scan: bool = False,
                      use_while_loop: bool = False,
                      profile_config: Optional['ProfileConfig'] = None) -> Tuple[jax.Array, Dict[int, jax.Array], Dict]:
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
            (times, voltages, stats) tuple where:
            - times: array of time points
            - voltages: dict mapping node index to voltage array
            - stats: dict with convergence info (total_timesteps, non_converged_count, etc.)
        """
        logger.debug("importing gpu backend")
        from jax_spice.analysis.gpu_backend import select_backend, is_gpu_available
        logger.debug("imported gpu backend")

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
        logger.debug("selecting gpu backend")
        if backend is None or backend == "auto":
            backend = select_backend(self.num_nodes)

        logger.info(f"Running transient: t_stop={t_stop:.2e}s, dt={dt:.2e}s, backend={backend}")

        # Use hybrid solver if we have OpenVAF devices
        if self._has_openvaf_devices:
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
                logger.info(f"Using BCOO/BCSR sparse solver ({self.num_nodes} nodes, OpenVAF devices)")
                # Use BCOO/BCSR + spsolve (direct sparse solver)
                # This is more robust for circuit simulation than matrix-free GMRES
                return self._run_transient_hybrid(t_stop, dt, backend=backend, use_dense=False)
            else:
                logger.info("Using dense hybrid solver (OpenVAF devices detected)")
                return self._run_transient_hybrid(t_stop, dt, backend=backend, use_dense=True)

        # Convert to MNA system
        logger.debug("Getting mna system")
        system = self.to_mna_system()

        # Run production transient analysis with backend selection
        logger.debug("Running transient analysis")
        times, voltages_array, stats = transient_analysis_jit(
            system=system,
            t_stop=t_stop,
            t_step=dt,
            t_start=0.0,
            backend=backend,
        )

        logger.debug("Creating voltage dict")
        # Create voltage dict from JAX arrays
        voltages = {}
        for i in range(self.num_nodes):
            if i < voltages_array.shape[1]:
                voltages[i] = voltages_array[:, i]
            else:
                voltages[i] = jnp.zeros(len(times))

        logger.info(f"Completed: {len(times)} timesteps, {stats.get('iterations', 'N/A')} total NR iterations")

        return times, voltages, stats

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

    def _get_psp103_collapse_pairs(self, device_params: Dict[str, float]) -> List[Tuple[int, int]]:
        """Determine which PSP103 node pairs should collapse based on parameters.

        PSP103 has 7 CollapsableR macro instances that control node collapse:
        - (1, 5) G-GP: Controlled by RG_i (gate resistance)
        - (2, 6) S-SI: Controlled by RSE_i (source resistance)
        - (0, 7) D-DI: Controlled by RDE_i (drain resistance)
        - (8, 9) BP-BI: Controlled by RBULK_i (bulk resistance)
        - (10, 9) BS-BI: Controlled by RJUNS_i (source junction resistance)
        - (11, 9) BD-BI: Controlled by RJUND_i (drain junction resistance)
        - (3, 9) B-BI: Controlled by RWELL_i (well resistance)

        The CollapsableR macro collapses when R=0: if (R > 0) I<+G*V else V<+0

        Based on VACASK behavior analysis, even when all R=0, the B-BI pair (3,9)
        doesn't collapse. This keeps BI as a separate internal node where BP/BS/BD
        collapse to it, giving 2 internal nodes per device (NOI and BI).

        Args:
            device_params: Device instance parameters

        Returns:
            List of (node_idx, collapse_to_idx) pairs to apply
        """
        # PSP103 resistance parameters and their controlling pairs
        # Format: (pair, resistance_params) where resistance is computed from params
        # Using simplified checks - if any contributing param is non-zero, don't collapse
        collapse_controls = [
            ((1, 5), ['rgo', 'rint', 'rvpoly', 'rshg']),  # G-GP: RG_i
            ((2, 6), ['nrs', 'rsh']),                      # S-SI: RSE_i = NRS * RSH_i
            ((0, 7), ['nrd', 'rsh']),                      # D-DI: RDE_i = NRD * RSHD_i
            ((8, 9), ['rbulko']),                          # BP-BI: RBULK_i = NF * RBULKO
            ((10, 9), ['rjunso']),                         # BS-BI: RJUNS_i = NF * RJUNSO
            ((11, 9), ['rjundo']),                         # BD-BI: RJUND_i = NF * RJUNDO
            # Note: B-BI (3, 9) controlled by RWELL_i is intentionally excluded
            # Based on VACASK behavior, this pair doesn't collapse even when RWELLO=0
            # This keeps BI as a separate internal node
        ]

        pairs = []
        for pair, param_names in collapse_controls:
            # Check if all controlling parameters are effectively zero
            should_collapse = True
            for pname in param_names:
                val = device_params.get(pname, 0.0)
                if val != 0.0:
                    should_collapse = False
                    break

            if should_collapse:
                pairs.append(pair)

        return pairs

    def _get_model_collapse_pairs(self, model_type: str, model_params: Dict[str, float],
                                     model_nodes: List[str]) -> List[Tuple[int, int]]:
        """Get collapse pairs based on model definition and parameters.

        Instead of blindly trusting OpenVAF's collapsible_pairs, we determine
        which nodes should collapse based on our understanding of the model.

        Args:
            model_type: The model type (e.g., 'psp103', 'diode')
            model_params: Model parameters dictionary
            model_nodes: List of node names from the compiled model

        Returns:
            List of (node_idx, collapse_to_idx) pairs
        """
        # PSP103 uses parameter-dependent collapse logic
        if model_type == 'psp103':
            return self._get_psp103_collapse_pairs(model_params)

        # For other models, use OpenVAF's collapsible_pairs
        compiled = self._compiled_models.get(model_type)
        if compiled:
            return compiled.get('collapsible_pairs', [])

        return []

    def _get_model_params_for_collapse(self, model_type: str) -> Dict[str, float]:
        """Get model parameters relevant for collapse decision.

        Searches through devices to find model parameters.
        """
        for dev in self.devices:
            if dev.get('model') == model_type and dev.get('is_openvaf'):
                return dev.get('params', {})
        return {}

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

        For PSP103, collapse decisions are made per-device based on device parameters.
        For other models, collapse pairs are shared across all devices of that type.

        Returns:
            (total_nodes, device_internal_nodes) where device_internal_nodes maps
            device name to dict of internal node name -> global index
        """
        n_external = self.num_nodes
        next_internal = n_external
        device_internal_nodes = {}

        # Cache collapse roots for non-PSP103 models (shared across devices)
        collapse_roots_cache: Dict[str, Dict[int, int]] = {}

        for dev in self.devices:
            if not dev.get('is_openvaf'):
                continue

            model_type = dev['model']
            compiled = self._compiled_models.get(model_type)
            if not compiled:
                continue

            model_nodes = compiled['nodes']
            n_model_nodes = len(model_nodes)

            # Get collapse pairs based on model definition and device parameters
            # For PSP103, compute per-device to handle parameter variations
            if model_type == 'psp103':
                device_params = dev.get('params', {})
                collapse_pairs = self._get_psp103_collapse_pairs(device_params)
                collapse_roots = self._compute_collapse_roots(collapse_pairs, n_model_nodes)
            else:
                # Other models: cache collapse roots per model type
                if model_type not in collapse_roots_cache:
                    model_params = self._get_model_params_for_collapse(model_type)
                    collapse_pairs = self._get_model_collapse_pairs(
                        model_type, model_params, model_nodes
                    )
                    collapse_roots_cache[model_type] = self._compute_collapse_roots(
                        collapse_pairs, n_model_nodes
                    )
                collapse_roots = collapse_roots_cache[model_type]

            # Determine if last node is a branch current node (skip it if so)
            # Branch current nodes have names like 'br[Branch(BranchId(N))]'
            last_is_branch = n_model_nodes > 0 and model_nodes[-1].startswith('br[')
            n_internal_end = n_model_nodes - 1 if last_is_branch else n_model_nodes

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
                               use_dense: bool = True) -> Tuple[jax.Array, Dict[int, jax.Array], Dict]:
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
            logger.info("Building source function")
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
                logger.debug(f"Getting compiled model for {model_type}")
                compiled = self._compiled_models.get(model_type)
                logger.debug(f"Got model:\n  module: {compiled['module']}\n  translator: {compiled['translator']}\n  jax_fn_arrary: {compiled['jax_fn_array']}\n  vmapped_fn: {compiled['vmapped_fn']}\n  array_metadata size: {len(compiled['array_metadata'])}")
                if compiled and 'vmapped_fn' in compiled:
                    vmapped_fns[model_type] = compiled['vmapped_fn']
                    logger.debug(f"Preparing static inputs: {model_type}")
                    static_inputs, voltage_indices, device_contexts = self._prepare_static_inputs(
                        model_type, openvaf_by_type[model_type], device_internal_nodes, ground
                    )
                    # Pre-compute stamp index mapping (once per model type)
                    logger.debug(f"building stamp index mapping for {model_type}")
                    stamp_indices = self._build_stamp_index_mapping(
                        model_type, device_contexts, ground
                    )
                    # Pre-compute voltage node arrays for vectorized update
                    n_devices = len(device_contexts)
                    n_voltages = len(voltage_indices)
                    # Build arrays directly from list comprehension (setup phase only)
                    logger.debug("building voltage model")
                    voltage_node1 = jnp.array([
                        [n1 for n1, n2 in ctx['voltage_node_pairs']]
                        for ctx in device_contexts
                    ], dtype=jnp.int32)
                    voltage_node2 = jnp.array([
                        [n2 for n1, n2 in ctx['voltage_node_pairs']]
                        for ctx in device_contexts
                    ], dtype=jnp.int32)


                    logger.debug("fetching static inputs")
                    if backend == "gpu":
                        with jax.default_device(device):
                            static_inputs = jnp.array(static_inputs, dtype=dtype)
                    else:
                        static_inputs = jnp.array(static_inputs, dtype=jnp.float64)
                    static_inputs_cache[model_type] = (
                        static_inputs, voltage_indices, stamp_indices, voltage_node1, voltage_node2
                    )
                    n_devs = len(openvaf_by_type[model_type])
                    logger.info(f"Prepared {model_type}: {n_devs} devices, stamp indices cached")

            # Pre-compute source device stamp indices
            logger.debug("Precomputing source device data")
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
            # Create GPU-resident build_system function (closure captures all static data)
            build_system_fn = self._make_gpu_resident_build_system_fn(
                source_device_data, vmapped_fns, static_inputs_cache, n_unknowns, use_dense
            )

            # JIT compile build_system_fn separately (layered compilation)
            # JAX handles nested JIT naturally - inner JIT is compiled as a call
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
                    build_system_jit, n_nodes, noi_indices=noi_indices,
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
                J_bcoo_probe, _, _ = build_system_fn(V_init, vsource_init, isource_init, Q_init, 0.0)

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
                        noi_indices=noi_indices,
                        max_iterations=MAX_NR_ITERATIONS, abstol=DEFAULT_ABSTOL, max_step=1.0
                    )
                else:
                    # Fallback to JAX spsolve (QR factorization, no caching)
                    nr_solve = self._make_sparse_jit_compiled_solver(
                        build_system_jit, n_nodes, nse, noi_indices=noi_indices,
                        max_iterations=MAX_NR_ITERATIONS, abstol=DEFAULT_ABSTOL, max_step=1.0
                    )

            # Cache JIT-wrapped solver and build_system (JAX handles compilation automatically)
            self._cached_nr_solve = nr_solve
            self._cached_solver_key = cache_key
            self._cached_build_system = build_system_jit
            logger.info(f"Cached {'dense' if use_dense else 'sparse'} NR solver")

        # Compute initial condition based on icmode
        icmode = self.analysis_params.get('icmode', 'op')
        if icmode == 'op':
            V = self._compute_dc_operating_point(
                n_nodes=n_nodes,
                n_vsources=n_vsources,
                n_isources=n_isources,
                nr_solve=nr_solve,
                backend=backend,
                use_dense=use_dense,
                device_internal_nodes=device_internal_nodes
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

        times = []

        logger.info(f"initialising voltages: {n_external}")
        voltages = {i: [] for i in range(n_external)}

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
            _, _, Q_prev = self._cached_build_system(V, vsource_dc, isource_dc, jnp.zeros(n_unknowns), 0.0)
            logger.debug(f"  Initialized Q_prev from DC operating point (max|Q|={float(jnp.max(jnp.abs(Q_prev))):.2e})")
        else:
            Q_prev = jnp.zeros(n_unknowns, dtype=jnp.float64)

        # Use integer-based iteration to avoid floating-point comparison issues
        # This ensures Python loop and lax.scan produce the same number of timesteps
        num_timesteps = int(round(t_stop / dt)) + 1

        logger.info(f"Starting NR iteration ({num_timesteps} timesteps, inv_dt={inv_dt:.2e})")
        for step_idx in range(num_timesteps):
            t = step_idx * dt
            logger.debug(f"Step time:{t}")
            source_values = source_fn(t)
            # Build source value arrays once per timestep (Python loop here, not in NR loop)
            vsource_vals, isource_vals = build_source_arrays(source_values)

            # GPU-resident NR solve - JIT compiled, runs on GPU via lax.while_loop
            # Backward Euler: f = f_resist + (Q - Q_prev)/dt, J = J_resist + C/dt
            V_new, iterations, converged, max_f, Q = nr_solve(V, vsource_vals, isource_vals, Q_prev, inv_dt)

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

            # Record state
            times.append(t)
            for i in range(n_external):
                voltages[i].append(float(V[i]))

            V_prev = V

        # Build stats dict
        stats = {
            'total_timesteps': len(times),
            'total_nr_iterations': total_nr_iters,
            'non_converged_count': len(non_converged_steps),
            'non_converged_steps': non_converged_steps,
            'convergence_rate': 1.0 - len(non_converged_steps) / max(len(times), 1),
        }

        logger.info(f"Completed: {len(times)} timesteps, {total_nr_iters} total NR iterations")
        if non_converged_steps:
            logger.info(f"  Non-converged: {len(non_converged_steps)} steps ({100*(1-stats['convergence_rate']):.1f}%)")

        return jnp.array(times), {k: jnp.array(v) for k, v in voltages.items()}, stats

    def _compute_dc_operating_point(self, n_nodes: int, n_vsources: int, n_isources: int,
                                     nr_solve: Callable, backend: str = "cpu",
                                     use_dense: bool = True,
                                     max_iterations: int = 100,
                                     device_internal_nodes: Optional[Dict[str, Dict[str, int]]] = None) -> jax.Array:
        """Compute DC operating point as initial condition for transient analysis.

        This finds the steady-state solution where all node currents sum to zero,
        using the same NR solver as transient analysis but with source values at t=0.

        For capacitors, the NR solver includes C/dt terms, but starting from a good
        initial guess (VDD nodes initialized to supply voltage) and running until
        convergence will find the DC equilibrium.

        Args:
            n_nodes: Number of nodes in the system
            n_vsources: Number of voltage sources
            n_isources: Number of current sources
            nr_solve: The cached NR solver function
            backend: 'gpu' or 'cpu'
            use_dense: Whether using dense solver
            max_iterations: Maximum DC NR iterations
            device_internal_nodes: Map of device name -> {node_name: circuit_node_idx}

        Returns:
            DC operating point voltages (shape: [n_nodes])
        """
        logger.info("Computing DC operating point (icmode='op')...")

        # Find VDD value from voltage sources
        vdd_value = 0.0
        for dev in self.devices:
            if dev['model'] == 'vsource':
                params = dev['params']
                dc_val = params.get('dc', 0.0)
                if dc_val > vdd_value:
                    vdd_value = dc_val

        # Initialize V with a better starting point for convergence:
        # - Ground (node 0) = 0V
        # - VDD nodes = vdd_value
        # - Other signal nodes = VDD/2 (mid-rail for CMOS)
        # This helps avoid the degenerate solution where all MOSFETs are in cutoff
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

        # Initialize PSP103 internal nodes properly:
        # 1. NOI nodes (node4) to 0V - they have extremely high conductance to ground
        # 2. Body internal nodes (node8-11 = BP/BI/BS/BD) to match external B terminal
        #    For PMOS with B=VDD, these must be at VDD for proper body-source voltage
        if device_internal_nodes:
            noi_nodes_initialized = 0
            body_nodes_initialized = 0

            # Build map from device name to external nodes
            device_external_nodes = {}
            for dev in self.devices:
                device_external_nodes[dev['name']] = dev.get('nodes', [])

            for dev_name, internal_nodes in device_internal_nodes.items():
                # NOI node initialization (node4) - must be 0V (has 1e40 conductance to ground)
                if 'node4' in internal_nodes:
                    noi_idx = internal_nodes['node4']
                    V = V.at[noi_idx].set(0.0)
                    noi_nodes_initialized += 1

                # Body internal nodes initialization (node8-11 = BP/BI/BS/BD)
                # These should match the external B terminal voltage
                ext_nodes = device_external_nodes.get(dev_name, [])
                if len(ext_nodes) >= 4:  # PSP103 has [D, G, S, B]
                    b_circuit_node = ext_nodes[3]  # External B terminal
                    b_voltage = float(V[b_circuit_node])  # Get B voltage (VDD for PMOS)

                    for body_node_name in ['node8', 'node9', 'node10', 'node11']:
                        if body_node_name in internal_nodes:
                            body_idx = internal_nodes[body_node_name]
                            # Only update if not already set (avoid overwriting ground or VDD)
                            if body_idx > 0 and abs(V[body_idx] - mid_rail) < 0.01:
                                V = V.at[body_idx].set(b_voltage)
                                body_nodes_initialized += 1

            if noi_nodes_initialized > 0:
                logger.debug(f"  Initialized {noi_nodes_initialized} NOI nodes to 0V")
            if body_nodes_initialized > 0:
                logger.debug(f"  Initialized {body_nodes_initialized} body internal nodes to match B terminal")

        logger.debug(f"  Initial V: ground=0V, VDD={vdd_value}V, others={mid_rail}V")

        # Get DC source values (at t=0)
        # - Voltage sources: use their DC value
        # - Current sources: use val0 (before pulse starts) or DC value
        vsource_vals = jnp.zeros(n_vsources, dtype=jnp.float64)
        isource_vals = jnp.zeros(n_isources, dtype=jnp.float64)

        # Build source values at t=0
        vsource_idx = 0
        isource_idx = 0
        for dev in self.devices:
            if dev['model'] == 'vsource':
                params = dev['params']
                dc_val = params.get('dc', 0.0)
                vsource_vals = vsource_vals.at[vsource_idx].set(float(dc_val))
                vsource_idx += 1
            elif dev['model'] == 'isource':
                params = dev['params']
                source_type = str(params.get('type', 'dc')).lower()
                if source_type == 'pulse':
                    # At t=0 (before delay), pulse is at val0
                    dc_val = params.get('val0', 0.0)
                else:
                    dc_val = params.get('dc', 0.0)
                isource_vals = isource_vals.at[isource_idx].set(float(dc_val))
                isource_idx += 1

        # Run NR iterations until convergence
        # For DC: inv_dt=0 means no reactive terms (dQ/dt = 0 at steady state)
        # Q_prev doesn't matter when inv_dt=0, but needs correct shape (n_unknowns,)
        n_unknowns = n_nodes - 1
        Q_prev = jnp.zeros(n_unknowns, dtype=jnp.float64)
        inv_dt = 0.0  # DC analysis - no time derivative terms

        converged = False
        for iteration in range(max_iterations):
            V_new, nr_iters, is_converged, max_f, _ = nr_solve(V, vsource_vals, isource_vals, Q_prev, inv_dt)

            if bool(is_converged):
                converged = True
                V = V_new
                logger.info(f"  DC operating point converged in {iteration + 1} outer iterations "
                           f"({int(nr_iters)} NR iters, residual={float(max_f):.2e})")
                break

            # Check if we're making progress
            delta = jnp.max(jnp.abs(V_new - V))
            V = V_new

            # Clamp NOI nodes to 0V - they have 1e40 conductance to ground
            # which causes numerical issues if they drift from 0V
            if device_internal_nodes:
                for dev_name, internal_nodes in device_internal_nodes.items():
                    if 'node4' in internal_nodes:  # NOI is node4 in PSP103
                        noi_idx = internal_nodes['node4']
                        V = V.at[noi_idx].set(0.0)

            if float(delta) < 1e-9:
                converged = True
                logger.info(f"  DC operating point converged (delta < 1e-9) in {iteration + 1} iterations")
                break

        if not converged:
            logger.warning(f"  DC operating point did not fully converge after {max_iterations} iterations "
                          f"(max_f={float(max_f):.2e})")

        # Log some key node voltages
        n_external = self.num_nodes
        logger.info(f"  DC solution: {min(n_external, 5)} node voltages:")
        for i in range(min(n_external, 5)):
            # Find node name for index
            name = next((n for n, idx in self.node_names.items() if idx == i), str(i))
            logger.info(f"    Node {name} (idx {i}): {float(V[i]):.6f}V")

        return V

    def _run_transient_while_loop(self, t_stop: float, dt: float,
                                   backend: str = "cpu",
                                   use_dense: bool = True,
                                   profile_config: Optional['ProfileConfig'] = None) -> Tuple[jax.Array, Dict[int, jax.Array], Dict]:
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
                backend=backend,
                use_dense=use_dense,
                device_internal_nodes=device_internal_nodes
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
                def run_simulation_with_outputs(V_init, Q_init, all_vsource, all_isource):
                    """Run simulation with time-varying sources using lax.scan.

                    Carry includes both V and Q for reactive term tracking.
                    Uses backward Euler: f = f_resist + (Q - Q_prev)/dt
                    """
                    def step_fn(carry, source_vals):
                        V, Q_prev = carry
                        vsource_vals, isource_vals = source_vals
                        V_new, iterations, converged, max_f, Q = nr_solve_fn(
                            V, vsource_vals, isource_vals, Q_prev, inv_dt_val
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
            _, _, Q0 = self._cached_build_system(V0, vsource_dc, isource_dc, jnp.zeros(n_unknowns), 0.0)
            logger.debug(f"  Initialized Q0 from DC operating point (max|Q0|={float(jnp.max(jnp.abs(Q0))):.2e})")
        else:
            Q0 = jnp.zeros(n_unknowns, dtype=jnp.float64)

        # Run the simulation (with optional profiling of just the core loop)
        logger.info("Running lax.scan simulation...")
        t0 = time_module.perf_counter()
        if profile_config:
            with profile_section("lax_scan_simulation", profile_config):
                all_V, all_iters, all_converged = run_simulation_with_outputs(V0, Q0, all_vsource_vals, all_isource_vals)
                jax.block_until_ready(all_V)
                # Measure time BEFORE profile_section.__exit__ (which saves trace to disk)
                t1 = time_module.perf_counter()
        else:
            all_V, all_iters, all_converged = run_simulation_with_outputs(V0, Q0, all_vsource_vals, all_isource_vals)
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

        return times, voltages, stats

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
        by_type: Dict[str, List[Dict]] = {}
        for dev in source_devices:
            model = dev['model']
            if model in ('vsource', 'isource'):
                if model not in by_type:
                    by_type[model] = []
                by_type[model].append(dev)

        result = {}
        for model, devs in by_type.items():
            n = len(devs)
            # Extract node indices as JAX arrays
            node_p = jnp.array([d['nodes'][0] for d in devs], dtype=jnp.int32)
            node_n = jnp.array([d['nodes'][1] for d in devs], dtype=jnp.int32)
            names = [d['name'] for d in devs]

            # Pre-compute stamp templates for 2-terminal devices
            # Residual indices: [p-1, n-1] for each device, -1 if grounded
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
    ) -> Callable:
        """Create a JIT-compilable function that builds J and f from V.

        This closure captures all the static data structures (stamp indices,
        vmapped functions) and returns a function that can be traced by JAX.
        The returned function is suitable for use inside lax.while_loop.

        Args:
            source_device_data: Pre-computed source device stamp templates
            vmapped_fns: Dict of vmapped OpenVAF functions per model type
            static_inputs_cache: Dict of (static_inputs, voltage_indices, stamp_indices,
                                          voltage_node1, voltage_node2) per model type
            n_unknowns: Number of unknowns (total nodes - 1 for ground)
            use_dense: Whether to use dense or sparse matrix assembly

        Returns:
            Function build_system(V, vsource_vals, isource_vals, Q_prev, inv_dt) -> (J, f, Q)

            For DC analysis: pass inv_dt=0.0 and Q_prev=zeros (reactive terms ignored)
            For transient: pass inv_dt=1/dt and Q_prev from previous timestep
        """
        from jax.experimental.sparse import BCOO, BCSR
        from jax_spice.analysis.sparse import sparse_solve_csr

        # Capture model types as static list (unrolled at trace time)
        model_types = list(static_inputs_cache.keys())

        def build_system(V: jax.Array, vsource_vals: jax.Array, isource_vals: jax.Array,
                        Q_prev: jax.Array, inv_dt: float | jax.Array
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
                static_inputs, voltage_indices, stamp_indices, voltage_node1, voltage_node2 = \
                    static_inputs_cache[model_type]
                vmapped_fn = vmapped_fns[model_type]

                # Vectorized voltage update
                voltage_updates = V[voltage_node1] - V[voltage_node2]
                batch_inputs = static_inputs.at[:, jnp.array(voltage_indices)].set(voltage_updates)

                # Batched device evaluation - now returns 4 arrays
                batch_res_resist, batch_res_react, batch_jac_resist, batch_jac_react = vmapped_fn(batch_inputs)

                # Mask out huge residuals from internal nodes with 1e40 conductance
                # These arise from numerical noise × 1e40 = 1e20+ residuals
                # PSP103 NOI and related internal nodes can produce such values
                huge_res_mask = jnp.abs(batch_res_resist) > 1e20
                batch_res_resist = jnp.where(huge_res_mask, 0.0, batch_res_resist)
                batch_res_react = jnp.where(huge_res_mask, 0.0, batch_res_react)

                # Also mask huge Jacobian entries (> 1e20) from NOI and similar internal nodes
                # These 1e40 conductance values pollute the circuit Jacobian and cause instability
                huge_jac_mask = jnp.abs(batch_jac_resist) > 1e20
                batch_jac_resist = jnp.where(huge_jac_mask, 0.0, batch_jac_resist)
                batch_jac_react = jnp.where(huge_jac_mask, 0.0, batch_jac_react)

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
                # Add regularization (1e-9 matches sparse solver)
                J = J + 1e-9 * jnp.eye(n_unknowns, dtype=jnp.float64)
            else:
                # Sparse path - build BCOO sparse matrix
                from jax.experimental.sparse import BCOO

                # Add diagonal regularization entries
                diag_idx = jnp.arange(n_unknowns, dtype=jnp.int32)
                all_j_rows = jnp.concatenate([all_j_rows, diag_idx])
                all_j_cols = jnp.concatenate([all_j_cols, diag_idx])
                # Large regularization needed for GPU spsolve (stricter than scipy)
                all_j_vals = jnp.concatenate([all_j_vals, jnp.full(n_unknowns, 1e-3)])

                # Build BCOO with duplicates (BCSR.from_bcoo handles them)
                indices = jnp.stack([all_j_rows, all_j_cols], axis=1)
                J = BCOO((all_j_vals, indices), shape=(n_unknowns, n_unknowns))

            return J, f, Q

        return build_system

    def _make_jit_compiled_solver(
        self,
        build_system_jit: Callable,
        n_nodes: int,
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
            build_system_jit: JIT-wrapped function (V, vsource_vals, isource_vals, Q_prev, inv_dt) -> (J, f, Q)
            n_nodes: Total node count including ground (V.shape[0])
            noi_indices: Optional array of NOI node indices to constrain to 0V
            max_iterations: Maximum NR iterations
            abstol: Absolute tolerance for convergence
            max_step: Maximum voltage step per iteration

        Returns:
            JIT-compiled function: (V, vsource_vals, isource_vals, Q_prev, inv_dt) -> (V, iters, converged, max_f, Q)
        """
        # Pre-compute residual mask if we have NOI nodes
        # NOI nodes have indices in the full V vector, but residuals use 0-indexed (ground excluded)
        # So NOI residual index = NOI node index - 1
        if noi_indices is not None and len(noi_indices) > 0:
            # Create mask: True for nodes to include in convergence check, False for NOI
            n_unknowns = n_nodes - 1
            residual_mask = jnp.ones(n_unknowns, dtype=jnp.bool_)
            noi_residual_indices = noi_indices - 1  # Convert to residual indices
            # Set mask to False for NOI residuals
            residual_mask = residual_mask.at[noi_residual_indices].set(False)
        else:
            residual_mask = None

        def nr_solve(V_init: jax.Array, vsource_vals: jax.Array, isource_vals: jax.Array,
                    Q_prev: jax.Array, inv_dt: float | jax.Array):
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
                J, f, Q = build_system_jit(V, vsource_vals, isource_vals, Q_prev, inv_dt)

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
                if noi_indices is not None and len(noi_indices) > 0:
                    # Convert NOI node indices to residual indices (ground excluded)
                    noi_res_idx = noi_indices - 1

                    # Zero out NOI rows and columns in J
                    n_unknowns = J.shape[0]
                    row_mask = jnp.ones(n_unknowns, dtype=jnp.bool_)
                    row_mask = row_mask.at[noi_res_idx].set(False)

                    # Create identity entries for NOI nodes
                    J_noi_fixed = jnp.where(
                        row_mask[:, None] & row_mask[None, :],  # Non-NOI entries
                        J,
                        jnp.where(
                            jnp.arange(n_unknowns)[:, None] == jnp.arange(n_unknowns)[None, :],  # Diagonal
                            jnp.where(~row_mask[:, None], 1.0, 0.0),  # 1.0 on NOI diagonal
                            0.0  # 0 elsewhere in NOI rows/cols
                        )
                    )
                    # Zero out NOI residuals
                    f_noi_fixed = jnp.where(row_mask, f, 0.0)

                    J = J_noi_fixed
                    f = f_noi_fixed

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
            _, _, Q_final = build_system_jit(V_final, vsource_vals, isource_vals, Q_prev, inv_dt)

            return V_final, iterations, converged, max_f, Q_final

        # Return JIT-wrapped function - compilation happens lazily on first call
        logger.info(f"Creating JIT-compiled NR solver: V({n_nodes}), NOI constrained: {noi_indices is not None}")
        return jax.jit(nr_solve)

    def _make_sparse_jit_compiled_solver(
        self,
        build_system_jit: Callable,
        n_nodes: int,
        nse: int,
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
            noi_indices: Optional array of NOI node indices to constrain to 0V
            max_iterations: Maximum NR iterations
            abstol: Absolute tolerance for convergence
            max_step: Maximum voltage step per iteration

        Returns:
            JIT-compiled sparse solver function: (V, vsrc, isrc, Q_prev, inv_dt) -> (V, iters, converged, max_f, Q)
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
                    Q_prev: jax.Array, inv_dt: float | jax.Array):
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

                # Build sparse system (J_bcoo, f, Q)
                J_bcoo, f, Q = build_system_jit(V, vsource_vals, isource_vals, Q_prev, inv_dt)

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
            _, _, Q_final = build_system_jit(V_final, vsource_vals, isource_vals, Q_prev, inv_dt)

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
            build_system_jit: JIT-wrapped function returning (J_bcoo, f)
            n_nodes: Total node count including ground
            nse: Number of stored elements after summing duplicates
            bcsr_indptr: Pre-computed BCSR row pointers
            bcsr_indices: Pre-computed BCSR column indices
            noi_indices: Optional array of NOI node indices to constrain to 0V
            max_iterations: Maximum NR iterations
            abstol: Absolute tolerance for convergence
            max_step: Maximum voltage step per iteration

        Returns:
            JIT-compiled sparse solver function using Spineax
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

        def nr_solve(V_init: jax.Array, vsource_vals: jax.Array, isource_vals: jax.Array):
            init_state = (
                V_init,
                jnp.array(0, dtype=jnp.int32),
                jnp.array(False),
                jnp.array(jnp.inf),
                jnp.array(jnp.inf),
            )

            def cond_fn(state):
                V, iteration, converged, max_f, max_delta = state
                return jnp.logical_and(~converged, iteration < max_iterations)

            def body_fn(state):
                V, iteration, _, _, _ = state

                # Build sparse system (J_bcoo and f)
                J_bcoo, f = build_system_jit(V, vsource_vals, isource_vals)

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

                return (V_new, iteration + 1, converged, max_f, max_delta)

            V_final, iterations, converged, max_f, max_delta = lax.while_loop(
                cond_fn, body_fn, init_state
            )

            return V_final, iterations, converged, max_f

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
