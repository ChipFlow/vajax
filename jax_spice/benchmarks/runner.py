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

            # Check if model has MULT as a separate param (like PSP103)
            # If so, mfactor sysfun should stay 0 (it's derived internally)
            # If not (like diode), mfactor should be 1.0 as the device multiplier
            has_mult_param = 'MULT' in param_names and param_kinds[param_names.index('MULT')] == 'param'

            # Fill input array for this device (static params only, voltages stay 0)
            for param_idx, (name, kind) in enumerate(zip(param_names, param_kinds)):
                if kind == 'voltage':
                    pass  # Already 0 from np.zeros
                elif kind == 'temperature':
                    # Temperature parameters (e.g., $temperature)
                    all_inputs[dev_idx, param_idx] = 300.15  # ~27°C in Kelvin
                elif kind == 'sysfun':
                    # System functions like mfactor
                    # For models with MULT param (PSP103), leave mfactor=0 (derived from MULT)
                    # For diode (which has no MULT), set mfactor=1.0 (device multiplier)
                    # For resistor/capacitor, leave mfactor=0 to preserve original behavior
                    # (their Jacobian contributions are handled differently)
                    if name.lower() == 'mfactor' and not has_mult_param and model_type == 'diode':
                        all_inputs[dev_idx, param_idx] = params.get('mfactor', 1.0)
                elif kind == 'param':
                    param_lower = name.lower()
                    if param_lower in params:
                        all_inputs[dev_idx, param_idx] = float(params[param_lower])
                    elif name in params:
                        all_inputs[dev_idx, param_idx] = float(params[name])
                    elif 'temperature' in param_lower or name == '$temperature':
                        all_inputs[dev_idx, param_idx] = 300.15
                    elif param_lower in ('tnom', 'tref', 'tr'):
                        all_inputs[dev_idx, param_idx] = 300.0
                    elif param_lower == 'mfactor':
                        all_inputs[dev_idx, param_idx] = params.get('mfactor', 1.0)
                    elif param_lower in model_defaults:
                        # Use model-specific default
                        all_inputs[dev_idx, param_idx] = model_defaults[param_lower]
                    else:
                        all_inputs[dev_idx, param_idx] = 1.0
                # hidden_state, current, param_given stay 0 from np.zeros
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
        internal_name_map = {
            # PSP103 MOSFET nodes
            'GP': 'node4', 'SI': 'node5', 'DI': 'node6', 'BP': 'node7',
            'BS': 'node8', 'BD': 'node9', 'BI': 'node10', 'NOI': 'node11',
            'G': 'node1', 'D': 'node0', 'S': 'node2', 'B': 'node3',
            # Diode nodes (A=anode, C=cathode, CI=internal cathode)
            'A': 'node0', 'C': 'node1', 'CI': 'node2',
        }

        match = re.match(r'V\(([^,)]+)(?:,([^)]+))?\)', name)
        if not match:
            return (ground, ground)

        node1_name = match.group(1).strip()
        node2_name = match.group(2).strip() if match.group(2) else None

        # Resolve node1
        if node1_name in internal_name_map:
            node1_name = internal_name_map[node1_name]
        node1_idx = node_map.get(node1_name, node_map.get(node1_name.lower(), ground))

        # Resolve node2
        if node2_name:
            if node2_name in internal_name_map:
                node2_name = internal_name_map[node2_name]
            node2_idx = node_map.get(node2_name, node_map.get(node2_name.lower(), ground))
        else:
            node2_idx = ground

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
                    if param_lower in params:
                        inputs.append(float(params[param_lower]))
                    elif name in params:
                        inputs.append(float(params[name]))
                    elif 'temperature' in param_lower or name == '$temperature':
                        inputs.append(300.15)
                    elif param_lower in ('tnom', 'tref', 'tr'):
                        inputs.append(300.0)
                    elif param_lower == 'mfactor':
                        inputs.append(params.get('mfactor', 1.0))
                    else:
                        inputs.append(1.0)
                elif kind == 'hidden_state':
                    inputs.append(0.0)
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

            if use_scan:
                # lax.scan version - pre-computes all source values
                logger.info(f"Using lax.scan solver ({self.num_nodes} nodes, "
                           f"{'sparse' if use_sparse else 'dense'})")
                return self._run_transient_fully_jit(t_stop, dt, backend=backend, use_dense=use_dense,
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

    def _should_collapse_all_pairs(self, model_type: str, model_params: Dict[str, float]) -> bool:
        """Determine if all collapsible pairs should be collapsed for a model.

        Always returns True - we always collapse internal nodes as specified by the
        model's collapsible_pairs. This reduces system size significantly for models
        like PSP103 where many internal nodes can be merged.

        Args:
            model_type: The model type (e.g., 'psp103')
            model_params: Model parameters dictionary (unused, kept for API compat)

        Returns:
            True - always collapse
        """
        return True

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

        Returns:
            (total_nodes, device_internal_nodes) where device_internal_nodes maps
            device name to dict of internal node name -> global index
        """
        n_external = self.num_nodes
        next_internal = n_external
        device_internal_nodes = {}

        # Cache collapse decisions and root mappings per model type
        collapse_decisions: Dict[str, bool] = {}
        collapse_roots_cache: Dict[str, Dict[int, int]] = {}

        for dev in self.devices:
            if not dev.get('is_openvaf'):
                continue

            model_type = dev['model']
            compiled = self._compiled_models.get(model_type)
            if not compiled:
                continue

            # Check if we should collapse nodes for this model type
            if model_type not in collapse_decisions:
                model_params = self._get_model_params_for_collapse(model_type)
                collapse_decisions[model_type] = self._should_collapse_all_pairs(
                    model_type, model_params
                )

                # Precompute collapse roots if collapsing
                if collapse_decisions[model_type]:
                    collapsible_pairs = compiled.get('collapsible_pairs', [])
                    n_model_nodes = len(compiled['nodes'])
                    collapse_roots_cache[model_type] = self._compute_collapse_roots(
                        collapsible_pairs, n_model_nodes
                    )

            should_collapse = collapse_decisions[model_type]
            model_nodes = compiled['nodes']
            n_model_nodes = len(model_nodes)

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

            if should_collapse and model_type in collapse_roots_cache:
                collapse_roots = collapse_roots_cache[model_type]

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
            else:
                # No collapse - allocate internal nodes normally
                node_mapping = {}
                for i in range(n_ext_terminals, n_internal_end):
                    node_mapping[i] = next_internal
                    next_internal += 1

            # Build internal_map: model node name -> circuit node index
            internal_map = {}
            for i in range(n_ext_terminals, n_internal_end):
                node_name = model_nodes[i]
                internal_map[node_name] = node_mapping[i]

            device_internal_nodes[dev['name']] = internal_map

        if device_internal_nodes:
            n_internal = next_internal - n_external
            n_collapsed = sum(1 for v in collapse_decisions.values() if v)
            logger.info(f"Allocated {n_internal} internal nodes for {len(device_internal_nodes)} OpenVAF devices")
            if n_collapsed > 0:
                logger.info(f"  Node collapse applied to {n_collapsed} model type(s)")

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

            # Create JIT-compiled NR solver
            if use_dense:
                # Dense solver for small/medium circuits
                nr_solve = self._make_jit_compiled_solver(
                    build_system_jit, n_nodes,
                    max_iterations=MAX_NR_ITERATIONS, abstol=DEFAULT_ABSTOL, max_step=1.0
                )
            else:
                # Sparse solver for large circuits
                # Pre-compute nse by running build_system once to get sparsity pattern
                logger.info("Pre-computing sparse matrix nse...")
                V_init = jnp.zeros(n_nodes, dtype=jnp.float64)
                vsource_init = jnp.zeros(n_vsources, dtype=jnp.float64)
                isource_init = jnp.zeros(n_isources, dtype=jnp.float64)
                J_bcoo_probe, _ = build_system_fn(V_init, vsource_init, isource_init)

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
                        max_iterations=MAX_NR_ITERATIONS, abstol=DEFAULT_ABSTOL, max_step=1.0
                    )
                else:
                    # Fallback to JAX spsolve (QR factorization, no caching)
                    nr_solve = self._make_sparse_jit_compiled_solver(
                        build_system_jit, n_nodes, nse,
                        max_iterations=MAX_NR_ITERATIONS, abstol=DEFAULT_ABSTOL, max_step=1.0
                    )

            # Cache JIT-wrapped solver (JAX handles compilation automatically)
            self._cached_nr_solve = nr_solve
            self._cached_solver_key = cache_key
            logger.info(f"Cached {'dense' if use_dense else 'sparse'} NR solver")

        times = []

        logger.info(f"initialising voltages: {n_external}")
        voltages = {i: [] for i in range(n_external)}

        total_nr_iters = 0
        non_converged_steps = []  # Track (time, max_residual) for non-converged steps

        # Use integer-based iteration to avoid floating-point comparison issues
        # This ensures Python loop and lax.scan produce the same number of timesteps
        num_timesteps = int(round(t_stop / dt)) + 1

        logger.info(f"Starting NR iteration ({num_timesteps} timesteps)")
        for step_idx in range(num_timesteps):
            t = step_idx * dt
            logger.debug(f"Step time:{t}")
            source_values = source_fn(t)
            # Build source value arrays once per timestep (Python loop here, not in NR loop)
            vsource_vals, isource_vals = build_source_arrays(source_values)

            # GPU-resident NR solve - JIT compiled, runs on GPU via lax.while_loop
            V_new, iterations, converged, max_f = nr_solve(V, vsource_vals, isource_vals)

            # Transfer results back to Python for logging/tracking (once per timestep)
            nr_iters = int(iterations)
            is_converged = bool(converged)
            residual = float(max_f)

            V = V_new
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
            self._run_transient_hybrid(t_stop=0, dt=dt, backend=backend, use_dense=use_dense)

        setup = self._transient_setup_cache
        n_total = setup['n_total']
        n_unknowns = setup['n_unknowns']
        source_device_data = setup['source_device_data']

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

        # Initial state
        V0 = jnp.zeros(n_nodes, dtype=jnp.float64)

        # Cache key for the scan function
        # Note: Does NOT include num_timesteps - lax.scan handles variable-length inputs
        scan_cache_key = (n_nodes, n_vsources, n_isources, n_external, use_dense)

        if hasattr(self, '_cached_scan_fn') and self._cached_scan_key == scan_cache_key:
            run_simulation_with_outputs = self._cached_scan_fn
            logger.info("Reusing cached lax.scan simulation function")
        else:
            # Create and cache the scan function
            # Source values are passed per-timestep via lax.scan's xs argument
            def make_scan_fn(nr_solve_fn, n_ext):
                @jax.jit
                def run_simulation_with_outputs(V_init, all_vsource, all_isource):
                    """Run simulation with time-varying sources using lax.scan."""
                    def step_fn(V, source_vals):
                        vsource_vals, isource_vals = source_vals
                        V_new, iterations, converged, max_f = nr_solve_fn(V, vsource_vals, isource_vals)
                        return V_new, (V_new[:n_ext], iterations, converged)

                    # Stack source arrays for scan input
                    source_inputs = (all_vsource, all_isource)
                    _, (all_V, all_iters, all_converged) = jax.lax.scan(
                        step_fn, V_init, source_inputs
                    )
                    return all_V, all_iters, all_converged
                return run_simulation_with_outputs

            run_simulation_with_outputs = make_scan_fn(nr_solve, n_external)
            self._cached_scan_fn = run_simulation_with_outputs
            self._cached_scan_key = scan_cache_key
            logger.info("Created and cached lax.scan simulation function")

        # Run the simulation (with optional profiling of just the core loop)
        logger.info("Running lax.scan simulation...")
        t0 = time_module.perf_counter()
        if profile_config:
            with profile_section("lax_scan_simulation", profile_config):
                all_V, all_iters, all_converged = run_simulation_with_outputs(V0, all_vsource_vals, all_isource_vals)
                jax.block_until_ready(all_V)
        else:
            all_V, all_iters, all_converged = run_simulation_with_outputs(V0, all_vsource_vals, all_isource_vals)
            jax.block_until_ready(all_V)
        total_time = time_module.perf_counter() - t0

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

    def _run_transient_fully_jit(self, t_stop: float, dt: float,
                                  backend: str = "cpu",
                                  use_dense: bool = True,
                                  profile_config: Optional[ProfileConfig] = None) -> Tuple[jax.Array, Dict[int, jax.Array], Dict]:
        """Fully JIT-compiled transient analysis using lax.scan.

        This version pre-computes all source values and uses lax.scan for the
        timestep loop, eliminating Python loop overhead entirely.

        Args:
            t_stop: Simulation stop time
            dt: Time step
            backend: 'gpu' or 'cpu' for device evaluation
            use_dense: If True, use dense solver; if False, use sparse solver
            profile_config: If provided, profile just the core simulation loop
        """
        from jax_spice.analysis.gpu_backend import get_device, get_default_dtype
        import time as time_module

        ground = 0
        device = get_device(backend)
        dtype = get_default_dtype(backend)

        # Reuse cached setup from _run_transient_hybrid
        setup_cache_key = f"{self.num_nodes}_{len(self.devices)}_{use_dense}_{backend}"

        if (self._transient_setup_cache is None or
            self._transient_setup_key != setup_cache_key):
            # Build setup by calling hybrid version with 0 steps
            logger.info("Building transient setup...")
            self._run_transient_hybrid(t_stop=0, dt=dt, backend=backend, use_dense=use_dense)

        setup = self._transient_setup_cache
        n_total = setup['n_total']
        n_unknowns = setup['n_unknowns']
        source_fn = setup['source_fn']
        source_device_data = setup['source_device_data']
        vmapped_fns = setup['vmapped_fns']
        static_inputs_cache = setup['static_inputs_cache']

        n_external = self.num_nodes
        n_vsources = len(source_device_data.get('vsource', {}).get('names', []))
        n_isources = len(source_device_data.get('isource', {}).get('names', []))
        n_nodes = n_unknowns + 1

        # Generate all time points
        num_timesteps = int(t_stop / dt)
        times = jnp.linspace(0.0, t_stop, num_timesteps + 1)
        logger.info(f"Fully JIT transient: {num_timesteps} timesteps, {n_total} nodes")

        # Pre-compute source values for all timesteps
        logger.info("Pre-computing source values for all timesteps...")
        t0 = time_module.perf_counter()

        # Build source value arrays for each timestep
        vsource_names = source_device_data.get('vsource', {}).get('names', [])
        isource_names = source_device_data.get('isource', {}).get('names', [])
        vsource_dc = source_device_data.get('vsource', {}).get('dc', jnp.array([]))
        isource_dc = source_device_data.get('isource', {}).get('dc', jnp.array([]))

        # Pre-compute source values using vectorized evaluation
        # Source functions are already JAX-compatible (use jnp.where)
        if n_vsources > 0:
            all_vsource = []
            for i, name in enumerate(vsource_names):
                src_fn = {k: v for d in self.devices
                         for k, v in [(d['name'], self._get_source_fn_for_device(d))]
                         if d['name'] == name}.get(name)
                if src_fn is not None:
                    # Vectorize over time
                    vals = jax.vmap(src_fn)(times[1:])  # Skip t=0 (initial condition)
                    all_vsource.append(vals)
                else:
                    all_vsource.append(jnp.full(num_timesteps, float(vsource_dc[i])))
            all_vsource_vals = jnp.stack(all_vsource, axis=1) if all_vsource else jnp.zeros((num_timesteps, 0))
        else:
            all_vsource_vals = jnp.zeros((num_timesteps, 0))

        if n_isources > 0:
            all_isource = []
            for i, name in enumerate(isource_names):
                src_fn = {k: v for d in self.devices
                         for k, v in [(d['name'], self._get_source_fn_for_device(d))]
                         if d['name'] == name}.get(name)
                if src_fn is not None:
                    vals = jax.vmap(src_fn)(times[1:])
                    all_isource.append(vals)
                else:
                    all_isource.append(jnp.full(num_timesteps, float(isource_dc[i])))
            all_isource_vals = jnp.stack(all_isource, axis=1) if all_isource else jnp.zeros((num_timesteps, 0))
        else:
            all_isource_vals = jnp.zeros((num_timesteps, 0))

        t1 = time_module.perf_counter()
        logger.info(f"Source pre-computation: {t1-t0:.3f}s")

        # Get or create the NR solver
        cache_key = (n_nodes, n_vsources, n_isources, use_dense)
        if hasattr(self, '_cached_nr_solve') and self._cached_solver_key == cache_key:
            nr_solve = self._cached_nr_solve
        else:
            # Need to build the solver - call hybrid to set it up
            self._run_transient_hybrid(t_stop=dt, dt=dt, backend=backend, use_dense=use_dense)
            nr_solve = self._cached_nr_solve

        # Initial condition (DC operating point = 0 for now)
        V0 = jnp.zeros(n_nodes, dtype=jnp.float64)

        # Create the scan function
        def step_fn(V_prev, source_inputs):
            """Single timestep for lax.scan - fully traceable."""
            vsource_vals, isource_vals = source_inputs
            V_new, iterations, converged, max_f = nr_solve(V_prev, vsource_vals, isource_vals)
            # Return new state and output (voltage at external nodes only)
            return V_new, (V_new[:n_external], iterations, converged)

        # JIT compile the entire scan
        @jax.jit
        def run_all_steps(V_init, vsource_all, isource_all):
            """Run all timesteps using lax.scan - fully JIT'd."""
            source_inputs = (vsource_all, isource_all)
            final_V, (all_V, all_iters, all_converged) = jax.lax.scan(
                step_fn, V_init, source_inputs
            )
            return all_V, all_iters, all_converged

        # Run the fully JIT'd simulation (with optional profiling of just the core loop)
        logger.info("Running full simulation (lax.scan)...")
        t2 = time_module.perf_counter()

        if profile_config:
            with profile_section("lax_scan_simulation", profile_config):
                all_V, all_iters, all_converged = run_all_steps(V0, all_vsource_vals, all_isource_vals)
                # Wait for computation to complete (JAX operations are async)
                all_V.block_until_ready()
        else:
            all_V, all_iters, all_converged = run_all_steps(V0, all_vsource_vals, all_isource_vals)
            all_V.block_until_ready()

        t3 = time_module.perf_counter()
        total_time = t3 - t2

        # Prepend initial condition
        all_V_with_ic = jnp.concatenate([V0[:n_external][None, :], all_V], axis=0)

        # Build stats
        total_iters = int(jnp.sum(all_iters))
        non_converged = int(jnp.sum(~all_converged))

        stats = {
            'total_timesteps': num_timesteps + 1,
            'total_nr_iterations': total_iters,
            'non_converged_count': non_converged,
            'non_converged_steps': [],  # Not tracking individual steps in JIT mode
            'convergence_rate': 1.0 - non_converged / max(num_timesteps, 1),
            'wall_time': total_time,
            'time_per_step_ms': total_time / num_timesteps * 1000,
            'fully_jit': True,
        }

        logger.info(f"Completed: {num_timesteps} steps in {total_time:.3f}s "
                   f"({stats['time_per_step_ms']:.2f}ms/step, {total_iters} NR iters)")

        # Convert to dict format for compatibility
        voltages = {i: all_V_with_ic[:, i] for i in range(n_external)}

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
            # Residual: -I at p, +I at n (note sign convention)
            f_vals = isource_vals[:, None] * jnp.array([-1.0, 1.0])[None, :]  # (n, 2)
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
            Function build_system(V, vsource_vals, isource_vals) -> (J, f)
        """
        from jax.experimental.sparse import BCOO, BCSR
        from jax_spice.analysis.sparse import sparse_solve_csr

        # Capture model types as static list (unrolled at trace time)
        model_types = list(static_inputs_cache.keys())

        def build_system(V: jax.Array, vsource_vals: jax.Array, isource_vals: jax.Array
                        ) -> Tuple[Any, jax.Array]:
            """Build Jacobian J and residual f from current voltages.

            Fully JAX-traceable - no Python lists or dynamic allocation.
            All device contributions are concatenated into fixed-size arrays.

            Returns:
                J: Jacobian matrix (dense jax.Array if use_dense=True, BCOO if use_dense=False)
                f: Residual vector (jax.Array)
            """
            f_parts = []
            j_parts = []

            # === Source devices contribution ===
            # Voltage sources: I = G * (Vp - Vn - Vtarget), G = 1e12
            if 'vsource' in source_device_data and vsource_vals.size > 0:
                d = source_device_data['vsource']
                G = 1e12
                Vp, Vn = V[d['node_p']], V[d['node_n']]
                I = G * (Vp - Vn - vsource_vals)
                G_arr = jnp.full(d['n'], G)

                # Residual contribution
                f_vals = I[:, None] * d['f_signs'][None, :]  # (n, 2)
                f_idx = d['f_indices'].ravel()
                f_val = f_vals.ravel()
                f_valid = f_idx >= 0
                f_parts.append((jnp.where(f_valid, f_idx, 0), jnp.where(f_valid, f_val, 0.0)))

                # Jacobian contribution
                j_vals_arr = G_arr[:, None] * d['j_signs'][None, :]  # (n, 4)
                j_row = d['j_rows'].ravel()
                j_col = d['j_cols'].ravel()
                j_val = j_vals_arr.ravel()
                j_valid = j_row >= 0
                j_parts.append((
                    jnp.where(j_valid, j_row, 0),
                    jnp.where(j_valid, j_col, 0),
                    jnp.where(j_valid, j_val, 0.0)
                ))

            # Current sources (residual only)
            if 'isource' in source_device_data and isource_vals.size > 0:
                d = source_device_data['isource']
                f_vals = isource_vals[:, None] * jnp.array([-1.0, 1.0])[None, :]
                f_idx = d['f_indices'].ravel()
                f_val = f_vals.ravel()
                f_valid = f_idx >= 0
                f_parts.append((jnp.where(f_valid, f_idx, 0), jnp.where(f_valid, f_val, 0.0)))

            # === OpenVAF devices contribution (unrolled at trace time) ===
            for model_type in model_types:
                static_inputs, voltage_indices, stamp_indices, voltage_node1, voltage_node2 = \
                    static_inputs_cache[model_type]
                vmapped_fn = vmapped_fns[model_type]

                # Vectorized voltage update
                voltage_updates = V[voltage_node1] - V[voltage_node2]
                batch_inputs = static_inputs.at[:, jnp.array(voltage_indices)].set(voltage_updates)

                # Batched device evaluation
                batch_residuals, batch_jacobian = vmapped_fn(batch_inputs)

                # Collect COO triplets using pre-computed indices
                res_idx = stamp_indices['res_indices']
                jac_row_idx = stamp_indices['jac_row_indices']
                jac_col_idx = stamp_indices['jac_col_indices']

                # Residuals - use masking instead of boolean indexing for static shapes
                flat_res_idx = res_idx.ravel()
                flat_res_val = batch_residuals.ravel()
                valid_res = flat_res_idx >= 0
                # Mask invalid indices to 0, mask invalid values to 0
                flat_res_idx_masked = jnp.where(valid_res, flat_res_idx, 0)
                flat_res_val_masked = jnp.where(valid_res, flat_res_val, 0.0)
                flat_res_val_masked = jnp.where(jnp.isnan(flat_res_val_masked), 0.0, flat_res_val_masked)
                f_parts.append((flat_res_idx_masked, flat_res_val_masked))

                # Jacobian - use masking instead of boolean indexing
                flat_jac_rows = jac_row_idx.ravel()
                flat_jac_cols = jac_col_idx.ravel()
                flat_jac_vals = batch_jacobian.ravel()
                valid_jac = (flat_jac_rows >= 0) & (flat_jac_cols >= 0)
                flat_jac_rows_masked = jnp.where(valid_jac, flat_jac_rows, 0)
                flat_jac_cols_masked = jnp.where(valid_jac, flat_jac_cols, 0)
                flat_jac_vals_masked = jnp.where(valid_jac, flat_jac_vals, 0.0)
                flat_jac_vals_masked = jnp.where(jnp.isnan(flat_jac_vals_masked), 0.0, flat_jac_vals_masked)
                j_parts.append((
                    flat_jac_rows_masked,
                    flat_jac_cols_masked,
                    flat_jac_vals_masked
                ))

            # === Build residual vector f using segment_sum ===
            if f_parts:
                all_f_idx = jnp.concatenate([p[0] for p in f_parts])
                all_f_val = jnp.concatenate([p[1] for p in f_parts])
                f = jax.ops.segment_sum(all_f_val, all_f_idx, num_segments=n_unknowns)
            else:
                f = jnp.zeros(n_unknowns, dtype=jnp.float64)

            # === Build Jacobian J ===
            if j_parts:
                all_j_rows = jnp.concatenate([p[0] for p in j_parts])
                all_j_cols = jnp.concatenate([p[1] for p in j_parts])
                all_j_vals = jnp.concatenate([p[2] for p in j_parts])

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
            else:
                J = jnp.eye(n_unknowns, dtype=jnp.float64) * 1e-12

            return J, f

        return build_system

    def _make_jit_compiled_solver(
        self,
        build_system_jit: Callable,
        n_nodes: int,
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
            build_system_jit: JIT-wrapped function (V, vsource_vals, isource_vals) -> (J, f)
            n_nodes: Total node count including ground (V.shape[0])
            max_iterations: Maximum NR iterations
            abstol: Absolute tolerance for convergence
            max_step: Maximum voltage step per iteration

        Returns:
            JIT-compiled function: (V, vsource_vals, isource_vals) -> (V, iters, converged, max_f)
        """

        def nr_solve(V_init: jax.Array, vsource_vals: jax.Array, isource_vals: jax.Array):
            # State: (V, iteration, converged, max_f, max_delta)
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

                # Build system (J and f) - calls JIT'd function
                J, f = build_system_jit(V, vsource_vals, isource_vals)

                # Check residual convergence (exclude ground at index 0)
                # Ground's residual includes current source stamps but ground is pinned
                max_f = jnp.max(jnp.abs(f[1:]))
                residual_converged = max_f < abstol

                # Solve: J @ delta = -f (only updating non-ground nodes)
                delta = jax.scipy.linalg.solve(J, -f)

                # Step limiting
                max_delta = jnp.max(jnp.abs(delta))
                scale = jnp.where(max_delta > max_step, max_step / max_delta, 1.0)
                delta = delta * scale

                # Update V (ground at index 0 stays fixed)
                V_new = V.at[1:].add(delta)

                # Check delta-based convergence
                delta_converged = max_delta < 1e-12

                converged = jnp.logical_or(residual_converged, delta_converged)

                return (V_new, iteration + 1, converged, max_f, max_delta)

            # Run NR loop on GPU
            V_final, iterations, converged, max_f, max_delta = lax.while_loop(
                cond_fn, body_fn, init_state
            )

            return V_final, iterations, converged, max_f

        # Return JIT-wrapped function - compilation happens lazily on first call
        logger.info(f"Creating JIT-compiled NR solver: V({n_nodes})")
        return jax.jit(nr_solve)

    def _make_sparse_jit_compiled_solver(
        self,
        build_system_jit: Callable,
        n_nodes: int,
        nse: int,
        max_iterations: int = MAX_NR_ITERATIONS,
        abstol: float = 1e-6,
        max_step: float = 1.0,
    ) -> Callable:
        """Create a JIT-compiled sparse NR solver using spsolve.

        Uses JAX's sparse direct solver (QR factorization) for large circuits
        where dense linear algebra would OOM.

        Args:
            build_system_jit: JIT-wrapped function returning (J_bcoo, f)
            n_nodes: Total node count including ground
            nse: Number of stored elements after summing duplicates
            max_iterations: Maximum NR iterations
            abstol: Absolute tolerance for convergence
            max_step: Maximum voltage step per iteration

        Returns:
            JIT-compiled sparse solver function
        """
        from jax.experimental.sparse import BCSR
        from jax.experimental.sparse.linalg import spsolve

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

                # Check residual convergence (exclude ground)
                max_f = jnp.max(jnp.abs(f[1:]))
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

                # Check delta-based convergence
                delta_converged = max_delta < 1e-12
                converged = jnp.logical_or(residual_converged, delta_converged)

                return (V_new, iteration + 1, converged, max_f, max_delta)

            V_final, iterations, converged, max_f, max_delta = lax.while_loop(
                cond_fn, body_fn, init_state
            )

            return V_final, iterations, converged, max_f

        logger.info(f"Creating sparse JIT-compiled NR solver: V({n_nodes})")
        return jax.jit(nr_solve)

    def _make_spineax_jit_compiled_solver(
        self,
        build_system_jit: Callable,
        n_nodes: int,
        nse: int,
        bcsr_indptr: jax.Array,
        bcsr_indices: jax.Array,
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
            max_iterations: Maximum NR iterations
            abstol: Absolute tolerance for convergence
            max_step: Maximum voltage step per iteration

        Returns:
            JIT-compiled sparse solver function using Spineax
        """
        from jax.experimental.sparse import BCSR
        from spineax.cudss.solver import CuDSSSolver

        n_unknowns = n_nodes - 1  # Exclude ground

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

                # Check residual convergence (exclude ground)
                max_f = jnp.max(jnp.abs(f[1:]))
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
        # PSP103 internal node mapping:
        # GP, SI, DI, BP, BS, BD, BI, NOI correspond to node4-node11
        internal_name_map = {
            'GP': 'node4', 'SI': 'node5', 'DI': 'node6', 'BP': 'node7',
            'BS': 'node8', 'BD': 'node9', 'BI': 'node10', 'NOI': 'node11',
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
