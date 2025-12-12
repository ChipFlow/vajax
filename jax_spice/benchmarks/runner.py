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
import numpy as np

from jax_spice.netlist.parser import VACASKParser
from jax_spice.netlist.circuit import Instance
from jax_spice.analysis.mna import MNASystem, DeviceInfo
from jax_spice.analysis.transient import transient_analysis_jit

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


class VACASKBenchmarkRunner:
    """Generic runner for VACASK benchmark circuits.

    Parses a benchmark .sim file and runs transient analysis using our solver.
    Handles resistors, capacitors, diodes, and voltage sources automatically.
    """

    # Map OSDI module names to device types
    MODULE_TO_DEVICE = {
        'sp_resistor': 'resistor',
        'sp_capacitor': 'capacitor',
        'sp_diode': 'diode',
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
    }

    # SPICE number suffixes
    SUFFIXES = {
        't': 1e12, 'g': 1e9, 'meg': 1e6, 'k': 1e3,
        'm': 1e-3, 'u': 1e-6, 'n': 1e-9, 'p': 1e-12, 'f': 1e-15
    }

    def __init__(self, sim_path: Path, verbose: bool = False):
        self.sim_path = Path(sim_path)
        self.verbose = verbose
        self.circuit = None
        self.devices = []
        self.node_names = {}
        self.num_nodes = 0
        self.analysis_params = {}
        self.flat_instances = []

        # OpenVAF compiled models cache
        self._compiled_models: Dict[str, Any] = {}
        self._has_openvaf_devices = False

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

        if self.verbose:
            print(f"parse(): starting...", flush=True)
            sys.stdout.flush()

        t0 = time.perf_counter()
        parser = VACASKParser()
        self.circuit = parser.parse_file(self.sim_path)
        t1 = time.perf_counter()

        if self.verbose:
            print(f"Parsed: {self.circuit.title} ({t1-t0:.1f}s)", flush=True)
            print(f"Models: {list(self.circuit.models.keys())}", flush=True)
            if self.circuit.subckts:
                print(f"Subcircuits: {list(self.circuit.subckts.keys())}", flush=True)

        # Flatten subcircuit instances to leaf devices
        if self.verbose:
            print(f"Flattening subcircuit instances...", flush=True)
        self.flat_instances = self._flatten_top_instances()
        t2 = time.perf_counter()

        if self.verbose:
            print(f"Flattened: {len(self.flat_instances)} leaf devices ({t2-t1:.1f}s)", flush=True)
            for name, terms, model, params in self.flat_instances[:10]:
                print(f"  {name}: {model} {terms}")
            if len(self.flat_instances) > 10:
                print(f"  ... and {len(self.flat_instances) - 10} more")

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

        if self.verbose:
            print(f"Node mapping: {self.num_nodes} nodes ({t3-t2:.1f}s)", flush=True)

        # Build devices
        self._build_devices()
        t4 = time.perf_counter()

        if self.verbose:
            print(f"Built devices: {len(self.devices)} ({t4-t3:.1f}s)", flush=True)

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
                if self.verbose:
                    print(f"Elaborating subcircuit: {elaborate_subckt}")
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

        if self.verbose:
            print(f"_build_devices(): starting with {len(self.flat_instances)} instances", flush=True)
            sys.stdout.flush()

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
        if self.verbose:
            print(f"_build_devices(): loop done in {t_loop - t_start:.1f}s", flush=True)
            sys.stdout.flush()
            print(f"Devices: {len(self.devices)}")
            for dev in self.devices[:10]:
                print(f"  {dev['name']}: {dev['model']} nodes={dev['nodes']}")
            if len(self.devices) > 10:
                print(f"  ... and {len(self.devices) - 10} more")

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
            import sys
            if log_fn:
                log_fn(msg)
            elif self.verbose:
                print(msg, flush=True)
            # Force flush for Cloud Run logging
            sys.stdout.flush()
            sys.stderr.flush()

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
            node_map = {}
            for i, model_node in enumerate(model_nodes[:4]):
                if i < len(ext_nodes):
                    node_map[model_node] = ext_nodes[i]
                else:
                    node_map[model_node] = ground

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

            # Fill input array for this device (static params only, voltages stay 0)
            for param_idx, (name, kind) in enumerate(zip(param_names, param_kinds)):
                if kind == 'voltage':
                    pass  # Already 0 from np.zeros
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
                    else:
                        all_inputs[dev_idx, param_idx] = 1.0
                # hidden_state and others stay 0 from np.zeros
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

        # PSP103 internal node mapping
        internal_name_map = {
            'GP': 'node4', 'SI': 'node5', 'DI': 'node6', 'BP': 'node7',
            'BS': 'node8', 'BD': 'node9', 'BI': 'node10', 'NOI': 'node11',
            'G': 'node1', 'D': 'node0', 'S': 'node2', 'B': 'node3',
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
            node_map = {}
            for i, model_node in enumerate(model_nodes[:4]):
                if i < len(ext_nodes):
                    node_map[model_node] = ext_nodes[i]
                else:
                    node_map[model_node] = ground

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

        if self.verbose:
            print(f"Analysis: {self.analysis_params}")

    def _build_source_fn(self):
        """Build time-varying source function from device parameters."""
        sources = {}

        for dev in self.devices:
            if dev['model'] not in ('vsource', 'isource'):
                continue

            params = dev['params']
            source_type = str(params.get('type', 'dc')).lower()

            if self.verbose:
                print(f"  Source {dev['name']}: type={source_type}")

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

    def run_transient(self, t_stop: Optional[float] = None, dt: Optional[float] = None,
                      max_steps: int = 10000, use_sparse: Optional[bool] = None,
                      backend: Optional[str] = None) -> Tuple[jax.Array, Dict[int, jax.Array], Dict]:
        """Run transient analysis.

        Uses the JIT-compiled solver for circuits with only simple devices.
        Uses a hybrid Python-based solver for circuits with OpenVAF devices.
        Automatically uses sparse matrices for large circuits (>1000 nodes).

        Args:
            t_stop: Stop time (default: from analysis params or 1ms)
            dt: Time step (default: from analysis params or 1µs)
            max_steps: Maximum number of time steps
            use_sparse: Force sparse (True) or dense (False) solver. If None, auto-detect.
            backend: 'gpu', 'cpu', or None (auto-select based on circuit size).
                     For circuits >500 nodes with GPU available, uses GPU acceleration.

        Returns:
            (times, voltages, stats) tuple where:
            - times: array of time points
            - voltages: dict mapping node index to voltage array
            - stats: dict with convergence info (total_timesteps, non_converged_count, etc.)
        """
        print("importing gpu backend", flush=True)
        from jax_spice.analysis.gpu_backend import select_backend, is_gpu_available
        print("imported gpu backend", flush=True)

        if t_stop is None:
            t_stop = self.analysis_params.get('stop', 1e-3)
        if dt is None:
            dt = self.analysis_params.get('step', 1e-6)

        # Limit number of steps
        num_steps = int(t_stop / dt)
        if num_steps > max_steps:
            dt = t_stop / max_steps
            if self.verbose:
                print(f"Limiting to {max_steps} steps, dt={dt:.2e}s")

        # Select backend if not specified

        print("selecting gpu backend", flush=True)
        if backend is None or backend == "auto":
            backend = select_backend(self.num_nodes)

        # if self.verbose:
        print(f"Running transient: t_stop={t_stop:.2e}s, dt={dt:.2e}s, backend={backend}", flush=True)

        # Use hybrid solver if we have OpenVAF devices
        if self._has_openvaf_devices:
            # Auto-detect sparse usage if not specified
            # Dense matrix for 86k nodes would be ~56GB, so use sparse for large circuits
            if use_sparse is None:
                use_sparse = self.num_nodes > 1000

            if use_sparse:
                # if self.verbose:
                print(f"Using BCOO/BCSR sparse solver ({self.num_nodes} nodes, OpenVAF devices)", flush=True)
                # Use BCOO/BCSR + spsolve (direct sparse solver)
                # This is more robust for circuit simulation than matrix-free GMRES
                return self._run_transient_hybrid(t_stop, dt, backend=backend, use_dense=False)
            else:
                #if self.verbose:
                print("Using dense hybrid solver (OpenVAF devices detected)", flush=True)
                return self._run_transient_hybrid(t_stop, dt, backend=backend, use_dense=True)

        # Convert to MNA system
        print("Getting mna system", flush=True)
        system = self.to_mna_system()

        # Run production transient analysis with backend selection
        print("Running transient analysis", flush=True)
        times, voltages_array, stats = transient_analysis_jit(
            system=system,
            t_stop=t_stop,
            t_step=dt,
            t_start=0.0,
            backend=backend,
        )

        print("Creating voltage dict", flush=True)
        # Create voltage dict from JAX arrays
        voltages = {}
        for i in range(self.num_nodes):
            if i < voltages_array.shape[1]:
                voltages[i] = voltages_array[:, i]
            else:
                voltages[i] = jnp.zeros(len(times))

        # if self.verbose:
        print(f"Completed: {len(times)} timesteps, {stats.get('iterations', 'N/A')} total NR iterations", flush=True)

        return times, voltages, stats

    def _setup_internal_nodes(self) -> Tuple[int, Dict[str, Dict[str, int]]]:
        """Set up internal nodes for OpenVAF devices.

        Returns:
            (total_nodes, device_internal_nodes) where device_internal_nodes maps
            device name to dict of internal node name -> global index
        """
        n_external = self.num_nodes
        next_internal = n_external
        device_internal_nodes = {}

        for dev in self.devices:
            if not dev.get('is_openvaf'):
                continue

            model_type = dev['model']
            compiled = self._compiled_models.get(model_type)
            if not compiled:
                continue

            # Get model nodes (first 4 are external: D, G, S, B)
            model_nodes = compiled['nodes']
            n_model_nodes = len(model_nodes)

            # Allocate internal nodes (skip first 4 external and last 1 branch)
            internal_map = {}
            for i in range(4, n_model_nodes - 1):  # Skip external nodes and branch
                node_name = model_nodes[i]
                internal_map[node_name] = next_internal
                next_internal += 1

            device_internal_nodes[dev['name']] = internal_map

        if self.verbose and device_internal_nodes:
            n_internal = next_internal - n_external
            print(f"Allocated {n_internal} internal nodes for {len(device_internal_nodes)} OpenVAF devices")

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

        print(f"Importing backend ({backend})")

        from jax_spice.analysis.gpu_backend import get_device, get_default_dtype
        from jax_spice.analysis.sparse import build_csr_arrays, sparse_solve_csr
        from jax.experimental.sparse import BCOO, BCSR

        ground = 0
    
        # Get target device for JAX operations

        print("getting device and dtype", flush=True)
        device = get_device(backend)
        dtype = get_default_dtype(backend)

        # Set up internal nodes for OpenVAF devices
        n_total, device_internal_nodes = self._setup_internal_nodes()
        n_external = self.num_nodes
        n_unknowns = n_total - 1

        solver_type = "dense batched scatter" if use_dense else "COO sparse"
        # if self.verbose:
        print(f"Total nodes: {n_total} ({n_external} external, {n_total - n_external} internal)")
        print(f"Backend: {backend}, device: {device.platform}")
        print(f"Using {solver_type} solver", flush=True)

        # Initialize voltages
        V = jnp.zeros(n_total, dtype=jnp.float64)
        V_prev = jnp.zeros(n_total, dtype=jnp.float64)

        # Build time-varying source function
        print("Building source function", flush=True)
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

        print(f"{len(source_devices)} source devices", flush=True)
        # Prepare OpenVAF: vmapped functions, static inputs, and stamp index mappings
        vmapped_fns: Dict[str, Callable] = {}
        static_inputs_cache: Dict[str, Tuple[Any, List[int], List[Dict], Dict]] = {}

        for model_type in openvaf_by_type:
            print(f"Getting compiled model for {model_type}", flush=True)
            compiled = self._compiled_models.get(model_type)
            print(f"Got model: {compiled}", flush=True)

            if compiled and 'vmapped_fn' in compiled:
                vmapped_fns[model_type] = compiled['vmapped_fn']
                print(f"Preparing static inputs: {model_type}, {openvaf_by_type[model_type]}, {device_internal_nodes}, {ground}", flush=True)
                static_inputs, voltage_indices, device_contexts = self._prepare_static_inputs(
                    model_type, openvaf_by_type[model_type], device_internal_nodes, ground
                )
                # Pre-compute stamp index mapping (once per model type)
                print("building stamp index mapping for {model_type}, {device_contexts}, {ground}", flush=True)
                stamp_indices = self._build_stamp_index_mapping(
                    model_type, device_contexts, ground
                )
                # Pre-compute voltage node arrays for vectorized update
                n_devices = len(device_contexts)
                n_voltages = len(voltage_indices)
                # Build arrays directly from list comprehension (setup phase only)
                print("building voltage model", flush=True)
                voltage_node1 = jnp.array([
                    [n1 for n1, n2 in ctx['voltage_node_pairs']]
                    for ctx in device_contexts
                ], dtype=jnp.int32)
                voltage_node2 = jnp.array([
                    [n2 for n1, n2 in ctx['voltage_node_pairs']]
                    for ctx in device_contexts
                ], dtype=jnp.int32)


                print("fetching static inputs", flush=True)
                if backend == "gpu":
                    with jax.default_device(device):
                        static_inputs = jnp.array(static_inputs, dtype=dtype)
                else:
                    static_inputs = jnp.array(static_inputs, dtype=jnp.float64)
                static_inputs_cache[model_type] = (
                    static_inputs, voltage_indices, stamp_indices, voltage_node1, voltage_node2
                )
                # if self.verbose:
                n_devs = len(openvaf_by_type[model_type])
                print(f"Prepared {model_type}: {n_devs} devices, stamp indices cached", flush=True)

        # Pre-compute source device stamp indices
        print(f"Precomputing source device data for {source_devices}, {ground}, {n_unknowns}", flush=True)
        source_device_data = self._prepare_source_devices_coo(source_devices, ground, n_unknowns)

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

        print("Running time steps", flush=True)

        times = []
        voltages = {i: [] for i in range(n_external)}

        t = 0.0
        total_nr_iters = 0
        non_converged_steps = []  # Track (time, max_residual) for non-converged steps

        while t <= t_stop:
            source_values = source_fn(t)
            # Build source value arrays once per timestep (Python loop here, not in NR loop)
            vsource_vals, isource_vals = build_source_arrays(source_values)

            converged = False
            V_iter = V

            for nr_iter in range(100):
                # === Pure JAX: COO collection for both dense and sparse paths ===
                f_indices_list = []
                f_values_list = []
                j_rows_list = []
                j_cols_list = []
                j_vals_list = []

                # Collect from source devices (vsource, isource) - fully vectorized
                self._collect_source_devices_coo(
                    source_device_data, V_iter, vsource_vals, isource_vals,
                    f_indices_list, f_values_list, j_rows_list, j_cols_list, j_vals_list
                )

                # Collect from OpenVAF devices
                for model_type in openvaf_by_type:
                    if model_type in vmapped_fns and model_type in static_inputs_cache:
                        static_inputs, voltage_indices, stamp_indices, voltage_node1, voltage_node2 = \
                            static_inputs_cache[model_type]

                        # Vectorized voltage update (no Python loops)
                        voltage_updates = V_iter[voltage_node1] - V_iter[voltage_node2]
                        batch_inputs = static_inputs.at[:, jnp.array(voltage_indices)].set(voltage_updates)

                        batch_residuals, batch_jacobian = vmapped_fns[model_type](batch_inputs)

                        self._collect_openvaf_coo(
                            batch_residuals, batch_jacobian, stamp_indices,
                            f_indices_list, f_values_list,
                            j_rows_list, j_cols_list, j_vals_list
                        )

                # Build residual vector using segment_sum
                if f_indices_list:
                    all_f_idx = jnp.concatenate(f_indices_list)
                    all_f_val = jnp.concatenate(f_values_list)
                    f = jax.ops.segment_sum(all_f_val, all_f_idx, num_segments=n_unknowns)
                else:
                    f = jnp.zeros(n_unknowns, dtype=jnp.float64)

                # Check convergence
                max_f = float(jnp.max(jnp.abs(f)))
                if max_f < 1e-9:
                    converged = True
                    break

                if use_dense:
                    # === DENSE PATH: COO -> dense matrix via segment_sum ===
                    if j_rows_list:
                        all_j_rows = jnp.concatenate(j_rows_list)
                        all_j_cols = jnp.concatenate(j_cols_list)
                        all_j_vals = jnp.concatenate(j_vals_list)

                        # Convert (row, col) to flat index and use segment_sum
                        flat_indices = all_j_rows * n_unknowns + all_j_cols
                        J_flat = jax.ops.segment_sum(
                            all_j_vals, flat_indices, num_segments=n_unknowns * n_unknowns
                        )
                        J = J_flat.reshape((n_unknowns, n_unknowns))
                    else:
                        J = jnp.zeros((n_unknowns, n_unknowns), dtype=jnp.float64)

                    # Dense solve with regularization
                    J_reg = J + 1e-12 * jnp.eye(n_unknowns, dtype=jnp.float64)
                    delta = jax.scipy.linalg.solve(J_reg, -f)

                else:
                    # === SPARSE PATH: Build BCOO from COO and solve ===
                    if j_rows_list:
                        all_j_rows = jnp.concatenate(j_rows_list)
                        all_j_cols = jnp.concatenate(j_cols_list)
                        all_j_vals = jnp.concatenate(j_vals_list)

                        # Add diagonal regularization (larger for sparse solver stability)
                        # Sparse direct solvers are more sensitive to near-singular matrices
                        diag_idx = jnp.arange(n_unknowns, dtype=jnp.int32)
                        all_j_rows = jnp.concatenate([all_j_rows, diag_idx])
                        all_j_cols = jnp.concatenate([all_j_cols, diag_idx])
                        all_j_vals = jnp.concatenate([all_j_vals, jnp.full(n_unknowns, 1e-9)])

                        # Build BCOO and sum duplicates (native JAX sparse)
                        indices = jnp.stack([all_j_rows, all_j_cols], axis=1)
                        J_bcoo = BCOO((all_j_vals, indices), shape=(n_unknowns, n_unknowns))
                        J_bcoo = J_bcoo.sum_duplicates()

                        # Convert to BCSR for solve (JAX spsolve expects CSR format)
                        J_bcsr = BCSR.from_bcoo(J_bcoo)
                        delta = sparse_solve_csr(
                            J_bcsr.data, J_bcsr.indices, J_bcsr.indptr, -f, (n_unknowns, n_unknowns)
                        )
                    else:
                        delta = -f

                # Limit voltage step
                max_delta = float(jnp.max(jnp.abs(delta)))
                if max_delta > 1.0:
                    delta = delta * (1.0 / max_delta)

                V_iter = V_iter.at[1:].add(delta)

                if max_delta < 1e-12:
                    converged = True
                    break

            V = V_iter
            total_nr_iters += nr_iter + 1

            if not converged:
                non_converged_steps.append((t, max_f))
                if self.verbose:
                    print(f"Warning: t={t:.2e}s did not converge (max_f={max_f:.2e})")

            # Record state
            times.append(t)
            for i in range(n_external):
                voltages[i].append(float(V[i]))

            V_prev = V
            t += dt

        # Build stats dict
        stats = {
            'total_timesteps': len(times),
            'total_nr_iterations': total_nr_iters,
            'non_converged_count': len(non_converged_steps),
            'non_converged_steps': non_converged_steps,
            'convergence_rate': 1.0 - len(non_converged_steps) / max(len(times), 1),
        }

        if self.verbose:
            print(f"Completed: {len(times)} timesteps, {total_nr_iters} total NR iterations")
            if non_converged_steps:
                print(f"  Non-converged: {len(non_converged_steps)} steps ({100*(1-stats['convergence_rate']):.1f}%)")

        return jnp.array(times), {k: jnp.array(v) for k, v in voltages.items()}, stats

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
