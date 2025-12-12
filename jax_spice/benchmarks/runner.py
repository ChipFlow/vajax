"""VACASK Benchmark Runner

Generic runner for VACASK benchmark circuits. Parses benchmark .sim files
and runs transient analysis using the production JAX-based solver.

For circuits with OpenVAF-compiled devices (like PSP103 MOSFETs), uses a
hybrid solver that combines the JIT-compiled solver for simple devices
with Python-based Newton-Raphson for complex Verilog-A models.
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

    # OpenVAF model sources (relative to openvaf-py/vendor/OpenVAF/integration_tests)
    OPENVAF_MODELS = {
        'psp103': 'PSP103/psp103.va',
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
        parser = VACASKParser()
        self.circuit = parser.parse_file(self.sim_path)

        if self.verbose:
            print(f"Parsed: {self.circuit.title}")
            print(f"Models: {list(self.circuit.models.keys())}")
            if self.circuit.subckts:
                print(f"Subcircuits: {list(self.circuit.subckts.keys())}")

        # Flatten subcircuit instances to leaf devices
        self.flat_instances = self._flatten_top_instances()

        if self.verbose:
            print(f"Flattened: {len(self.flat_instances)} leaf devices")
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

        # Build devices
        self._build_devices()

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

        if self.verbose:
            print(f"Devices: {len(self.devices)}")
            for dev in self.devices[:10]:
                print(f"  {dev['name']}: {dev['model']} nodes={dev['nodes']}")
            if len(self.devices) > 10:
                print(f"  ... and {len(self.devices) - 10} more")

        # Compile OpenVAF models if needed
        if self._has_openvaf_devices:
            self._compile_openvaf_models()

    def _compile_openvaf_models(self):
        """Compile OpenVAF models needed by the circuit."""
        if not HAS_OPENVAF:
            raise ImportError("OpenVAF support required but openvaf_py not available")

        # Find unique OpenVAF model types
        openvaf_types = set()
        for dev in self.devices:
            if dev.get('is_openvaf'):
                openvaf_types.add(dev['model'])

        if self.verbose:
            print(f"Compiling OpenVAF models: {openvaf_types}")

        # Compile each model type
        openvaf_base = Path(__file__).parent.parent.parent / "openvaf-py" / "vendor" / "OpenVAF" / "integration_tests"

        for model_type in openvaf_types:
            if model_type in self._compiled_models:
                continue

            va_path = self.OPENVAF_MODELS.get(model_type)
            if not va_path:
                raise ValueError(f"Unknown OpenVAF model type: {model_type}")

            full_path = openvaf_base / va_path
            if not full_path.exists():
                raise FileNotFoundError(f"VA model not found: {full_path}")

            if self.verbose:
                print(f"  Compiling {model_type} from {va_path}...")

            modules = openvaf_py.compile_va(str(full_path))
            if not modules:
                raise ValueError(f"Failed to compile {va_path}")

            module = modules[0]
            translator = openvaf_jax.OpenVAFToJAX(module)
            jax_fn = translator.translate()

            # Also generate array-based function for batched evaluation
            jax_fn_array, array_metadata = translator.translate_array()

            # Create JIT-compiled vmapped function for fast batched evaluation
            # JIT compilation is now possible after fixing boolean constants in openvaf_jax
            vmapped_fn = jax.jit(jax.vmap(jax_fn_array))

            self._compiled_models[model_type] = {
                'module': module,
                'translator': translator,
                'jax_fn': jax_fn,
                'jax_fn_array': jax_fn_array,
                'vmapped_fn': vmapped_fn,
                'array_metadata': array_metadata,
                'param_names': list(module.param_names),
                'param_kinds': list(module.param_kinds),
                'nodes': list(module.nodes),
            }

            if self.verbose:
                print(f"    {model_type}: {len(module.param_names)} params, {len(module.nodes)} nodes")

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

            # Pre-compute voltage node pairs for fast update
            voltage_node_pairs = []
            for idx in voltage_indices:
                name = param_names[idx]
                node_pair = self._parse_voltage_param(name, node_map, model_nodes, ground)
                voltage_node_pairs.append(node_pair)

            # Build input array for this device (static params only, voltages set to 0)
            inputs = []
            for name, kind in zip(param_names, param_kinds):
                if kind == 'voltage':
                    inputs.append(0.0)  # Will be updated dynamically
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
                'voltage_node_pairs': voltage_node_pairs,
            })

        return jnp.array(all_inputs), voltage_indices, device_contexts

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
                # Pulse source
                val0 = params.get('val0', 0)
                val1 = params.get('val1', 1)
                rise = params.get('rise', 1e-9)
                fall = params.get('fall', 1e-9)
                width = params.get('width', 1e-6)
                period = params.get('period', 2e-6)
                delay = params.get('delay', 0)

                def pulse_fn(t, v0=val0, v1=val1, r=rise, f=fall, w=width, p=period, d=delay):
                    if t < d:
                        return v0
                    t_in_period = (t - d) % p
                    if t_in_period < r:
                        return v0 + (v1 - v0) * t_in_period / r
                    elif t_in_period < r + w:
                        return v1
                    elif t_in_period < r + w + f:
                        return v1 - (v1 - v0) * (t_in_period - r - w) / f
                    else:
                        return v0

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
                      backend: Optional[str] = None) -> Tuple[jax.Array, Dict[int, jax.Array]]:
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
            (times, voltages) tuple where voltages is dict mapping node index to voltage array
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
            if self.verbose:
                print(f"Limiting to {max_steps} steps, dt={dt:.2e}s")

        # Select backend if not specified
        if backend is None or backend == "auto":
            backend = select_backend(self.num_nodes)

        if self.verbose:
            print(f"Running transient: t_stop={t_stop:.2e}s, dt={dt:.2e}s, backend={backend}")

        # Use hybrid solver if we have OpenVAF devices
        if self._has_openvaf_devices:
            # Auto-detect sparse usage if not specified
            # Dense matrix for 86k nodes would be ~56GB, so use sparse for large circuits
            if use_sparse is None:
                use_sparse = self.num_nodes > 1000

            if use_sparse:
                if self.verbose:
                    print(f"Using sparse hybrid solver ({self.num_nodes} nodes, OpenVAF devices)")
                return self._run_transient_hybrid_sparse(t_stop, dt, backend=backend)
            else:
                if self.verbose:
                    print("Using dense hybrid solver (OpenVAF devices detected)")
                return self._run_transient_hybrid(t_stop, dt, backend=backend)

        # Convert to MNA system
        system = self.to_mna_system()

        # Run production transient analysis with backend selection
        times, voltages_array, stats = transient_analysis_jit(
            system=system,
            t_stop=t_stop,
            t_step=dt,
            t_start=0.0,
            backend=backend,
        )

        # Create voltage dict from JAX arrays
        voltages = {}
        for i in range(self.num_nodes):
            if i < voltages_array.shape[1]:
                voltages[i] = voltages_array[:, i]
            else:
                voltages[i] = jnp.zeros(len(times))

        if self.verbose:
            print(f"Completed: {len(times)} timesteps, {stats.get('iterations', 'N/A')} total NR iterations")

        return times, voltages

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
                               use_dense: bool = True) -> Tuple[jax.Array, Dict[int, jax.Array]]:
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
        from jax_spice.analysis.gpu_backend import get_device, get_default_dtype
        from jax_spice.analysis.sparse import build_csr_arrays, sparse_solve_csr
        from jax.experimental.sparse import BCOO

        ground = 0

        # Get target device for JAX operations
        device = get_device(backend)
        dtype = get_default_dtype(backend)

        # Set up internal nodes for OpenVAF devices
        n_total, device_internal_nodes = self._setup_internal_nodes()
        n_external = self.num_nodes
        n_unknowns = n_total - 1

        solver_type = "dense batched scatter" if use_dense else "COO sparse"
        if self.verbose:
            print(f"Total nodes: {n_total} ({n_external} external, {n_total - n_external} internal)")
            print(f"Backend: {backend}, device: {device.platform}")
            print(f"Using {solver_type} solver")

        # Initialize voltages
        V = jnp.zeros(n_total, dtype=jnp.float64)
        V_prev = jnp.zeros(n_total, dtype=jnp.float64)

        # Build time-varying source function
        source_fn = self._build_source_fn()

        # Group devices by type
        openvaf_by_type: Dict[str, List[Dict]] = {}
        simple_devices = []
        for dev in self.devices:
            if dev.get('is_openvaf'):
                model_type = dev['model']
                if model_type not in openvaf_by_type:
                    openvaf_by_type[model_type] = []
                openvaf_by_type[model_type].append(dev)
            else:
                simple_devices.append(dev)

        # Prepare OpenVAF: vmapped functions, static inputs, and stamp index mappings
        vmapped_fns: Dict[str, Callable] = {}
        static_inputs_cache: Dict[str, Tuple[Any, List[int], List[Dict], Dict]] = {}

        for model_type in openvaf_by_type:
            compiled = self._compiled_models.get(model_type)
            if compiled and 'vmapped_fn' in compiled:
                vmapped_fns[model_type] = compiled['vmapped_fn']
                static_inputs, voltage_indices, device_contexts = self._prepare_static_inputs(
                    model_type, openvaf_by_type[model_type], device_internal_nodes, ground
                )
                # Pre-compute stamp index mapping (once per model type)
                stamp_indices = self._build_stamp_index_mapping(
                    model_type, device_contexts, ground
                )
                # Pre-compute voltage node arrays for vectorized update
                n_devices = len(device_contexts)
                n_voltages = len(voltage_indices)
                voltage_node1 = jnp.zeros((n_devices, n_voltages), dtype=jnp.int32)
                voltage_node2 = jnp.zeros((n_devices, n_voltages), dtype=jnp.int32)
                for dev_idx, ctx in enumerate(device_contexts):
                    for i, (n1, n2) in enumerate(ctx['voltage_node_pairs']):
                        voltage_node1 = voltage_node1.at[dev_idx, i].set(n1)
                        voltage_node2 = voltage_node2.at[dev_idx, i].set(n2)

                if backend == "gpu":
                    with jax.default_device(device):
                        static_inputs = jnp.array(static_inputs, dtype=dtype)
                else:
                    static_inputs = jnp.array(static_inputs, dtype=jnp.float64)
                static_inputs_cache[model_type] = (
                    static_inputs, voltage_indices, stamp_indices, voltage_node1, voltage_node2
                )
                if self.verbose:
                    n_devs = len(openvaf_by_type[model_type])
                    print(f"Prepared {model_type}: {n_devs} devices, stamp indices cached")

        # Pre-compute simple device stamp indices
        simple_device_data = self._prepare_simple_devices_coo(simple_devices, ground, n_unknowns)

        # Time stepping
        times = []
        voltages = {i: [] for i in range(n_external)}

        t = 0.0
        total_nr_iters = 0

        while t <= t_stop:
            source_values = source_fn(t)

            converged = False
            V_iter = V

            for nr_iter in range(100):
                # === Pure JAX: COO collection for both dense and sparse paths ===
                f_indices_list = []
                f_values_list = []
                j_rows_list = []
                j_cols_list = []
                j_vals_list = []

                # Collect from simple devices
                self._collect_simple_devices_coo(
                    simple_device_data, V_iter, V_prev, dt, source_values, ground,
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

                        # Add diagonal regularization
                        diag_idx = jnp.arange(n_unknowns, dtype=jnp.int32)
                        all_j_rows = jnp.concatenate([all_j_rows, diag_idx])
                        all_j_cols = jnp.concatenate([all_j_cols, diag_idx])
                        all_j_vals = jnp.concatenate([all_j_vals, jnp.full(n_unknowns, 1e-12)])

                        # Build BCOO and sum duplicates (native JAX sparse)
                        indices = jnp.stack([all_j_rows, all_j_cols], axis=1)
                        J_bcoo = BCOO((all_j_vals, indices), shape=(n_unknowns, n_unknowns))
                        J_bcoo = J_bcoo.sum_duplicates()

                        # Convert to CSR for solve (JAX spsolve expects CSR)
                        J_csr = J_bcoo.tocsr()
                        delta = sparse_solve_csr(
                            J_csr.data, J_csr.indices, J_csr.indptr, -f, (n_unknowns, n_unknowns)
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

            if not converged and self.verbose:
                print(f"Warning: t={t:.2e}s did not converge (max_f={max_f:.2e})")

            # Record state
            times.append(t)
            for i in range(n_external):
                voltages[i].append(float(V[i]))

            V_prev = V
            t += dt

        if self.verbose:
            print(f"Completed: {len(times)} timesteps, {total_nr_iters} total NR iterations")

        return jnp.array(times), {k: jnp.array(v) for k, v in voltages.items()}

    def _prepare_simple_devices_coo(
        self,
        simple_devices: List[Dict],
        ground: int,
        n_unknowns: int,
    ) -> Dict[str, Any]:
        """Pre-compute data structures and stamp templates for simple devices.

        Pre-computes static index arrays so runtime collection is fully vectorized
        with no Python loops.

        For 2-terminal devices (p, n), the stamp pattern is:
        - Residual: f[p] += I, f[n] -= I (2 entries, masked by ground)
        - Jacobian: J[p,p] += G, J[p,n] -= G, J[n,p] -= G, J[n,n] += G (4 entries)

        Returns dict with device data and pre-computed stamp templates.
        """
        # Group by model type
        by_type: Dict[str, List[Dict]] = {}
        for dev in simple_devices:
            model = dev['model']
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

            if model == 'resistor':
                R = jnp.array([d['params'].get('r', 1e3) for d in devs], dtype=jnp.float64)
                result['resistor'] = {**base_data, 'R': R}
            elif model == 'capacitor':
                C = jnp.array([d['params'].get('c', 1e-12) for d in devs], dtype=jnp.float64)
                result['capacitor'] = {**base_data, 'C': C}
            elif model == 'vsource':
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
            elif model == 'diode':
                Is = jnp.array([d['params'].get('is', 1e-14) for d in devs], dtype=jnp.float64)
                n_factor = jnp.array([d['params'].get('n', 1.0) for d in devs], dtype=jnp.float64)
                result['diode'] = {**base_data, 'Is': Is, 'n_factor': n_factor}

        return result

    def _collect_simple_devices_coo(
        self,
        device_data: Dict[str, Any],
        V: jax.Array,
        V_prev: jax.Array,
        dt: float,
        source_values: Dict,
        ground: int,
        f_indices: List,
        f_values: List,
        j_rows: List,
        j_cols: List,
        j_vals: List,
    ):
        """Collect COO triplets from simple devices using fully vectorized operations.

        Uses pre-computed stamp templates from _prepare_simple_devices_coo.
        No Python loops - all operations are batched JAX operations.
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

        # Resistors: I = G * (Vp - Vn), G = 1/R
        if 'resistor' in device_data:
            d = device_data['resistor']
            G = 1.0 / d['R']
            Vp = V[d['node_p']]
            Vn = V[d['node_n']]
            I = G * (Vp - Vn)
            _stamp_two_terminal(d, I, G)

        # Capacitors: I = G_eq * (V - V_prev), G_eq = C/dt
        if 'capacitor' in device_data:
            d = device_data['capacitor']
            G_eq = d['C'] / dt
            Vp, Vn = V[d['node_p']], V[d['node_n']]
            Vp_prev, Vn_prev = V_prev[d['node_p']], V_prev[d['node_n']]
            I_eq = G_eq * (Vp_prev - Vn_prev)
            I = G_eq * (Vp - Vn) - I_eq
            _stamp_two_terminal(d, I, G_eq)

        # Voltage sources: I = G * (Vp - Vn - Vtarget), G = 1e12
        if 'vsource' in device_data:
            d = device_data['vsource']
            G = 1e12
            Vp, Vn = V[d['node_p']], V[d['node_n']]
            # Build target voltage array from source_values dict
            V_target = jnp.array([
                source_values.get(name, float(dc))
                for name, dc in zip(d['names'], d['dc'])
            ])
            I = G * (Vp - Vn - V_target)
            G_arr = jnp.full(d['n'], G)
            _stamp_two_terminal(d, I, G_arr)

        # Current sources (residual only, no Jacobian)
        if 'isource' in device_data:
            d = device_data['isource']
            # Build current array from source_values dict
            I_arr = jnp.array([
                source_values.get(name, float(dc))
                for name, dc in zip(d['names'], d['dc'])
            ])
            # Residual: -I at p, +I at n (note sign convention)
            f_vals = I_arr[:, None] * jnp.array([-1.0, 1.0])[None, :]  # (n, 2)
            f_idx = d['f_indices'].ravel()
            f_val = f_vals.ravel()
            f_valid = f_idx >= 0
            f_indices.append(jnp.where(f_valid, f_idx, 0))
            f_values.append(jnp.where(f_valid, f_val, 0.0))

        # Diodes: I = Is * (exp(Vd/nVt) - 1), G = Is * exp(Vd/nVt) / nVt
        if 'diode' in device_data:
            d = device_data['diode']
            Is, n_factor = d['Is'], d['n_factor']
            Vt = 0.0258
            nVt = n_factor * Vt
            Vd = V[d['node_p']] - V[d['node_n']]
            Vd_norm = Vd / nVt

            # Vectorized diode current with limiting (no Python loop)
            exp_40 = jnp.exp(40.0)
            # Use jnp.where for vectorized conditional
            I = jnp.where(
                Vd_norm > 40,
                Is * (exp_40 + exp_40 * (Vd_norm - 40) - 1),
                jnp.where(
                    Vd_norm < -40,
                    -Is,
                    Is * (jnp.exp(Vd_norm) - 1)
                )
            )
            G = jnp.where(
                Vd_norm > 40,
                Is * exp_40 / nVt,
                jnp.where(
                    Vd_norm < -40,
                    jnp.zeros_like(Is),
                    Is * jnp.exp(Vd_norm) / nVt
                )
            )
            _stamp_two_terminal(d, I, G)

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

    def _stamp_simple_device_jax_dense(self, dev: Dict, V: jax.Array, V_prev: jax.Array,
                                        dt: float, f: jax.Array, J: jax.Array,
                                        ground: int, source_values: Dict) -> Tuple[jax.Array, jax.Array]:
        """Stamp a simple device into dense JAX matrices (f and J)."""
        model = dev['model']
        nodes = dev['nodes']
        params = dev['params']

        if model == 'vsource':
            if dev['name'] in source_values:
                V_target = source_values[dev['name']]
            else:
                V_target = params.get('dc', 0)

            np_idx, nn_idx = nodes[0], nodes[1]
            G = 1e12
            I = G * (V[np_idx] - V[nn_idx] - V_target)

            if np_idx != ground:
                f = f.at[np_idx - 1].add(I)
                J = J.at[np_idx - 1, np_idx - 1].add(G)
                if nn_idx != ground:
                    J = J.at[np_idx - 1, nn_idx - 1].add(-G)
            if nn_idx != ground:
                f = f.at[nn_idx - 1].add(-I)
                J = J.at[nn_idx - 1, nn_idx - 1].add(G)
                if np_idx != ground:
                    J = J.at[nn_idx - 1, np_idx - 1].add(-G)

        elif model == 'isource':
            if dev['name'] in source_values:
                I_val = source_values[dev['name']]
            else:
                I_val = params.get('dc', 0)

            np_idx, nn_idx = nodes[0], nodes[1]
            if np_idx != ground:
                f = f.at[np_idx - 1].add(-I_val)
            if nn_idx != ground:
                f = f.at[nn_idx - 1].add(I_val)

        elif model == 'resistor':
            R = params.get('r', 1e3)
            G = 1.0 / R
            np_idx, nn_idx = nodes[0], nodes[1]
            I = G * (V[np_idx] - V[nn_idx])

            if np_idx != ground:
                f = f.at[np_idx - 1].add(I)
                J = J.at[np_idx - 1, np_idx - 1].add(G)
                if nn_idx != ground:
                    J = J.at[np_idx - 1, nn_idx - 1].add(-G)
            if nn_idx != ground:
                f = f.at[nn_idx - 1].add(-I)
                J = J.at[nn_idx - 1, nn_idx - 1].add(G)
                if np_idx != ground:
                    J = J.at[nn_idx - 1, np_idx - 1].add(-G)

        elif model == 'capacitor':
            C = params.get('c', 1e-12)
            np_idx, nn_idx = nodes[0], nodes[1]
            G_eq = C / dt
            I_eq = G_eq * (V_prev[np_idx] - V_prev[nn_idx])
            I = G_eq * (V[np_idx] - V[nn_idx]) - I_eq

            if np_idx != ground:
                f = f.at[np_idx - 1].add(I)
                J = J.at[np_idx - 1, np_idx - 1].add(G_eq)
                if nn_idx != ground:
                    J = J.at[np_idx - 1, nn_idx - 1].add(-G_eq)
            if nn_idx != ground:
                f = f.at[nn_idx - 1].add(-I)
                J = J.at[nn_idx - 1, nn_idx - 1].add(G_eq)
                if np_idx != ground:
                    J = J.at[nn_idx - 1, np_idx - 1].add(-G_eq)

        elif model == 'diode':
            np_idx, nn_idx = nodes[0], nodes[1]
            Is = params.get('is', 1e-14)
            n = params.get('n', 1.0)
            Vt = 0.0258
            nVt = n * Vt

            Vd = V[np_idx] - V[nn_idx]
            Vd_norm = Vd / nVt
            exp_40 = jnp.exp(40.0)

            # Diode current with limiting
            I = jax.lax.cond(
                Vd_norm > 40,
                lambda: Is * (exp_40 + exp_40 * (Vd_norm - 40) - 1),
                lambda: jax.lax.cond(
                    Vd_norm < -40,
                    lambda: -Is,
                    lambda: Is * (jnp.exp(Vd_norm) - 1)
                )
            )

            # Diode conductance
            G = jax.lax.cond(
                Vd_norm > 40,
                lambda: Is * exp_40 / nVt,
                lambda: jax.lax.cond(
                    Vd_norm < -40,
                    lambda: 0.0,
                    lambda: Is * jnp.exp(Vd_norm) / nVt
                )
            )

            if np_idx != ground:
                f = f.at[np_idx - 1].add(I)
                J = J.at[np_idx - 1, np_idx - 1].add(G)
                if nn_idx != ground:
                    J = J.at[np_idx - 1, nn_idx - 1].add(-G)
            if nn_idx != ground:
                f = f.at[nn_idx - 1].add(-I)
                J = J.at[nn_idx - 1, nn_idx - 1].add(G)
                if np_idx != ground:
                    J = J.at[nn_idx - 1, np_idx - 1].add(-G)

        return f, J

    def _run_transient_hybrid_sparse(self, t_stop: float, dt: float,
                                       backend: str = "cpu") -> Tuple[jax.Array, Dict[int, jax.Array]]:
        """Run transient analysis with sparse matrix support for large circuits.

        Uses pure JAX with matrix-free GMRES for efficient memory usage
        with large circuits like c6288.

        Args:
            t_stop: Simulation stop time
            dt: Time step
            backend: 'gpu' or 'cpu' for device evaluation. GPU keeps OpenVAF
                     evaluation arrays on GPU device for reduced transfer overhead.
        """
        from jax.scipy.sparse.linalg import gmres
        from jax_spice.analysis.gpu_backend import get_device, get_default_dtype

        ground = 0

        # Get target device for JAX operations
        device = get_device(backend)
        dtype = get_default_dtype(backend)

        # Set up internal nodes for OpenVAF devices
        n_total, device_internal_nodes = self._setup_internal_nodes()
        n_external = self.num_nodes
        n_unknowns = n_total - 1

        if self.verbose:
            print(f"Total nodes: {n_total} ({n_external} external, {n_total - n_external} internal)")
            print(f"Backend: {backend}, device: {device.platform}")

        # Initialize voltages as JAX arrays for GPU compatibility
        V = jnp.zeros(n_total, dtype=jnp.float64)
        V_prev = jnp.zeros(n_total, dtype=jnp.float64)

        # Build time-varying source function
        source_fn = self._build_source_fn()

        # Group devices
        openvaf_by_type: Dict[str, List[Dict]] = {}
        simple_devices = []
        for dev in self.devices:
            if dev.get('is_openvaf'):
                model_type = dev['model']
                if model_type not in openvaf_by_type:
                    openvaf_by_type[model_type] = []
                openvaf_by_type[model_type].append(dev)
            else:
                simple_devices.append(dev)

        if self.verbose:
            print(f"Using matrix-free GMRES solver ({n_unknowns} unknowns)")

        # Get cached JIT-compiled vmapped functions and prepare static inputs
        vmapped_fns: Dict[str, Callable] = {}
        static_inputs_cache: Dict[str, Tuple[Any, List[int], List[Dict]]] = {}
        for model_type in openvaf_by_type:
            compiled = self._compiled_models.get(model_type)
            if compiled and 'vmapped_fn' in compiled:
                vmapped_fns[model_type] = compiled['vmapped_fn']
                static_inputs, voltage_indices, device_contexts = self._prepare_static_inputs(
                    model_type, openvaf_by_type[model_type], device_internal_nodes, ground
                )
                # For GPU backend, keep static inputs as JAX array on device
                if backend == "gpu":
                    with jax.default_device(device):
                        static_inputs = jnp.array(static_inputs, dtype=dtype)
                static_inputs_cache[model_type] = (static_inputs, voltage_indices, device_contexts)
                if self.verbose:
                    n_devs = len(openvaf_by_type[model_type])
                    print(f"Using JIT-compiled vmapped function for {model_type} ({n_devs} devices)")

        def build_residual(V_curr: jax.Array, V_prev_step: jax.Array,
                           source_values: Dict) -> jax.Array:
            """Build residual vector using JAX arrays."""
            f = jnp.zeros(n_unknowns, dtype=jnp.float64)

            # Stamp simple devices
            for dev in simple_devices:
                f = self._stamp_simple_device_jax(dev, V_curr, V_prev_step, dt, f, ground, source_values)

            # Stamp OpenVAF devices
            for model_type, devices in openvaf_by_type.items():
                if model_type in vmapped_fns and model_type in static_inputs_cache:
                    static_inputs, voltage_indices, device_contexts = static_inputs_cache[model_type]

                    # Update voltage inputs
                    if backend == "gpu":
                        batch_inputs = self._update_voltage_inputs_jax_gpu(
                            static_inputs, voltage_indices, device_contexts, V_curr,
                            device, dtype
                        )
                    else:
                        batch_inputs = self._update_voltage_inputs_jax(
                            static_inputs, voltage_indices, device_contexts, V_curr
                        )

                    # Evaluate all devices in parallel
                    batch_residuals, _ = vmapped_fns[model_type](batch_inputs)

                    # Stamp residuals
                    f = self._stamp_residuals_jax(
                        model_type, batch_residuals, device_contexts, f, ground
                    )

            return f

        # Time stepping
        times = []
        voltages = {i: [] for i in range(n_external)}

        t = 0.0
        total_nr_iters = 0

        while t <= t_stop:
            source_values = source_fn(t)

            converged = False
            V_iter = V

            for nr_iter in range(100):
                # Compute residual
                f = build_residual(V_iter, V_prev, source_values)
                max_f = float(jnp.max(jnp.abs(f)))

                if max_f < 1e-9:
                    converged = True
                    break

                # Matrix-free GMRES: solve J @ delta = -f
                # where J @ v is computed via jvp of residual
                def matvec(v: jax.Array) -> jax.Array:
                    # Pad v to include ground node (which doesn't change)
                    v_padded = jnp.concatenate([jnp.array([0.0], dtype=v.dtype), v])
                    # Compute Jacobian-vector product via forward-mode AD
                    _, jvp_result = jax.jvp(
                        lambda x: build_residual(x, V_prev, source_values),
                        (V_iter,),
                        (v_padded,)
                    )
                    return jvp_result

                # Solve with GMRES
                delta, info = gmres(matvec, -f, tol=1e-6, maxiter=100)

                # Limit voltage step for stability
                max_delta = float(jnp.max(jnp.abs(delta)))
                if max_delta > 1.0:
                    delta = delta * (1.0 / max_delta)

                # Update voltages (ground at index 0 stays fixed)
                V_iter = V_iter.at[1:].add(delta)

                if max_delta < 1e-12:
                    converged = True
                    break

            V = V_iter
            total_nr_iters += nr_iter + 1

            if not converged and self.verbose:
                print(f"Warning: t={t:.2e}s did not converge (max_f={max_f:.2e})")

            # Record state (only external nodes)
            times.append(t)
            for i in range(n_external):
                voltages[i].append(float(V[i]))

            V_prev = V
            t += dt

        if self.verbose:
            print(f"Completed: {len(times)} timesteps, {total_nr_iters} total NR iterations")

        return jnp.array(times), {k: jnp.array(v) for k, v in voltages.items()}

    def _stamp_simple_device_jax(self, dev: Dict, V: jax.Array, V_prev: jax.Array,
                                  dt: float, f: jax.Array, ground: int,
                                  source_values: Dict) -> jax.Array:
        """Stamp a simple device into residual vector (JAX version)."""
        model = dev['model']
        nodes = dev['nodes']
        params = dev['params']

        if model == 'vsource':
            if dev['name'] in source_values:
                V_target = source_values[dev['name']]
            else:
                V_target = params.get('dc', 0)

            np_idx, nn_idx = nodes[0], nodes[1]
            G = 1e12
            I = G * (V[np_idx] - V[nn_idx] - V_target)

            if np_idx != ground:
                f = f.at[np_idx - 1].add(I)
            if nn_idx != ground:
                f = f.at[nn_idx - 1].add(-I)

        elif model == 'isource':
            if dev['name'] in source_values:
                I_val = source_values[dev['name']]
            else:
                I_val = params.get('dc', 0)

            np_idx, nn_idx = nodes[0], nodes[1]
            if np_idx != ground:
                f = f.at[np_idx - 1].add(-I_val)
            if nn_idx != ground:
                f = f.at[nn_idx - 1].add(I_val)

        elif model == 'resistor':
            R = params.get('r', 1e3)
            G = 1.0 / R
            np_idx, nn_idx = nodes[0], nodes[1]
            I = G * (V[np_idx] - V[nn_idx])

            if np_idx != ground:
                f = f.at[np_idx - 1].add(I)
            if nn_idx != ground:
                f = f.at[nn_idx - 1].add(-I)

        elif model == 'capacitor':
            C = params.get('c', 1e-12)
            np_idx, nn_idx = nodes[0], nodes[1]
            G_eq = C / dt
            I_eq = G_eq * (V_prev[np_idx] - V_prev[nn_idx])
            I = G_eq * (V[np_idx] - V[nn_idx]) - I_eq

            if np_idx != ground:
                f = f.at[np_idx - 1].add(I)
            if nn_idx != ground:
                f = f.at[nn_idx - 1].add(-I)

        elif model == 'diode':
            np_idx, nn_idx = nodes[0], nodes[1]
            Is = params.get('is', 1e-14)
            n = params.get('n', 1.0)
            Vt = 0.0258
            nVt = n * Vt

            Vd = V[np_idx] - V[nn_idx]

            # Diode current with limiting (JAX-compatible)
            Vd_norm = Vd / nVt
            exp_40 = jnp.exp(40.0)

            # Use jax.lax.cond for JIT compatibility
            I = jax.lax.cond(
                Vd_norm > 40,
                lambda: Is * (exp_40 + exp_40 * (Vd_norm - 40) - 1),
                lambda: jax.lax.cond(
                    Vd_norm < -40,
                    lambda: -Is,
                    lambda: Is * (jnp.exp(Vd_norm) - 1)
                )
            )

            if np_idx != ground:
                f = f.at[np_idx - 1].add(I)
            if nn_idx != ground:
                f = f.at[nn_idx - 1].add(-I)

        return f

    def _update_voltage_inputs_jax(self, static_inputs: jax.Array,
                                    voltage_indices: List[int],
                                    device_contexts: List[Dict],
                                    V: jax.Array) -> jax.Array:
        """Update voltage parameters using JAX arrays (CPU path)."""
        inputs = static_inputs  # Already JAX array

        for dev_idx, ctx in enumerate(device_contexts):
            voltage_node_pairs = ctx['voltage_node_pairs']
            for i, (n1, n2) in enumerate(voltage_node_pairs):
                v1 = jax.lax.cond(n1 < len(V), lambda: V[n1], lambda: 0.0)
                v2 = jax.lax.cond(n2 < len(V), lambda: V[n2], lambda: 0.0)
                inputs = inputs.at[dev_idx, voltage_indices[i]].set(v1 - v2)

        return inputs

    def _update_voltage_inputs_jax_gpu(self, static_inputs: jax.Array,
                                        voltage_indices: List[int],
                                        device_contexts: List[Dict],
                                        V: jax.Array,
                                        device: Any, dtype: Any) -> jax.Array:
        """Update voltage parameters using JAX arrays (GPU path)."""
        n_devices = len(device_contexts)
        n_voltages = len(voltage_indices)

        # Build voltage updates as JAX array
        voltage_updates = jnp.zeros((n_devices, n_voltages), dtype=dtype)

        for dev_idx, ctx in enumerate(device_contexts):
            voltage_node_pairs = ctx['voltage_node_pairs']
            for i, (n1, n2) in enumerate(voltage_node_pairs):
                v1 = jax.lax.cond(n1 < len(V), lambda: V[n1], lambda: 0.0)
                v2 = jax.lax.cond(n2 < len(V), lambda: V[n2], lambda: 0.0)
                voltage_updates = voltage_updates.at[dev_idx, i].set(v1 - v2)

        # Update voltage columns
        inputs = static_inputs
        for i, vid in enumerate(voltage_indices):
            inputs = inputs.at[:, vid].set(voltage_updates[:, i])

        return inputs

    def _stamp_residuals_jax(self, model_type: str, batch_residuals: jax.Array,
                              device_contexts: List[Dict], f: jax.Array,
                              ground: int) -> jax.Array:
        """Stamp batched residuals into f vector (JAX version)."""
        compiled = self._compiled_models.get(model_type)
        if not compiled:
            return f

        metadata = compiled['array_metadata']
        node_names = metadata['node_names']

        for dev_idx, ctx in enumerate(device_contexts):
            node_map = ctx['node_map']

            for res_idx, node_name in enumerate(node_names):
                if node_name.startswith('sim_'):
                    model_node = node_name[4:]
                else:
                    model_node = node_name

                node_idx = node_map.get(model_node, node_map.get(node_name, None))
                if node_idx is None or node_idx == ground:
                    continue

                resist = float(batch_residuals[dev_idx, res_idx])
                if not jnp.isnan(resist) and node_idx > 0 and node_idx - 1 < len(f):
                    f = f.at[node_idx - 1].add(resist)

        return f

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
