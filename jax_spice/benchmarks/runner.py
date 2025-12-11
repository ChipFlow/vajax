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
from typing import Dict, List, Tuple, Any, Optional

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

    def _flatten_top_instances(self) -> List[Tuple[str, List[str], str, Dict[str, str]]]:
        """Flatten subcircuit instances to leaf devices.

        Returns list of (name, terminals, model, params) tuples for leaf devices.
        """
        flat_instances = []
        ground = self.circuit.ground or '0'

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

            self._compiled_models[model_type] = {
                'module': module,
                'translator': translator,
                'jax_fn': jax_fn,
                'param_names': list(module.param_names),
                'param_kinds': list(module.param_kinds),
                'nodes': list(module.nodes),
            }

            if self.verbose:
                print(f"    {model_type}: {len(module.param_names)} params, {len(module.nodes)} nodes")

    def _extract_analysis_params(self):
        """Extract analysis parameters from control block."""
        text = self.sim_path.read_text()

        # Find tran analysis: analysis <name> tran step=X stop=Y [maxstep=Z] [icmode="..."]
        match = re.search(
            r'analysis\s+\w+\s+tran\s+'
            r'(?:step=(\S+)\s+)?'
            r'(?:stop=(\S+)\s*)?'
            r'(?:maxstep=(\S+)\s*)?'
            r'(?:icmode="(\w+)")?',
            text
        )
        if match:
            step = self.parse_spice_number(match.group(1) or '1u')
            stop = self.parse_spice_number(match.group(2) or '1m')
            self.analysis_params = {
                'type': 'tran',
                'step': step,
                'stop': stop,
                'icmode': match.group(4) or 'op',
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
                    return dc + a * np.sin(2 * np.pi * f * t + ph)

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
                      max_steps: int = 10000) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
        """Run transient analysis.

        Uses the JIT-compiled solver for circuits with only simple devices.
        Uses a hybrid Python-based solver for circuits with OpenVAF devices.

        Args:
            t_stop: Stop time (default: from analysis params or 1ms)
            dt: Time step (default: from analysis params or 1µs)
            max_steps: Maximum number of time steps

        Returns:
            (times, voltages) tuple where voltages is dict mapping node index to voltage array
        """
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

        if self.verbose:
            print(f"Running transient: t_stop={t_stop:.2e}s, dt={dt:.2e}s")

        # Use hybrid solver if we have OpenVAF devices
        if self._has_openvaf_devices:
            if self.verbose:
                print("Using hybrid solver (OpenVAF devices detected)")
            return self._run_transient_hybrid(t_stop, dt)

        # Convert to MNA system
        system = self.to_mna_system()

        # Run production transient analysis
        times, voltages_array, stats = transient_analysis_jit(
            system=system,
            t_stop=t_stop,
            t_step=dt,
            t_start=0.0
        )

        # Convert JAX arrays to numpy and create voltage dict
        times_np = np.array(times)
        voltages = {}
        for i in range(self.num_nodes):
            if i < voltages_array.shape[1]:
                voltages[i] = np.array(voltages_array[:, i])
            else:
                voltages[i] = np.zeros(len(times_np))

        if self.verbose:
            print(f"Completed: {len(times_np)} timesteps, {stats.get('iterations', 'N/A')} total NR iterations")

        return times_np, voltages

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

    def _run_transient_hybrid(self, t_stop: float, dt: float) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
        """Run transient analysis with OpenVAF devices using Python-based Newton-Raphson.

        This solver handles a mix of simple devices (resistor, capacitor, etc.)
        and OpenVAF-compiled devices (like PSP103 MOSFETs).

        Internal nodes of OpenVAF devices are added to the system matrix and
        solved simultaneously with external circuit nodes.
        """
        ground = 0

        # Set up internal nodes for OpenVAF devices
        n_total, device_internal_nodes = self._setup_internal_nodes()
        n_external = self.num_nodes

        if self.verbose:
            print(f"Total nodes: {n_total} ({n_external} external, {n_total - n_external} internal)")

        # Initialize voltages (external + internal nodes)
        V = np.zeros(n_total)
        V_prev = np.zeros(n_total)

        # Build time-varying source function
        source_fn = self._build_source_fn()

        # Time stepping
        times = []
        voltages = {i: [] for i in range(n_external)}  # Only track external node voltages

        t = 0.0
        total_nr_iters = 0
        n_unknowns = n_total - 1  # Exclude ground

        while t <= t_stop:
            # Update time-varying sources
            source_values = source_fn(t)

            # Newton-Raphson iteration
            converged = False
            for nr_iter in range(100):
                # Build system: J * dV = -f
                J = np.zeros((n_unknowns, n_unknowns))
                f = np.zeros(n_unknowns)

                # Stamp all devices
                for dev in self.devices:
                    if dev.get('is_openvaf'):
                        internal_nodes = device_internal_nodes.get(dev['name'], {})
                        self._stamp_openvaf_device(dev, V, V_prev, dt, f, J, ground,
                                                   source_values, internal_nodes, n_total)
                    else:
                        self._stamp_simple_device(dev, V, V_prev, dt, f, J, ground, source_values)

                # Check convergence
                max_f = np.max(np.abs(f))
                if max_f < 1e-9:
                    converged = True
                    break

                # Add small diagonal for numerical stability
                J_reg = J + 1e-12 * np.eye(n_unknowns)

                # Solve and update
                try:
                    delta = np.linalg.solve(J_reg, -f)
                except np.linalg.LinAlgError:
                    delta = np.linalg.lstsq(J_reg, -f, rcond=None)[0]

                # Limit voltage step for convergence
                max_delta = np.max(np.abs(delta))
                if max_delta > 1.0:
                    delta = delta * (1.0 / max_delta)

                V[1:] += delta

                if max_delta < 1e-12:
                    converged = True
                    break

            total_nr_iters += nr_iter + 1

            if not converged and self.verbose:
                print(f"Warning: t={t:.2e}s did not converge (max_f={max_f:.2e})")

            # Record state (only external nodes)
            times.append(t)
            for i in range(n_external):
                voltages[i].append(V[i])

            # Advance time
            V_prev = V.copy()
            t += dt

        if self.verbose:
            print(f"Completed: {len(times)} timesteps, {total_nr_iters} total NR iterations")

        return np.array(times), {k: np.array(v) for k, v in voltages.items()}

    def _stamp_simple_device(self, dev: Dict, V: np.ndarray, V_prev: np.ndarray,
                             dt: float, f: np.ndarray, J: np.ndarray,
                             ground: int, source_values: Dict):
        """Stamp a simple device (resistor, capacitor, diode, source) into the system."""
        model = dev['model']
        nodes = dev['nodes']
        params = dev['params']

        if model == 'vsource':
            # Check for time-varying value
            if dev['name'] in source_values:
                V_target = source_values[dev['name']]
            else:
                V_target = params.get('dc', 0)

            np_idx, nn_idx = nodes[0], nodes[1]
            G = 1e12

            I = G * (V[np_idx] - V[nn_idx] - V_target)

            if np_idx != ground:
                f[np_idx - 1] += I
                J[np_idx - 1, np_idx - 1] += G
                if nn_idx != ground:
                    J[np_idx - 1, nn_idx - 1] -= G
            if nn_idx != ground:
                f[nn_idx - 1] -= I
                J[nn_idx - 1, nn_idx - 1] += G
                if np_idx != ground:
                    J[nn_idx - 1, np_idx - 1] -= G

        elif model == 'isource':
            # Current source
            if dev['name'] in source_values:
                I_dc = source_values[dev['name']]
            else:
                I_dc = params.get('dc', params.get('i', 0))

            np_idx, nn_idx = nodes[0], nodes[1]

            if np_idx != ground:
                f[np_idx - 1] += I_dc
            if nn_idx != ground:
                f[nn_idx - 1] -= I_dc

        elif model == 'resistor':
            R = params.get('r', 1000)
            G = 1.0 / max(R, 1e-12)
            np_idx, nn_idx = nodes[0], nodes[1]
            Vd = V[np_idx] - V[nn_idx]
            I = G * Vd

            if np_idx != ground:
                f[np_idx - 1] += I
                J[np_idx - 1, np_idx - 1] += G
                if nn_idx != ground:
                    J[np_idx - 1, nn_idx - 1] -= G
            if nn_idx != ground:
                f[nn_idx - 1] -= I
                J[nn_idx - 1, nn_idx - 1] += G
                if np_idx != ground:
                    J[nn_idx - 1, np_idx - 1] -= G

        elif model == 'capacitor':
            C = params.get('c', 1e-12)
            G_eq = C / max(dt, 1e-15)
            np_idx, nn_idx = nodes[0], nodes[1]

            Vd = V[np_idx] - V[nn_idx]
            Vd_prev = V_prev[np_idx] - V_prev[nn_idx]
            I = G_eq * (Vd - Vd_prev)

            if np_idx != ground:
                f[np_idx - 1] += I
                J[np_idx - 1, np_idx - 1] += G_eq
                if nn_idx != ground:
                    J[np_idx - 1, nn_idx - 1] -= G_eq
            if nn_idx != ground:
                f[nn_idx - 1] -= I
                J[nn_idx - 1, nn_idx - 1] += G_eq
                if np_idx != ground:
                    J[nn_idx - 1, np_idx - 1] -= G_eq

        elif model == 'diode':
            np_idx, nn_idx = nodes[0], nodes[1]
            Is = params.get('is', 1e-14)
            n = params.get('n', 1.0)
            Vt = 0.0258

            Vd = V[np_idx] - V[nn_idx]
            nVt = n * Vt

            # Limit exponential
            Vd_norm = Vd / nVt
            if Vd_norm > 40:
                exp_40 = np.exp(40)
                I = Is * (exp_40 + exp_40 * (Vd_norm - 40) - 1)
                gd = Is * exp_40 / nVt
            elif Vd_norm < -40:
                I = -Is
                gd = 1e-12
            else:
                exp_term = np.exp(Vd_norm)
                I = Is * (exp_term - 1)
                gd = Is * exp_term / nVt

            gd = max(gd, 1e-12)

            if np_idx != ground:
                f[np_idx - 1] += I
                J[np_idx - 1, np_idx - 1] += gd
                if nn_idx != ground:
                    J[np_idx - 1, nn_idx - 1] -= gd
            if nn_idx != ground:
                f[nn_idx - 1] -= I
                J[nn_idx - 1, nn_idx - 1] += gd
                if np_idx != ground:
                    J[nn_idx - 1, np_idx - 1] -= gd

    def _stamp_openvaf_device(self, dev: Dict, V: np.ndarray, V_prev: np.ndarray,
                              dt: float, f: np.ndarray, J: np.ndarray,
                              ground: int, source_values: Dict,
                              internal_nodes: Dict[str, int], n_total: int):
        """Stamp an OpenVAF-compiled device (like PSP103) into the system.

        Args:
            dev: Device dictionary with name, model, nodes, params
            V: Current voltage vector (includes internal nodes)
            V_prev: Previous timestep voltage vector
            dt: Time step
            f: Residual vector to stamp into
            J: Jacobian matrix to stamp into
            ground: Ground node index
            source_values: Time-varying source values
            internal_nodes: Map of internal node names to global indices for this device
            n_total: Total number of nodes in the system
        """
        model_type = dev['model']
        ext_nodes = dev['nodes']  # External node indices [d, g, s, b]
        params = dev['params']

        compiled = self._compiled_models.get(model_type)
        if not compiled:
            raise ValueError(f"OpenVAF model {model_type} not compiled")

        jax_fn = compiled['jax_fn']
        param_names = compiled['param_names']
        param_kinds = compiled['param_kinds']
        model_nodes = compiled['nodes']  # e.g., ['node0', 'node1', ..., 'node11', 'br[...]']

        # Build node map: model node name -> global circuit node index
        # External nodes (0-3): D, G, S, B
        node_map = {}
        for i, model_node in enumerate(model_nodes[:4]):
            if i < len(ext_nodes):
                node_map[model_node] = ext_nodes[i]
            else:
                node_map[model_node] = ground

        # Internal nodes (4-11): GP, SI, DI, BP, BS, BD, BI, NOI
        for model_node, global_idx in internal_nodes.items():
            node_map[model_node] = global_idx

        # Also map sim_node names used in residuals/Jacobian
        for i, model_node in enumerate(model_nodes[:-1]):  # Skip branch
            node_map[f'sim_{model_node}'] = node_map.get(model_node, ground)

        # Build input array for the JAX function
        inputs = []
        for name, kind in zip(param_names, param_kinds):
            if kind == 'voltage':
                # Parse voltage parameter names like "V(GP,SI)" or "V(DI)"
                voltage_val = self._compute_voltage_param(name, V, node_map, model_nodes, ground)
                inputs.append(voltage_val)
            elif kind == 'param':
                # Look up parameter from model card
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
                    # Default to 1.0 for unknown params
                    inputs.append(1.0)
            elif kind == 'hidden_state':
                inputs.append(0.0)
            else:
                inputs.append(0.0)

        # Evaluate device
        try:
            residuals, jacobian = jax_fn(inputs)
        except Exception as e:
            if self.verbose:
                print(f"Warning: OpenVAF evaluation failed for {dev['name']}: {e}")
            return

        # Stamp residuals into f
        for node_name, res in residuals.items():
            # Map sim_nodeX to global index
            if node_name.startswith('sim_'):
                model_node = node_name[4:]  # Remove 'sim_' prefix
            else:
                model_node = node_name

            node_idx = node_map.get(model_node, None)
            if node_idx is None:
                # Try direct lookup
                node_idx = node_map.get(node_name, None)
            if node_idx is None or node_idx == ground:
                continue

            resist = float(res.get('resist', 0))
            if not np.isnan(resist) and node_idx > 0 and node_idx - 1 < len(f):
                f[node_idx - 1] += resist

        # Stamp Jacobian
        for entry_key, entry in jacobian.items():
            row_name, col_name = entry_key

            # Map names to global indices
            row_model = row_name[4:] if row_name.startswith('sim_') else row_name
            col_model = col_name[4:] if col_name.startswith('sim_') else col_name

            row_idx = node_map.get(row_model, node_map.get(row_name, None))
            col_idx = node_map.get(col_model, node_map.get(col_name, None))

            if row_idx is None or col_idx is None:
                continue
            if row_idx == ground or col_idx == ground:
                continue

            resist = float(entry.get('resist', 0))
            if not np.isnan(resist):
                ri = row_idx - 1
                ci = col_idx - 1
                if 0 <= ri < len(f) and 0 <= ci < len(f):
                    J[ri, ci] += resist

    def _compute_voltage_param(self, name: str, V: np.ndarray,
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
