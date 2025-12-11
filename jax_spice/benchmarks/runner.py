"""VACASK Benchmark Runner

Generic runner for VACASK benchmark circuits. Parses benchmark .sim files
and runs transient analysis using our solver.
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np

from jax_spice.netlist.parser import VACASKParser
from jax_spice.netlist.circuit import Instance


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

        def eval_param_expr(expr: str, param_env: Dict[str, float]) -> float:
            """Evaluate a parameter expression like 'w*pfact' or '2*(w+ld)'."""
            if not isinstance(expr, str):
                return float(expr)

            # Try direct parse first
            val = self.parse_spice_number(expr)
            if val != 0.0 or expr.strip() in ('0', '0.0'):
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
                    inst_params[k] = str(eval_param_expr(v, param_env))

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
                    new_param_env[k] = eval_param_expr(v, new_param_env)
                # Override with instance params
                for k, v in inst.params.items():
                    new_param_env[k] = eval_param_expr(v, param_env)

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

            self.devices.append({
                'name': inst_name,
                'model': device_type,
                'nodes': nodes,
                'params': params,
            })

        if self.verbose:
            print(f"Devices: {len(self.devices)}")
            for dev in self.devices[:10]:
                print(f"  {dev['name']}: {dev['model']} nodes={dev['nodes']}")
            if len(self.devices) > 10:
                print(f"  ... and {len(self.devices) - 10} more")

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

    def run_transient(self, t_stop=None, dt=None, max_steps=10000):
        """Run transient analysis.

        Args:
            t_stop: Stop time (default: from analysis params or 1ms)
            dt: Time step (default: from analysis params or 1µs)
            max_steps: Maximum number of time steps

        Returns:
            (times, voltages) tuple
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

        source_fn = self._build_source_fn()

        # Run simple transient solver
        V = np.zeros(self.num_nodes)
        V_prev = np.zeros(self.num_nodes)

        times = []
        voltages = {i: [] for i in range(self.num_nodes)}

        t = 0.0
        while t <= t_stop:
            # Update sources
            source_values = source_fn(t)
            for dev in self.devices:
                if dev['name'] in source_values:
                    dev['params']['_time_value'] = source_values[dev['name']]

            # Newton-Raphson
            for nr_iter in range(100):
                J = np.zeros((self.num_nodes - 1, self.num_nodes - 1))
                f = np.zeros(self.num_nodes - 1)

                for dev in self.devices:
                    self._stamp_device(dev, V, V_prev, dt, f, J)

                if np.max(np.abs(f)) < 1e-9:
                    break

                delta = np.linalg.solve(J + 1e-15 * np.eye(J.shape[0]), -f)
                V[1:] += delta

            # Record
            times.append(t)
            for i in range(self.num_nodes):
                voltages[i].append(V[i])

            V_prev = V.copy()
            t += dt

        return np.array(times), {k: np.array(v) for k, v in voltages.items()}

    def _stamp_device(self, dev, V, V_prev, dt, f, J):
        """Stamp device into system matrices."""
        model = dev['model']
        nodes = dev['nodes']
        params = dev['params']
        ground = 0

        if model == 'vsource':
            V_target = params.get('_time_value', params.get('dc', 0))
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
            I_target = params.get('_time_value', params.get('dc', 0))
            np_idx, nn_idx = nodes[0], nodes[1]

            if np_idx != ground:
                f[np_idx - 1] -= I_target
            if nn_idx != ground:
                f[nn_idx - 1] += I_target

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
            np_idx, nn_idx = nodes[0], nodes[1]
            G_eq = C / dt
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
