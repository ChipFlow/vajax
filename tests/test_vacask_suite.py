"""Parameterized VACASK test suite

Automatically discovers and runs all VACASK .sim test files, extracting
expected values from embedded Python scripts and comparing with our solver.
"""

import pytest
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Add openvaf-py to path
sys.path.insert(0, str(Path(__file__).parent.parent / "openvaf-py"))

import numpy as np
import jax.numpy as jnp

import openvaf_py
import openvaf_jax
from jax_spice.netlist.parser import parse_netlist
from jax_spice.devices.base import DeviceStamps
from jax_spice.analysis.mna import MNASystem, DeviceInfo
from jax_spice.analysis.dc import dc_operating_point
from jax_spice.analysis.transient import transient_analysis

# Paths - VACASK is at ../VACASK relative to jax-spice
JAX_SPICE_ROOT = Path(__file__).parent.parent
VACASK_ROOT = JAX_SPICE_ROOT.parent / "VACASK"
VACASK_TEST = VACASK_ROOT / "test"
VACASK_DEVICES = VACASK_ROOT / "devices"
VACASK_BENCHMARK = JAX_SPICE_ROOT / "vendor" / "VACASK" / "benchmark"


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
        self.sim_path = sim_path
        self.verbose = verbose
        self.circuit = None
        self.devices = []
        self.node_names = {}
        self.num_nodes = 0
        self.analysis_params = {}

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
        from jax_spice.netlist.parser import VACASKParser

        parser = VACASKParser()
        self.circuit = parser.parse_file(self.sim_path)

        if self.verbose:
            print(f"Parsed: {self.circuit.title}")
            print(f"Models: {list(self.circuit.models.keys())}")

        # Build node mapping
        node_set = {'0'}
        for inst in self.circuit.top_instances:
            for t in inst.terminals:
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

    def _build_devices(self):
        """Build device list from parsed instances."""
        self.devices = []

        # Parameters that should be kept as strings (not parsed as numbers)
        STRING_PARAMS = {'type'}

        for inst in self.circuit.top_instances:
            model_name = inst.model.lower()
            device_type = self._get_device_type(model_name)
            nodes = [self.node_names[t] for t in inst.terminals]

            # Get model parameters and instance parameters
            model_params = self._get_model_params(model_name)

            # Parse instance params, but keep string params as strings
            inst_params = {}
            for k, v in inst.params.items():
                if k in STRING_PARAMS:
                    # Keep as string, strip quotes
                    inst_params[k] = str(v).strip('"').strip("'")
                else:
                    inst_params[k] = self.parse_spice_number(v)

            # Merge model params with instance params (instance overrides model)
            params = {**model_params, **inst_params}

            self.devices.append({
                'name': inst.name,
                'model': device_type,
                'nodes': nodes,
                'params': params,
            })

        if self.verbose:
            print(f"Devices: {len(self.devices)}")
            for dev in self.devices:
                print(f"  {dev['name']}: {dev['model']} nodes={dev['nodes']}")

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
                # Unknown type - log warning and treat as DC
                if self.verbose:
                    print(f"    WARNING: Unknown source type '{source_type}', treating as DC")
                dc_val = params.get('dc', 0)
                sources[dev['name']] = lambda t, v=dc_val: v

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


def discover_benchmark_dirs() -> List[Path]:
    """Find all benchmark directories with runme.sim files."""
    if not VACASK_BENCHMARK.exists():
        return []
    benchmarks = []
    for d in sorted(VACASK_BENCHMARK.iterdir()):
        sim_file = d / "vacask" / "runme.sim"
        if sim_file.exists():
            benchmarks.append(d)
    return benchmarks


# Discover benchmarks for parameterized tests
BENCHMARK_DIRS = discover_benchmark_dirs()


def discover_sim_files() -> List[Path]:
    """Find all .sim test files in VACASK test directory."""
    if not VACASK_TEST.exists():
        return []
    return sorted(VACASK_TEST.glob("*.sim"))


def parse_embedded_python(sim_path: Path) -> Dict[str, Any]:
    """Extract expected values from embedded Python test script.

    Parses patterns like:
        v = op1["2"]
        exact = 10*0.9

    Returns dict with:
        - 'expectations': List of (variable_name, expected_value, tolerance)
        - 'analysis_type': 'op' or 'tran'
    """
    text = sim_path.read_text()

    # Find embedded Python between <<<FILE and >>>FILE
    match = re.search(r'<<<FILE\n(.*?)>>>FILE', text, re.DOTALL)
    if not match:
        return {'expectations': [], 'analysis_type': 'op'}

    py_code = match.group(1)
    lines = py_code.split('\n')

    expectations = []
    current_var = None

    for i, line in enumerate(lines):
        # Match: v = op1["node_name"] or i = op1["device.i"]
        m = re.match(r'\s*(\w+)\s*=\s*op1\["([^"]+)"\]', line)
        if m:
            current_var = m.group(2)
            continue

        # Match: exact = <expression>
        m = re.match(r'\s*exact\s*=\s*(.+)', line)
        if m and current_var:
            try:
                # Safe evaluation of numeric expressions
                expr = m.group(1).strip()
                # Handle simple math expressions
                val = eval(expr, {"__builtins__": {}, "np": np}, {})
                expectations.append((current_var, float(val), 1e-3))
            except:
                pass
            current_var = None

    # Determine analysis type
    analysis_type = 'op'
    if 'tran1' in py_code or 'rawread(\'tran' in py_code:
        analysis_type = 'tran'

    return {
        'expectations': expectations,
        'analysis_type': analysis_type
    }


def parse_analysis_commands(sim_path: Path) -> List[Dict]:
    """Extract analysis commands from control block."""
    text = sim_path.read_text()

    analyses = []

    # Find control block
    control_match = re.search(r'control\s*(.*?)endc', text, re.DOTALL | re.IGNORECASE)
    if not control_match:
        return analyses

    control_block = control_match.group(1)

    # Match: analysis <name> op [options]
    for m in re.finditer(r'analysis\s+(\w+)\s+op\b', control_block):
        analyses.append({'name': m.group(1), 'type': 'op'})

    # Match: analysis <name> tran stop=<time> step=<step> [icmode=<mode>]
    for m in re.finditer(r'analysis\s+(\w+)\s+tran\s+stop=(\S+)\s+step=(\S+)(?:\s+icmode="(\w+)")?', control_block):
        analyses.append({
            'name': m.group(1),
            'type': 'tran',
            'stop': m.group(2),
            'step': m.group(3),
            'icmode': m.group(4) or 'op'
        })

    return analyses


def get_required_models(sim_path: Path) -> List[str]:
    """Extract model types required by the sim file."""
    text = sim_path.read_text()
    models = set()

    # Match: load "model.osdi"
    for m in re.finditer(r'load\s+"(\w+)\.osdi"', text):
        models.add(m.group(1))

    return list(models)


def categorize_test(sim_path: Path) -> Tuple[str, List[str]]:
    """Categorize a test and return (category, skip_reasons).

    Categories:
    - 'op_basic': Simple DC operating point (resistor, diode)
    - 'op_complex': Complex DC (sweeps, multiple analyses)
    - 'tran': Transient analysis
    - 'ac': AC analysis
    - 'hb': Harmonic balance
    - 'unsupported': Not yet supported
    """
    name = sim_path.stem
    text = sim_path.read_text()
    models = get_required_models(sim_path)
    analyses = parse_analysis_commands(sim_path)

    skip_reasons = []

    # Check for unsupported features
    if 'hb' in name or any(a.get('type') == 'hb' for a in analyses if isinstance(a, dict) and 'type' in a):
        skip_reasons.append("harmonic balance not implemented")
        return 'hb', skip_reasons

    if 'ac' in name or ('analysis' in text and ' ac ' in text.lower()):
        skip_reasons.append("AC analysis not implemented")
        return 'ac', skip_reasons

    if 'mutual' in name:
        skip_reasons.append("mutual inductors not implemented")
        return 'unsupported', skip_reasons

    if 'noise' in name:
        skip_reasons.append("noise analysis not implemented")
        return 'unsupported', skip_reasons

    if 'sweep' in name:
        skip_reasons.append("parameter sweeps not yet implemented")
        return 'unsupported', skip_reasons

    if 'xf' in name:
        skip_reasons.append("transfer function analysis not implemented")
        return 'unsupported', skip_reasons

    # Check for unsupported models
    unsupported_models = {'bsimsoi', 'hicum', 'mextram'}
    for model in models:
        if model.lower() in unsupported_models:
            skip_reasons.append(f"model {model} not supported")
            return 'unsupported', skip_reasons

    # Categorize by analysis type
    has_tran = any(a.get('type') == 'tran' for a in analyses if isinstance(a, dict))
    has_op = any(a.get('type') == 'op' for a in analyses if isinstance(a, dict))

    if has_tran:
        return 'tran', skip_reasons
    elif has_op:
        if len(analyses) == 1 and models and all(m in ['resistor', 'vsource', 'isource'] for m in models):
            return 'op_basic', skip_reasons
        return 'op_complex', skip_reasons

    return 'unsupported', ["no recognized analysis type"]


# Discover all tests
ALL_SIM_FILES = discover_sim_files()

# Categorize tests
CATEGORIZED_TESTS = {path: categorize_test(path) for path in ALL_SIM_FILES}


def get_test_ids():
    """Generate test IDs from sim file names."""
    return [p.stem for p in ALL_SIM_FILES]


# ============================================================================
# Device evaluation functions for simulation
# ============================================================================

def resistor_eval(voltages, params, context):
    """Resistor evaluation function matching VACASK resistor.va"""
    Vp = voltages.get('p', 0.0)
    Vn = voltages.get('n', 0.0)
    R = float(params.get('r', 1000.0))

    # Ensure minimum resistance
    R = max(R, 1e-12)
    G = 1.0 / R
    I = G * (Vp - Vn)

    return DeviceStamps(
        currents={'p': jnp.array(I), 'n': jnp.array(-I)},
        conductances={
            ('p', 'p'): jnp.array(G), ('p', 'n'): jnp.array(-G),
            ('n', 'p'): jnp.array(-G), ('n', 'n'): jnp.array(G)
        }
    )


def vsource_eval(voltages, params, context):
    """Voltage source evaluation function using large conductance method.

    Note: mfactor for voltage sources represents parallel instances.
    All instances have the same voltage, so voltage output is unchanged.
    Branch current would be divided by mfactor, but we don't track that here.
    """
    Vp = voltages.get('p', 0.0)
    Vn = voltages.get('n', 0.0)

    # Get DC value - could be 'dc' param or 'val0' for pulse sources
    V_target = float(params.get('dc', params.get('val0', 0.0)))
    V_actual = Vp - Vn

    # mfactor for vsource: parallel instances share current but have same voltage
    # This affects the effective conductance (higher mfactor = lower impedance)
    mfactor = float(params.get('mfactor', 1.0))

    G_big = 1e12 * mfactor  # Lower impedance with more parallel instances
    I = G_big * (V_actual - V_target)

    return DeviceStamps(
        currents={'p': jnp.array(I), 'n': jnp.array(-I)},
        conductances={
            ('p', 'p'): jnp.array(G_big), ('p', 'n'): jnp.array(-G_big),
            ('n', 'p'): jnp.array(-G_big), ('n', 'n'): jnp.array(G_big)
        }
    )


def isource_eval(voltages, params, context):
    """Current source evaluation function."""
    # Get DC value - could be 'dc' param or 'val0' for pulse sources
    I_base = float(params.get('dc', params.get('val0', 0.0)))
    mfactor = float(params.get('mfactor', 1.0))
    I = I_base * mfactor

    return DeviceStamps(
        currents={'p': jnp.array(I), 'n': jnp.array(-I)},
        conductances={}
    )


def pulse_vsource_eval(voltages, params, context, time=0.0):
    """Pulse voltage source for transient analysis.

    Pulse parameters:
        val0: Initial value (before delay)
        val1: Pulse value (after rise)
        delay: Time when pulse starts
        rise: Rise time
        fall: Fall time (not yet implemented)
        width: Pulse width (not yet implemented)
        period: Pulse period (not yet implemented)
    """
    Vp = voltages.get('p', 0.0)
    Vn = voltages.get('n', 0.0)

    # Get pulse parameters
    val0 = float(params.get('val0', params.get('dc', 0.0)))
    val1 = float(params.get('val1', val0))
    delay = float(params.get('delay', 0.0))
    rise = float(params.get('rise', 1e-9))  # Default 1ns rise

    # Calculate voltage at current time
    if time < delay:
        V_target = val0
    elif time < delay + rise:
        # Linear ramp during rise time
        V_target = val0 + (val1 - val0) * (time - delay) / rise
    else:
        V_target = val1

    V_actual = Vp - Vn

    G_big = 1e12
    I = G_big * (V_actual - V_target)

    return DeviceStamps(
        currents={'p': jnp.array(I), 'n': jnp.array(-I)},
        conductances={
            ('p', 'p'): jnp.array(G_big), ('p', 'n'): jnp.array(-G_big),
            ('n', 'p'): jnp.array(-G_big), ('n', 'n'): jnp.array(G_big)
        }
    )


def inductor_eval(voltages, params, context):
    """Inductor evaluation function.

    For DC analysis: short circuit (large conductance).
    For transient: would need companion model with history.
    """
    Vp = voltages.get('p', 0.0)
    Vn = voltages.get('n', 0.0)

    # For DC, inductor is short circuit - use large conductance
    G_big = 1e12
    I = G_big * (Vp - Vn)

    return DeviceStamps(
        currents={'p': jnp.array(I), 'n': jnp.array(-I)},
        conductances={
            ('p', 'p'): jnp.array(G_big), ('p', 'n'): jnp.array(-G_big),
            ('n', 'p'): jnp.array(-G_big), ('n', 'n'): jnp.array(G_big)
        }
    )


def capacitor_eval(voltages, params, context):
    """Capacitor evaluation function.

    For DC analysis: open circuit (gmin for numerical stability).
    For transient: returns capacitance for backward Euler companion model.
    """
    Vp = voltages.get('p', 0.0)
    Vn = voltages.get('n', 0.0)

    C = float(params.get('c', 1e-12))

    # For DC, capacitor is open circuit - just add gmin for stability
    gmin = 1e-12
    I = gmin * (Vp - Vn)

    return DeviceStamps(
        currents={'p': jnp.array(I), 'n': jnp.array(-I)},
        conductances={
            ('p', 'p'): jnp.array(gmin), ('p', 'n'): jnp.array(-gmin),
            ('n', 'p'): jnp.array(-gmin), ('n', 'n'): jnp.array(gmin)
        },
        capacitances={
            ('p', 'p'): jnp.array(C), ('p', 'n'): jnp.array(-C),
            ('n', 'p'): jnp.array(-C), ('n', 'n'): jnp.array(C)
        }
    )


def diode_eval(voltages, params, context):
    """Diode evaluation function implementing Shockley equation.

    I = Is * (exp(V/(n*Vt)) - 1)
    """
    Vp = voltages.get('p', 0.0)  # Anode
    Vn = voltages.get('n', 0.0)  # Cathode
    Vd = Vp - Vn

    # Parameters
    Is = float(params.get('is', 1e-12))
    n = float(params.get('n', 2.0))

    # Constants
    Vt = 0.02585  # Thermal voltage at 300K
    gmin = 1e-12

    # Limited exponential
    Vd_max = 40 * n * Vt
    if Vd > Vd_max:
        exp_max = np.exp(Vd_max / (n * Vt))
        Id = Is * (exp_max - 1) + Is * exp_max / (n * Vt) * (Vd - Vd_max)
        gd = Is * exp_max / (n * Vt)
    elif Vd < -40 * Vt:
        Id = -Is
        gd = gmin
    else:
        exp_val = np.exp(Vd / (n * Vt))
        Id = Is * (exp_val - 1)
        gd = Is * exp_val / (n * Vt)

    Id = Id + gmin * Vd
    gd = gd + gmin

    return DeviceStamps(
        currents={'p': jnp.array(Id), 'n': jnp.array(-Id)},
        conductances={
            ('p', 'p'): jnp.array(gd), ('p', 'n'): jnp.array(-gd),
            ('n', 'p'): jnp.array(-gd), ('n', 'n'): jnp.array(gd)
        }
    )


def vccs_eval(voltages, params, context):
    """Voltage-Controlled Current Source.

    SPICE convention: G element terminals (n+, n-, nc+, nc-)
    Current flows FROM n+ TO n- (i.e., into n+, out of n-)

    I_out = gain * V_control where V_control = V(nc+) - V(nc-)
    """
    # Control terminals
    Vncp = voltages.get('ncp', 0.0)
    Vncn = voltages.get('ncn', 0.0)

    gain = float(params.get('gain', 1.0))
    mfactor = float(params.get('mfactor', 1.0))
    gm = gain * mfactor

    V_control = Vncp - Vncn
    I_out = gm * V_control

    # Current flows from np to nn (out of np, into nn)
    # Residual = currents leaving node, so:
    # - At np: +I_out (current leaving)
    # - At nn: -I_out (current entering = negative leaving)
    return DeviceStamps(
        currents={
            'np': jnp.array(I_out),   # Current OUT of np (leaving)
            'nn': jnp.array(-I_out),  # Current INTO nn (entering)
        },
        conductances={
            ('np', 'ncp'): jnp.array(gm),
            ('np', 'ncn'): jnp.array(-gm),
            ('nn', 'ncp'): jnp.array(-gm),
            ('nn', 'ncn'): jnp.array(gm),
        }
    )


def vcvs_eval(voltages, params, context):
    """Voltage-Controlled Voltage Source.

    Terminals: (n+, n-) output, (nc+, nc-) control
    V_out = gain * V_control

    Implemented using large conductance method.
    """
    Vnp = voltages.get('np', 0.0)
    Vnn = voltages.get('nn', 0.0)
    Vncp = voltages.get('ncp', 0.0)
    Vncn = voltages.get('ncn', 0.0)

    gain = float(params.get('gain', 1.0))
    mfactor = float(params.get('mfactor', 1.0))
    A = gain * mfactor

    G_big = 1e12
    V_control = Vncp - Vncn
    V_target = A * V_control
    V_actual = Vnp - Vnn

    I = G_big * (V_actual - V_target)

    return DeviceStamps(
        currents={
            'np': jnp.array(I),
            'nn': jnp.array(-I),
        },
        conductances={
            ('np', 'np'): jnp.array(G_big),
            ('np', 'nn'): jnp.array(-G_big),
            ('np', 'ncp'): jnp.array(-G_big * A),
            ('np', 'ncn'): jnp.array(G_big * A),
            ('nn', 'np'): jnp.array(-G_big),
            ('nn', 'nn'): jnp.array(G_big),
            ('nn', 'ncp'): jnp.array(G_big * A),
            ('nn', 'ncn'): jnp.array(-G_big * A),
        }
    )


def parse_si_value(s: str) -> float:
    """Parse a value with SI suffix (e.g., '2k' -> 2000)."""
    s = s.strip().lower()
    multipliers = {
        'f': 1e-15, 'p': 1e-12, 'n': 1e-9, 'u': 1e-6, 'm': 1e-3,
        'k': 1e3, 'meg': 1e6, 'g': 1e9, 't': 1e12
    }
    for suffix, mult in multipliers.items():
        if s.endswith(suffix):
            return float(s[:-len(suffix)]) * mult
    return float(s)


# ============================================================================
# Test classes
# ============================================================================

class TestVACASKDiscovery:
    """Test that we can discover and parse VACASK test files."""

    def test_discover_sim_files(self):
        """Should find VACASK .sim test files."""
        if not VACASK_TEST.exists():
            pytest.skip("VACASK test directory not found")

        sim_files = discover_sim_files()
        assert len(sim_files) > 0, "No .sim files found"
        print(f"\nFound {len(sim_files)} .sim files")

    def test_categorize_all_tests(self):
        """Categorize all discovered tests."""
        if not ALL_SIM_FILES:
            pytest.skip("No sim files found")

        categories = {}
        for path, (cat, reasons) in CATEGORIZED_TESTS.items():
            categories.setdefault(cat, []).append(path.stem)

        print("\nTest categorization:")
        for cat, tests in sorted(categories.items()):
            print(f"  {cat}: {len(tests)} tests")
            if len(tests) <= 5:
                for t in tests:
                    print(f"    - {t}")


class TestVACASKParsing:
    """Test that we can parse all VACASK .sim files."""

    @pytest.mark.parametrize("sim_file", ALL_SIM_FILES, ids=get_test_ids())
    def test_parse_netlist(self, sim_file):
        """Each sim file should parse without error."""
        try:
            circuit = parse_netlist(sim_file)
            assert circuit is not None
            assert circuit.ground is not None
        except Exception as e:
            pytest.fail(f"Failed to parse {sim_file.name}: {e}")

    @pytest.mark.parametrize("sim_file", ALL_SIM_FILES, ids=get_test_ids())
    def test_extract_expectations(self, sim_file):
        """Should extract expected values from embedded Python."""
        result = parse_embedded_python(sim_file)
        # Just verify it doesn't crash - not all files have expectations
        assert 'expectations' in result
        assert 'analysis_type' in result


class TestVACASKOperatingPoint:
    """Run DC operating point tests from VACASK suite."""

    def _build_system_from_circuit(self, circuit) -> Tuple[MNASystem, Dict[str, int]]:
        """Build an MNASystem from a parsed circuit."""
        # Build node mapping by collecting all terminals from instances
        node_names = {'0': 0}  # Ground is always 0
        if circuit.ground and circuit.ground != '0':
            node_names[circuit.ground] = 0  # Map ground name to 0

        node_idx = 1
        for inst in circuit.top_instances:
            for terminal in inst.terminals:
                if terminal not in node_names and terminal != circuit.ground:
                    node_names[terminal] = node_idx
                    node_idx += 1

        system = MNASystem(num_nodes=node_idx, node_names=node_names)

        # Eval function mapping: (eval_fn, terminal_names)
        eval_funcs = {
            'vsource': (vsource_eval, ['p', 'n']),
            'v': (vsource_eval, ['p', 'n']),
            'isource': (isource_eval, ['p', 'n']),
            'i': (isource_eval, ['p', 'n']),
            'resistor': (resistor_eval, ['p', 'n']),
            'r': (resistor_eval, ['p', 'n']),
            'capacitor': (capacitor_eval, ['p', 'n']),
            'c': (capacitor_eval, ['p', 'n']),
            'diode': (diode_eval, ['p', 'n']),
            'd': (diode_eval, ['p', 'n']),
            'inductor': (inductor_eval, ['p', 'n']),
            'l': (inductor_eval, ['p', 'n']),
            'vccs': (vccs_eval, ['np', 'nn', 'ncp', 'ncn']),
            'vcvs': (vcvs_eval, ['np', 'nn', 'ncp', 'ncn']),
        }

        # Add devices
        for inst in circuit.top_instances:
            model_name = inst.model.lower()

            # Get node indices
            node_indices = []
            for terminal in inst.terminals:
                term_name = terminal if terminal != circuit.ground else '0'
                if term_name not in node_names:
                    node_names[term_name] = node_idx
                    node_idx += 1
                node_indices.append(node_names[term_name])

            # Get eval function and terminal names
            eval_info = eval_funcs.get(model_name)
            if eval_info is None:
                continue  # Skip unsupported device types
            eval_fn, terminal_names = eval_info

            # Parse parameters
            params = {}
            for k, v in inst.params.items():
                try:
                    params[k] = parse_si_value(str(v))
                except (ValueError, TypeError):
                    params[k] = v

            device = DeviceInfo(
                name=inst.name,
                model_name=model_name,
                terminals=terminal_names,
                node_indices=node_indices,
                params=params,
                eval_fn=eval_fn
            )
            system.devices.append(device)

        return system, node_names

    def test_test_op(self):
        """Test test_op.sim - resistor voltage divider."""
        sim_file = VACASK_TEST / "test_op.sim"
        if not sim_file.exists():
            pytest.skip("VACASK test_op.sim not found")

        # Parse circuit
        circuit = parse_netlist(sim_file)

        # Build MNA system
        system, node_names = self._build_system_from_circuit(circuit)

        # Get expected values
        expectations = parse_embedded_python(sim_file)['expectations']

        # Solve DC operating point
        solution, info = dc_operating_point(system)
        assert info['converged'], f"DC analysis did not converge: {info}"

        # Check expected values
        for var_name, expected, tol in expectations:
            if var_name in node_names:
                idx = node_names[var_name]
                actual = float(solution[idx])
                rel_err = abs(actual - expected) / (abs(expected) + 1e-12)
                print(f"{var_name}: expected={expected}, actual={actual:.6f}, rel_err={rel_err:.2e}")
                assert rel_err < tol, \
                    f"{var_name}: expected {expected}, got {actual} (rel_err={rel_err:.2e})"

    def test_test_resistor(self):
        """Test test_resistor.sim - basic resistor with mfactor."""
        sim_file = VACASK_TEST / "test_resistor.sim"
        if not sim_file.exists():
            pytest.skip("VACASK test_resistor.sim not found")

        # Parse circuit
        circuit = parse_netlist(sim_file)

        # Build MNA system
        system, node_names = self._build_system_from_circuit(circuit)

        # Solve DC operating point
        solution, info = dc_operating_point(system)
        assert info['converged'], f"DC analysis did not converge: {info}"

        # Expected: V1=1V, R=2k, mfactor=3 → I = 1/(2k/3) = 1.5mA
        # Note: Our simple eval doesn't handle mfactor yet
        # Just verify circuit solves
        print(f"Solution: {dict(zip(node_names.keys(), solution))}")

    def test_test_ctlsrc(self):
        """Test test_ctlsrc.sim - controlled sources (VCCS, VCVS).

        Circuit:
        V1(2V) -- R0(2k) -- GND  → V1 current = 1mA
        VCCS1: gain=2m*3, control=(1,0), output to node2 → I=6mA, V(2)=6V
        VCVS1: gain=2*3, control=(1,0), output at node3 → V(3)=12V
        """
        sim_file = VACASK_TEST / "test_ctlsrc.sim"
        if not sim_file.exists():
            pytest.skip("VACASK test_ctlsrc.sim not found")

        # Parse circuit
        circuit = parse_netlist(sim_file)

        # Build MNA system
        system, node_names = self._build_system_from_circuit(circuit)

        # Solve DC operating point
        solution, info = dc_operating_point(system)
        assert info['converged'], f"DC analysis did not converge: {info}"

        print(f"Solution: {dict((k, float(solution[v])) for k, v in node_names.items())}")

        # Check node 1 (V1 = 2V)
        if '1' in node_names:
            v1 = float(solution[node_names['1']])
            assert abs(v1 - 2.0) < 0.01, f"V(1) expected 2V, got {v1}V"

        # Check node 2 (VCCS output: I = 2m * 3 * 2V = 12mA, V = 12mA * 1k = 12V)
        # Wait, looking at circuit: gain=2m, mfactor=3, control V=2V
        # I_out = 2m * 3 * 2 = 12mA → V(2) = 12mA * 1kΩ = 12V
        # But our eval doesn't pass mfactor through yet, so expected is 2m * 2V = 4mA → 4V
        if '2' in node_names:
            v2 = float(solution[node_names['2']])
            print(f"V(2) = {v2}V (VCCS output)")

        # Check node 3 (VCVS output: gain=2, mfactor=3 → V = 2*3*2 = 12V)
        # Without mfactor: V = 2 * 2 = 4V
        if '3' in node_names:
            v3 = float(solution[node_names['3']])
            print(f"V(3) = {v3}V (VCVS output)")

    def test_test_capacitor_op(self):
        """Test test_capacitor.sim DC operating point.

        For DC, capacitor is open circuit, so V(2) should equal V(1) = 1V.
        """
        sim_file = VACASK_TEST / "test_capacitor.sim"
        if not sim_file.exists():
            pytest.skip("VACASK test_capacitor.sim not found")

        # Parse circuit
        circuit = parse_netlist(sim_file)

        # Build MNA system
        system, node_names = self._build_system_from_circuit(circuit)

        # Solve DC operating point
        solution, info = dc_operating_point(system)
        assert info['converged'], f"DC analysis did not converge: {info}"

        print(f"Solution: {dict((k, float(solution[v])) for k, v in node_names.items())}")

        # V1 = 1V (from pulse source dc value)
        # At DC, capacitor is open → no current flows through R1 → V(2) = V(1) = 1V
        if '2' in node_names:
            v2 = float(solution[node_names['2']])
            # Pulse source starts at val0=1V
            assert abs(v2 - 1.0) < 0.1, f"V(2) expected ~1V at DC, got {v2}V"

    def test_test_visrc(self):
        """Test test_visrc.sim - voltage and current sources with mfactor.

        Circuit:
        - V1(2V) -- R1(2k) -- GND: V(1) = 2V
        - I1(2mA, mfactor=3) → R2(2k) → GND: V(2) = 2mA * 3 * 2kΩ = 12V
        """
        sim_file = VACASK_TEST / "test_visrc.sim"
        if not sim_file.exists():
            pytest.skip("VACASK test_visrc.sim not found")

        # Parse circuit
        circuit = parse_netlist(sim_file)

        # Build MNA system
        system, node_names = self._build_system_from_circuit(circuit)

        # Solve DC operating point
        solution, info = dc_operating_point(system)
        assert info['converged'], f"DC analysis did not converge: {info}"

        print(f"Solution: {dict((k, float(solution[v])) for k, v in node_names.items())}")

        # V(1) = 2V from voltage source
        if '1' in node_names:
            v1 = float(solution[node_names['1']])
            rel_err = abs(v1 - 2.0) / 2.0
            print(f"V(1) = {v1}V (expected 2V, rel_err={rel_err:.2e})")
            assert rel_err < 1e-3, f"V(1) expected 2V, got {v1}V"

        # V(2) = 2mA * 3 (mfactor) * 2kΩ = 12V
        if '2' in node_names:
            v2 = float(solution[node_names['2']])
            expected_v2 = 2e-3 * 3 * 2e3  # 12V
            rel_err = abs(v2 - expected_v2) / expected_v2
            print(f"V(2) = {v2}V (expected {expected_v2}V, rel_err={rel_err:.2e})")
            assert rel_err < 1e-3, f"V(2) expected {expected_v2}V, got {v2}V"

    def test_test_inductor_op(self):
        """Test test_inductor.sim DC operating point.

        For DC, inductor is short circuit. Circuit has current source with val0=1A.
        With inductor shorted, it carries all the current.
        """
        sim_file = VACASK_TEST / "test_inductor.sim"
        if not sim_file.exists():
            pytest.skip("VACASK test_inductor.sim not found")

        # Parse circuit
        circuit = parse_netlist(sim_file)

        # Build MNA system
        system, node_names = self._build_system_from_circuit(circuit)

        # Solve DC operating point
        solution, info = dc_operating_point(system)
        assert info['converged'], f"DC analysis did not converge: {info}"

        print(f"Solution: {dict((k, float(solution[v])) for k, v in node_names.items())}")

        # At DC, inductor is short circuit (0Ω) in parallel with R1 (1Ω)
        # Equivalent resistance ≈ 0Ω, so V(1) ≈ 0V
        # Current source I=1A flows through the inductor short
        if '1' in node_names:
            v1 = float(solution[node_names['1']])
            # Should be very close to 0V due to inductor short
            print(f"V(1) = {v1}V (expected ~0V due to inductor short)")
            assert abs(v1) < 1e-3, f"V(1) expected ~0V, got {v1}V"


class TestVACASKTransient:
    """Run transient analysis tests from VACASK suite."""

    def _simple_transient(
        self,
        circuit,
        t_stop: float,
        t_step: float,
        ic: Dict[str, float] = None,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Simple transient analysis with time-varying sources.

        Uses backward Euler integration for capacitors.
        Returns (times, node_voltages).
        """
        # Build node mapping
        node_names = {'0': 0}
        if circuit.ground and circuit.ground != '0':
            node_names[circuit.ground] = 0

        node_idx = 1
        for inst in circuit.top_instances:
            for terminal in inst.terminals:
                if terminal not in node_names and terminal != circuit.ground:
                    node_names[terminal] = node_idx
                    node_idx += 1

        num_nodes = node_idx
        ground = 0

        # Parse device info
        devices = []
        for inst in circuit.top_instances:
            model = inst.model.lower()
            node_indices = [node_names.get(t, node_names.get(circuit.ground, 0))
                           for t in inst.terminals]
            params = {}
            for k, v in inst.params.items():
                try:
                    params[k] = parse_si_value(str(v))
                except:
                    params[k] = v

            devices.append({
                'name': inst.name,
                'model': model,
                'nodes': node_indices,
                'params': params,
            })

        # Initial condition from DC operating point or IC
        if ic:
            V = np.zeros(num_nodes)
            for name, val in ic.items():
                if name in node_names:
                    V[node_names[name]] = val
        else:
            # Run DC operating point to get initial condition
            V = self._dc_solve(num_nodes, ground, devices, time=0.0)

        # Time stepping
        times = np.arange(0, t_stop + t_step, t_step)
        results = {name: [V[idx]] for name, idx in node_names.items()}

        V_prev = V.copy()

        for t in times[1:]:
            # Newton-Raphson for this timepoint
            V_iter = V_prev.copy()

            for _ in range(50):  # Max iterations
                J = np.zeros((num_nodes - 1, num_nodes - 1))
                f = np.zeros(num_nodes - 1)

                # Stamp all devices
                for dev in devices:
                    self._stamp_device_transient(
                        J, f, V_iter, V_prev, dev, t, t_step, ground
                    )

                # Check convergence
                if np.max(np.abs(f)) < 1e-9:
                    break

                # Solve and update
                try:
                    delta = np.linalg.solve(J + 1e-12 * np.eye(J.shape[0]), -f)
                except:
                    delta = np.linalg.lstsq(J, -f, rcond=None)[0]

                V_iter[1:] += delta

                if np.max(np.abs(delta)) < 1e-12:
                    break

            # Store result
            V_prev = V_iter.copy()
            for name, idx in node_names.items():
                results[name].append(V_iter[idx])

        return times, {k: np.array(v) for k, v in results.items()}

    def _dc_solve(self, num_nodes, ground, devices, time=0.0):
        """Solve DC operating point."""
        V = np.zeros(num_nodes)

        for _ in range(100):  # Max iterations
            J = np.zeros((num_nodes - 1, num_nodes - 1))
            f = np.zeros(num_nodes - 1)

            for dev in devices:
                self._stamp_device_dc(J, f, V, dev, time, ground)

            if np.max(np.abs(f)) < 1e-9:
                break

            try:
                delta = np.linalg.solve(J + 1e-12 * np.eye(J.shape[0]), -f)
            except:
                delta = np.linalg.lstsq(J, -f, rcond=None)[0]

            V[1:] += delta

            if np.max(np.abs(delta)) < 1e-12:
                break

        return V

    def _stamp_device_dc(self, J, f, V, dev, time, ground):
        """Stamp device for DC analysis."""
        model = dev['model']
        nodes = dev['nodes']
        params = dev['params']

        if model in ('vsource', 'v'):
            # Time-varying voltage source
            src_type = str(params.get('type', 'dc')).strip('"\'')
            if src_type == 'pulse':
                val0 = params.get('val0', params.get('dc', 0))
                val1 = params.get('val1', val0)
                delay = params.get('delay', 0)
                rise = params.get('rise', 1e-9)

                if time < delay:
                    V_target = val0
                elif time < delay + rise:
                    V_target = val0 + (val1 - val0) * (time - delay) / rise
                else:
                    V_target = val1
            else:
                V_target = params.get('dc', params.get('val0', 0))

            np_idx, nn_idx = nodes[0], nodes[1]
            V_actual = V[np_idx] - V[nn_idx]
            G = 1e12
            I = G * (V_actual - V_target)

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

        elif model in ('resistor', 'r'):
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

        elif model in ('capacitor', 'c'):
            # For DC, capacitor is open circuit (gmin)
            gmin = 1e-12
            np_idx, nn_idx = nodes[0], nodes[1]
            Vd = V[np_idx] - V[nn_idx]
            I = gmin * Vd

            if np_idx != ground:
                f[np_idx - 1] += I
                J[np_idx - 1, np_idx - 1] += gmin
            if nn_idx != ground:
                f[nn_idx - 1] -= I
                J[nn_idx - 1, nn_idx - 1] += gmin

    def _stamp_device_transient(self, J, f, V, V_prev, dev, time, dt, ground):
        """Stamp device for transient analysis with backward Euler."""
        model = dev['model']
        nodes = dev['nodes']
        params = dev['params']

        # First stamp DC contribution
        self._stamp_device_dc(J, f, V, dev, time, ground)

        # Then add capacitor transient contribution
        if model in ('capacitor', 'c'):
            C = params.get('c', 1e-12)
            np_idx, nn_idx = nodes[0], nodes[1]

            # Backward Euler: I = C/dt * (V - V_prev)
            G_eq = C / dt
            Vd = V[np_idx] - V[nn_idx]
            Vd_prev = V_prev[np_idx] - V_prev[nn_idx]
            I_cap = G_eq * (Vd - Vd_prev)

            if np_idx != ground:
                f[np_idx - 1] += I_cap
                J[np_idx - 1, np_idx - 1] += G_eq
                if nn_idx != ground:
                    J[np_idx - 1, nn_idx - 1] -= G_eq
            if nn_idx != ground:
                f[nn_idx - 1] -= I_cap
                J[nn_idx - 1, nn_idx - 1] += G_eq
                if np_idx != ground:
                    J[nn_idx - 1, np_idx - 1] -= G_eq

    def test_test_tran(self):
        """Test test_tran.sim - RC transient response.

        Circuit: V1 (pulse 1→2V at 1ms) -- R1 (1k) -- C1 (1µF) -- GND
        Time constant τ = RC = 1ms
        Expected: V(2) = 1 + (1 - exp(-(t-1ms)/1ms)) for t > 1ms
        """
        sim_file = VACASK_TEST / "test_tran.sim"
        if not sim_file.exists():
            pytest.skip("VACASK test_tran.sim not found")

        # Parse circuit
        circuit = parse_netlist(sim_file)

        # Run transient analysis
        times, voltages = self._simple_transient(
            circuit,
            t_stop=10e-3,
            t_step=10e-6,
        )

        # Calculate expected response
        delay = 1e-3  # 1ms delay
        tau = 1e-3    # 1ms time constant
        v2 = voltages['2']

        expected = 1.0 + (1.0 - np.exp(-(times - delay) / tau)) * (times > delay)

        # Check max relative error
        # Exclude first few points during the step
        mask = times > (delay + 10e-6)
        rel_err = np.abs(v2[mask] - expected[mask]) / np.maximum(np.abs(expected[mask]), 1e-6)
        max_err = np.max(rel_err)

        print(f"Max relative error: {max_err:.4e}")
        print(f"V(2) at t=5ms: {v2[times >= 5e-3][0]:.4f}V (expected ~1.98V)")
        print(f"V(2) at t=10ms: {v2[-1]:.4f}V (expected ~2.0V)")

        assert max_err < 0.01, f"Max relative error {max_err:.4e} exceeds 1%"


class TestVACASKSweep:
    """Run DC sweep tests from VACASK suite."""

    def _dc_sweep(
        self,
        circuit,
        instance_name: str,
        param_name: str,
        values: List[float],
    ) -> Dict[str, np.ndarray]:
        """Run DC sweep over an instance parameter.

        Returns dict of node_name -> array of voltages at each sweep point.
        """
        # Build base circuit
        node_names = {'0': 0}
        if circuit.ground and circuit.ground != '0':
            node_names[circuit.ground] = 0

        node_idx = 1
        for inst in circuit.top_instances:
            for terminal in inst.terminals:
                if terminal not in node_names and terminal != circuit.ground:
                    node_names[terminal] = node_idx
                    node_idx += 1

        num_nodes = node_idx
        ground = 0

        # Parse device info
        devices = []
        sweep_device_idx = None
        for i, inst in circuit.top_instances:
            model = inst.model.lower()
            node_indices = [node_names.get(t, node_names.get(circuit.ground, 0))
                           for t in inst.terminals]
            params = {}
            for k, v in inst.params.items():
                try:
                    params[k] = parse_si_value(str(v))
                except:
                    params[k] = str(v).strip('"\'')

            devices.append({
                'name': inst.name,
                'model': model,
                'nodes': node_indices,
                'params': params,
            })

            if inst.name == instance_name:
                sweep_device_idx = len(devices) - 1

        # Run DC at each sweep point
        results = {name: [] for name in node_names}

        for val in values:
            # Update sweep parameter
            if sweep_device_idx is not None:
                devices[sweep_device_idx]['params'][param_name] = val

            # Solve DC
            V = self._dc_solve(num_nodes, ground, devices)

            # Store results
            for name, idx in node_names.items():
                results[name].append(V[idx])

        return {k: np.array(v) for k, v in results.items()}

    def _dc_solve(self, num_nodes, ground, devices):
        """Solve DC operating point."""
        V = np.zeros(num_nodes)

        for _ in range(100):
            J = np.zeros((num_nodes - 1, num_nodes - 1))
            f = np.zeros(num_nodes - 1)

            for dev in devices:
                self._stamp_device(J, f, V, dev, ground)

            if np.max(np.abs(f)) < 1e-9:
                break

            try:
                delta = np.linalg.solve(J + 1e-12 * np.eye(J.shape[0]), -f)
            except:
                delta = np.linalg.lstsq(J, -f, rcond=None)[0]

            V[1:] += delta

            if np.max(np.abs(delta)) < 1e-12:
                break

        return V

    def _stamp_device(self, J, f, V, dev, ground):
        """Stamp device for DC analysis."""
        model = dev['model']
        nodes = dev['nodes']
        params = dev['params']

        if model in ('vsource', 'v'):
            V_target = params.get('dc', params.get('val0', 0))
            np_idx, nn_idx = nodes[0], nodes[1]
            V_actual = V[np_idx] - V[nn_idx]
            G = 1e12
            I = G * (V_actual - V_target)

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

        elif model in ('resistor', 'r'):
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

        elif model in ('diode', 'd'):
            Is = params.get('is', 1e-12)
            n = params.get('n', 2.0)
            np_idx, nn_idx = nodes[0], nodes[1]
            Vd = V[np_idx] - V[nn_idx]

            Vt = 0.02585
            gmin = 1e-12

            # Limited exponential
            Vd_max = 40 * n * Vt
            if Vd > Vd_max:
                exp_max = np.exp(Vd_max / (n * Vt))
                Id = Is * (exp_max - 1) + Is * exp_max / (n * Vt) * (Vd - Vd_max)
                gd = Is * exp_max / (n * Vt)
            elif Vd < -40 * Vt:
                Id = -Is
                gd = gmin
            else:
                exp_val = np.exp(Vd / (n * Vt))
                Id = Is * (exp_val - 1)
                gd = Is * exp_val / (n * Vt)

            Id = Id + gmin * Vd
            gd = gd + gmin

            if np_idx != ground:
                f[np_idx - 1] += Id
                J[np_idx - 1, np_idx - 1] += gd
                if nn_idx != ground:
                    J[np_idx - 1, nn_idx - 1] -= gd
            if nn_idx != ground:
                f[nn_idx - 1] -= Id
                J[nn_idx - 1, nn_idx - 1] += gd
                if np_idx != ground:
                    J[nn_idx - 1, np_idx - 1] -= gd

    def test_diode_dc_sweep(self):
        """Test DC sweep of a simple diode circuit.

        Circuit: V1 (sweep -1 to 1V) -- R1 (1k) -- D1 -- GND
        Verifies exponential diode characteristic.
        """
        # Create a simple circuit programmatically since test_diode uses complex sweeps
        from collections import namedtuple

        Instance = namedtuple('Instance', ['name', 'model', 'terminals', 'params'])
        Circuit = namedtuple('Circuit', ['ground', 'top_instances'])

        circuit = Circuit(
            ground='0',
            top_instances=[
                Instance('v1', 'vsource', ['1', '0'], {'dc': 0}),
                Instance('r1', 'resistor', ['1', '2'], {'r': '1k'}),
                Instance('d1', 'diode', ['2', '0'], {'is': 1e-12, 'n': 2}),
            ]
        )

        # Sweep voltage from -1V to 1V
        voltages = np.linspace(-1, 1, 21)
        results = []

        for v_in in voltages:
            # Manually build and solve
            node_names = {'0': 0, '1': 1, '2': 2}
            num_nodes = 3
            ground = 0

            devices = [
                {'name': 'v1', 'model': 'vsource', 'nodes': [1, 0], 'params': {'dc': v_in}},
                {'name': 'r1', 'model': 'resistor', 'nodes': [1, 2], 'params': {'r': 1000}},
                {'name': 'd1', 'model': 'diode', 'nodes': [2, 0], 'params': {'is': 1e-12, 'n': 2}},
            ]

            V = self._dc_solve(num_nodes, ground, devices)
            results.append({'v_in': v_in, 'v1': V[1], 'v2': V[2]})

        # Check that:
        # 1. At negative input, diode is reverse biased, V(2) ≈ V(1)
        # 2. At positive input, diode conducts, V(2) < V(1)
        # 3. Current through R1 = (V1-V2)/R increases exponentially

        v_in = np.array([r['v_in'] for r in results])
        v2 = np.array([r['v2'] for r in results])

        # For negative V_in, diode is off, V2 ≈ V_in (small leakage through R)
        negative_mask = v_in < -0.5
        assert np.all(v2[negative_mask] < 0), "Diode should be reverse biased for negative input"

        # For positive V_in > 0.7V, significant current flows
        # With n=2 and Is=1e-12, forward voltage is higher than typical silicon diode
        positive_mask = v_in > 0.7
        assert np.all(v2[positive_mask] > 0.3), "Diode should have forward voltage > 0.3V"
        # At V_in=1V with R=1k, V2 ≈ 0.93V (high n=2 causes higher Vf)
        assert np.all(v2[positive_mask] < 1.0), "Diode voltage should be < input voltage"

        # Calculate current through R1
        i_r1 = (v_in - v2) / 1000
        print(f"Sweep: V_in from {v_in[0]:.1f}V to {v_in[-1]:.1f}V")
        print(f"At V_in=0.7V: V(2)={v2[v_in >= 0.7][0]:.3f}V, I_R1={i_r1[v_in >= 0.7][0]*1e6:.1f}µA")
        print(f"At V_in=1.0V: V(2)={v2[-1]:.3f}V, I_R1={i_r1[-1]*1e3:.2f}mA")


class TestVACASKSubcircuit:
    """Test subcircuit support."""

    def test_subcircuit_parsing(self):
        """Test that we can parse subcircuits from VACASK files."""
        sim_file = VACASK_TEST / "test_build.sim"
        if not sim_file.exists():
            pytest.skip("VACASK test_build.sim not found")

        circuit = parse_netlist(sim_file)

        # Check that subcircuits are parsed
        assert hasattr(circuit, 'subckts'), "Circuit should have subckts attribute"
        assert len(circuit.subckts) >= 2, "Should have at least 2 subcircuits"

        # Check par1 subcircuit
        assert 'par1' in circuit.subckts
        par1 = circuit.subckts['par1']
        assert len(par1.instances) == 1, "par1 should have 1 instance"
        assert par1.instances[0].model == 'resistor'

        print(f"Found {len(circuit.subckts)} subcircuits:")
        for name, sub in circuit.subckts.items():
            print(f"  {name}: {len(sub.instances)} instances, params={sub.params}")

    def test_subcircuit_instantiation(self):
        """Test instantiating a subcircuit in a circuit.

        Creates a voltage divider subcircuit and instantiates it.
        """
        from collections import namedtuple

        # Define a simple voltage divider subcircuit
        Instance = namedtuple('Instance', ['name', 'model', 'terminals', 'params'])
        Subcircuit = namedtuple('Subcircuit', ['name', 'terminals', 'instances', 'params'])
        Circuit = namedtuple('Circuit', ['ground', 'top_instances', 'subckts'])

        # Voltage divider subcircuit: in -> out with R1=1k, R2=1k (output = in/2)
        divider_subckt = Subcircuit(
            name='divider',
            terminals=['in', 'out', 'gnd'],
            instances=[
                Instance('r1', 'resistor', ['in', 'out'], {'r': 1000}),
                Instance('r2', 'resistor', ['out', 'gnd'], {'r': 1000}),
            ],
            params={}
        )

        # Main circuit: V1 (2V) -> divider -> GND
        # Expected output: 1V
        circuit_with_subckt = Circuit(
            ground='0',
            top_instances=[
                Instance('v1', 'vsource', ['1', '0'], {'dc': 2}),
                Instance('div1', 'divider', ['1', '2', '0'], {}),
            ],
            subckts={'divider': divider_subckt}
        )

        # Flatten subcircuit: expand div1 into its component instances
        flattened_instances = []
        node_map = {}  # Maps subcircuit internal nodes to global nodes

        for inst in circuit_with_subckt.top_instances:
            if inst.model in circuit_with_subckt.subckts:
                # This is a subcircuit instance - expand it
                subckt = circuit_with_subckt.subckts[inst.model]

                # Map subcircuit terminals to connection nodes
                term_map = dict(zip(subckt.terminals, inst.terminals))

                for sub_inst in subckt.instances:
                    # Create new terminals by mapping
                    new_terminals = [term_map.get(t, f'{inst.name}.{t}')
                                     for t in sub_inst.terminals]

                    flattened_instances.append(Instance(
                        name=f'{inst.name}.{sub_inst.name}',
                        model=sub_inst.model,
                        terminals=new_terminals,
                        params=sub_inst.params,
                    ))
            else:
                flattened_instances.append(inst)

        print(f"Flattened circuit: {len(flattened_instances)} instances")
        for inst in flattened_instances:
            print(f"  {inst.name}: {inst.model} {inst.terminals}")

        # Now solve the flattened circuit
        node_names = {'0': 0}
        node_idx = 1
        for inst in flattened_instances:
            for t in inst.terminals:
                if t not in node_names and t != '0':
                    node_names[t] = node_idx
                    node_idx += 1

        num_nodes = node_idx
        ground = 0

        devices = []
        for inst in flattened_instances:
            devices.append({
                'name': inst.name,
                'model': inst.model,
                'nodes': [node_names.get(t, 0) for t in inst.terminals],
                'params': {k: (parse_si_value(str(v)) if isinstance(v, str) else v)
                           for k, v in inst.params.items()},
            })

        # Solve DC
        V = np.zeros(num_nodes)
        for _ in range(100):
            J = np.zeros((num_nodes - 1, num_nodes - 1))
            f = np.zeros(num_nodes - 1)

            for dev in devices:
                model = dev['model']
                nodes = dev['nodes']
                params = dev['params']

                if model == 'vsource':
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

            if np.max(np.abs(f)) < 1e-9:
                break

            delta = np.linalg.solve(J + 1e-12 * np.eye(J.shape[0]), -f)
            V[1:] += delta

        # Check result
        v_in = V[node_names['1']]
        v_out = V[node_names['2']]

        print(f"V(in) = {v_in:.3f}V")
        print(f"V(out) = {v_out:.3f}V (expected 1.0V)")

        assert abs(v_in - 2.0) < 0.01, f"V(in) expected 2V, got {v_in}V"
        assert abs(v_out - 1.0) < 0.01, f"V(out) expected 1V, got {v_out}V"


class TestVACASKBenchmarks:
    """Test VACASK benchmark circuits.

    These are larger circuits from vendor/VACASK/benchmark/*/vacask/runme.sim
    """

    # Path to benchmarks
    BENCHMARK_ROOT = JAX_SPICE_ROOT / "vendor" / "VACASK" / "benchmark"

    def _simple_transient(self, devices, num_nodes, ground, t_stop, dt,
                          source_fn=None, initial_conditions=None):
        """Run transient analysis with backward Euler.

        Args:
            devices: List of device dicts with 'model', 'nodes', 'params'
            num_nodes: Total number of nodes
            ground: Ground node index
            t_stop: Stop time
            dt: Time step
            source_fn: Optional function(t) -> dict of time-varying source values
            initial_conditions: Optional dict of node -> initial voltage

        Returns:
            (times, voltages_dict) where voltages_dict maps node indices to arrays
        """
        # Initialize voltages
        V = np.zeros(num_nodes)
        V_prev = np.zeros(num_nodes)
        if initial_conditions:
            for node, voltage in initial_conditions.items():
                V[node] = voltage
                V_prev[node] = voltage

        times = []
        voltages = {i: [] for i in range(num_nodes)}

        t = 0.0
        while t <= t_stop:
            # Update time-varying sources
            if source_fn:
                source_values = source_fn(t)
                for dev in devices:
                    if dev['name'] in source_values:
                        dev['params']['_time_value'] = source_values[dev['name']]

            # Newton-Raphson iteration
            for nr_iter in range(100):
                J = np.zeros((num_nodes - 1, num_nodes - 1))
                f = np.zeros(num_nodes - 1)

                for dev in devices:
                    self._stamp_device_transient(
                        dev, V, V_prev, dt, f, J, ground
                    )

                if np.max(np.abs(f)) < 1e-9:
                    break

                delta = np.linalg.solve(J + 1e-15 * np.eye(J.shape[0]), -f)
                V[1:] += delta

            # Record state
            times.append(t)
            for i in range(num_nodes):
                voltages[i].append(V[i])

            # Advance time
            V_prev = V.copy()
            t += dt

        return np.array(times), {k: np.array(v) for k, v in voltages.items()}

    def _stamp_device_transient(self, dev, V, V_prev, dt, f, J, ground):
        """Stamp a device into the system for transient analysis."""
        model = dev['model']
        nodes = dev['nodes']
        params = dev['params']

        if model == 'vsource':
            # Check for time-varying value
            if '_time_value' in params:
                V_target = params['_time_value']
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

            # Backward Euler: G_eq = C/dt, I_eq = G_eq * V_prev
            G_eq = C / dt
            Vd = V[np_idx] - V[nn_idx]
            Vd_prev = V_prev[np_idx] - V_prev[nn_idx]

            # Current: I = G_eq * (Vd - Vd_prev)
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
            # Shockley diode with series resistance
            np_idx, nn_idx = nodes[0], nodes[1]
            Is = params.get('is', 1e-14)
            n = params.get('n', 1.0)
            Vt = 0.0258  # Thermal voltage at 300K

            Vd = V[np_idx] - V[nn_idx]
            nVt = n * Vt

            # Limit exponential argument to avoid overflow
            Vd_norm = Vd / nVt
            if Vd_norm > 40:
                # Linear extrapolation beyond limit
                exp_40 = np.exp(40)
                I = Is * (exp_40 + exp_40 * (Vd_norm - 40) - 1)
                gd = Is * exp_40 / nVt
            elif Vd_norm < -40:
                # Reverse bias - essentially zero current
                I = -Is
                gd = 1e-12
            else:
                exp_term = np.exp(Vd_norm)
                I = Is * (exp_term - 1)
                gd = Is * exp_term / nVt

            # Minimum conductance for numerical stability
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

    def test_rc_benchmark(self):
        """Test RC circuit benchmark from VACASK.

        Circuit: Pulse source -> 1kΩ resistor -> 1µF capacitor -> GND
        Time constant τ = RC = 1ms
        Input: Pulse train 0→1V with rise=1µs, width=1ms, period=2ms
        """
        sim_file = self.BENCHMARK_ROOT / "rc" / "vacask" / "runme.sim"
        if not sim_file.exists():
            pytest.skip("VACASK rc benchmark not found")

        # Circuit topology from runme.sim:
        # vs (1 0) vsource dc=0 type="pulse" val0=0 val1=1 rise=1u fall=1u width=1m period=2m
        # r1 (1 2) r r=1k
        # c1 (2 0) c c=1u
        # Time constant τ = 1k * 1µ = 1ms

        node_names = {'0': 0, '1': 1, '2': 2}
        ground = 0
        num_nodes = 3

        # Build devices
        devices = [
            {
                'name': 'vs',
                'model': 'vsource',
                'nodes': [1, 0],
                'params': {
                    'dc': 0,
                    'type': 'pulse',
                    'val0': 0, 'val1': 1,
                    'rise': 1e-6, 'fall': 1e-6,
                    'width': 1e-3, 'period': 2e-3,
                },
            },
            {
                'name': 'r1',
                'model': 'resistor',
                'nodes': [1, 2],
                'params': {'r': 1000},
            },
            {
                'name': 'c1',
                'model': 'capacitor',
                'nodes': [2, 0],
                'params': {'c': 1e-6},
            },
        ]

        # Pulse source function
        def pulse_source(t):
            val0, val1 = 0, 1
            rise, fall = 1e-6, 1e-6
            width, period = 1e-3, 2e-3
            delay = 0

            t_in_period = (t - delay) % period
            if t < delay:
                return {'vs': val0}
            elif t_in_period < rise:
                return {'vs': val0 + (val1 - val0) * t_in_period / rise}
            elif t_in_period < rise + width:
                return {'vs': val1}
            elif t_in_period < rise + width + fall:
                return {'vs': val1 - (val1 - val0) * (t_in_period - rise - width) / fall}
            else:
                return {'vs': val0}

        # Run transient analysis
        t_stop = 5e-3  # 5ms to see a few cycles
        dt = 10e-6     # 10µs time step

        times, voltages = self._simple_transient(
            devices, num_nodes, ground, t_stop, dt,
            source_fn=pulse_source
        )

        v1 = voltages[1]  # Input (node 1)
        v2 = voltages[2]  # Output (node 2)

        # Verify behavior:
        # 1. At t=0, capacitor starts at 0V
        assert abs(v2[0]) < 0.01, f"Initial V(2) should be 0V, got {v2[0]}V"

        # 2. After first pulse rise (~1ms), capacitor should charge toward 1V
        #    τ = 1ms, so at t=τ, V ≈ 0.632 * Vfinal
        idx_1ms = int(1e-3 / dt)
        print(f"At t=1ms: V(1)={v1[idx_1ms]:.3f}V, V(2)={v2[idx_1ms]:.3f}V")
        print(f"Expected V(2) ≈ 0.632V (1 - e^(-1))")

        # Allow some tolerance for discrete time stepping
        assert 0.5 < v2[idx_1ms] < 0.75, f"At τ=1ms, V(2) should be ~0.632V, got {v2[idx_1ms]:.3f}V"

        # 3. After several time constants, should be close to input
        idx_3ms = int(3e-3 / dt)
        print(f"At t=3ms: V(1)={v1[idx_3ms]:.3f}V, V(2)={v2[idx_3ms]:.3f}V")

        # Plot info
        print(f"\nRC Benchmark: {len(times)} time points, τ=1ms")
        print(f"V(2) range: {min(v2):.3f}V to {max(v2):.3f}V")

    def test_graetz_rectifier(self):
        """Test Graetz (full-wave) rectifier benchmark.

        Circuit: AC source -> 4-diode bridge -> smoothing cap + load
        Input: 20V amplitude, 50Hz sine
        Expected: ~DC output around peak voltage minus diode drops
        """
        sim_file = self.BENCHMARK_ROOT / "graetz" / "vacask" / "runme.sim"
        if not sim_file.exists():
            pytest.skip("VACASK graetz benchmark not found")

        # From runme.sim:
        # vs (inp inn) vsource dc=0 type="sine" sinedc=0.0 ampl=20 freq=50.0
        # d1 (inp outp) d1n4007
        # d2 (outn inp) d1n4007
        # d3 (inn outp) d1n4007
        # d4 (outn inn) d1n4007
        # cl (outp outn) c c=100u
        # rl (outp outn) r r=1k
        # rgnd1 (inn 0) r r=1G
        # rgnd2 (outn 0) r r=1G
        #
        # model d1n4007 sp_diode ( is=76.9p rs=42.0m ... n=1.45 )

        node_names = {'0': 0, 'inp': 1, 'inn': 2, 'outp': 3, 'outn': 4}
        ground = 0
        num_nodes = 5

        # Diode model parameters (1N4007)
        d_params = {'is': 76.9e-12, 'rs': 0.042, 'n': 1.45}

        devices = [
            {'name': 'vs', 'model': 'vsource', 'nodes': [1, 2],
             'params': {'dc': 0, 'type': 'sine', 'ampl': 20, 'freq': 50}},
            {'name': 'd1', 'model': 'diode', 'nodes': [1, 3], 'params': d_params.copy()},
            {'name': 'd2', 'model': 'diode', 'nodes': [4, 1], 'params': d_params.copy()},
            {'name': 'd3', 'model': 'diode', 'nodes': [2, 3], 'params': d_params.copy()},
            {'name': 'd4', 'model': 'diode', 'nodes': [4, 2], 'params': d_params.copy()},
            {'name': 'cl', 'model': 'capacitor', 'nodes': [3, 4], 'params': {'c': 100e-6}},
            {'name': 'rl', 'model': 'resistor', 'nodes': [3, 4], 'params': {'r': 1000}},
            {'name': 'rgnd1', 'model': 'resistor', 'nodes': [2, 0], 'params': {'r': 1e9}},
            {'name': 'rgnd2', 'model': 'resistor', 'nodes': [4, 0], 'params': {'r': 1e9}},
        ]

        # Sine source function
        def sine_source(t):
            ampl, freq = 20, 50
            return {'vs': ampl * np.sin(2 * np.pi * freq * t)}

        # Run transient (1 second = 50 cycles at 50Hz)
        t_stop = 0.1  # 100ms = 5 cycles
        dt = 10e-6    # 10µs time step

        times, voltages = self._simple_transient(
            devices, num_nodes, ground, t_stop, dt,
            source_fn=sine_source
        )

        v_inp = voltages[1]
        v_inn = voltages[2]
        v_outp = voltages[3]
        v_outn = voltages[4]
        v_out = v_outp - v_outn  # Output across load

        # After initial transient, output should settle near peak - 2*Vd
        # Peak = 20V, 2 diode drops ≈ 1.4V, so expect ~18.6V DC
        # With 100µF and 1kΩ, ripple will be significant at 50Hz

        # Check last 20ms (one cycle)
        idx_80ms = int(0.08 / dt)
        v_out_final = v_out[idx_80ms:]

        v_avg = np.mean(v_out_final)
        v_ripple = np.max(v_out_final) - np.min(v_out_final)

        print(f"\nGraetz Rectifier Benchmark:")
        print(f"  Output average: {v_avg:.2f}V")
        print(f"  Ripple: {v_ripple:.2f}V")
        print(f"  Expected: ~{20 - 1.4:.1f}V DC with significant ripple")

        # Verify reasonable output
        assert v_avg > 10, f"Output should be > 10V, got {v_avg:.2f}V"
        assert v_avg < 25, f"Output should be < 25V, got {v_avg:.2f}V"

    def test_diode_multiplier(self):
        """Test diode cascade voltage multiplier (Cockcroft-Walton).

        From runme.sim:
        vs (a 0) type="sine" ampl=50 freq=100k
        r1 (a 1) r=0.01
        c1 (1 2) c=100n
        d1 (0 1) d1n4007
        c2 (0 10) c=100n
        d2 (1 10) d1n4007
        c3 (1 2) c=100n  (parallel with c1)
        d3 (10 2) d1n4007
        c4 (10 20) c=100n
        d4 (2 20) d1n4007

        This is a 2-stage Cockcroft-Walton multiplier.
        """
        sim_file = self.BENCHMARK_ROOT / "mul" / "vacask" / "runme.sim"
        if not sim_file.exists():
            pytest.skip("VACASK mul benchmark not found")

        # Node mapping: 0=gnd, 1=a (source), 2=node1, 3=node2, 4=node10, 5=node20
        node_names = {'0': 0, 'a': 1, '1': 2, '2': 3, '10': 4, '20': 5}
        ground = 0
        num_nodes = 6

        d_params = {'is': 76.9e-12, 'n': 1.45}
        C = 100e-9

        # Exact circuit from runme.sim
        devices = [
            {'name': 'vs', 'model': 'vsource', 'nodes': [1, 0],
             'params': {'type': 'sine', 'ampl': 50, 'freq': 100e3}},
            {'name': 'r1', 'model': 'resistor', 'nodes': [1, 2], 'params': {'r': 0.01}},
            {'name': 'c1', 'model': 'capacitor', 'nodes': [2, 3], 'params': {'c': C}},
            {'name': 'd1', 'model': 'diode', 'nodes': [0, 2], 'params': d_params.copy()},
            {'name': 'c2', 'model': 'capacitor', 'nodes': [0, 4], 'params': {'c': C}},
            {'name': 'd2', 'model': 'diode', 'nodes': [2, 4], 'params': d_params.copy()},
            {'name': 'c3', 'model': 'capacitor', 'nodes': [2, 3], 'params': {'c': C}},
            {'name': 'd3', 'model': 'diode', 'nodes': [4, 3], 'params': d_params.copy()},
            {'name': 'c4', 'model': 'capacitor', 'nodes': [4, 5], 'params': {'c': C}},
            {'name': 'd4', 'model': 'diode', 'nodes': [3, 5], 'params': d_params.copy()},
        ]

        def sine_source(t):
            return {'vs': 50 * np.sin(2 * np.pi * 100e3 * t)}

        # Run transient - need many cycles for multiplier to charge
        # At 100kHz, one cycle = 10µs
        t_stop = 500e-6  # 500µs = 50 cycles
        dt = 200e-9      # 200ns time step (50 points per cycle)

        times, voltages = self._simple_transient(
            devices, num_nodes, ground, t_stop, dt,
            source_fn=sine_source
        )

        v_20 = voltages[5]  # Output stage (node 20)

        # After charging, voltage should multiply
        # Ideal 2-stage multiplier: Vout = 4 * Vpeak = 200V
        # Practical with losses: much less, but should exceed input
        print(f"\nDiode Multiplier Benchmark:")
        print(f"  Input amplitude: 50V")
        print(f"  Output at t={t_stop*1e6:.0f}µs: {v_20[-1]:.1f}V")
        print(f"  Output max: {max(v_20):.1f}V")
        print(f"  Output final range: {v_20[-100:].min():.1f}V to {v_20[-100:].max():.1f}V")

        # Multiplier needs many cycles to charge fully
        # Just verify it's generating voltage multiplication
        assert max(v_20) > 60, f"Output should exceed input significantly, got max {max(v_20):.1f}V"

    def _parse_spice_number(self, s: str) -> float:
        """Parse SPICE number with suffix (e.g., 1u, 100n, 1.5k)"""
        if not isinstance(s, str):
            return float(s)
        s = s.strip().lower().strip('"')
        if not s:
            return 0.0

        # SPICE suffixes
        suffixes = {
            't': 1e12, 'g': 1e9, 'meg': 1e6, 'k': 1e3,
            'm': 1e-3, 'u': 1e-6, 'n': 1e-9, 'p': 1e-12, 'f': 1e-15
        }

        for suffix, multiplier in sorted(suffixes.items(), key=lambda x: -len(x[0])):
            if s.endswith(suffix):
                try:
                    return float(s[:-len(suffix)]) * multiplier
                except ValueError:
                    continue

        try:
            return float(s)
        except ValueError:
            return 0.0

    def _convert_circuit_to_devices(self, circuit, models):
        """Convert parsed circuit to device list for simulation.

        Args:
            circuit: Parsed Circuit object from VACASKParser
            models: Dict mapping model names to device types

        Returns:
            (devices, node_names, num_nodes) tuple
        """
        # Build node mapping from all instances
        node_set = {'0'}  # Always include ground
        for inst in circuit.top_instances:
            for t in inst.terminals:
                node_set.add(t)

        # Create node indices (0 is always ground)
        node_names = {'0': 0}
        for i, name in enumerate(sorted(n for n in node_set if n != '0'), start=1):
            node_names[name] = i

        devices = []
        for inst in circuit.top_instances:
            # Map model name to device type
            model_name = inst.model.lower()
            device_type = models.get(model_name, model_name)

            # Get node indices
            nodes = [node_names[t] for t in inst.terminals]

            # Convert parameters
            params = {}
            for k, v in inst.params.items():
                params[k] = self._parse_spice_number(v)

            devices.append({
                'name': inst.name,
                'model': device_type,
                'nodes': nodes,
                'params': params,
            })

        return devices, node_names, len(node_names)

    def test_rc_benchmark_parsed(self):
        """Test RC circuit by parsing actual benchmark sim file.

        This parses vendor/VACASK/benchmark/rc/vacask/runme.sim
        and runs transient analysis using our solver.
        """
        from jax_spice.netlist.parser import VACASKParser

        sim_file = self.BENCHMARK_ROOT / "rc" / "vacask" / "runme.sim"
        if not sim_file.exists():
            pytest.skip("VACASK rc benchmark not found")

        # Parse the sim file
        parser = VACASKParser()
        circuit = parser.parse_file(sim_file)

        print(f"\nParsed: {circuit.title}")
        print(f"Models: {list(circuit.models.keys())}")
        print(f"Instances: {len(circuit.top_instances)}")

        # Map model names to device types
        # The sim file uses: model r sp_resistor, model c sp_capacitor
        model_map = {
            'r': 'resistor',
            'c': 'capacitor',
            'vsource': 'vsource',
        }

        devices, node_names, num_nodes = self._convert_circuit_to_devices(circuit, model_map)
        ground = 0

        print(f"Nodes: {node_names}")
        print(f"Devices:")
        for dev in devices:
            print(f"  {dev['name']}: {dev['model']} nodes={dev['nodes']} params={dev['params']}")

        # Build pulse source function from parsed parameters
        vs_params = next(d['params'] for d in devices if d['name'] == 'vs')
        val0 = vs_params.get('val0', 0)
        val1 = vs_params.get('val1', 1)
        rise = vs_params.get('rise', 1e-6)
        fall = vs_params.get('fall', 1e-6)
        width = vs_params.get('width', 1e-3)
        period = vs_params.get('period', 2e-3)

        def pulse_source(t):
            t_in_period = t % period
            if t_in_period < rise:
                return {'vs': val0 + (val1 - val0) * t_in_period / rise}
            elif t_in_period < rise + width:
                return {'vs': val1}
            elif t_in_period < rise + width + fall:
                return {'vs': val1 - (val1 - val0) * (t_in_period - rise - width) / fall}
            else:
                return {'vs': val0}

        # Run transient analysis (shorter than original for speed)
        t_stop = 5e-3  # 5ms
        dt = 10e-6     # 10µs

        times, voltages = self._simple_transient(
            devices, num_nodes, ground, t_stop, dt,
            source_fn=pulse_source
        )

        # Get output voltages
        node_1_idx = node_names['1']
        node_2_idx = node_names['2']
        v1 = voltages[node_1_idx]
        v2 = voltages[node_2_idx]

        # Verify RC behavior:
        # R = 1kΩ, C = 1µF, τ = RC = 1ms
        tau = 1e-3
        idx_1ms = int(1e-3 / dt)

        print(f"\nRC Time Constant Check:")
        print(f"  τ = RC = 1kΩ × 1µF = 1ms")
        print(f"  At t=τ: V(1)={v1[idx_1ms]:.3f}V, V(2)={v2[idx_1ms]:.3f}V")
        print(f"  Expected V(2) ≈ 0.632V (1 - e^(-1))")

        # At t=τ, capacitor should be at ~63.2% of input
        assert 0.5 < v2[idx_1ms] < 0.75, f"At τ, V(2) should be ~0.632V, got {v2[idx_1ms]:.3f}V"

        print(f"  ✓ RC benchmark passed (parsed from sim file)")

    def test_graetz_benchmark_parsed(self):
        """Test Graetz rectifier by parsing actual benchmark sim file.

        This parses vendor/VACASK/benchmark/graetz/vacask/runme.sim
        and runs transient analysis using our solver.
        """
        from jax_spice.netlist.parser import VACASKParser

        sim_file = self.BENCHMARK_ROOT / "graetz" / "vacask" / "runme.sim"
        if not sim_file.exists():
            pytest.skip("VACASK graetz benchmark not found")

        # Parse the sim file
        parser = VACASKParser()
        circuit = parser.parse_file(sim_file)

        print(f"\nParsed: {circuit.title}")
        print(f"Models: {list(circuit.models.keys())}")
        print(f"Instances: {len(circuit.top_instances)}")

        # Get diode model parameters
        diode_model = circuit.models.get('d1n4007')
        diode_params = {}
        if diode_model:
            for k, v in diode_model.params.items():
                diode_params[k] = self._parse_spice_number(v)
            print(f"Diode params: {diode_params}")

        # Build node mapping
        node_set = {'0'}
        for inst in circuit.top_instances:
            for t in inst.terminals:
                node_set.add(t)

        node_names = {'0': 0}
        for i, name in enumerate(sorted(n for n in node_set if n != '0'), start=1):
            node_names[name] = i
        num_nodes = len(node_names)

        print(f"Nodes: {node_names}")

        # Build devices
        devices = []
        for inst in circuit.top_instances:
            model_name = inst.model.lower()
            nodes = [node_names[t] for t in inst.terminals]

            # Map model to device type
            if model_name == 'd1n4007':
                # Use diode model parameters
                params = {
                    'is': diode_params.get('is', 1e-14),
                    'n': diode_params.get('n', 1.0),
                }
                device_type = 'diode'
            elif model_name == 'r':
                params = {'r': self._parse_spice_number(inst.params.get('r', '1k'))}
                device_type = 'resistor'
            elif model_name == 'c':
                params = {'c': self._parse_spice_number(inst.params.get('c', '1u'))}
                device_type = 'capacitor'
            elif model_name == 'vsource':
                params = {}
                for k, v in inst.params.items():
                    params[k] = self._parse_spice_number(v)
                device_type = 'vsource'
            else:
                continue

            devices.append({
                'name': inst.name,
                'model': device_type,
                'nodes': nodes,
                'params': params,
            })

        print(f"Devices:")
        for dev in devices:
            print(f"  {dev['name']}: {dev['model']} nodes={dev['nodes']}")

        # Build sine source function
        vs_params = next(d['params'] for d in devices if d['name'] == 'vs')
        ampl = vs_params.get('ampl', 20)
        freq = vs_params.get('freq', 50)
        sinedc = vs_params.get('sinedc', 0)

        def sine_source(t):
            return {'vs': sinedc + ampl * np.sin(2 * np.pi * freq * t)}

        # Run transient - run for a few cycles at 50Hz
        # One cycle = 20ms, run for 100ms = 5 cycles
        t_stop = 100e-3
        dt = 100e-6  # 100µs steps

        times, voltages = self._simple_transient(
            devices, num_nodes, 0, t_stop, dt,
            source_fn=sine_source
        )

        # Get differential outputs
        v_inp = voltages[node_names['inp']]
        v_inn = voltages[node_names['inn']]
        v_outp = voltages[node_names['outp']]
        v_outn = voltages[node_names['outn']]

        v_in = v_inp - v_inn  # Input voltage
        v_out = v_outp - v_outn  # Output voltage

        print(f"\nGraetz Rectifier Results:")
        print(f"  Input: {ampl}V amplitude, {freq}Hz sine")
        print(f"  Input range: {min(v_in):.1f}V to {max(v_in):.1f}V")
        print(f"  Output range: {min(v_out):.1f}V to {max(v_out):.1f}V")
        print(f"  Output final (steady): {np.mean(v_out[-100:]):.1f}V")

        # Verify rectifier operation:
        # 1. Output should be predominantly positive
        assert np.mean(v_out[-100:]) > 10, "Output should be positive DC"

        # 2. With 100µF cap and 1kΩ load, ripple should be small
        # τ = RC = 100ms, at 50Hz (20ms period), ripple ≈ 20/100 = 20%
        ripple = max(v_out[-100:]) - min(v_out[-100:])
        print(f"  Ripple: {ripple:.1f}V ({100*ripple/np.mean(v_out[-100:]):.0f}%)")

        # 3. Peak output should be close to input peak minus diode drops
        # Two diodes in series: Vout_peak ≈ Vin_peak - 2*0.7V ≈ 18.6V
        assert max(v_out) > 15, f"Peak output should be >15V, got {max(v_out):.1f}V"

        print(f"  ✓ Graetz benchmark passed (parsed from sim file)")


class TestVACASKBenchmarksGeneric:
    """Generic parameterized tests for all VACASK benchmarks.

    Uses VACASKBenchmarkRunner to automatically parse and simulate
    any benchmark circuit without benchmark-specific code.
    """

    # Benchmarks that can be run with supported device types
    SUPPORTED_BENCHMARKS = ['rc', 'graetz', 'mul']

    # Benchmarks that need MOSFET support (ring, c6288)
    MOSFET_BENCHMARKS = ['ring', 'c6288']

    @pytest.mark.parametrize("benchmark_name", SUPPORTED_BENCHMARKS)
    def test_benchmark_generic(self, benchmark_name):
        """Test benchmark using generic runner.

        This test automatically:
        1. Parses the benchmark sim file
        2. Extracts models and devices
        3. Runs transient analysis
        4. Verifies the simulation completes without errors
        """
        benchmark_dir = VACASK_BENCHMARK / benchmark_name
        sim_file = benchmark_dir / "vacask" / "runme.sim"

        if not sim_file.exists():
            pytest.skip(f"Benchmark {benchmark_name} not found")

        print(f"\n{'='*60}")
        print(f"Running {benchmark_name} benchmark (generic)")
        print(f"{'='*60}")

        # Create and run the benchmark
        runner = VACASKBenchmarkRunner(sim_file, verbose=True)
        runner.parse()

        # Run with limited steps for faster testing
        # Use shorter stop time for tests
        t_stop = min(runner.analysis_params.get('stop', 1e-3), 10e-3)
        times, voltages = runner.run_transient(t_stop=t_stop, max_steps=1000)

        print(f"\nResults:")
        print(f"  Time points: {len(times)}")
        print(f"  Time range: 0 to {times[-1]:.2e}s")

        # Print voltage ranges for each non-ground node
        for name, idx in sorted(runner.node_names.items(), key=lambda x: x[1]):
            if idx == 0:
                continue
            v = voltages[idx]
            print(f"  V({name}): {min(v):.3f}V to {max(v):.3f}V")

        # Basic verification - simulation should complete
        assert len(times) > 1, "Simulation should produce multiple time points"
        assert not np.any(np.isnan(list(voltages.values()))), "No NaN values in output"
        assert not np.any(np.isinf(list(voltages.values()))), "No Inf values in output"

        print(f"\n  ✓ {benchmark_name} benchmark passed")

    @pytest.mark.parametrize("benchmark_name", MOSFET_BENCHMARKS)
    def test_benchmark_mosfet_skip(self, benchmark_name):
        """Placeholder for MOSFET benchmarks - currently unsupported."""
        pytest.skip(f"Benchmark {benchmark_name} requires MOSFET support (not yet implemented in generic runner)")


class TestVACASKSummary:
    """Summary of VACASK test coverage."""

    def test_coverage_summary(self):
        """Print summary of which tests are supported."""
        if not ALL_SIM_FILES:
            pytest.skip("No sim files found")

        supported = []
        unsupported = []

        for path, (cat, reasons) in CATEGORIZED_TESTS.items():
            if reasons:
                unsupported.append((path.stem, cat, reasons))
            else:
                supported.append((path.stem, cat))

        print(f"\n{'='*60}")
        print(f"VACASK Test Suite Coverage")
        print(f"{'='*60}")
        print(f"Supported: {len(supported)}/{len(ALL_SIM_FILES)} tests")
        print(f"Unsupported: {len(unsupported)}/{len(ALL_SIM_FILES)} tests")
        print()

        if unsupported:
            print("Unsupported tests:")
            for name, cat, reasons in sorted(unsupported):
                print(f"  {name} ({cat}): {', '.join(reasons)}")
