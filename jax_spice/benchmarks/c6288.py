"""C6288 16x16 Multiplier Benchmark for JAX-SPICE

This module provides the c6288 multiplier benchmark circuit for testing
sparse and dense DC operating point analysis.

The c6288 is a 16x16 bit multiplier with:
- 2416 gates (256 AND + 2128 NOR + 32 NOT)
- ~10,112 transistors (using simplified MOSFET model)
- 5,123 nodes after flattening
"""

import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from importlib import resources

import jax
import jax.numpy as jnp
from jax import Array

from jax_spice.netlist.parser import VACASKParser
from jax_spice.netlist.circuit import Circuit, Instance
from jax_spice.analysis.mna import MNASystem, DeviceInfo
from jax_spice.analysis.context import AnalysisContext
from jax_spice.analysis.dc import (
    dc_operating_point,
    dc_operating_point_sparse,
    dc_operating_point_gmin_stepping,
)
from jax_spice.devices.base import DeviceStamps


# Global circuit parameters (set during parsing)
_circuit_params: Dict[str, Any] = {}

# Default MOSFET parameters from subcircuit definitions
_mosfet_defaults: Dict[str, Any] = {
    'w': 1e-6,    # 1u from nmos/pmos subcircuit
    'l': 0.2e-6,  # 0.2u from nmos/pmos subcircuit
    'ld': 0.5e-6, # 0.5u
    'ls': 0.5e-6, # 0.5u
}


def get_data_path() -> Path:
    """Get path to benchmark data files"""
    # Use importlib.resources for package data
    try:
        # Python 3.9+
        with resources.files("jax_spice.benchmarks") as pkg_path:
            return Path(pkg_path) / "data"
    except AttributeError:
        # Fallback for older Python
        import pkg_resources
        return Path(pkg_resources.resource_filename("jax_spice.benchmarks", "data"))


def parse_spice_number(s: str) -> float:
    """Parse SPICE number with suffix (e.g., 1u, 100n, 1.5k)"""
    s = s.strip().lower()
    if not s:
        return 0.0

    # SPICE suffixes
    suffixes = {
        't': 1e12, 'g': 1e9, 'meg': 1e6, 'k': 1e3,
        'm': 1e-3, 'u': 1e-6, 'n': 1e-9, 'p': 1e-12, 'f': 1e-15
    }

    # Try to find a suffix
    for suffix, multiplier in sorted(suffixes.items(), key=lambda x: -len(x[0])):
        if s.endswith(suffix):
            try:
                return float(s[:-len(suffix)]) * multiplier
            except ValueError:
                continue

    # No suffix, try direct conversion
    try:
        return float(s)
    except ValueError:
        return 0.0


def eval_param(value: Any, params: Dict[str, Any], defaults: Dict[str, Any] = None) -> float:
    """Evaluate a parameter value, resolving references

    Args:
        value: The parameter value (can be number, string, or reference)
        params: Circuit-level parameters for resolution
        defaults: Optional default values for unresolved references
    """
    if defaults is None:
        defaults = _mosfet_defaults

    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        # Check if it's a reference to another parameter
        if value in params:
            return eval_param(params[value], params, defaults)
        # Check defaults for unresolved references (e.g., 'w', 'l')
        if value in defaults:
            return float(defaults[value])
        # Try to parse as SPICE number with suffix
        result = parse_spice_number(value)
        if result != 0.0:
            return result
        # Try direct float conversion
        try:
            return float(value)
        except ValueError:
            # Could be an expression - try simple evaluation
            try:
                # Merge params and defaults for expression evaluation
                eval_context = {**defaults, **params}
                return float(eval(value, {"__builtins__": {}}, eval_context))
            except Exception:
                return 0.0
    return 0.0


def create_vsource_eval():
    """Create voltage source evaluation function"""
    def vsource_eval(voltages: Dict[str, float], params: Dict[str, Any],
                     context: AnalysisContext) -> DeviceStamps:
        Vp = voltages.get('p', 0.0)
        Vn = voltages.get('n', 0.0)
        # Check various parameter keys for DC value:
        # - 'v': generic voltage value
        # - 'dc': SPICE DC value
        # - 'val0': PULSE source initial value (used for DC analysis)
        V_target = eval_param(
            params.get('v', params.get('dc', params.get('val0', 0.0))),
            _circuit_params
        )

        # Support source stepping: scale voltage sources by vdd_scale from context
        # This is used during dc_operating_point_source_stepping() to ramp Vdd
        vdd_scale = getattr(context, 'vdd_scale', 1.0)
        if vdd_scale != 1.0 and V_target != 0.0:
            V_target = V_target * vdd_scale

        V_actual = Vp - Vn
        G_big = 1e6  # Reduced from 1e9 for better matrix conditioning
        I = G_big * (V_actual - V_target)
        return DeviceStamps(
            currents={'p': jnp.array(I), 'n': jnp.array(-I)},
            conductances={
                ('p', 'p'): jnp.array(G_big), ('p', 'n'): jnp.array(-G_big),
                ('n', 'p'): jnp.array(-G_big), ('n', 'n'): jnp.array(G_big)
            }
        )
    return vsource_eval


def create_resistor_eval():
    """Create resistor evaluation function"""
    def resistor_eval(voltages: Dict[str, float], params: Dict[str, Any],
                      context: AnalysisContext) -> DeviceStamps:
        Vp = voltages.get('p', 0.0)
        Vn = voltages.get('n', 0.0)
        R = eval_param(params.get('r', 1000.0), _circuit_params)
        if R <= 0:
            R = 1e-6
        G = 1.0 / R
        I = G * (Vp - Vn)
        return DeviceStamps(
            currents={'p': jnp.array(I), 'n': jnp.array(-I)},
            conductances={
                ('p', 'p'): jnp.array(G), ('p', 'n'): jnp.array(-G),
                ('n', 'p'): jnp.array(-G), ('n', 'n'): jnp.array(G)
            }
        )
    return resistor_eval


def create_simple_mosfet_eval(is_pmos: bool = False, vth0: float = 0.4,
                               kp: float = 200e-6, lambda_: float = 0.01):
    """Create simplified MOSFET (level 1) evaluation function"""
    def mosfet_eval(voltages: Dict[str, float], params: Dict[str, Any],
                    context: AnalysisContext) -> DeviceStamps:
        Vd = float(voltages.get('D', 0.0))
        Vg = float(voltages.get('G', 0.0))
        Vs = float(voltages.get('S', 0.0))
        Vb = float(voltages.get('B', 0.0))

        W = eval_param(params.get('w', 1e-6), _circuit_params)
        L = eval_param(params.get('l', 100e-9), _circuit_params)

        # Fail gracefully if W or L are invalid
        if W <= 0:
            raise ValueError(f"MOSFET W parameter must be positive, got {W} (raw: {params.get('w')})")
        if L <= 0:
            raise ValueError(f"MOSFET L parameter must be positive, got {L} (raw: {params.get('l')})")

        # For PMOS, invert voltages
        if is_pmos:
            Vgs = Vs - Vg
            Vds = Vs - Vd
        else:
            Vgs = Vg - Vs
            Vds = Vd - Vs

        Vth = vth0
        Kp = kp * (W / L)

        # Operating region
        Vov = Vgs - Vth

        if Vov <= 0:
            # Cutoff - add subthreshold leakage for numerical stability
            # Use larger minimum conductance to help Newton-Raphson convergence
            gds_min = 1e-9  # Minimum off-state drain conductance
            Id = gds_min * Vds  # Small leakage current
            gm = 1e-12
            gds = gds_min
        elif Vds < Vov:
            # Linear (triode)
            Id = Kp * (Vov * Vds - 0.5 * Vds * Vds) * (1 + lambda_ * Vds)
            gm = Kp * Vds * (1 + lambda_ * Vds)
            gds = Kp * (Vov - Vds) * (1 + lambda_ * Vds) + Kp * (Vov * Vds - 0.5 * Vds * Vds) * lambda_
        else:
            # Saturation
            Id = 0.5 * Kp * Vov * Vov * (1 + lambda_ * Vds)
            gm = Kp * Vov * (1 + lambda_ * Vds)
            gds = 0.5 * Kp * Vov * Vov * lambda_

        # Ensure minimum conductance for numerical stability
        gm = max(gm, 1e-12)
        gds = max(gds, 1e-9)

        # For PMOS, current flows in opposite direction
        if is_pmos:
            Id = -Id
            # Also need to adjust conductance signs for PMOS
            # The Jacobian entries need adjustment

        # Currents: positive into drain, negative out of source
        Id_drain = Id
        Id_source = -Id

        # Build conductance matrix (Jacobian contributions)
        # For simplified model: dId/dVg = gm, dId/dVd = gds
        # GMIN for gate and body nodes to prevent floating nodes
        # Use larger GMIN (1e-9) for better convergence with digital circuits
        gmin = 1e-9

        if is_pmos:
            # PMOS: current flows S->D when on
            return DeviceStamps(
                currents={
                    'D': jnp.array(Id_drain),
                    'G': jnp.array(0.0),
                    'S': jnp.array(Id_source),
                    'B': jnp.array(0.0)
                },
                conductances={
                    ('D', 'D'): jnp.array(gds),
                    ('D', 'G'): jnp.array(gm),
                    ('D', 'S'): jnp.array(-gds - gm),
                    ('S', 'D'): jnp.array(-gds),
                    ('S', 'G'): jnp.array(-gm),
                    ('S', 'S'): jnp.array(gds + gm),
                    ('G', 'G'): jnp.array(gmin),  # Gate GMIN
                    ('B', 'B'): jnp.array(gmin),  # Body GMIN
                }
            )
        else:
            # NMOS: current flows D->S when on
            return DeviceStamps(
                currents={
                    'D': jnp.array(Id_drain),
                    'G': jnp.array(0.0),
                    'S': jnp.array(Id_source),
                    'B': jnp.array(0.0)
                },
                conductances={
                    ('D', 'D'): jnp.array(gds),
                    ('D', 'G'): jnp.array(gm),
                    ('D', 'S'): jnp.array(-gds - gm),
                    ('S', 'D'): jnp.array(-gds),
                    ('S', 'G'): jnp.array(-gm),
                    ('S', 'S'): jnp.array(gds + gm),
                    ('G', 'G'): jnp.array(gmin),  # Gate GMIN
                    ('B', 'B'): jnp.array(gmin),  # Body GMIN
                }
            )

    return mosfet_eval


class C6288Benchmark:
    """C6288 16x16 multiplier benchmark circuit"""

    def __init__(self, verbose: bool = False):
        """Initialize the benchmark

        Args:
            verbose: Print progress messages
        """
        self.verbose = verbose
        self.circuit: Optional[Circuit] = None
        self.system: Optional[MNASystem] = None
        self.nodes: Optional[Dict[str, int]] = None
        self.vdd: float = 1.2
        self._parsed = False
        self._built = False

    def parse(self) -> "C6288Benchmark":
        """Parse the c6288 netlist"""
        data_path = get_data_path()
        netlist_path = data_path / "c6288.sim"

        if self.verbose:
            print(f"Parsing {netlist_path}...")

        start = time.perf_counter()
        parser = VACASKParser()
        self.circuit = parser.parse_file(str(netlist_path))
        elapsed = time.perf_counter() - start

        _circuit_params.clear()
        _circuit_params.update(self.circuit.params)
        self.vdd = eval_param(self.circuit.params.get('vdd', 1.2), _circuit_params)

        if self.verbose:
            print(f"  Parsed in {elapsed:.3f}s")
            print(f"  Subcircuits: {len(self.circuit.subckts)}")
            print(f"  Models: {list(self.circuit.models.keys())}")

        self._parsed = True
        return self

    def flatten(self, circuit_name: str = 'c6288_test') -> "C6288Benchmark":
        """Flatten the circuit hierarchy

        Args:
            circuit_name: Top-level subcircuit to flatten
        """
        if not self._parsed:
            self.parse()

        if self.verbose:
            print(f"Flattening {circuit_name}...")

        start = time.perf_counter()
        instances, nodes = self.circuit.flatten(circuit_name)
        elapsed = time.perf_counter() - start

        self.instances = instances
        self.nodes = nodes

        if self.verbose:
            print(f"  Flattened in {elapsed:.3f}s")
            print(f"  Instances: {len(instances)}")
            print(f"  Nodes: {len(nodes)}")

        return self

    def build_system(self, circuit_name: str = 'c6288_test') -> "C6288Benchmark":
        """Build MNA system from circuit

        Args:
            circuit_name: Top-level subcircuit name
        """
        if self.nodes is None:
            self.flatten(circuit_name)

        if self.verbose:
            print("Building MNA system...")

        start = time.perf_counter()

        system = MNASystem(num_nodes=len(self.nodes), node_names=self.nodes)

        # Create evaluation functions
        vsource_eval = create_vsource_eval()
        resistor_eval = create_resistor_eval()
        nmos_eval = create_simple_mosfet_eval(is_pmos=False, vth0=0.35, kp=300e-6, lambda_=0.02)
        pmos_eval = create_simple_mosfet_eval(is_pmos=True, vth0=0.35, kp=100e-6, lambda_=0.02)

        for inst in self.instances:
            model_name = inst.model.lower()
            node_indices = [self.nodes[t] for t in inst.terminals]

            if model_name == 'v':
                eval_fn = vsource_eval
                terminals = ['p', 'n']
            elif model_name == 'r':
                eval_fn = resistor_eval
                terminals = ['p', 'n']
            elif model_name == 'psp103n':
                eval_fn = nmos_eval
                terminals = ['D', 'G', 'S', 'B']
            elif model_name == 'psp103p':
                eval_fn = pmos_eval
                terminals = ['D', 'G', 'S', 'B']
            else:
                continue

            if len(node_indices) != len(terminals):
                continue

            device = DeviceInfo(
                name=inst.name,
                model_name=inst.model,
                terminals=terminals,
                node_indices=node_indices,
                params=inst.params,
                eval_fn=eval_fn
            )
            system.devices.append(device)

        self.system = system
        elapsed = time.perf_counter() - start

        if self.verbose:
            print(f"  Built in {elapsed:.3f}s")
            print(f"  Devices: {len(system.devices)}")
            print(f"  Matrix size: {system.num_nodes - 1} x {system.num_nodes - 1}")

        self._built = True
        return self

    def run_sparse_dc(self, max_iterations: int = 200, abstol: float = 1e-6,
                      verbose: bool = False) -> Tuple[Array, Dict]:
        """Run sparse DC operating point analysis

        Args:
            max_iterations: Maximum Newton-Raphson iterations
            abstol: Absolute convergence tolerance
            verbose: Print iteration progress

        Returns:
            (voltages, info) tuple
        """
        if not self._built:
            self.build_system()

        return dc_operating_point_sparse(
            self.system,
            vdd=self.vdd,
            max_iterations=max_iterations,
            abstol=abstol,
            verbose=verbose
        )

    def run_dense_dc(self, max_iterations: int = 50, abstol: float = 1e-12,
                     verbose: bool = False) -> Tuple[Array, Dict]:
        """Run dense DC operating point analysis

        Note: This may be very slow or run out of memory for large circuits.

        Args:
            max_iterations: Maximum Newton-Raphson iterations
            abstol: Absolute convergence tolerance
            verbose: Print iteration progress (not supported in dense solver)

        Returns:
            (voltages, info) tuple
        """
        if not self._built:
            self.build_system()

        return dc_operating_point(
            self.system,
            max_iterations=max_iterations,
            abstol=abstol
        )

    def run_gmin_stepping_dc(
        self,
        start_gmin: float = 1e-2,
        target_gmin: float = 1e-12,
        gmin_factor: float = 10.0,
        max_gmin_steps: int = 20,
        max_iterations_per_step: int = 50,
        abstol: float = 1e-9,
        verbose: bool = False
    ) -> Tuple[Array, Dict]:
        """Run DC operating point analysis with GMIN stepping

        GMIN stepping is a homotopy method for converging difficult circuits.
        It starts with a large GMIN value making the matrix well-conditioned,
        then gradually reduces GMIN to the target value.

        Args:
            start_gmin: Initial large GMIN value (default 1e-2)
            target_gmin: Final small GMIN value (default 1e-12)
            gmin_factor: Factor to reduce GMIN by each step (default 10.0)
            max_gmin_steps: Maximum number of GMIN stepping iterations
            max_iterations_per_step: Max NR iterations per GMIN step
            abstol: Absolute convergence tolerance
            verbose: Print progress information

        Returns:
            (voltages, info) tuple
        """
        if not self._built:
            self.build_system()

        return dc_operating_point_gmin_stepping(
            self.system,
            start_gmin=start_gmin,
            target_gmin=target_gmin,
            gmin_factor=gmin_factor,
            max_gmin_steps=max_gmin_steps,
            max_iterations_per_step=max_iterations_per_step,
            abstol=abstol,
            vdd=self.vdd,
            verbose=verbose
        )


def run_c6288_sparse_dc(circuit_name: str = 'c6288_test',
                        max_iterations: int = 200,
                        abstol: float = 1e-6,
                        verbose: bool = True) -> Tuple[Array, Dict, float]:
    """Convenience function to run sparse DC benchmark

    Args:
        circuit_name: Circuit to simulate ('inv_test', 'c6288_test', etc.)
        max_iterations: Maximum NR iterations
        abstol: Convergence tolerance
        verbose: Print progress

    Returns:
        (voltages, info, elapsed_time) tuple
    """
    print("=" * 70, flush=True)
    print(f"JAX-SPICE Sparse DC Benchmark: {circuit_name}", flush=True)
    print("=" * 70, flush=True)
    print(flush=True)
    print(f"JAX Backend: {jax.default_backend()}", flush=True)
    print(f"Devices: {jax.devices()}", flush=True)
    print(flush=True)

    bench = C6288Benchmark(verbose=verbose)
    print("Parsing netlist...", flush=True)
    bench.parse()
    print("Flattening circuit...", flush=True)
    bench.flatten(circuit_name)
    print("Building MNA system...", flush=True)
    bench.build_system(circuit_name)

    print(flush=True)
    print("Running sparse DC solver...", flush=True)
    start = time.perf_counter()
    V, info = bench.run_sparse_dc(max_iterations=max_iterations, abstol=abstol, verbose=verbose)
    elapsed = time.perf_counter() - start

    print()
    print("=" * 70)
    print("Results:")
    print(f"  Converged: {info['converged']}")
    print(f"  Iterations: {info['iterations']}")
    print(f"  Residual: {info['residual_norm']:.2e}")
    print(f"  Time: {elapsed:.2f}s")
    print()

    # Sample voltages
    print("Sample voltages:")
    for name in ['vdd', 'vss', f'{circuit_name}.p0']:
        if name in bench.nodes:
            print(f"  {name}: {float(V[bench.nodes[name]]):.4f}V")

    # Memory estimate
    n = bench.system.num_nodes - 1
    nnz = len(bench.system.devices) * 16
    sparse_mem = nnz * 8 / 1e6
    dense_mem = n * n * 8 / 1e6
    print()
    print("Memory estimate:")
    print(f"  Dense Jacobian: {dense_mem:.1f} MB")
    print(f"  Sparse Jacobian: ~{sparse_mem:.1f} MB")
    print(f"  Savings: {dense_mem/sparse_mem:.0f}x")
    print("=" * 70)

    return V, info, elapsed


def run_c6288_dense_dc(circuit_name: str = 'inv_test',
                       max_iterations: int = 50,
                       abstol: float = 1e-12,
                       verbose: bool = True) -> Tuple[Array, Dict, float]:
    """Convenience function to run dense DC benchmark

    Note: Only suitable for small circuits (inv_test, nor_test, etc.)

    Args:
        circuit_name: Circuit to simulate
        max_iterations: Maximum NR iterations
        abstol: Convergence tolerance
        verbose: Print progress

    Returns:
        (voltages, info, elapsed_time) tuple
    """
    bench = C6288Benchmark(verbose=verbose)
    bench.parse()
    bench.flatten(circuit_name)
    bench.build_system(circuit_name)

    start = time.perf_counter()
    V, info = bench.run_dense_dc(max_iterations=max_iterations, abstol=abstol, verbose=verbose)
    elapsed = time.perf_counter() - start

    return V, info, elapsed


def run_c6288_gmin_stepping_dc(
    circuit_name: str = 'c6288_test',
    start_gmin: float = 1e-2,
    target_gmin: float = 1e-12,
    gmin_factor: float = 10.0,
    abstol: float = 1e-9,
    verbose: bool = True
) -> Tuple[Array, Dict, float]:
    """Convenience function to run GMIN stepping DC benchmark

    GMIN stepping is a homotopy method for difficult circuits that starts
    with large GMIN and gradually reduces it to the target value.

    Args:
        circuit_name: Circuit to simulate ('inv_test', 'c6288_test', etc.)
        start_gmin: Initial large GMIN value
        target_gmin: Final small GMIN value
        gmin_factor: Factor to reduce GMIN by each step
        abstol: Convergence tolerance
        verbose: Print progress

    Returns:
        (voltages, info, elapsed_time) tuple
    """
    print("=" * 70, flush=True)
    print(f"JAX-SPICE GMIN Stepping DC Benchmark: {circuit_name}", flush=True)
    print("=" * 70, flush=True)
    print(flush=True)
    print(f"JAX Backend: {jax.default_backend()}", flush=True)
    print(f"Devices: {jax.devices()}", flush=True)
    print(flush=True)

    bench = C6288Benchmark(verbose=verbose)
    print("Parsing netlist...", flush=True)
    bench.parse()
    print("Flattening circuit...", flush=True)
    bench.flatten(circuit_name)
    print("Building MNA system...", flush=True)
    bench.build_system(circuit_name)

    print(flush=True)
    print("Running GMIN stepping DC solver...", flush=True)
    start = time.perf_counter()
    V, info = bench.run_gmin_stepping_dc(
        start_gmin=start_gmin,
        target_gmin=target_gmin,
        gmin_factor=gmin_factor,
        abstol=abstol,
        verbose=verbose
    )
    elapsed = time.perf_counter() - start

    print()
    print("=" * 70)
    print("Results:")
    print(f"  Converged: {info['converged']}")
    print(f"  Total Iterations: {info['iterations']}")
    print(f"  GMIN Steps: {info['gmin_steps']}")
    print(f"  Final GMIN: {info['final_gmin']:.2e}")
    print(f"  Residual: {info['residual_norm']:.2e}")
    print(f"  Time: {elapsed:.2f}s")
    print()

    # Sample voltages
    print("Sample voltages:")
    for name in ['vdd', 'vss', f'{circuit_name}.p0']:
        if name in bench.nodes:
            print(f"  {name}: {float(V[bench.nodes[name]]):.4f}V")

    # Memory estimate
    n = bench.system.num_nodes - 1
    nnz = len(bench.system.devices) * 16
    sparse_mem = nnz * 8 / 1e6
    dense_mem = n * n * 8 / 1e6
    print()
    print("Memory estimate:")
    print(f"  Dense Jacobian: {dense_mem:.1f} MB")
    print(f"  Sparse Jacobian: ~{sparse_mem:.1f} MB")
    print(f"  Savings: {dense_mem/sparse_mem:.0f}x")
    print("=" * 70)

    return V, info, elapsed


# CLI entry point
def main():
    """Command-line interface for running benchmarks"""
    import argparse

    parser = argparse.ArgumentParser(description='JAX-SPICE C6288 Benchmark')
    parser.add_argument('--circuit', '-c', default='c6288_test',
                        choices=['inv_test', 'nor_test', 'and_test', 'gatedrv_test', 'c6288_test'],
                        help='Circuit to simulate')
    parser.add_argument('--dense', action='store_true',
                        help='Use dense solver (only for small circuits)')
    parser.add_argument('--gmin-stepping', action='store_true',
                        help='Use GMIN stepping for difficult circuits')
    parser.add_argument('--max-iter', type=int, default=200,
                        help='Maximum iterations')
    parser.add_argument('--abstol', type=float, default=1e-6,
                        help='Absolute tolerance')
    parser.add_argument('--start-gmin', type=float, default=1e-2,
                        help='Starting GMIN for GMIN stepping (default 1e-2)')
    parser.add_argument('--target-gmin', type=float, default=1e-12,
                        help='Target GMIN for GMIN stepping (default 1e-12)')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Minimal output')

    args = parser.parse_args()

    # Enable float64
    jax.config.update('jax_enable_x64', True)

    if args.dense:
        run_c6288_dense_dc(args.circuit, args.max_iter, args.abstol, not args.quiet)
    elif args.gmin_stepping:
        run_c6288_gmin_stepping_dc(
            args.circuit,
            start_gmin=args.start_gmin,
            target_gmin=args.target_gmin,
            abstol=args.abstol,
            verbose=not args.quiet
        )
    else:
        run_c6288_sparse_dc(args.circuit, args.max_iter, args.abstol, not args.quiet)


if __name__ == '__main__':
    main()
