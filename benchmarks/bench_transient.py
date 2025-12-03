"""Benchmark transient analysis performance

Compares performance of RC circuit simulation across different:
- Circuit sizes (number of RC stages)
- Simulation lengths (number of timesteps)

Note: jax-metal does NOT support triangular_solve operations required
for direct matrix solvers. GPU acceleration would require implementing
iterative solvers (conjugate gradient, GMRES, etc.) instead of LU decomposition.
For now, benchmarks run on CPU with JAX JIT compilation.
"""

import os
import time
from typing import Tuple, List

# Force CPU backend since Metal doesn't support linear solve operations
os.environ['JAX_PLATFORMS'] = 'cpu'

import jax
import jax.numpy as jnp
from jax import Array

# Enable float64 for numerical precision on CPU
jax.config.update('jax_enable_x64', True)

from jax_spice.devices.base import DeviceStamps
from jax_spice.analysis.context import AnalysisContext
from jax_spice.analysis.mna import MNASystem, DeviceInfo
from jax_spice.analysis.transient import transient_analysis


def resistor_eval(voltages, params, context):
    """Resistor evaluation function"""
    Vp = voltages.get('p', 0.0)
    Vn = voltages.get('n', 0.0)
    R = float(params.get('r', 1000.0))
    G = 1.0 / R
    I = G * (Vp - Vn)
    return DeviceStamps(
        currents={'p': jnp.array(I), 'n': jnp.array(-I)},
        conductances={
            ('p', 'p'): jnp.array(G), ('p', 'n'): jnp.array(-G),
            ('n', 'p'): jnp.array(-G), ('n', 'n'): jnp.array(G)
        }
    )


def capacitor_eval(voltages, params, context):
    """Capacitor evaluation function"""
    Vp = voltages.get('p', 0.0)
    Vn = voltages.get('n', 0.0)
    C = float(params.get('c', 1e-6))
    V = Vp - Vn
    Q = C * V
    G_small = 1e-15
    return DeviceStamps(
        currents={'p': jnp.array(0.0), 'n': jnp.array(0.0)},
        conductances={
            ('p', 'p'): jnp.array(G_small), ('p', 'n'): jnp.array(-G_small),
            ('n', 'p'): jnp.array(-G_small), ('n', 'n'): jnp.array(G_small)
        },
        charges={'p': jnp.array(Q), 'n': jnp.array(-Q)},
        capacitances={
            ('p', 'p'): jnp.array(C), ('p', 'n'): jnp.array(-C),
            ('n', 'p'): jnp.array(-C), ('n', 'n'): jnp.array(C)
        }
    )


def vsource_eval(voltages, params, context):
    """Voltage source evaluation function"""
    Vp = voltages.get('p', 0.0)
    Vn = voltages.get('n', 0.0)
    V_target = float(params.get('v', 5.0))
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


def create_rc_ladder(num_stages: int, R: float = 1000.0, C: float = 1e-6, V_s: float = 5.0) -> MNASystem:
    """Create an RC ladder circuit with N stages

    Vs -- R1 -- C1 -- R2 -- C2 -- ... -- RN -- CN -- GND

    Args:
        num_stages: Number of RC stages
        R: Resistance per stage (ohms)
        C: Capacitance per stage (farads)
        V_s: Source voltage

    Returns:
        MNASystem ready for simulation
    """
    # Nodes: GND(0), Vs(1), after_R1(2), after_R2(3), ..., after_RN(N+1)
    num_nodes = num_stages + 2
    node_names = {'0': 0, 'vs': 1}
    for i in range(num_stages):
        node_names[f'n{i+1}'] = i + 2

    system = MNASystem(num_nodes=num_nodes, node_names=node_names)

    # Voltage source
    system.devices.append(DeviceInfo(
        name='Vs', model_name='vsource', terminals=['p', 'n'],
        node_indices=[1, 0], params={'v': V_s}, eval_fn=vsource_eval
    ))

    # RC stages
    for i in range(num_stages):
        # Resistor from previous node to this node
        prev_node = 1 if i == 0 else i + 1
        curr_node = i + 2

        system.devices.append(DeviceInfo(
            name=f'R{i+1}', model_name='resistor', terminals=['p', 'n'],
            node_indices=[prev_node, curr_node], params={'r': R}, eval_fn=resistor_eval
        ))

        # Capacitor from this node to ground
        system.devices.append(DeviceInfo(
            name=f'C{i+1}', model_name='capacitor', terminals=['p', 'n'],
            node_indices=[curr_node, 0], params={'c': C}, eval_fn=capacitor_eval
        ))

    return system


def benchmark_single_simulation(num_stages: int, num_timesteps: int) -> Tuple[float, float]:
    """Benchmark a single RC ladder simulation

    Args:
        num_stages: Number of RC stages in ladder
        num_timesteps: Number of simulation timesteps

    Returns:
        Tuple of (simulation_time_ms, setup_time_ms)
    """
    R, C = 1000.0, 1e-6
    tau = R * C

    # Setup
    setup_start = time.perf_counter()
    system = create_rc_ladder(num_stages, R, C)
    t_step = tau / 10  # 10 points per time constant
    t_stop = t_step * num_timesteps

    # Initial conditions: all caps at 0V
    initial_conditions = {'vs': 5.0}
    for i in range(num_stages):
        initial_conditions[f'n{i+1}'] = 0.0
    setup_time = (time.perf_counter() - setup_start) * 1000

    # Simulation
    sim_start = time.perf_counter()
    times, solutions, info = transient_analysis(
        system, t_stop=t_stop, t_step=t_step,
        initial_conditions=initial_conditions
    )
    # Force computation to complete
    solutions.block_until_ready()
    sim_time = (time.perf_counter() - sim_start) * 1000

    return sim_time, setup_time


def run_benchmarks():
    """Run comprehensive benchmarks"""
    print("=" * 70)
    print("JAX-SPICE Transient Analysis Benchmark")
    print("=" * 70)
    print()

    # System info
    devices = jax.devices()
    print(f"JAX Backend: {jax.default_backend()}")
    print(f"Devices: {devices}")
    print()

    # Warm up JIT compilation
    print("Warming up JIT compilation...")
    _ = benchmark_single_simulation(2, 10)
    print()

    # Benchmark different circuit sizes
    print("-" * 70)
    print("Circuit Size Scaling (100 timesteps)")
    print("-" * 70)
    print(f"{'Stages':>8} | {'Nodes':>8} | {'Devices':>8} | {'Sim (ms)':>10} | {'Setup (ms)':>10}")
    print("-" * 70)

    for num_stages in [1, 2, 5, 10, 20, 50]:
        sim_time, setup_time = benchmark_single_simulation(num_stages, 100)
        num_nodes = num_stages + 2
        num_devices = 1 + 2 * num_stages  # 1 vsource + N resistors + N capacitors
        print(f"{num_stages:>8} | {num_nodes:>8} | {num_devices:>8} | {sim_time:>10.2f} | {setup_time:>10.2f}")

    print()

    # Benchmark different simulation lengths
    print("-" * 70)
    print("Simulation Length Scaling (10-stage RC ladder)")
    print("-" * 70)
    print(f"{'Timesteps':>10} | {'Sim (ms)':>10} | {'ms/step':>10}")
    print("-" * 70)

    for num_timesteps in [10, 50, 100, 500, 1000]:
        sim_time, _ = benchmark_single_simulation(10, num_timesteps)
        per_step = sim_time / num_timesteps
        print(f"{num_timesteps:>10} | {sim_time:>10.2f} | {per_step:>10.4f}")

    print()
    print("=" * 70)
    print("Benchmark complete!")


if __name__ == '__main__':
    run_benchmarks()
