#!/usr/bin/env python3
"""Test GPU-native DC solver on inverter circuit

Compares the new sparsejac-based GPU solver with the existing sparse solver
to validate correctness and measure performance improvement.
"""

import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import jax
import jax.numpy as jnp
import numpy as np

# Enable float64
jax.config.update('jax_enable_x64', True)

from jax_spice.benchmarks.c6288 import C6288Benchmark
from jax_spice.analysis.dc_gpu import dc_operating_point_gpu, HAS_SPARSEJAC


def test_inverter():
    """Test GPU DC solver on inverter circuit (inv_test from C6288 benchmark)"""
    print("=" * 70)
    print("Testing GPU-native DC solver on inverter circuit")
    print("=" * 70)
    print()

    if not HAS_SPARSEJAC:
        print("ERROR: sparsejac not installed. Install with: pip install sparsejac")
        return False

    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")
    print()

    # Load the benchmark
    print("Loading C6288 benchmark...")
    bench = C6288Benchmark(verbose=False)
    bench.parse()

    # Use the inverter test circuit (smallest)
    circuit_name = 'inv_test'
    print(f"Flattening circuit: {circuit_name}")
    bench.flatten(circuit_name)

    print(f"Building MNA system...")
    bench.build_system(circuit_name)

    system = bench.system
    print(f"  Nodes: {system.num_nodes}")
    print(f"  Devices: {len(system.devices)}")
    print()

    # Count device types
    def is_vsource(m):
        return 'vsource' in m or 'vdc' in m or m == 'v' or (m.startswith('v') and len(m) <= 2)
    def is_mosfet(m):
        return 'mos' in m or ('psp' in m and (m.endswith('n') or m.endswith('p')))

    vsources = sum(1 for d in system.devices if is_vsource(d.model_name.lower()))
    mosfets = sum(1 for d in system.devices if is_mosfet(d.model_name.lower()))
    print(f"  Voltage sources: {vsources}")
    print(f"  MOSFETs: {mosfets}")
    print()

    # Test 1: Run existing sparse solver
    print("-" * 40)
    print("Test 1: Existing sparse DC solver")
    print("-" * 40)

    t0 = time.perf_counter()
    V_sparse, info_sparse = bench.run_sparse_dc(
        max_iterations=100,
        abstol=1e-9,
        verbose=False
    )
    t_sparse = time.perf_counter() - t0

    print(f"  Converged: {info_sparse['converged']}")
    print(f"  Iterations: {info_sparse['iterations']}")
    print(f"  Residual: {info_sparse['residual_norm']:.2e}")
    print(f"  Time: {t_sparse*1000:.1f} ms")
    print()

    # Test 2: Run GPU solver
    print("-" * 40)
    print("Test 2: GPU-native DC solver (sparsejac)")
    print("-" * 40)

    t0 = time.perf_counter()
    V_gpu, info_gpu = dc_operating_point_gpu(
        system,
        max_iterations=100,
        abstol=1e-9,
        vdd=1.2,
        verbose=True
    )
    t_gpu = time.perf_counter() - t0

    print(f"  Converged: {info_gpu['converged']}")
    print(f"  Iterations: {info_gpu['iterations']}")
    print(f"  Residual: {info_gpu['residual_norm']:.2e}")
    print(f"  Time: {t_gpu*1000:.1f} ms")
    print()

    # Compare solutions
    print("-" * 40)
    print("Comparison")
    print("-" * 40)

    # Both should have same length
    if len(V_sparse) != len(V_gpu):
        print(f"  ERROR: Solution lengths differ: {len(V_sparse)} vs {len(V_gpu)}")
        return False

    # Compare node voltages
    max_diff = float(jnp.max(jnp.abs(V_sparse - V_gpu)))
    print(f"  Max voltage difference: {max_diff:.2e}")

    # Print some node voltages
    print()
    print("  Node voltages:")
    for name, idx in list(system.node_names.items())[:10]:
        v_sparse = float(V_sparse[idx]) if idx < len(V_sparse) else 0.0
        v_gpu = float(V_gpu[idx]) if idx < len(V_gpu) else 0.0
        print(f"    {name}: sparse={v_sparse:.4f}, gpu={v_gpu:.4f}, diff={abs(v_sparse-v_gpu):.2e}")

    print()
    print(f"  Speedup: {t_sparse/t_gpu:.2f}x (negative means slower)")
    print()

    # Success criteria
    if info_gpu['converged'] and max_diff < 0.1:
        print("SUCCESS: GPU solver produces correct results!")
        return True
    else:
        print("FAILED: Results don't match or didn't converge")
        return False


def test_nor_gate():
    """Test GPU DC solver on NOR gate (nor_test from C6288 benchmark)"""
    print()
    print("=" * 70)
    print("Testing GPU-native DC solver on NOR gate")
    print("=" * 70)
    print()

    if not HAS_SPARSEJAC:
        print("ERROR: sparsejac not installed")
        return False

    bench = C6288Benchmark(verbose=False)
    bench.parse()

    circuit_name = 'nor_test'
    print(f"Flattening circuit: {circuit_name}")
    bench.flatten(circuit_name)
    bench.build_system(circuit_name)

    system = bench.system
    print(f"  Nodes: {system.num_nodes}")
    print(f"  Devices: {len(system.devices)}")
    print()

    # Sparse solver
    t0 = time.perf_counter()
    V_sparse, info_sparse = bench.run_sparse_dc(max_iterations=100, abstol=1e-9)
    t_sparse = time.perf_counter() - t0

    print(f"Sparse: converged={info_sparse['converged']}, iter={info_sparse['iterations']}, "
          f"residual={info_sparse['residual_norm']:.2e}, time={t_sparse*1000:.1f}ms")

    # GPU solver
    t0 = time.perf_counter()
    V_gpu, info_gpu = dc_operating_point_gpu(system, max_iterations=100, abstol=1e-9, vdd=1.2)
    t_gpu = time.perf_counter() - t0

    print(f"GPU:    converged={info_gpu['converged']}, iter={info_gpu['iterations']}, "
          f"residual={info_gpu['residual_norm']:.2e}, time={t_gpu*1000:.1f}ms")

    max_diff = float(jnp.max(jnp.abs(V_sparse - V_gpu)))
    print(f"Max voltage difference: {max_diff:.2e}")
    print(f"Speedup: {t_sparse/t_gpu:.2f}x")

    return info_gpu['converged'] and max_diff < 0.1


def benchmark_iteration_time():
    """Benchmark per-iteration time for larger circuit"""
    print()
    print("=" * 70)
    print("Benchmarking iteration time on c6288_test")
    print("=" * 70)
    print()

    if not HAS_SPARSEJAC:
        print("ERROR: sparsejac not installed")
        return

    bench = C6288Benchmark(verbose=False)
    bench.parse()

    circuit_name = 'c6288_test'
    print(f"Flattening circuit: {circuit_name}")
    bench.flatten(circuit_name)
    bench.build_system(circuit_name)

    system = bench.system
    print(f"  Nodes: {system.num_nodes}")
    print(f"  Devices: {len(system.devices)}")
    print()

    # Count device types (matches dc_gpu.py detection)
    def is_vsource(m):
        m = m.lower()
        return 'vsource' in m or 'vdc' in m or m == 'v' or (m.startswith('v') and len(m) <= 2)
    def is_mosfet(m):
        m = m.lower()
        return 'mos' in m or ('psp' in m and (m.endswith('n') or m.endswith('p')))
    def is_resistor(m):
        m = m.lower()
        return m == 'r' or m == 'resistor' or (m.startswith('r') and len(m) <= 2)

    vsources = sum(1 for d in system.devices if is_vsource(d.model_name))
    mosfets = sum(1 for d in system.devices if is_mosfet(d.model_name))
    resistors = sum(1 for d in system.devices if is_resistor(d.model_name))
    print(f"  Voltage sources: {vsources}")
    print(f"  MOSFETs: {mosfets}")
    print(f"  Resistors: {resistors}")
    print()

    # Run a few iterations with each solver (not full convergence)
    max_iter = 5

    print(f"Running {max_iter} iterations with each solver...")
    print()

    # Sparse solver
    t0 = time.perf_counter()
    V_sparse, info_sparse = bench.run_sparse_dc(max_iterations=max_iter, abstol=1e-20)
    t_sparse = time.perf_counter() - t0

    print(f"Sparse solver:")
    print(f"  Total time: {t_sparse*1000:.1f} ms")
    print(f"  Per-iteration: {t_sparse*1000/max_iter:.1f} ms")
    print()

    # GPU solver
    t0 = time.perf_counter()
    V_gpu, info_gpu = dc_operating_point_gpu(
        system,
        max_iterations=max_iter,
        abstol=1e-20,  # Won't converge, just measure time
        vdd=1.2
    )
    t_gpu = time.perf_counter() - t0

    print(f"GPU solver (sparsejac):")
    print(f"  Total time: {t_gpu*1000:.1f} ms")
    print(f"  Per-iteration: {t_gpu*1000/max_iter:.1f} ms")
    print()

    print(f"Speedup: {t_sparse/t_gpu:.2f}x")


if __name__ == '__main__':
    success = test_inverter()
    if success:
        test_nor_gate()
        benchmark_iteration_time()
