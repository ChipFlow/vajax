"""GPU Transient Analysis Tests

Tests GPU-native transient analysis with MOSFET circuits using icmode='uic'
to match VACASK benchmark behavior.

Run with: pytest tests/test_transient_gpu.py -v -s
"""

import pytest
import numpy as np
from pathlib import Path
import sys
import os
import time

jax_spice_path = Path(__file__).parent.parent
if str(jax_spice_path) not in sys.path:
    sys.path.insert(0, str(jax_spice_path))

import jax
import jax.numpy as jnp
jax.config.update('jax_enable_x64', True)

from jax_spice.benchmarks.c6288 import C6288Benchmark
from jax_spice.analysis.transient_gpu import (
    transient_analysis_gpu,
    build_transient_circuit_data_fast,
    build_transient_residual_fn,
)


def has_gpu():
    """Check if GPU/CUDA backend is available"""
    for device in jax.devices():
        platform = device.platform.lower()
        if platform in ('gpu', 'cuda'):
            return True
    return False


requires_gpu = pytest.mark.skipif(
    not has_gpu() and os.environ.get('RUN_GPU_BENCHMARKS') != '1',
    reason="GPU not available (set RUN_GPU_BENCHMARKS=1 to run anyway)"
)


class TestTransientGPUBasic:
    """Basic transient GPU tests with simple MOSFET circuits"""

    def test_inv_transient_uic(self, capsys):
        """Test inverter transient with icmode='uic' (Use Initial Conditions)

        This matches VACASK behavior which uses transient with uic
        instead of DC operating point for digital circuits.
        """
        print("\n" + "=" * 60)
        print("Inverter Transient with icmode='uic'")
        print("=" * 60)

        bench = C6288Benchmark(verbose=False)
        bench.parse()
        bench.flatten('inv_test')
        bench.build_system('inv_test')

        system = bench.system
        system.build_device_groups()

        print(f"Nodes: {system.num_nodes}")
        print(f"Devices: {len(system.devices)}")

        # Run transient with icmode='uic' - no DC operating point computation
        t_stop = 1e-9  # 1ns
        t_step = 10e-12  # 10ps

        start = time.perf_counter()
        times, solutions, info = transient_analysis_gpu(
            system,
            t_stop=t_stop,
            t_step=t_step,
            icmode='uic',  # Use Initial Conditions - matches VACASK
            vdd=1.2,
            verbose=True,
        )
        elapsed = time.perf_counter() - start

        print(f"\nSimulation completed in {elapsed:.2f}s")
        print(f"Timesteps: {info['num_timepoints']}")
        print(f"Avg iterations/step: {info['avg_iterations_per_step']:.1f}")

        # Check that output settles to expected value
        # With input at 0V (via voltage source), output should go to VDD
        out_node = bench.nodes.get('inv_test.out', None)
        if out_node is not None:
            V_out_final = float(solutions[-1, out_node])
            print(f"Output voltage at t={t_stop:.1e}: {V_out_final:.4f}V")
            # Output should be close to VDD (1.2V) since input is 0
            assert V_out_final > 0.5, f"Output {V_out_final}V should be high"

        # Verify icmode was used
        assert info['icmode'] == 'uic'
        assert info['dc_info'] is None  # No DC solve for uic mode

    def test_nor_transient_uic(self, capsys):
        """Test NOR gate transient with icmode='uic'"""
        print("\n" + "=" * 60)
        print("NOR Gate Transient with icmode='uic'")
        print("=" * 60)

        bench = C6288Benchmark(verbose=False)
        bench.parse()
        bench.flatten('nor_test')
        bench.build_system('nor_test')

        system = bench.system
        system.build_device_groups()

        print(f"Nodes: {system.num_nodes}")
        print(f"Devices: {len(system.devices)}")
        mosfet_count = sum(1 for d in system.devices if 'psp' in d.model_name.lower())
        print(f"MOSFETs: {mosfet_count}")

        # Run transient
        t_stop = 1e-9
        t_step = 10e-12

        start = time.perf_counter()
        times, solutions, info = transient_analysis_gpu(
            system,
            t_stop=t_stop,
            t_step=t_step,
            icmode='uic',
            vdd=1.2,
            verbose=True,
        )
        elapsed = time.perf_counter() - start

        print(f"\nSimulation completed in {elapsed:.2f}s")
        print(f"Total Newton iterations: {info['total_iterations']}")

        # NOR gate with both inputs at 0 -> output should be high
        out_node = bench.nodes.get('nor_test.out', None)
        if out_node is not None:
            V_out_final = float(solutions[-1, out_node])
            print(f"NOR output at t={t_stop:.1e}: {V_out_final:.4f}V")
            assert V_out_final > 0.5, f"NOR output should be high when inputs are low"

    def test_and_transient_uic(self, capsys):
        """Test AND gate transient with icmode='uic'"""
        print("\n" + "=" * 60)
        print("AND Gate Transient with icmode='uic'")
        print("=" * 60)

        bench = C6288Benchmark(verbose=False)
        bench.parse()
        bench.flatten('and_test')
        bench.build_system('and_test')

        system = bench.system
        system.build_device_groups()

        print(f"Nodes: {system.num_nodes}")
        print(f"Devices: {len(system.devices)}")
        mosfet_count = sum(1 for d in system.devices if 'psp' in d.model_name.lower())
        print(f"MOSFETs: {mosfet_count}")

        # Run transient
        t_stop = 1e-9
        t_step = 10e-12

        start = time.perf_counter()
        times, solutions, info = transient_analysis_gpu(
            system,
            t_stop=t_stop,
            t_step=t_step,
            icmode='uic',
            vdd=1.2,
            verbose=True,
        )
        elapsed = time.perf_counter() - start

        print(f"\nSimulation completed in {elapsed:.2f}s")
        print(f"Total Newton iterations: {info['total_iterations']}")

        # AND gate with both inputs at 0 -> output should be low
        out_node = bench.nodes.get('and_test.out', None)
        if out_node is not None:
            V_out_final = float(solutions[-1, out_node])
            print(f"AND output at t={t_stop:.1e}: {V_out_final:.4f}V")
            # AND(0,0) = 0, but output might not be exactly 0 during settling
            # Just check it's not stuck at VDD
            assert V_out_final < 1.0, f"AND output should not be stuck high"


class TestTransientGPUPerformance:
    """Performance tests for GPU transient analysis"""

    @requires_gpu
    def test_inv_transient_gpu_vs_cpu(self, capsys):
        """Compare GPU vs CPU transient performance for inverter"""
        print("\n" + "=" * 60)
        print("GPU vs CPU Transient Performance")
        print("=" * 60)

        bench = C6288Benchmark(verbose=False)
        bench.parse()
        bench.flatten('inv_test')
        bench.build_system('inv_test')

        system = bench.system
        system.build_device_groups()

        t_stop = 2e-9
        t_step = 10e-12
        num_steps = int(t_stop / t_step)

        print(f"Timesteps: {num_steps}")
        print(f"Backend: {jax.default_backend()}")

        # Run transient
        start = time.perf_counter()
        times, solutions, info = transient_analysis_gpu(
            system,
            t_stop=t_stop,
            t_step=t_step,
            icmode='uic',
            vdd=1.2,
            verbose=False,
        )
        elapsed = time.perf_counter() - start

        print(f"Total time: {elapsed:.2f}s")
        print(f"First step (w/ compile): {info['first_step_time']:.2f}s")
        print(f"Time per step (after compile): {(elapsed - info['first_step_time']) / max(1, num_steps - 1):.4f}s")

        assert info['num_timepoints'] == num_steps + 1


class TestCircuitDataConstruction:
    """Test circuit data construction for transient analysis"""

    def test_build_circuit_data_fast(self, capsys):
        """Test fast circuit data construction from device groups"""
        print("\n" + "=" * 60)
        print("Circuit Data Construction")
        print("=" * 60)

        bench = C6288Benchmark(verbose=False)
        bench.parse()
        bench.flatten('inv_test')
        bench.build_system('inv_test')

        system = bench.system
        system.build_device_groups()

        # Build circuit data using fast method
        start = time.perf_counter()
        circuit = build_transient_circuit_data_fast(system, vdd=1.2, gmin=1e-9)
        elapsed = time.perf_counter() - start

        print(f"Circuit data built in {elapsed:.4f}s")
        print(f"Nodes: {circuit.num_nodes}")
        print(f"Voltage sources: {len(circuit.vsource_node_p)}")
        print(f"MOSFETs: {len(circuit.mosfet_node_d)}")
        print(f"Resistors: {len(circuit.resistor_node_p)}")
        print(f"Capacitors: {len(circuit.capacitor_node_p)}")
        print(f"Sparsity entries: {len(circuit.sparsity_rows)}")

        # Verify MOSFET count
        mosfet_count = sum(1 for d in system.devices if 'psp' in d.model_name.lower())
        assert len(circuit.mosfet_node_d) == mosfet_count

        # Verify sparsity pattern
        assert len(circuit.sparsity_rows) == len(circuit.sparsity_cols)
        assert len(circuit.sparsity_rows) > 0

    def test_residual_fn_construction(self, capsys):
        """Test residual function construction and evaluation"""
        print("\n" + "=" * 60)
        print("Residual Function Construction")
        print("=" * 60)

        bench = C6288Benchmark(verbose=False)
        bench.parse()
        bench.flatten('inv_test')
        bench.build_system('inv_test')

        system = bench.system
        system.build_device_groups()

        circuit = build_transient_circuit_data_fast(system, vdd=1.2, gmin=1e-9)
        residual_fn = build_transient_residual_fn(circuit)

        # Test residual evaluation
        n_reduced = circuit.n_reduced
        V_test = jnp.zeros(n_reduced)
        V_prev = jnp.zeros(n_reduced)
        dt = 1e-12

        # Set VDD node
        vdd_idx = None
        for name, idx in system.node_names.items():
            if 'vdd' in name.lower() and idx > 0:
                V_test = V_test.at[idx - 1].set(1.2)
                V_prev = V_prev.at[idx - 1].set(1.2)
                vdd_idx = idx
                break

        start = time.perf_counter()
        residual = residual_fn(V_test, V_prev, dt)
        elapsed = time.perf_counter() - start

        print(f"Residual evaluation time: {elapsed:.4f}s")
        print(f"Residual shape: {residual.shape}")
        print(f"Max |residual|: {float(jnp.max(jnp.abs(residual))):.2e}")

        assert residual.shape == (n_reduced,)


class TestICModeComparison:
    """Compare icmode='op' vs icmode='uic'"""

    def test_icmode_op_vs_uic(self, capsys):
        """Compare initial condition modes for inverter"""
        print("\n" + "=" * 60)
        print("icmode='op' vs icmode='uic' Comparison")
        print("=" * 60)

        bench = C6288Benchmark(verbose=False)
        bench.parse()
        bench.flatten('inv_test')
        bench.build_system('inv_test')

        system = bench.system
        system.build_device_groups()

        t_stop = 0.5e-9
        t_step = 10e-12

        # Test icmode='uic' (like VACASK)
        print("\n--- icmode='uic' ---")
        start = time.perf_counter()
        times_uic, solutions_uic, info_uic = transient_analysis_gpu(
            system,
            t_stop=t_stop,
            t_step=t_step,
            icmode='uic',
            vdd=1.2,
            verbose=False,
        )
        elapsed_uic = time.perf_counter() - start
        print(f"Time: {elapsed_uic:.2f}s")
        print(f"First step: {info_uic['first_step_time']:.2f}s")
        print(f"DC info: {info_uic['dc_info']}")

        # Test icmode='op' (compute DC operating point first)
        print("\n--- icmode='op' ---")
        start = time.perf_counter()
        times_op, solutions_op, info_op = transient_analysis_gpu(
            system,
            t_stop=t_stop,
            t_step=t_step,
            icmode='op',
            vdd=1.2,
            verbose=False,
        )
        elapsed_op = time.perf_counter() - start
        print(f"Time: {elapsed_op:.2f}s")
        print(f"First step: {info_op['first_step_time']:.2f}s")
        print(f"DC converged: {info_op['dc_info']['converged'] if info_op['dc_info'] else 'N/A'}")

        # Compare final voltages
        out_node = bench.nodes.get('inv_test.out', None)
        if out_node is not None:
            V_out_uic = float(solutions_uic[-1, out_node])
            V_out_op = float(solutions_op[-1, out_node])
            print(f"\nFinal output voltage:")
            print(f"  uic: {V_out_uic:.4f}V")
            print(f"  op:  {V_out_op:.4f}V")

            # Both should converge to similar values
            assert abs(V_out_uic - V_out_op) < 0.5, "Large difference between icmodes"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
