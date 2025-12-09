#!/usr/bin/env python3
"""GPU Profiling Script for JAX-SPICE

Profiles CPU/GPU utilization and data movement during circuit simulation.
Outputs a report suitable for GitHub Actions job summaries.
"""

import os
import sys
import time
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from contextlib import contextmanager

# Ensure jax-spice is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

import jax
import jax.numpy as jnp
import numpy as np

# Enable float64
jax.config.update('jax_enable_x64', True)


@dataclass
class TimingStats:
    """Timing statistics for a profiled operation"""
    name: str
    total_time_ms: float = 0.0
    call_count: int = 0
    min_time_ms: float = float('inf')
    max_time_ms: float = 0.0
    times: List[float] = field(default_factory=list)

    def record(self, time_ms: float):
        self.times.append(time_ms)
        self.total_time_ms += time_ms
        self.call_count += 1
        self.min_time_ms = min(self.min_time_ms, time_ms)
        self.max_time_ms = max(self.max_time_ms, time_ms)

    @property
    def avg_time_ms(self) -> float:
        return self.total_time_ms / self.call_count if self.call_count > 0 else 0.0


class GPUProfiler:
    """Profiles JAX operations for CPU/GPU utilization analysis"""

    def __init__(self):
        self.timings: Dict[str, TimingStats] = {}
        self.transfers: List[Dict] = []
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

    @contextmanager
    def measure(self, name: str):
        """Context manager to measure execution time of a block"""
        # Ensure any pending operations complete
        jax.block_until_ready(jnp.array(0.0))

        start = time.perf_counter()
        yield
        # Block until all JAX operations complete
        jax.block_until_ready(jnp.array(0.0))
        elapsed_ms = (time.perf_counter() - start) * 1000

        if name not in self.timings:
            self.timings[name] = TimingStats(name=name)
        self.timings[name].record(elapsed_ms)

    def record_transfer(self, name: str, size_bytes: int, direction: str):
        """Record a data transfer between CPU and GPU"""
        self.transfers.append({
            'name': name,
            'size_bytes': size_bytes,
            'size_mb': size_bytes / (1024 * 1024),
            'direction': direction,  # 'to_gpu' or 'from_gpu'
        })

    def start(self):
        """Mark the start of profiling"""
        self.start_time = time.perf_counter()

    def stop(self):
        """Mark the end of profiling"""
        self.end_time = time.perf_counter()

    @property
    def total_time_s(self) -> float:
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0

    def generate_report(self) -> str:
        """Generate a markdown report for GitHub Actions"""
        lines = []
        lines.append("# JAX-SPICE GPU Profiling Report\n")

        # Environment info
        lines.append("## Environment\n")
        lines.append(f"- **JAX Backend**: `{jax.default_backend()}`")
        lines.append(f"- **Devices**: `{[str(d) for d in jax.devices()]}`")

        # Check for GPU
        gpu_devices = [d for d in jax.devices() if d.platform != 'cpu']
        if gpu_devices:
            lines.append(f"- **GPU**: `{gpu_devices[0]}`")
        else:
            lines.append("- **GPU**: None (CPU only)")
        lines.append("")

        # Overall timing
        lines.append("## Overall Timing\n")
        lines.append(f"- **Total Simulation Time**: {self.total_time_s:.3f}s")
        lines.append("")

        # Timing breakdown
        if self.timings:
            lines.append("## Operation Timing Breakdown\n")
            lines.append("| Operation | Total (ms) | Calls | Avg (ms) | Min (ms) | Max (ms) | % of Total |")
            lines.append("|-----------|------------|-------|----------|----------|----------|------------|")

            total_measured = sum(t.total_time_ms for t in self.timings.values())
            sorted_timings = sorted(self.timings.values(), key=lambda x: -x.total_time_ms)

            for t in sorted_timings:
                pct = (t.total_time_ms / total_measured * 100) if total_measured > 0 else 0
                lines.append(
                    f"| {t.name} | {t.total_time_ms:.1f} | {t.call_count} | "
                    f"{t.avg_time_ms:.2f} | {t.min_time_ms:.2f} | {t.max_time_ms:.2f} | {pct:.1f}% |"
                )
            lines.append("")

        # Data transfers
        if self.transfers:
            lines.append("## Data Transfers\n")

            to_gpu = [t for t in self.transfers if t['direction'] == 'to_gpu']
            from_gpu = [t for t in self.transfers if t['direction'] == 'from_gpu']

            total_to_gpu_mb = sum(t['size_mb'] for t in to_gpu)
            total_from_gpu_mb = sum(t['size_mb'] for t in from_gpu)

            lines.append(f"- **To GPU**: {len(to_gpu)} transfers, {total_to_gpu_mb:.2f} MB total")
            lines.append(f"- **From GPU**: {len(from_gpu)} transfers, {total_from_gpu_mb:.2f} MB total")
            lines.append("")

            # Show largest transfers
            all_transfers = sorted(self.transfers, key=lambda x: -x['size_bytes'])[:10]
            if all_transfers:
                lines.append("### Largest Transfers\n")
                lines.append("| Name | Size (MB) | Direction |")
                lines.append("|------|-----------|-----------|")
                for t in all_transfers:
                    lines.append(f"| {t['name']} | {t['size_mb']:.3f} | {t['direction']} |")
                lines.append("")

        # CPU vs GPU analysis
        lines.append("## CPU vs GPU Analysis\n")

        if self.timings:
            # Categorize operations
            cpu_ops = ['parse', 'flatten', 'build_system', 'numpy_ops']
            gpu_ops = ['jacobian_build', 'sparse_solve', 'device_eval', 'jax_ops', 'gpu_native', 'transient_gpu']

            cpu_time = sum(
                t.total_time_ms for name, t in self.timings.items()
                if any(op in name.lower() for op in cpu_ops)
            )
            gpu_time = sum(
                t.total_time_ms for name, t in self.timings.items()
                if any(op in name.lower() for op in gpu_ops)
            )
            other_time = sum(t.total_time_ms for t in self.timings.values()) - cpu_time - gpu_time

            total = cpu_time + gpu_time + other_time
            if total > 0:
                lines.append(f"- **CPU-bound operations**: {cpu_time:.1f}ms ({cpu_time/total*100:.1f}%)")
                lines.append(f"- **GPU-bound operations**: {gpu_time:.1f}ms ({gpu_time/total*100:.1f}%)")
                lines.append(f"- **Other/Mixed**: {other_time:.1f}ms ({other_time/total*100:.1f}%)")
            lines.append("")

        return "\n".join(lines)


def profile_c6288_simulation(profiler: GPUProfiler, circuit_name: str = 'c6288_test',
                              max_iterations: int = 100):
    """Profile a full C6288 circuit simulation

    Args:
        profiler: GPUProfiler instance
        circuit_name: Circuit to profile
        max_iterations: Maximum iterations (to avoid timeout on non-converging circuits)
    """
    from jax_spice.benchmarks.c6288 import C6288Benchmark

    profiler.start()

    # Phase 1: Parse netlist (CPU)
    bench = C6288Benchmark(verbose=False)
    with profiler.measure("parse"):
        bench.parse()

    # Phase 2: Flatten hierarchy (CPU)
    with profiler.measure("flatten"):
        bench.flatten(circuit_name)

    # Phase 3: Build MNA system (CPU)
    with profiler.measure("build_system"):
        bench.build_system(circuit_name)

    # Record system size for transfer estimation
    n_nodes = bench.system.num_nodes
    n_devices = len(bench.system.devices)

    # Phase 4: Run DC analysis (mixed CPU/GPU)
    # Use sparse DC solver with limited iterations to avoid timeout
    with profiler.measure("dc_analysis_total"):
        V, info = bench.run_sparse_dc(
            max_iterations=max_iterations,
            abstol=1e-9,
            verbose=False
        )

    profiler.stop()

    # Estimate data transfers based on circuit size
    # Each iteration: V vector to GPU, result back
    n_iterations = info.get('iterations', 0)
    v_size = n_nodes * 8  # float64
    jacobian_nnz_estimate = n_devices * 16  # ~16 entries per device
    jacobian_size = jacobian_nnz_estimate * 8  # float64 data

    profiler.record_transfer("voltage_vectors", v_size * n_iterations, "to_gpu")
    profiler.record_transfer("jacobian_data", jacobian_size * n_iterations, "to_gpu")
    profiler.record_transfer("solution_vectors", v_size * n_iterations, "from_gpu")

    return {
        'circuit': circuit_name,
        'nodes': n_nodes,
        'devices': n_devices,
        'converged': info.get('converged', False),
        'iterations': n_iterations,
        'residual': info.get('residual_norm', 0),
    }


def profile_iteration_breakdown(profiler: GPUProfiler):
    """Profile individual operations within Newton-Raphson iterations"""
    from jax_spice.benchmarks.c6288 import C6288Benchmark
    from jax_spice.analysis.context import AnalysisContext
    from jax_spice.analysis.sparse import sparse_solve_csr

    bench = C6288Benchmark(verbose=False)
    bench.parse()
    bench.flatten('inv_test')  # Use small circuit for detailed profiling
    bench.build_system('inv_test')

    # Initial voltage
    V = jnp.zeros(bench.system.num_nodes, dtype=jnp.float64)
    context = AnalysisContext(time=0.0, dt=1e-9, analysis_type='dc', gmin=1e-9)

    # Profile 10 iterations of old CPU-based solver
    for i in range(10):
        # Jacobian build
        with profiler.measure("jacobian_build"):
            (data, indices, indptr, shape), f = bench.system.build_sparse_jacobian_and_residual(V, context)

        # Sparse solve
        with profiler.measure("sparse_solve"):
            delta_V = sparse_solve_csr(
                jnp.array(data),
                jnp.array(indices),
                jnp.array(indptr),
                jnp.array(-f),
                shape
            )

        # Update
        with profiler.measure("voltage_update"):
            V = V.at[1:].add(delta_V)


def profile_gpu_native_solver(profiler: GPUProfiler, circuit_name: str = 'inv_test'):
    """Profile the GPU-native DC solver using sparsejac

    This uses the new GPU-native solver that:
    1. Builds a vectorized JAX residual function
    2. Uses sparsejac for automatic sparse Jacobian computation
    3. Uses jax.experimental.sparse.linalg.spsolve (cuSOLVER on GPU)
    """
    from jax_spice.benchmarks.c6288 import C6288Benchmark
    from jax_spice.analysis.dc_gpu import dc_operating_point_gpu

    bench = C6288Benchmark(verbose=False)
    bench.parse()
    bench.flatten(circuit_name)
    bench.build_system(circuit_name)

    print(f"  GPU-native solver on {circuit_name}:")
    print(f"    Nodes: {bench.system.num_nodes}, Devices: {len(bench.system.devices)}")

    # Run GPU-native solver
    with profiler.measure(f"gpu_native_{circuit_name}"):
        V, info = dc_operating_point_gpu(
            bench.system,
            vdd=1.2,
            max_iterations=100,
            abstol=1e-9,
            verbose=False
        )

    print(f"    Converged: {info.get('converged', False)}")
    print(f"    Iterations: {info.get('iterations', 0)}")

    return {
        'circuit': circuit_name,
        'nodes': bench.system.num_nodes,
        'devices': len(bench.system.devices),
        'converged': info.get('converged', False),
        'iterations': info.get('iterations', 0),
    }


def profile_gpu_transient_solver(profiler: GPUProfiler, circuit_name: str = 'inv_test',
                                  num_timesteps: int = 10, t_step: float = 1e-9):
    """Profile the GPU-native transient solver using sparsejac

    This profiles:
    1. Circuit data setup time
    2. First timestep (includes JIT compilation)
    3. Subsequent timesteps (steady-state performance)
    """
    from jax_spice.benchmarks.c6288 import C6288Benchmark
    from jax_spice.analysis.transient_gpu import transient_analysis_gpu

    bench = C6288Benchmark(verbose=False)
    bench.parse()
    bench.flatten(circuit_name)
    bench.build_system(circuit_name)

    print(f"  GPU-native transient solver on {circuit_name}:")
    print(f"    Nodes: {bench.system.num_nodes}, Devices: {len(bench.system.devices)}")
    print(f"    Timesteps: {num_timesteps}, t_step: {t_step*1e9:.2f}ns")

    # Run GPU-native transient solver
    with profiler.measure(f"transient_gpu_{circuit_name}"):
        t_points, V_history, info = transient_analysis_gpu(
            bench.system,
            t_stop=t_step * num_timesteps,
            t_step=t_step,
            vdd=bench.vdd,
            verbose=False
        )

    print(f"    Completed: {len(t_points)} timesteps")
    if 'first_step_time' in info:
        print(f"    First step (JIT compile): {info['first_step_time']:.3f}s")
    if 'avg_step_time' in info:
        print(f"    Avg step time: {info['avg_step_time']*1000:.2f}ms")
    if 'total_iterations' in info:
        print(f"    Total iterations: {info['total_iterations']}")

    return {
        'circuit': circuit_name,
        'nodes': bench.system.num_nodes,
        'devices': len(bench.system.devices),
        'timesteps': len(t_points),
        'first_step_time': info.get('first_step_time', 0),
        'avg_step_time': info.get('avg_step_time', 0),
        'total_iterations': info.get('total_iterations', 0),
    }


def main():
    # Use explicit flush for all prints to ensure output is visible in CI logs
    def log(msg=""):
        print(msg)
        sys.stdout.flush()

    log("=" * 70)
    log("JAX-SPICE GPU Profiling")
    log("=" * 70)
    log()

    log("[Stage 1/7] Checking JAX configuration...")
    log(f"  JAX backend: {jax.default_backend()}")
    log(f"  JAX devices: {jax.devices()}")
    log(f"  Float64 enabled: {jax.config.jax_enable_x64}")
    log()

    log("[Stage 2/7] Creating profiler...")
    profiler = GPUProfiler()
    profiler.start()  # Track total time
    log("  Profiler started")
    log()

    # Profile GPU-native solver on circuits that converge for DC analysis
    # Note: c6288_test is excluded because it doesn't converge for DC operating point
    # with our simplified Level-1 MOSFET model. The VACASK benchmarks for c6288 use
    # transient simulation (not DC), which requires MOSFET support in transient analysis.
    log("[Stage 3/7] Profiling GPU-native DC solver (sparsejac + cuSOLVER)...")
    for circuit in ['inv_test', 'nor_test']:
        log(f"  Starting {circuit}...")
        try:
            gpu_info = profile_gpu_native_solver(profiler, circuit)
            log(f"  Completed {circuit}")
        except Exception as e:
            log(f"  {circuit}: Error - {e}")
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
    log()

    # Profile GPU transient solver on small circuits
    log("[Stage 4/7] Profiling GPU-native transient solver on small circuits...")
    for circuit in ['inv_test', 'nor_test']:
        log(f"  Starting {circuit}...")
        try:
            transient_info = profile_gpu_transient_solver(profiler, circuit, num_timesteps=10)
            log(f"  Completed {circuit}")
        except Exception as e:
            log(f"  {circuit}: Error - {e}")
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
    log()

    # Profile GPU transient solver on C6288 (the main benchmark)
    # Note: Full VACASK benchmark uses 1021 timepoints, but we reduce to 50 for CI timeout constraints
    # The 31-minute Cloud Run timeout is tight for large circuits with JIT compilation overhead
    log("[Stage 5/7] Profiling GPU-native transient solver on C6288...")
    log("  Running 50 timesteps (reduced for CI timeout)...")
    try:
        c6288_info = profile_gpu_transient_solver(
            profiler,
            'c6288_test',
            num_timesteps=50,   # Reduced from 1200 to fit within CI timeout
            t_step=0.1e-9       # 0.1ns timestep like VACASK
        )
        log("  Completed c6288_test")
    except Exception as e:
        log(f"  c6288_test: Error - {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
    log()

    # Profile iteration breakdown on smaller circuit (old CPU solver for comparison)
    log("[Stage 6/7] Profiling CPU-based iteration breakdown (inv_test)...")
    try:
        profile_iteration_breakdown(profiler)
        log("  Completed iteration breakdown")
    except Exception as e:
        log(f"  Error in iteration breakdown: {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
    log()

    profiler.stop()  # End total time tracking

    # Skip the slow C6288 CPU-based profiling by default
    # It takes ~77s per iteration with the old solver
    run_c6288_cpu_profile = False
    sim_info = None
    if run_c6288_cpu_profile:
        log("Profiling C6288 circuit simulation (max 10 iterations)...")
        sim_info = profile_c6288_simulation(profiler, 'c6288_test', max_iterations=10)
        log(f"  Circuit: {sim_info['circuit']}")
        log(f"  Nodes: {sim_info['nodes']}")
        log(f"  Devices: {sim_info['devices']}")
        log(f"  Converged: {sim_info['converged']}")
        log(f"  Iterations: {sim_info['iterations']}")
        log(f"  Residual: {sim_info['residual']:.2e}")
        log()

    # Generate report
    log("[Stage 7/7] Generating report...")
    try:
        report = profiler.generate_report()
        log("  Report generated")
    except Exception as e:
        log(f"  Error generating report: {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        report = f"Error generating report: {e}"

    # Print to stdout
    log()
    log(report)

    # Write to file for CI
    log()
    log("Writing report files...")
    report_path = Path(__file__).parent.parent / "profile_report.md"
    report_path.write_text(report)
    log(f"  Report written to: {report_path}")

    # Write to GitHub step summary if available
    if 'GITHUB_STEP_SUMMARY' in os.environ:
        with open(os.environ['GITHUB_STEP_SUMMARY'], 'a') as f:
            f.write(report)
        log("  Report appended to GitHub step summary")

    # Also output JSON for programmatic use
    json_report = {
        'environment': {
            'backend': jax.default_backend(),
            'devices': [str(d) for d in jax.devices()],
        },
        'simulation': sim_info,
        'timings': {
            name: {
                'total_ms': t.total_time_ms,
                'calls': t.call_count,
                'avg_ms': t.avg_time_ms,
            }
            for name, t in profiler.timings.items()
        },
        'total_time_s': profiler.total_time_s,
    }

    json_path = Path(__file__).parent.parent / "profile_report.json"
    json_path.write_text(json.dumps(json_report, indent=2))
    log(f"  JSON report written to: {json_path}")

    log()
    log("=" * 70)
    log("Profiling complete!")
    log("=" * 70)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n*** FATAL ERROR ***")
        print(f"Unhandled exception: {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        sys.exit(1)
