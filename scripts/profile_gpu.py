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
            gpu_ops = ['jacobian_build', 'sparse_solve', 'device_eval', 'jax_ops']

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

    # Profile 10 iterations
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


def main():
    print("=" * 70)
    print("JAX-SPICE GPU Profiling")
    print("=" * 70)
    print()

    profiler = GPUProfiler()

    # Profile full simulation with limited iterations to avoid timeout
    # The C6288 circuit may not converge, but we want to measure GPU performance
    # Use only 10 iterations to stay within Cloud Run's 30-minute timeout
    print("Profiling C6288 circuit simulation (max 10 iterations)...")
    sim_info = profile_c6288_simulation(profiler, 'c6288_test', max_iterations=10)

    print(f"  Circuit: {sim_info['circuit']}")
    print(f"  Nodes: {sim_info['nodes']}")
    print(f"  Devices: {sim_info['devices']}")
    print(f"  Converged: {sim_info['converged']}")
    print(f"  Iterations: {sim_info['iterations']}")
    print(f"  Residual: {sim_info['residual']:.2e}")
    print()

    # Profile iteration breakdown on smaller circuit
    print("Profiling iteration breakdown (inv_test)...")
    profile_iteration_breakdown(profiler)
    print()

    # Generate report
    report = profiler.generate_report()

    # Print to stdout
    print(report)

    # Write to file for CI
    report_path = Path(__file__).parent.parent / "profile_report.md"
    report_path.write_text(report)
    print(f"\nReport written to: {report_path}")

    # Write to GitHub step summary if available
    if 'GITHUB_STEP_SUMMARY' in os.environ:
        with open(os.environ['GITHUB_STEP_SUMMARY'], 'a') as f:
            f.write(report)
        print("Report appended to GitHub step summary")

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
    print(f"JSON report written to: {json_path}")


if __name__ == '__main__':
    main()
