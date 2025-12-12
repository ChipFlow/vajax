#!/usr/bin/env python3
"""GPU Profiling Script for JAX-SPICE

Profiles CPU/GPU performance of VACASK benchmark circuits using VACASKBenchmarkRunner.
Compares dense vs sparse solvers across different circuit sizes.
Outputs a report suitable for GitHub Actions job summaries.

Usage:
    # Local benchmarking
    uv run scripts/profile_gpu.py

    # Single benchmark with Perfetto tracing
    uv run scripts/profile_gpu.py --benchmark ring --trace --trace-dir /tmp/traces

    # Run specific benchmarks
    uv run scripts/profile_gpu.py --benchmark ring,c6288
"""

import argparse
import os
import sys
import time
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from contextlib import nullcontext

# Ensure jax-spice is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure JAX memory allocation BEFORE importing JAX
# Disable preallocation to avoid grabbing all GPU memory at startup
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
# Use memory growth instead of fixed allocation
os.environ.setdefault('XLA_PYTHON_CLIENT_ALLOCATOR', 'platform')

import jax
import jax.numpy as jnp
import numpy as np

# Enable float64
jax.config.update('jax_enable_x64', True)

from jax_spice.benchmarks.runner import VACASKBenchmarkRunner


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run"""
    name: str
    nodes: int
    devices: int
    openvaf_devices: int
    timesteps: int
    total_time_s: float
    time_per_step_ms: float
    solver: str  # 'dense' or 'sparse'
    backend: str  # 'cpu' or 'gpu'
    converged: bool = True
    error: Optional[str] = None


class GPUProfiler:
    """Profiles VACASK benchmark circuits for CPU/GPU performance analysis"""

    def __init__(self):
        self.results: List[BenchmarkResult] = []
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

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

    def run_benchmark(self, sim_path: Path, name: str, use_sparse: bool,
                      num_steps: int = 20, warmup_steps: int = 5,
                      trace_ctx=None) -> BenchmarkResult:
        """Run a single benchmark configuration.

        Args:
            sim_path: Path to the .sim file
            name: Benchmark name
            use_sparse: Whether to use sparse solver
            num_steps: Number of timesteps for timing
            warmup_steps: Number of warmup steps (includes JIT compilation)
            trace_ctx: Optional JAX profiler trace context for Perfetto

        Returns:
            BenchmarkResult with timing information
        """
        import sys
        log(f"      starting...")
        sys.stdout.flush()
        sys.stderr.flush()

        backend = jax.default_backend()
        log(f"      backend: {backend}")
        sys.stdout.flush()
        sys.stderr.flush()

        try:
            # Parse circuit
            log(f"      parsing...")
            sys.stdout.flush()
            sys.stderr.flush()
            runner = VACASKBenchmarkRunner(sim_path, verbose=True)
            runner.parse()
            log("      parsing done")

            nodes = runner.num_nodes
            devices = len(runner.devices)
            openvaf_devices = sum(1 for d in runner.devices if d.get('is_openvaf'))

            # Estimate total nodes including internal nodes from OpenVAF devices
            # PSP103 has 8 internal nodes per device (13 total - 4 external - 1 branch)
            estimated_internal = openvaf_devices * 8  # Conservative estimate for PSP103
            total_nodes_estimate = nodes + estimated_internal
            log(f"      nodes: {nodes} external, ~{estimated_internal} internal ({total_nodes_estimate} total)")

            # Skip sparse for non-OpenVAF circuits (they use JIT solver)
            if use_sparse and not runner._has_openvaf_devices:
                return BenchmarkResult(
                    name=name,
                    nodes=nodes,
                    devices=devices,
                    openvaf_devices=openvaf_devices,
                    timesteps=0,
                    total_time_s=0,
                    time_per_step_ms=0,
                    solver='sparse',
                    backend=backend,
                    converged=True,
                    error="Sparse not applicable (no OpenVAF devices)"
                )

            # Get analysis params
            dt = runner.analysis_params.get('step', 1e-12)

            # Warmup run (includes JIT compilation)
            log(f"      warmup ({warmup_steps} steps, includes JIT)...")
            sys.stdout.flush()
            sys.stderr.flush()
            warmup_start = time.perf_counter()
            runner.run_transient(t_stop=dt * warmup_steps, dt=dt,
                                max_steps=warmup_steps, use_sparse=use_sparse)
            warmup_time = time.perf_counter() - warmup_start
            log(f"      warmup done ({warmup_time:.1f}s)")

            # Create fresh runner for timing (reuse compiled models)
            runner2 = VACASKBenchmarkRunner(sim_path, verbose=True)
            runner2.parse()
            if runner._has_openvaf_devices:
                runner2._compiled_models = runner._compiled_models

            log(f"_compiled_models = {runner2._compiled_models}")

            # Timed run (optionally with tracing)
            ctx = trace_ctx if trace_ctx else nullcontext()
            with ctx:
                start = time.perf_counter()
                times, voltages, stats = runner2.run_transient(
                    t_stop=dt * num_steps, dt=dt,
                    max_steps=num_steps, use_sparse=use_sparse
                )
                elapsed = time.perf_counter() - start

            actual_steps = len(times)
            time_per_step = (elapsed / actual_steps * 1000) if actual_steps > 0 else 0

            solver='sparse' if use_sparse else 'dense',
            log("Running benchmark... ({solver})")

            return BenchmarkResult(
                name=name,
                nodes=nodes,
                devices=devices,
                openvaf_devices=openvaf_devices,
                timesteps=actual_steps,
                total_time_s=elapsed,
                time_per_step_ms=time_per_step,
                solver=solver,
                backend=backend,
                converged=True,
            )

        except Exception as e:
            import traceback
            traceback.print_exc()
            return BenchmarkResult(
                name=name,
                nodes=0,
                devices=0,
                openvaf_devices=0,
                timesteps=0,
                total_time_s=0,
                time_per_step_ms=0,
                solver='sparse' if use_sparse else 'dense',
                backend=backend,
                converged=False,
                error=str(e)
            )

    def generate_report(self) -> str:
        """Generate a markdown report for GitHub Actions"""
        lines = []
        lines.append("# JAX-SPICE GPU Profiling Report\n")

        # Environment info
        lines.append("## Environment\n")
        lines.append(f"- **JAX Backend**: `{jax.default_backend()}`")
        lines.append(f"- **Devices**: `{[str(d) for d in jax.devices()]}`")

        gpu_devices = [d for d in jax.devices() if d.platform != 'cpu']
        if gpu_devices:
            lines.append(f"- **GPU**: `{gpu_devices[0]}`")
        else:
            lines.append("- **GPU**: None (CPU only)")
        lines.append("")

        # Overall timing
        lines.append("## Overall Timing\n")
        lines.append(f"- **Total Profiling Time**: {self.total_time_s:.3f}s")
        lines.append("")

        # Results table
        if self.results:
            lines.append("## Benchmark Results\n")
            lines.append("| Benchmark | Nodes | Devices | OpenVAF | Solver | Steps | Total (s) | Per Step (ms) | Status |")
            lines.append("|-----------|-------|---------|---------|--------|-------|-----------|---------------|--------|")

            for r in self.results:
                status = "OK" if r.converged and not r.error else (r.error or "Failed")
                if len(status) > 20:
                    status = status[:17] + "..."
                lines.append(
                    f"| {r.name} | {r.nodes} | {r.devices} | {r.openvaf_devices} | "
                    f"{r.solver} | {r.timesteps} | {r.total_time_s:.3f} | "
                    f"{r.time_per_step_ms:.1f} | {status} |"
                )
            lines.append("")

            # Dense vs Sparse comparison
            dense_results = [r for r in self.results if r.solver == 'dense' and r.converged and not r.error]
            sparse_results = [r for r in self.results if r.solver == 'sparse' and r.converged and not r.error]

            if dense_results and sparse_results:
                lines.append("## Dense vs Sparse Comparison\n")
                lines.append("| Benchmark | Dense (ms/step) | Sparse (ms/step) | Speedup |")
                lines.append("|-----------|-----------------|------------------|---------|")

                for dr in dense_results:
                    sr = next((r for r in sparse_results if r.name == dr.name), None)
                    if sr and sr.time_per_step_ms > 0:
                        speedup = dr.time_per_step_ms / sr.time_per_step_ms
                        lines.append(
                            f"| {dr.name} | {dr.time_per_step_ms:.1f} | "
                            f"{sr.time_per_step_ms:.1f} | {speedup:.2f}x |"
                        )
                lines.append("")

        return "\n".join(lines)


def get_vacask_benchmarks(names: Optional[List[str]] = None) -> List[Tuple[str, Path]]:
    """Get list of VACASK benchmark .sim files"""
    base = Path(__file__).parent.parent / "vendor" / "VACASK" / "benchmark"
    all_benchmarks = ['rc', 'graetz', 'mul', 'ring', 'c6288']

    if names:
        all_benchmarks = [n for n in names if n in all_benchmarks]

    benchmarks = []
    for name in all_benchmarks:
        sim_path = base / name / "vacask" / "runme.sim"
        if sim_path.exists():
            benchmarks.append((name, sim_path))

    return benchmarks


def log(msg="", end="\n"):
    """Print with flush for CI logs"""
    print(msg, end=end, flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Profile VACASK benchmarks on CPU/GPU"
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default=None,
        help="Comma-separated list of benchmarks to run (default: all)",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=20,
        help="Number of timesteps per benchmark (default: 20)",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=5,
        help="Number of warmup steps (default: 5)",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Enable Perfetto tracing (for Cloud Run profiling)",
    )
    parser.add_argument(
        "--trace-dir",
        type=str,
        default="/tmp/jax-trace",
        help="Directory for Perfetto traces (default: /tmp/jax-trace)",
    )
    parser.add_argument(
        "--sparse-only",
        action="store_true",
        help="Only run sparse solver (skip dense comparison)",
    )
    parser.add_argument(
        "--dense-only",
        action="store_true",
        help="Only run dense solver (skip sparse comparison)",
    )
    args = parser.parse_args()

    log("=" * 70)
    log("JAX-SPICE GPU Profiling")
    log("=" * 70)
    log()

    log("[Stage 1/4] Checking JAX configuration...")
    log(f"  JAX backend: {jax.default_backend()}")
    log(f"  JAX devices: {jax.devices()}")
    log(f"  Float64 enabled: {jax.config.jax_enable_x64}")
    log()

    # Parse benchmark names
    benchmark_names = None
    if args.benchmark:
        benchmark_names = [b.strip() for b in args.benchmark.split(',')]

    log("[Stage 2/4] Discovering benchmarks...")
    benchmarks = get_vacask_benchmarks(benchmark_names)
    log(f"  Found {len(benchmarks)} benchmarks: {[b[0] for b in benchmarks]}")
    if args.trace:
        log(f"  Perfetto tracing enabled, output: {args.trace_dir}")
    log()

    profiler = GPUProfiler()
    profiler.start()

    # Setup tracing context if requested
    trace_ctx = None
    if args.trace:
        os.makedirs(args.trace_dir, exist_ok=True)
        log(f"  Created trace directory: {args.trace_dir}")

    log("[Stage 3/4] Running benchmarks...")
    log()

    # Use trace context for entire benchmark suite if tracing
    trace_manager = jax.profiler.trace(args.trace_dir, create_perfetto_link=False) if args.trace else nullcontext()

    with trace_manager:
        for name, sim_path in benchmarks:
            log(f"  {name}:")
            sys.stdout.flush()
            sys.stderr.flush()

            # Determine which solvers to run
            # c6288 has ~86k total nodes (5k external + 81k internal from PSP103)
            # Dense matrix would be 86k x 86k x 8 bytes = ~56GB - skip dense
            run_dense = not args.sparse_only and name != 'c6288'
            run_sparse = not args.dense_only

            if name == 'c6288':
                log(f"    Note: ~86k total nodes (5k external + 81k internal)")
                if not args.sparse_only:
                    log(f"    Skipping dense (would need ~56GB memory)")

            # Run dense
            if run_dense:
                result_dense = profiler.run_benchmark(
                    sim_path, name, use_sparse=False,
                    num_steps=args.timesteps, warmup_steps=args.warmup_steps
                )
                profiler.results.append(result_dense)
                if result_dense.error:
                    log(f"    dense:  ERROR - {result_dense.error}")
                else:
                    log(f"    dense:  {result_dense.time_per_step_ms:.1f}ms/step ({result_dense.timesteps} steps)")

            # Run sparse
            if run_sparse:
                result_sparse = profiler.run_benchmark(
                    sim_path, name, use_sparse=True,
                    num_steps=args.timesteps, warmup_steps=args.warmup_steps
                )
                profiler.results.append(result_sparse)
                if result_sparse.error:
                    log(f"    sparse: {result_sparse.error}")
                else:
                    log(f"    sparse: {result_sparse.time_per_step_ms:.1f}ms/step ({result_sparse.timesteps} steps)")

            log()

    profiler.stop()

    log("[Stage 4/4] Generating report...")
    report = profiler.generate_report()

    log()
    log(report)

    # Write to file
    report_path = Path(__file__).parent.parent / "profile_report.md"
    report_path.write_text(report)
    log(f"Report written to: {report_path}")

    # Write to GitHub step summary if available
    if 'GITHUB_STEP_SUMMARY' in os.environ:
        with open(os.environ['GITHUB_STEP_SUMMARY'], 'a') as f:
            f.write(report)
        log("Report appended to GitHub step summary")

    # JSON report
    json_report = {
        'environment': {
            'backend': jax.default_backend(),
            'devices': [str(d) for d in jax.devices()],
        },
        'results': [
            {
                'name': r.name,
                'nodes': r.nodes,
                'devices': r.devices,
                'openvaf_devices': r.openvaf_devices,
                'timesteps': r.timesteps,
                'total_time_s': r.total_time_s,
                'time_per_step_ms': r.time_per_step_ms,
                'solver': r.solver,
                'backend': r.backend,
                'converged': r.converged,
                'error': r.error,
            }
            for r in profiler.results
        ],
        'total_time_s': profiler.total_time_s,
    }

    json_path = Path(__file__).parent.parent / "profile_report.json"
    json_path.write_text(json.dumps(json_report, indent=2))
    log(f"JSON report written to: {json_path}")

    if args.trace:
        log()
        log(f"Perfetto traces saved to: {args.trace_dir}")
        log("To view traces:")
        log("  1. Open https://ui.perfetto.dev/")
        log(f"  2. Load trace files from: {args.trace_dir}")

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
