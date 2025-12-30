#!/usr/bin/env python3
"""GPU Profiling Script for JAX-SPICE

Profiles CPU/GPU performance of VACASK benchmark circuits using CircuitEngine.
Compares dense vs sparse solvers across different circuit sizes.
Outputs a report suitable for GitHub Actions job summaries.

Each benchmark runs in a separate subprocess to ensure complete memory cleanup
between benchmarks and avoid OOM from GPU memory accumulation.

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
import subprocess
from pathlib import Path
from dataclasses import asdict
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

# Import jax_spice first to auto-configure precision based on backend
# (Metal/TPU use f32, CPU/CUDA use f64)
from jax_spice.analysis import CircuitEngine
from jax_spice.logging import enable_performance_logging, logger
from scripts.benchmark_utils import BenchmarkResult, get_vacask_benchmarks

# Enable verbose logging with flush and memory stats for profiling visibility
enable_performance_logging()


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
                      num_steps: int = 20, trace_ctx=None, full: bool = False,
                      warmup_steps: int = 5) -> BenchmarkResult:
        """Run a single benchmark configuration.

        Args:
            sim_path: Path to the .sim file
            name: Benchmark name
            use_sparse: Whether to use sparse solver
            num_steps: Number of timesteps for timing (ignored if full=True)
            trace_ctx: Optional JAX profiler trace context for Perfetto
            full: If True, use original VACASK parameters for comparable results
            warmup_steps: Number of warmup steps for JIT compilation

        Returns:
            BenchmarkResult with timing information
        """
        import sys
        logger.info("      starting...")
        sys.stdout.flush()
        sys.stderr.flush()

        backend = jax.default_backend()
        logger.info(f"      backend: {backend}")
        sys.stdout.flush()
        sys.stderr.flush()

        try:
            # Parse circuit
            logger.info(f"      parsing...")
            sys.stdout.flush()
            sys.stderr.flush()
            engine = CircuitEngine(sim_path)
            engine.parse()
            logger.info("      parsing done")

            nodes = engine.num_nodes
            devices = len(engine.devices)
            openvaf_devices = sum(1 for d in engine.devices if d.get('is_openvaf'))

            # Estimate total nodes including internal nodes from OpenVAF devices
            # PSP103 has 8 internal nodes per device (13 total - 4 external - 1 branch)
            estimated_internal = openvaf_devices * 8  # Conservative estimate for PSP103
            total_nodes_estimate = nodes + estimated_internal
            logger.info(f"      nodes: {nodes} external, ~{estimated_internal} internal ({total_nodes_estimate} total)")

            # Use GPU if available, otherwise CPU
            selected_backend = "gpu" if any(d.platform == 'gpu' for d in jax.devices()) else "cpu"

            # Skip sparse for non-OpenVAF circuits (they use JIT solver)
            if use_sparse and not engine._has_openvaf_devices:
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

            # Get analysis params from .sim file
            dt = engine.analysis_params.get('step', 1e-12)
            t_stop_original = engine.analysis_params.get('stop', 1e-9)

            if full:
                # Use original VACASK parameters for comparable benchmarking
                t_stop = t_stop_original
                expected_steps = int(t_stop / dt)
                logger.info(f"      FULL mode: t_stop={t_stop:.2e}s, dt={dt:.2e}s ({expected_steps} steps)")
            else:
                # Use reduced parameters for quick testing
                t_stop = dt * num_steps
                expected_steps = num_steps

            # Warmup run (includes JIT compilation) - optional
            if warmup_steps > 0:
                logger.info(f"      warmup ({warmup_steps} steps, includes JIT)...")
                sys.stdout.flush()
                sys.stderr.flush()
                warmup_start = time.perf_counter()
                engine.run_transient(t_stop=dt * warmup_steps, dt=dt,
                                    max_steps=warmup_steps, use_sparse=use_sparse,
                                    backend=selected_backend)
                warmup_time = time.perf_counter() - warmup_start
                logger.info(f"      warmup done ({warmup_time:.1f}s)")
            else:
                logger.info("      skipping warmup (JIT included in timing)")

            # Timed run (optionally with tracing)
            # This is the measurement comparable to VACASK benchmark.py "Runtime"
            ctx = trace_ctx if trace_ctx else nullcontext()
            logger.info(f"      timed run: t_stop={t_stop:.2e}s, dt={dt:.2e}s...")
            sys.stdout.flush()
            with ctx:
                start = time.perf_counter()
                result = engine.run_transient(
                    t_stop=t_stop, dt=dt,
                    max_steps=expected_steps * 2,  # Allow some margin
                    use_sparse=use_sparse,
                    backend=selected_backend
                )
                elapsed = time.perf_counter() - start

            actual_steps = result.num_steps
            time_per_step = (elapsed / actual_steps * 1000) if actual_steps > 0 else 0

            solver = 'sparse' if use_sparse else 'dense'

            # Log in VACASK-comparable format
            logger.info(f"")
            logger.info(f"      === VACASK-comparable metrics ===")
            logger.info(f"      Runtime:        {elapsed:.3f}s")
            logger.info(f"      Timesteps:      {actual_steps}")
            logger.info(f"      Time per step:  {time_per_step:.3f}ms")
            logger.info(f"      Solver:         {solver}")
            logger.info(f"      Backend:        {selected_backend}")

            return BenchmarkResult(
                name=name,
                nodes=nodes,
                devices=devices,
                openvaf_devices=openvaf_devices,
                timesteps=actual_steps,
                total_time_s=elapsed,
                time_per_step_ms=time_per_step,
                solver=solver,
                backend=selected_backend,
                t_stop=t_stop,
                dt=dt,
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

        # VACASK-comparable results (main table)
        if self.results:
            lines.append("## VACASK-Comparable Results\n")
            lines.append("These metrics are directly comparable with VACASK benchmark.py output.\n")
            lines.append("| Benchmark | Solver | t_stop | dt | Steps | Runtime (s) | Status |")
            lines.append("|-----------|--------|--------|-----|-------|-------------|--------|")

            for r in self.results:
                status = "OK" if r.converged and not r.error else (r.error or "Failed")
                if len(status) > 15:
                    status = status[:12] + "..."
                t_stop_str = f"{r.t_stop:.2e}" if r.t_stop else "N/A"
                dt_str = f"{r.dt:.2e}" if r.dt else "N/A"
                lines.append(
                    f"| {r.name} | {r.solver} | {t_stop_str} | {dt_str} | "
                    f"{r.timesteps} | **{r.total_time_s:.3f}** | {status} |"
                )
            lines.append("")

            # Detailed results table
            lines.append("## Detailed Results\n")
            lines.append("| Benchmark | Nodes | Devices | OpenVAF | Solver | Per Step (ms) | Backend |")
            lines.append("|-----------|-------|---------|---------|--------|---------------|---------|")

            for r in self.results:
                if r.converged and not r.error:
                    lines.append(
                        f"| {r.name} | {r.nodes} | {r.devices} | {r.openvaf_devices} | "
                        f"{r.solver} | {r.time_per_step_ms:.3f} | {r.backend} |"
                    )
            lines.append("")

            # Dense vs Sparse comparison
            dense_results = [r for r in self.results if r.solver == 'dense' and r.converged and not r.error]
            sparse_results = [r for r in self.results if r.solver == 'sparse' and r.converged and not r.error]

            if dense_results and sparse_results:
                lines.append("## Dense vs Sparse Comparison\n")
                lines.append("| Benchmark | Dense Runtime (s) | Sparse Runtime (s) | Speedup |")
                lines.append("|-----------|-------------------|--------------------|---------| ")

                for dr in dense_results:
                    sr = next((r for r in sparse_results if r.name == dr.name), None)
                    if sr and sr.total_time_s > 0:
                        speedup = dr.total_time_s / sr.total_time_s
                        lines.append(
                            f"| {dr.name} | {dr.total_time_s:.3f} | "
                            f"{sr.total_time_s:.3f} | {speedup:.2f}x |"
                        )
                lines.append("")

        return "\n".join(lines)


# get_vacask_benchmarks is imported from scripts.benchmark_utils


def run_single_benchmark(args):
    """Run a single benchmark in subprocess mode and output JSON result."""
    # Parse the single benchmark spec: "name:dense" or "name:sparse"
    parts = args.single.split(':')
    if len(parts) != 2:
        print(json.dumps({'error': f'Invalid --single format: {args.single}'}))
        sys.exit(1)

    name, solver = parts
    use_sparse = solver == 'sparse'

    # Find the benchmark
    benchmarks = get_vacask_benchmarks([name])
    if not benchmarks:
        print(json.dumps({'error': f'Benchmark not found: {name}'}))
        sys.exit(1)

    _, sim_path = benchmarks[0]

    # Run the benchmark
    profiler = GPUProfiler()
    result = profiler.run_benchmark(
        sim_path, name, use_sparse=use_sparse,
        num_steps=args.timesteps,
        full=args.full,
        warmup_steps=args.warmup_steps,
    )

    # Output JSON result
    print(json.dumps(asdict(result)))


def run_benchmark_subprocess(name: str, solver: str, timesteps: int, warmup_steps: int,
                              full: bool = False, cpu: bool = False) -> Optional[BenchmarkResult]:
    """Run a benchmark in a separate subprocess to ensure memory cleanup.

    Streams stdout/stderr in real-time and captures the JSON result from the last line.
    """
    script_path = Path(__file__)
    cmd = [
        sys.executable, str(script_path),
        '--single', f'{name}:{solver}',
        '--timesteps', str(timesteps),
        '--warmup-steps', str(warmup_steps),
    ]
    if full:
        cmd.append('--full')
    if cpu:
        cmd.append('--cpu')

    try:
        # Start subprocess with pipes for streaming
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Merge stderr into stdout
            text=True,
            bufsize=1,  # Line buffered
            env={**os.environ},
        )

        # Stream output and capture lines
        output_lines = []
        for line in proc.stdout:
            line = line.rstrip('\n')
            output_lines.append(line)
            # Print with indent to show it's from subprocess
            print(f"      {line}", flush=True)

        proc.wait(timeout=1800)  # 30 minute timeout

        if not output_lines:
            logger.error(f"  No output from subprocess")
            return None

        # Try to parse the last line as JSON
        json_line = output_lines[-1]
        try:
            data = json.loads(json_line)
            if 'error' in data and data.get('nodes') is None:
                logger.error(f"  Subprocess error: {data['error']}")
                return None
            return BenchmarkResult(**data)
        except json.JSONDecodeError:
            logger.error(f"  Failed to parse JSON from subprocess")
            logger.error(f"  Last line: {json_line[:200]}")
            return None

    except subprocess.TimeoutExpired:
        proc.kill()
        logger.error(f"  Benchmark timed out after 30 minutes")
        return BenchmarkResult(
            name=name, nodes=0, devices=0, openvaf_devices=0,
            timesteps=0, total_time_s=0, time_per_step_ms=0,
            solver=solver, backend='gpu', converged=False,
            error="Timeout after 30 minutes"
        )
    except Exception as e:
        logger.error(f"  Subprocess failed: {e}")
        return None


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
    parser.add_argument(
        "--single",
        type=str,
        default=None,
        help="Internal: run single benchmark (format: name:sparse|dense)",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=0,
        help="Number of warmup steps for JIT compilation (default: 0, timing includes JIT)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run with original VACASK parameters (full simulation, comparable results)",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Allow running on CPU (skip GPU requirement check)",
    )
    args = parser.parse_args()

    # Single benchmark mode (run in subprocess)
    if args.single:
        run_single_benchmark(args)
        return

    logger.info("=" * 70)
    logger.info("JAX-SPICE GPU Profiling")
    logger.info("=" * 70)
    logger.info("")

    logger.info("[Stage 1/4] Checking JAX configuration...")
    logger.info(f"  JAX backend: {jax.default_backend()}")
    logger.info(f"  JAX devices: {jax.devices()}")
    logger.info(f"  Float64 enabled: {jax.config.jax_enable_x64}")

    # Check for GPU backend
    has_gpu = any(d.platform == 'gpu' for d in jax.devices())
    if not has_gpu and not args.cpu:
        logger.error("ERROR: No GPU device found. Use --cpu flag for CPU-only profiling.")
        sys.exit(1)
    if not has_gpu:
        logger.info("  Running in CPU-only mode (--cpu flag)")
    logger.info("")

    # Parse benchmark names
    benchmark_names = None
    if args.benchmark:
        benchmark_names = [b.strip() for b in args.benchmark.split(',')]

    logger.info("[Stage 2/4] Discovering benchmarks...")
    benchmarks = get_vacask_benchmarks(benchmark_names)
    logger.info(f"  Found {len(benchmarks)} benchmarks: {[b[0] for b in benchmarks]}")
    if args.trace:
        logger.info(f"  Perfetto tracing enabled, output: {args.trace_dir}")
    logger.info("")

    profiler = GPUProfiler()
    profiler.start()

    # Setup tracing context if requested
    trace_ctx = None
    if args.trace:
        os.makedirs(args.trace_dir, exist_ok=True)
        logger.info(f"  Created trace directory: {args.trace_dir}")

    logger.info("[Stage 3/4] Running benchmarks...")
    logger.info("  (Each benchmark runs in a separate subprocess for memory isolation)")
    logger.info("")

    # Use trace context for entire benchmark suite if tracing
    trace_manager = jax.profiler.trace(args.trace_dir, create_perfetto_link=False) if args.trace else nullcontext()

    with trace_manager:
        for name, sim_path in benchmarks:
            logger.info(f"  {name}:")
            sys.stdout.flush()
            sys.stderr.flush()

            # Determine which solvers to run
            # c6288 has ~86k total nodes (5k external + 81k internal from PSP103)
            # Dense matrix would be 86k x 86k x 8 bytes = ~56GB - skip dense
            run_dense = not args.sparse_only and name != 'c6288'
            run_sparse = not args.dense_only

            if name == 'c6288':
                logger.info(f"    Note: ~86k total nodes (5k external + 81k internal)")
                if not args.sparse_only:
                    logger.info(f"    Skipping dense (would need ~56GB memory)")

            # Run dense (in subprocess)
            if run_dense:
                logger.info(f"    dense:  running in subprocess...")
                sys.stdout.flush()
                result_dense = run_benchmark_subprocess(name, 'dense', args.timesteps, args.warmup_steps, args.full, args.cpu)
                if result_dense:
                    profiler.results.append(result_dense)
                    if result_dense.error:
                        logger.info(f"    dense:  ERROR - {result_dense.error}")
                    else:
                        logger.info(f"    dense:  {result_dense.total_time_s:.3f}s total, {result_dense.timesteps} steps")
                else:
                    logger.info(f"    dense:  FAILED (subprocess error)")

            # Run sparse (in subprocess)
            if run_sparse:
                logger.info(f"    sparse: running in subprocess...")
                sys.stdout.flush()
                result_sparse = run_benchmark_subprocess(name, 'sparse', args.timesteps, args.warmup_steps, args.full, args.cpu)
                if result_sparse:
                    profiler.results.append(result_sparse)
                    if result_sparse.error:
                        logger.info(f"    sparse: {result_sparse.error}")
                    else:
                        logger.info(f"    sparse: {result_sparse.total_time_s:.3f}s total, {result_sparse.timesteps} steps")
                else:
                    logger.info(f"    sparse: FAILED (subprocess error)")

            logger.info("")

    profiler.stop()

    logger.info("[Stage 4/4] Generating report...")
    report = profiler.generate_report()

    logger.info("")
    # Print report directly (no memory prefix for multi-line report)
    print(report)

    # Write to file
    report_path = Path(__file__).parent.parent / "profile_report.md"
    report_path.write_text(report)
    logger.info(f"Report written to: {report_path}")

    # Write to GitHub step summary if available
    if 'GITHUB_STEP_SUMMARY' in os.environ:
        with open(os.environ['GITHUB_STEP_SUMMARY'], 'a') as f:
            f.write(report)
        logger.info("Report appended to GitHub step summary")

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
    logger.info(f"JSON report written to: {json_path}")

    if args.trace:
        logger.info("")
        logger.info(f"Perfetto traces saved to: {args.trace_dir}")
        logger.info("To view traces:")
        logger.info("  1. Open https://ui.perfetto.dev/")
        logger.info(f"  2. Load trace files from: {args.trace_dir}")

    logger.info("")
    logger.info("=" * 70)
    logger.info("Profiling complete!")
    logger.info("=" * 70)


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
