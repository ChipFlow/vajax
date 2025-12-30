#!/usr/bin/env python3
"""CPU Profiling Script for JAX-SPICE

Profiles CPU performance of VACASK benchmark circuits using CircuitEngine.
Compares dense vs sparse solvers across different circuit sizes.

Usage:
    # Run all benchmarks
    uv run scripts/profile_cpu.py

    # Run specific benchmarks
    uv run scripts/profile_cpu.py --benchmark ring,rc

    # Sparse only (for large circuits)
    uv run scripts/profile_cpu.py --benchmark c6288 --sparse-only
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List

# Ensure jax-spice is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

# Force CPU backend
os.environ['JAX_PLATFORMS'] = 'cpu'

# Import jax_spice first to auto-configure precision based on backend
# (Metal/TPU use f32, CPU/CUDA use f64)
from jax_spice.analysis import CircuitEngine
from scripts.benchmark_utils import BenchmarkResult, get_vacask_benchmarks, log


def run_benchmark(sim_path: Path, name: str, use_sparse: bool,
                  num_steps: int = 20, use_scan: bool = False) -> BenchmarkResult:
    """Run a single benchmark configuration."""
    try:
        # Parse circuit
        log(f"      parsing...")
        engine = CircuitEngine(sim_path)
        engine.parse()
        log("      parsing done")

        nodes = engine.num_nodes
        devices = len(engine.devices)
        openvaf_devices = sum(1 for d in engine.devices if d.get('is_openvaf'))

        # Skip sparse for non-OpenVAF circuits
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
                converged=True,
                error="Sparse not applicable (no OpenVAF devices)"
            )

        # Get analysis params
        dt = engine.analysis_params.get('step', 1e-12)

        # Warmup run (includes JIT compilation)
        # IMPORTANT: Use same num_steps as timed run to avoid JAX re-tracing
        # JAX traces based on array shapes, so different timestep counts cause recompilation
        mode_str = "lax.scan" if use_scan else "Python loop"
        log(f"      warmup ({num_steps} steps, {mode_str}, includes JIT)...")
        warmup_start = time.perf_counter()
        engine.run_transient(t_stop=dt * num_steps, dt=dt,
                            max_steps=num_steps, use_sparse=use_sparse,
                            use_while_loop=use_scan)
        warmup_time = time.perf_counter() - warmup_start
        log(f"      warmup done ({warmup_time:.1f}s)")

        # Timed run - use same engine with cached JIT functions
        start = time.perf_counter()
        result = engine.run_transient(
            t_stop=dt * num_steps, dt=dt,
            max_steps=num_steps, use_sparse=use_sparse,
            use_while_loop=use_scan
        )
        elapsed = time.perf_counter() - start

        actual_steps = result.num_steps
        time_per_step = (elapsed / actual_steps * 1000) if actual_steps > 0 else 0

        return BenchmarkResult(
            name=name,
            nodes=nodes,
            devices=devices,
            openvaf_devices=openvaf_devices,
            timesteps=actual_steps,
            total_time_s=elapsed,
            time_per_step_ms=time_per_step,
            solver='sparse' if use_sparse else 'dense',
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
            converged=False,
            error=str(e)
        )


def main():
    parser = argparse.ArgumentParser(
        description="Profile VACASK benchmarks on CPU"
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
        "--use-scan",
        action="store_true",
        help="Use lax.scan for fully JIT-compiled simulation (faster but less debugging info)",
    )
    args = parser.parse_args()

    log("=" * 70)
    log("JAX-SPICE CPU Profiling")
    log("=" * 70)
    log()

    log("[Stage 1/3] Checking JAX configuration...")
    log(f"  JAX backend: {jax.default_backend()}")
    log(f"  JAX devices: {jax.devices()}")
    log(f"  Float64 enabled: {jax.config.jax_enable_x64}")
    log()

    # Parse benchmark names
    benchmark_names = None
    if args.benchmark:
        benchmark_names = [b.strip() for b in args.benchmark.split(',')]

    log("[Stage 2/3] Discovering benchmarks...")
    benchmarks = get_vacask_benchmarks(benchmark_names)
    log(f"  Found {len(benchmarks)} benchmarks: {[b[0] for b in benchmarks]}")
    log()

    results: List[BenchmarkResult] = []
    start_time = time.perf_counter()

    log("[Stage 3/3] Running benchmarks...")
    log()

    for name, sim_path in benchmarks:
        log(f"  {name}:")

        # Determine which solvers to run
        run_dense = not args.sparse_only and name != 'c6288'  # c6288 too large for dense
        run_sparse = not args.dense_only

        if name == 'c6288' and not args.sparse_only:
            log(f"    Skipping dense (86k nodes would need ~56GB)")

        # Run dense
        if run_dense:
            result_dense = run_benchmark(
                sim_path, name, use_sparse=False,
                num_steps=args.timesteps, use_scan=args.use_scan
            )
            results.append(result_dense)
            if result_dense.error:
                log(f"    dense:  ERROR - {result_dense.error}")
            else:
                log(f"    dense:  {result_dense.time_per_step_ms:.1f}ms/step ({result_dense.timesteps} steps)")

        # Run sparse
        if run_sparse:
            result_sparse = run_benchmark(
                sim_path, name, use_sparse=True,
                num_steps=args.timesteps, use_scan=args.use_scan
            )
            results.append(result_sparse)
            if result_sparse.error:
                log(f"    sparse: {result_sparse.error}")
            else:
                log(f"    sparse: {result_sparse.time_per_step_ms:.1f}ms/step ({result_sparse.timesteps} steps)")

        log()

    total_time = time.perf_counter() - start_time

    # Print summary
    log("=" * 70)
    log("Summary")
    log("=" * 70)
    log()
    log(f"Total time: {total_time:.1f}s")
    log()

    # Results table
    log("| Benchmark | Nodes | Solver | Steps | Total (s) | Per Step (ms) | Status |")
    log("|-----------|-------|--------|-------|-----------|---------------|--------|")
    for r in results:
        status = "OK" if r.converged and not r.error else (r.error or "Failed")[:15]
        log(f"| {r.name:9} | {r.nodes:5} | {r.solver:6} | {r.timesteps:5} | "
            f"{r.total_time_s:9.3f} | {r.time_per_step_ms:13.1f} | {status:6} |")

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
