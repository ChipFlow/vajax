#!/usr/bin/env python3
"""Compare JAX-SPICE vs VACASK Performance

Runs both simulators on the same benchmarks with matching parameters
and compares performance.

Usage:
    # Run all benchmarks
    uv run scripts/compare_vacask.py

    # Run specific benchmark with custom steps
    uv run scripts/compare_vacask.py --benchmark ring --max-steps 1000

    # Use lax.scan (faster)
    uv run scripts/compare_vacask.py --benchmark ring --use-scan

    # Enable JAX profiling (GPU only, creates Perfetto traces)
    uv run scripts/compare_vacask.py --profile-mode jax --profile-dir /tmp/traces

    # Enable nsys-jax profiling (GPU only, creates .nsys-rep + analysis zip)
    # Note: Requires nsys-jax to be installed (from JAX-Toolbox)
    uv run scripts/compare_vacask.py --profile-mode nsys --profile-dir /tmp/traces

    # Enable both profiling modes (run separately to avoid interference)
    uv run scripts/compare_vacask.py --profile-mode both --profile-dir /tmp/traces
"""

import argparse
import os
import subprocess

# Enable jax_spice performance logging with perf_counter timestamps
# This helps correlate log messages with Perfetto trace timestamps
from jax_spice.logging import enable_performance_logging
enable_performance_logging(with_memory=False, with_perf_counter=True)
import sys
import time
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# Ensure jax-spice is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure JAX memory allocation BEFORE importing JAX
# Disable preallocation to avoid grabbing all GPU memory at startup
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
# Use memory growth instead of fixed allocation
os.environ.setdefault('XLA_PYTHON_CLIENT_ALLOCATOR', 'platform')
# Note: Set JAX_PLATFORMS=cpu before running for CPU-only mode

import jax
import jax.numpy as jnp

# Precision is auto-configured by jax_spice import (imported above via logging)
# Metal/TPU use f32, CPU/CUDA use f64

from jax_spice.analysis import CircuitEngine
from jax_spice.profiling import enable_profiling, ProfileConfig


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark."""
    name: str
    sim_path: Path
    dt: float  # timestep
    t_stop: float  # stop time from VACASK
    max_steps: int  # limit for practical testing


# Project root for relative paths
PROJECT_ROOT = Path(__file__).parent.parent

# Benchmark configurations matching VACASK parameters
BENCHMARKS = {
    'rc': BenchmarkConfig(
        name='rc',
        sim_path=PROJECT_ROOT / 'vendor/VACASK/benchmark/rc/vacask/runme.sim',
        dt=1e-6,  # 1µs
        t_stop=1.0,  # 1s (1M steps)
        max_steps=1000,  # limit for testing
    ),
    'graetz': BenchmarkConfig(
        name='graetz',
        sim_path=PROJECT_ROOT / 'vendor/VACASK/benchmark/graetz/vacask/runme.sim',
        dt=1e-6,  # 1µs
        t_stop=1.0,  # 1s (1M steps)
        max_steps=1000,
    ),
    'ring': BenchmarkConfig(
        name='ring',
        sim_path=PROJECT_ROOT / 'vendor/VACASK/benchmark/ring/vacask/runme.sim',
        dt=5e-11,  # 0.05ns
        t_stop=1e-6,  # 1µs (20k steps)
        max_steps=1000,
    ),
    'c6288': BenchmarkConfig(
        name='c6288',
        sim_path=PROJECT_ROOT / 'vendor/VACASK/benchmark/c6288/vacask/runme.sim',
        dt=2e-12,  # 2ps
        t_stop=2e-9,  # 2ns (1k steps)
        max_steps=1000,
    ),
}

# Published VACASK benchmark times (ms/step) from vendor/VACASK/benchmark/README.md
# Run on AMD Threadripper 7970, single-threaded, KLU solver
# These serve as reference when VACASK binary is not available (e.g., GPU-only CI)
VACASK_REFERENCE_TIMES = {
    # rc: 0.94s / 1005006 steps = 0.935µs/step
    'rc': 0.94 / 1005006 * 1000,  # 0.000935 ms/step
    # graetz: 1.89s / 1000003 steps = 1.89µs/step
    'graetz': 1.89 / 1000003 * 1000,  # 0.00189 ms/step
    # ring: 1.18s / 26066 steps = 45.3µs/step
    'ring': 1.18 / 26066 * 1000,  # 0.0453 ms/step
    # c6288: 57.98s / 1021 steps = 56.8ms/step
    'c6288': 57.98 / 1021 * 1000,  # 56.8 ms/step
}


# find_vacask_binary is imported from scripts.benchmark_utils
from scripts.benchmark_utils import find_vacask_binary


def run_vacask(config: BenchmarkConfig, num_steps: int) -> Optional[Tuple[float, float]]:
    """Run VACASK and return (time_per_step_ms, wall_time_s).

    Returns None if VACASK is not available or fails.
    """
    vacask_bin = find_vacask_binary()
    if vacask_bin is None:
        return None

    # Create a temporary sim file with modified stop time
    t_stop = config.dt * num_steps
    sim_dir = config.sim_path.parent

    # Read original sim file
    with open(config.sim_path) as f:
        sim_content = f.read()

    # Modify the analysis line to use our t_stop
    # Pattern: analysis <name> tran step=... stop=... maxstep=...
    modified = re.sub(
        r'(analysis\s+\w+\s+tran\s+.*?stop=)[^\s]+',
        f'\\g<1>{t_stop:.2e}',
        sim_content
    )

    # Write to temp file
    temp_sim = sim_dir / 'compare_temp.sim'
    with open(temp_sim, 'w') as f:
        f.write(modified)

    try:
        # Run VACASK with temp sim file
        start = time.perf_counter()
        result = subprocess.run(
            [str(vacask_bin), 'compare_temp.sim'],  # Use relative path in cwd
            cwd=sim_dir,
            capture_output=True,
            text=True,
            timeout=300
        )
        elapsed = time.perf_counter() - start

        # Parse output - VACASK may output Python errors from post-processing scripts
        # but still succeed. Look for elapsed time in stdout.

        # Check for VACASK elapsed time (most reliable)
        elapsed_match = re.search(r'Elapsed time:\s*([\d.]+)', result.stdout)
        if elapsed_match:
            vacask_elapsed = float(elapsed_match.group(1))
            # Use the number of timesteps we requested for fair comparison
            # NR solver calls != timesteps (multiple NR iters per timestep)
            time_per_step = vacask_elapsed / num_steps * 1000  # ms per timestep
            return time_per_step, vacask_elapsed
        elif result.returncode != 0:
            # Only report failure if no elapsed time found AND returncode is non-zero
            print(f"VACASK failed (rc={result.returncode}): {result.stdout[:200]}")
            return None
        else:
            print(f"VACASK: could not parse output")
            return None

    except subprocess.TimeoutExpired:
        print("VACASK timed out")
        return None
    except Exception as e:
        print(f"VACASK error: {e}")
        return None
    finally:
        # Clean up temp file
        if temp_sim.exists():
            temp_sim.unlink()


def run_jax_spice(config: BenchmarkConfig, num_steps: int, use_scan: bool,
                  use_sparse: bool = False, profile_config: Optional[ProfileConfig] = None,
                  profile_full: bool = False
                  ) -> Tuple[float, float, Dict]:
    """Run JAX-SPICE and return (time_per_step_ms, wall_time_s, stats).

    Args:
        config: Benchmark configuration
        num_steps: Number of timesteps
        use_scan: Use lax.scan for faster execution
        use_sparse: Use sparse solver
        profile_config: If provided, profile the simulation
        profile_full: If True, profile entire run including warmup (helps identify overhead)
    """
    from jax_spice.profiling import profile_section

    # Create benchmark-specific profile config if profiling is enabled
    full_profile_config = None
    scan_profile_config = None
    if profile_config:
        base_dir = Path(profile_config.trace_dir) / f"benchmark_{config.name}"
        if profile_full:
            # Profile entire run - don't also profile scan to avoid nested profiles
            full_profile_config = ProfileConfig(
                jax=profile_config.jax,
                cuda=profile_config.cuda,
                trace_dir=str(base_dir / "full_run"),
                create_perfetto_link=profile_config.create_perfetto_link,
            )
        else:
            # Profile just the scan portion
            scan_profile_config = ProfileConfig(
                jax=profile_config.jax,
                cuda=profile_config.cuda,
                trace_dir=str(base_dir / "lax_scan_simulation"),
                create_perfetto_link=profile_config.create_perfetto_link,
            )

    def do_run():
        # Use CircuitEngine API
        engine = CircuitEngine(config.sim_path)
        engine.parse()

        t_stop = config.dt * num_steps

        # Warmup (includes JIT compilation)
        startup_start = time.perf_counter()
        engine.run_transient(
            t_stop=t_stop, dt=config.dt,
            use_sparse=use_sparse, use_while_loop=use_scan
        )
        startup_time = time.perf_counter() - startup_start

        # Timed run - print perf_counter for correlation with Perfetto traces
        start = time.perf_counter()
        print(f"TIMED_RUN_START: {start:.6f}")
        result = engine.run_transient(
            t_stop=t_stop, dt=config.dt,
            use_sparse=use_sparse, use_while_loop=use_scan,
            profile_config=scan_profile_config,
        )
        after_transient = time.perf_counter()
        print(f"AFTER_RUN_TRANSIENT: {after_transient:.6f} (elapsed: {after_transient - start:.6f}s)")
        # Force completion of async JAX operations
        _ = float(result.voltages[0][0])
        end = time.perf_counter()
        external_elapsed = end - start
        print(f"TIMED_RUN_END: {end:.6f} (elapsed: {external_elapsed:.6f}s, sync took: {end - after_transient:.6f}s)")

        # Use wall_time from stats (excludes trace saving overhead)
        # Fall back to external timing if not available
        stats = result.stats
        elapsed = stats.get('wall_time', external_elapsed)
        actual_steps = result.num_steps - 1  # Exclude t=0 initial condition
        time_per_step = elapsed / actual_steps * 1000

        # Add startup time to stats
        stats['startup_time'] = startup_time

        return time_per_step, elapsed, stats

    if full_profile_config:
        with profile_section("full_benchmark", full_profile_config):
            return do_run()
    else:
        return do_run()


def main():
    parser = argparse.ArgumentParser(description="Compare JAX-SPICE vs VACASK")
    parser.add_argument(
        "--benchmark",
        type=str,
        default=None,
        help="Comma-separated list of benchmarks (default: all except c6288)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Maximum number of timesteps (default: 1000)",
    )
    parser.add_argument(
        "--use-scan",
        action="store_true",
        help="Use lax.scan for JAX-SPICE (faster)",
    )
    parser.add_argument(
        "--use-sparse",
        action="store_true",
        help="Use sparse solver (auto-enabled for c6288 unless --force-dense)",
    )
    parser.add_argument(
        "--force-dense",
        action="store_true",
        help="Force dense solver even for large circuits (for GPU performance testing)",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="[DEPRECATED] Use --profile-mode=jax instead",
    )
    parser.add_argument(
        "--profile-mode",
        type=str,
        choices=["none", "jax", "nsys", "both"],
        default="none",
        help="Profiling mode: none, jax (Perfetto traces), nsys (nsys-jax), or both",
    )
    parser.add_argument(
        "--profile-dir",
        type=str,
        default="/tmp/jax-spice-traces",
        help="Directory for profiling traces (default: /tmp/jax-spice-traces)",
    )
    parser.add_argument(
        "--profile-full",
        action="store_true",
        help="Profile entire run including warmup/setup (helps identify overhead)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("JAX-SPICE vs VACASK Performance Comparison")
    print("=" * 70)
    print()

    # Check for VACASK
    vacask_bin = find_vacask_binary()
    if vacask_bin:
        print(f"VACASK binary: {vacask_bin}")
    else:
        print("VACASK binary: NOT FOUND (set VACASK_BIN or build in vendor/VACASK/build)")
    print(f"JAX backend: {jax.default_backend()}")
    print(f"Mode: {'lax.scan' if args.use_scan else 'Python loop'}")
    print(f"Solver: {'sparse' if args.use_sparse else 'dense'}")
    print(f"Max steps: {args.max_steps}")

    # Handle deprecated --profile flag
    profile_mode = args.profile_mode
    if args.profile and profile_mode == "none":
        print("Warning: --profile is deprecated, use --profile-mode=jax instead")
        profile_mode = "jax"

    # Check if we're running under nsys-jax (it sets this env var)
    running_under_nsys = os.environ.get("NSYS_PROFILING_SESSION_ID") is not None

    # For nsys mode, re-exec under nsys-jax if not already running under it
    if profile_mode in ("nsys", "both") and not running_under_nsys:
        nsys_output = Path(args.profile_dir) / "nsys-jax-output"
        nsys_output.mkdir(parents=True, exist_ok=True)

        # Build nsys-jax command
        nsys_cmd = [
            "nsys-jax",
            "-o", str(nsys_output / "profile"),
            "--capture-range=cudaProfilerApi",
            "--capture-range-end=stop",
            "python", sys.argv[0],
        ]
        # Pass through arguments, but change profile-mode to jax if both
        for i, arg in enumerate(sys.argv[1:]):
            if arg == "--profile-mode":
                continue  # Skip, we'll add our own
            if i > 0 and sys.argv[i] == "--profile-mode":
                continue  # Skip the value too
            nsys_cmd.append(arg)

        # If mode is 'both', run JAX profiling in this process first
        if profile_mode == "both":
            print("Running JAX profiling first, then nsys-jax...")
            nsys_cmd.extend(["--profile-mode", "none"])  # nsys run won't do JAX profiling
        else:
            nsys_cmd.extend(["--profile-mode", "none"])  # Just use CUDA markers

        # Mark that we want CUDA profiler markers
        os.environ["JAX_SPICE_NSYS_MARKERS"] = "1"

        print(f"Re-executing under nsys-jax: {' '.join(nsys_cmd[:5])}...")
        result = subprocess.run(nsys_cmd)
        if result.returncode != 0:
            print(f"nsys-jax failed with code {result.returncode}")
            sys.exit(result.returncode)

        print(f"\nnsys-jax output saved to: {nsys_output}")
        print("To analyze, unzip and open Jupyter notebooks:")
        print(f"  unzip {nsys_output}/profile.zip -d {nsys_output}/")
        print(f"  jupyter notebook {nsys_output}/")

        if profile_mode == "nsys":
            sys.exit(0)  # Done, nsys-jax already ran the benchmarks
        # If 'both', continue to run JAX profiling below

    # Setup JAX profiling if requested
    profile_config = None
    use_cuda_markers = os.environ.get("JAX_SPICE_NSYS_MARKERS") == "1"

    if profile_mode in ("jax", "both") or use_cuda_markers:
        has_gpu = any(d.platform != 'cpu' for d in jax.devices())
        if has_gpu:
            # JAX profiling for jax/both modes, CUDA markers for nsys mode
            enable_jax = profile_mode in ("jax", "both")
            profile_config = ProfileConfig(
                jax=enable_jax,
                cuda=use_cuda_markers or running_under_nsys,
                trace_dir=args.profile_dir
            )
            if enable_jax:
                print(f"JAX Profiling: ENABLED (traces -> {args.profile_dir})")
            if use_cuda_markers or running_under_nsys:
                print("CUDA Profiler Markers: ENABLED (for nsys-jax)")
        else:
            print("Profiling: SKIPPED (no GPU available)")
    print()

    # Select benchmarks
    if args.benchmark:
        benchmark_names = [b.strip() for b in args.benchmark.split(',')]
    else:
        # Default: all benchmarks including c6288 (uses sparse solver automatically)
        benchmark_names = ['rc', 'graetz', 'ring', 'c6288']

    # Run benchmarks
    results = []

    for name in benchmark_names:
        if name not in BENCHMARKS:
            print(f"Unknown benchmark: {name}")
            continue

        config = BENCHMARKS[name]
        num_steps = min(args.max_steps, int(config.t_stop / config.dt))

        print(f"--- {name} ({num_steps} steps, dt={config.dt:.2e}) ---")

        # Run JAX-SPICE
        # Auto-enable sparse for c6288 unless --force-dense is specified
        use_sparse = args.use_sparse or (name == 'c6288' and not args.force_dense)
        print("  JAX-SPICE warmup...", end=" ", flush=True)
        jax_ms, jax_wall, stats = run_jax_spice(
            config, num_steps, args.use_scan, use_sparse, profile_config,
            profile_full=args.profile_full
        )
        startup_time = stats.get('startup_time', 0)
        print(f"done")
        print(f"  JAX-SPICE: {jax_ms:.3f} ms/step ({jax_wall:.3f}s total, startup: {startup_time:.1f}s)")

        # Run VACASK or use reference times
        vacask_ms = None
        vacask_wall = None
        vacask_source = None  # 'run' or 'reference'
        if vacask_bin:
            print("  VACASK...", end=" ", flush=True)
            vacask_result = run_vacask(config, num_steps)
            if vacask_result is not None:
                vacask_ms, vacask_wall = vacask_result
                vacask_source = 'run'
                print(f"done")
                print(f"  VACASK: {vacask_ms:.3f} ms/step ({vacask_wall:.3f}s total)")
                ratio = jax_ms / vacask_ms
                print(f"  Ratio: {ratio:.2f}x {'slower' if ratio > 1 else 'faster'}")
            else:
                print("failed")
        elif name in VACASK_REFERENCE_TIMES:
            # Use published reference times when VACASK binary isn't available
            vacask_ms = VACASK_REFERENCE_TIMES[name]
            vacask_wall = vacask_ms * num_steps / 1000  # Convert to seconds
            vacask_source = 'reference'
            print(f"  VACASK (reference): {vacask_ms:.3f} ms/step (AMD Threadripper 7970)")
            ratio = jax_ms / vacask_ms
            print(f"  Ratio: {ratio:.2f}x {'slower' if ratio > 1 else 'faster'}")

        results.append({
            'name': name,
            'steps': num_steps,
            'jax_ms': jax_ms,
            'jax_wall': jax_wall,
            'startup_time': startup_time,
            'vacask_ms': vacask_ms,
            'vacask_wall': vacask_wall,
            'vacask_source': vacask_source,
        })
        print()

    # Summary table
    has_reference = any(r.get('vacask_source') == 'reference' for r in results)
    print("=" * 70)
    print("Summary (per-step timing)")
    print("=" * 70)
    print()
    print("| Benchmark | Steps | JAX-SPICE (ms) | VACASK (ms) | Ratio |")
    print("|-----------|-------|----------------|-------------|-------|")
    for r in results:
        if r['vacask_ms']:
            # Add asterisk for reference times
            ref_marker = "*" if r.get('vacask_source') == 'reference' else ""
            vacask_str = f"{r['vacask_ms']:.3f}{ref_marker}"
            ratio = r['jax_ms'] / r['vacask_ms']
            ratio_str = f"{ratio:.2f}x"
        else:
            vacask_str = "N/A"
            ratio_str = "N/A"
        print(f"| {r['name']:9} | {r['steps']:5} | {r['jax_ms']:14.3f} | {vacask_str:11} | {ratio_str:5} |")

    print()
    print("=" * 70)
    print("Summary (total wall time)")
    print("=" * 70)
    print()
    print("| Benchmark | Steps | JAX-SPICE (ms) | VACASK (ms) | Ratio |")
    print("|-----------|-------|----------------|-------------|-------|")
    for r in results:
        # Convert wall time from seconds to milliseconds for better precision
        jax_wall_ms = r['jax_wall'] * 1000
        if r['vacask_wall']:
            vacask_wall_ms = r['vacask_wall'] * 1000
            ref_marker = "*" if r.get('vacask_source') == 'reference' else ""
            vacask_str = f"{vacask_wall_ms:.1f}{ref_marker}"
            ratio = r['jax_wall'] / r['vacask_wall']
            ratio_str = f"{ratio:.2f}x"
        else:
            vacask_str = "N/A"
            ratio_str = "N/A"
        print(f"| {r['name']:9} | {r['steps']:5} | {jax_wall_ms:14.1f} | {vacask_str:11} | {ratio_str:5} |")

    if has_reference:
        print()
        print("* VACASK reference times from AMD Threadripper 7970 (single-threaded)")

    # Startup time summary (shows JIT compilation overhead and breakeven analysis)
    print()
    print("=" * 70)
    print("Startup overhead (JIT compilation)")
    print("=" * 70)
    print()
    print("| Benchmark | Startup (s) | Per-step speedup | Breakeven steps |")
    print("|-----------|-------------|------------------|-----------------|")
    for r in results:
        startup_s = r.get('startup_time', 0)
        if r['vacask_ms'] and r['jax_ms'] < r['vacask_ms']:
            # JAX is faster per-step, calculate breakeven
            # breakeven = startup / (vacask_time - jax_time) per step
            speedup_per_step_ms = r['vacask_ms'] - r['jax_ms']
            breakeven = int(startup_s * 1000 / speedup_per_step_ms) if speedup_per_step_ms > 0 else float('inf')
            speedup_str = f"{r['vacask_ms'] / r['jax_ms']:.1f}x faster"
            breakeven_str = f"{breakeven:,}"
        elif r['vacask_ms']:
            # JAX is slower per-step
            speedup_str = f"{r['jax_ms'] / r['vacask_ms']:.1f}x slower"
            breakeven_str = "N/A (slower)"
        else:
            speedup_str = "N/A"
            breakeven_str = "N/A"
        print(f"| {r['name']:9} | {startup_s:11.1f} | {speedup_str:16} | {breakeven_str:15} |")

    print()

    # Report profiling traces location
    if profile_config and profile_config.jax:
        trace_dir = Path(args.profile_dir)
        if trace_dir.exists():
            # Find JAX traces (directories with .xplane.pb files)
            jax_traces = list(trace_dir.glob("benchmark_*"))
            if jax_traces:
                print(f"JAX profiling traces saved to: {trace_dir}")
                print(f"  To view in Perfetto: https://ui.perfetto.dev/")
                for t in jax_traces[:5]:  # Show first 5
                    print(f"    - {t.name}/")
                if len(jax_traces) > 5:
                    print(f"    ... and {len(jax_traces) - 5} more")

            # Check for nsys-jax output
            nsys_output = trace_dir / "nsys-jax-output"
            if nsys_output.exists():
                nsys_zips = list(nsys_output.glob("*.zip"))
                if nsys_zips:
                    print(f"\nnsys-jax traces saved to: {nsys_output}")
                    print("  To analyze:")
                    print(f"    unzip {nsys_zips[0]} -d {nsys_output}/analysis")
                    print(f"    jupyter notebook {nsys_output}/analysis/")
    print("=" * 70)


if __name__ == '__main__':
    main()
