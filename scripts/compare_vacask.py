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

    # Enable JAX profiling (GPU only)
    uv run scripts/compare_vacask.py --profile --profile-dir /tmp/traces
"""

import argparse
import os
import subprocess
import sys
import time
import re
from contextlib import nullcontext
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

# Enable float64
jax.config.update('jax_enable_x64', True)

from jax_spice.benchmarks.runner import VACASKBenchmarkRunner
from jax_spice.profiling import enable_profiling, profile_section, ProfileConfig


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


def find_vacask_binary() -> Optional[Path]:
    """Find the VACASK binary (returns absolute path)."""
    # Check environment variable
    if 'VACASK_BIN' in os.environ:
        path = Path(os.environ['VACASK_BIN']).resolve()
        if path.exists():
            return path

    # Check common build locations (relative to project root)
    project_root = Path(__file__).parent.parent
    candidates = [
        project_root / 'vendor/VACASK/build/simulator/vacask',
        project_root / 'vendor/VACASK/build/Release/simulator/vacask',
        project_root / 'vendor/VACASK/build/Debug/simulator/vacask',
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    return None


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
                  use_sparse: bool = False, profile_config: Optional[ProfileConfig] = None
                  ) -> Tuple[float, float, Dict]:
    """Run JAX-SPICE and return (time_per_step_ms, wall_time_s, stats).

    Args:
        config: Benchmark configuration
        num_steps: Number of timesteps
        use_scan: Use lax.scan for faster execution
        use_sparse: Use sparse solver
        profile_config: If provided, enable JAX/CUDA profiling for this run
    """
    runner = VACASKBenchmarkRunner(config.sim_path)
    runner.parse()

    t_stop = config.dt * num_steps

    # Warmup (use same num_steps to avoid re-tracing)
    _ = runner.run_transient(
        t_stop=t_stop, dt=config.dt,
        max_steps=num_steps, use_sparse=use_sparse,
        use_while_loop=use_scan
    )

    # Timed run - measure full transient analysis including source pre-computation
    # This is a fair comparison with VACASK which also evaluates sources during simulation
    # Optionally wrap in profiling context
    section_name = f"benchmark_{config.name}"

    start = time.perf_counter()
    with profile_section(section_name, profile_config) if profile_config else nullcontext():
        times, voltages, stats = runner.run_transient(
            t_stop=t_stop, dt=config.dt,
            max_steps=num_steps, use_sparse=use_sparse,
            use_while_loop=use_scan
        )
        # Force completion of async JAX operations
        _ = float(voltages[0][0])
    elapsed = time.perf_counter() - start

    actual_steps = len(times) - 1  # Exclude t=0 initial condition
    time_per_step = elapsed / actual_steps * 1000

    return time_per_step, elapsed, stats


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
        help="Use sparse solver (required for large circuits like c6288)",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable JAX/CUDA profiling (GPU only, creates Perfetto traces)",
    )
    parser.add_argument(
        "--profile-dir",
        type=str,
        default="/tmp/jax-spice-traces",
        help="Directory for profiling traces (default: /tmp/jax-spice-traces)",
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

    # Setup profiling if requested
    profile_config = None
    if args.profile:
        has_gpu = any(d.platform != 'cpu' for d in jax.devices())
        if has_gpu:
            profile_config = ProfileConfig(jax=True, cuda=True, trace_dir=args.profile_dir)
            print(f"Profiling: ENABLED (traces -> {args.profile_dir})")
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
        # Auto-enable sparse for c6288 (too large for dense)
        use_sparse = args.use_sparse or name == 'c6288'
        print("  JAX-SPICE warmup...", end=" ", flush=True)
        jax_ms, jax_wall, stats = run_jax_spice(
            config, num_steps, args.use_scan, use_sparse, profile_config
        )
        print(f"done")
        print(f"  JAX-SPICE: {jax_ms:.3f} ms/step ({jax_wall:.3f}s total)")

        # Run VACASK
        vacask_ms = None
        vacask_wall = None
        if vacask_bin:
            print("  VACASK...", end=" ", flush=True)
            vacask_result = run_vacask(config, num_steps)
            if vacask_result is not None:
                vacask_ms, vacask_wall = vacask_result
                print(f"done")
                print(f"  VACASK: {vacask_ms:.3f} ms/step ({vacask_wall:.3f}s total)")
                ratio = jax_ms / vacask_ms
                print(f"  Ratio: {ratio:.2f}x {'slower' if ratio > 1 else 'faster'}")
            else:
                print("failed")

        results.append({
            'name': name,
            'steps': num_steps,
            'jax_ms': jax_ms,
            'jax_wall': jax_wall,
            'vacask_ms': vacask_ms,
            'vacask_wall': vacask_wall,
        })
        print()

    # Summary table
    print("=" * 70)
    print("Summary (per-step timing)")
    print("=" * 70)
    print()
    print("| Benchmark | Steps | JAX-SPICE (ms) | VACASK (ms) | Ratio |")
    print("|-----------|-------|----------------|-------------|-------|")
    for r in results:
        vacask_str = f"{r['vacask_ms']:.3f}" if r['vacask_ms'] else "N/A"
        if r['vacask_ms']:
            ratio = r['jax_ms'] / r['vacask_ms']
            ratio_str = f"{ratio:.2f}x"
        else:
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
        vacask_wall_ms = r['vacask_wall'] * 1000 if r['vacask_wall'] else None
        vacask_str = f"{vacask_wall_ms:.1f}" if vacask_wall_ms else "N/A"
        if r['vacask_wall']:
            ratio = r['jax_wall'] / r['vacask_wall']
            ratio_str = f"{ratio:.2f}x"
        else:
            ratio_str = "N/A"
        print(f"| {r['name']:9} | {r['steps']:5} | {jax_wall_ms:14.1f} | {vacask_str:11} | {ratio_str:5} |")

    print()

    # Report profiling traces location
    if profile_config:
        trace_dir = Path(args.profile_dir)
        if trace_dir.exists():
            traces = list(trace_dir.glob("*"))
            if traces:
                print(f"Profiling traces saved to: {trace_dir}")
                print(f"  To view in Perfetto: https://ui.perfetto.dev/")
                for t in traces[:5]:  # Show first 5
                    print(f"    - {t.name}")
                if len(traces) > 5:
                    print(f"    ... and {len(traces) - 5} more")
    print("=" * 70)


if __name__ == '__main__':
    main()
