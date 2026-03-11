#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["jax"]
# ///
"""Sweep XLA flag combinations to find optimal CUDA performance.

Runs a benchmark circuit with different XLA flag configurations and
reports timing for each. Each configuration runs in a separate subprocess
to ensure clean XLA state.

Usage:
    # Run on GPU (auto-detects CUDA)
    uv run scripts/sweep_xla_flags.py

    # Specific benchmark
    uv run scripts/sweep_xla_flags.py --benchmark ring

    # Specific configurations only
    uv run scripts/sweep_xla_flags.py --configs baseline,autotune2,command_buffer

    # Also include large circuit (needs sparse solver)
    uv run scripts/sweep_xla_flags.py --benchmark ring,c6288 --include-sparse
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# XLA flag configurations to test
FLAG_CONFIGS = {
    "baseline": {
        "description": "Current CI config (autotune=0)",
        "xla_flags": "--xla_gpu_autotune_level=0",
        "env": {},
    },
    "autotune2": {
        "description": "Autotune level 2 (enables cuBLAS algorithm selection)",
        "xla_flags": "--xla_gpu_autotune_level=2",
        "env": {},
    },
    "autotune4": {
        "description": "Autotune level 4 (full autotuning)",
        "xla_flags": "--xla_gpu_autotune_level=4",
        "env": {},
    },
    "command_buffer": {
        "description": "Command buffers enabled (batch kernel launches)",
        "xla_flags": "--xla_gpu_autotune_level=0",
        "env": {},
    },
    "double_buffer": {
        "description": "While-loop double buffering",
        "xla_flags": (
            "--xla_gpu_autotune_level=0 --xla_gpu_enable_while_loop_double_buffering=true"
        ),
        "env": {},
    },
    "pgle": {
        "description": "Profile-guided latency estimation (3 profiling runs)",
        "xla_flags": "--xla_gpu_autotune_level=0",
        "env": {
            "JAX_ENABLE_PGLE": "true",
            "JAX_PGLE_PROFILING_RUNS": "3",
        },
    },
    "combined_safe": {
        "description": "Autotune 2 + double buffering",
        "xla_flags": (
            "--xla_gpu_autotune_level=2 --xla_gpu_enable_while_loop_double_buffering=true"
        ),
        "env": {},
    },
    "combined_aggressive": {
        "description": "Autotune 4 + double buffering + PGLE",
        "xla_flags": (
            "--xla_gpu_autotune_level=4 --xla_gpu_enable_while_loop_double_buffering=true"
        ),
        "env": {
            "JAX_ENABLE_PGLE": "true",
            "JAX_PGLE_PROFILING_RUNS": "3",
        },
    },
}

# The subprocess script that runs a single benchmark
BENCHMARK_RUNNER = """
import os
import sys
import time
import json

sys.path.insert(0, os.environ["PROJECT_ROOT"])

# Memory config
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import jax
import jax.numpy as jnp

from vajax.analysis import CircuitEngine

benchmark_name = os.environ["BENCHMARK_NAME"]
use_sparse = os.environ.get("USE_SPARSE", "0") == "1"
use_scan = True
force_gpu = os.environ.get("FORCE_GPU", "0") == "1"
n_warmup = int(os.environ.get("N_WARMUP", "1"))
n_runs = int(os.environ.get("N_RUNS", "3"))

from scripts.benchmark_utils import get_vacask_benchmarks

benchmarks = get_vacask_benchmarks([benchmark_name])
if not benchmarks:
    print(json.dumps({"error": f"Benchmark {benchmark_name} not found"}))
    sys.exit(1)

name, sim_path = benchmarks[0]

# Report JAX config
devices = jax.devices()
backend = devices[0].platform if devices else "unknown"
print(f"JAX backend: {backend}, devices: {[d.platform for d in devices]}", file=sys.stderr)
print(f"XLA_FLAGS: {os.environ.get('XLA_FLAGS', '(not set)')}", file=sys.stderr)

engine = CircuitEngine.from_sim_file(str(sim_path))
engine.prepare(use_sparse=use_sparse, force_gpu=force_gpu, use_scan=use_scan)

# Get step count from sim parameters
dt = engine.sim_params.get("dt", 1e-6)
t_stop = engine.sim_params.get("tstop", engine.sim_params.get("t_stop", 1e-3))
n_steps = int(t_stop / dt) if dt > 0 else 100

timings = []

for run_idx in range(n_warmup + n_runs):
    # Re-prepare to reset state
    if run_idx > 0:
        engine.prepare(use_sparse=use_sparse, force_gpu=force_gpu, use_scan=use_scan)

    start = time.perf_counter()
    result = engine.run_transient()
    # Block until computation complete
    if hasattr(result, 'voltages') and result.voltages is not None:
        jax.block_until_ready(result.voltages)
    elapsed = time.perf_counter() - start

    actual_steps = result.n_steps if hasattr(result, 'n_steps') else n_steps
    ms_per_step = (elapsed * 1000.0) / max(actual_steps, 1)

    label = "warmup" if run_idx < n_warmup else f"run {run_idx - n_warmup}"
    print(f"  {label}: {elapsed:.3f}s ({actual_steps} steps, {ms_per_step:.3f} ms/step)", file=sys.stderr)

    if run_idx >= n_warmup:
        timings.append({
            "elapsed_s": elapsed,
            "n_steps": actual_steps,
            "ms_per_step": ms_per_step,
        })

# Report median timing
timings.sort(key=lambda t: t["ms_per_step"])
median = timings[len(timings) // 2]

print(json.dumps({
    "benchmark": benchmark_name,
    "backend": backend,
    "n_steps": median["n_steps"],
    "ms_per_step": median["ms_per_step"],
    "elapsed_s": median["elapsed_s"],
    "n_runs": n_runs,
    "all_timings": [t["ms_per_step"] for t in timings],
}))
"""


def run_config(
    config_name: str,
    config: dict,
    benchmark: str,
    project_root: Path,
    use_sparse: bool,
    force_gpu: bool,
    n_warmup: int = 1,
    n_runs: int = 3,
) -> dict:
    """Run a single benchmark with a specific XLA flag configuration."""
    env = os.environ.copy()
    env["PROJECT_ROOT"] = str(project_root)
    env["BENCHMARK_NAME"] = benchmark
    env["USE_SPARSE"] = "1" if use_sparse else "0"
    env["FORCE_GPU"] = "1" if force_gpu else "0"
    env["N_WARMUP"] = str(n_warmup)
    env["N_RUNS"] = str(n_runs)
    env["JAX_PLATFORMS"] = "cuda,cpu" if force_gpu else "cpu"
    env["JAX_ENABLE_X64"] = "1"

    # Set XLA flags
    env["XLA_FLAGS"] = config["xla_flags"]

    # Set additional env vars
    for k, v in config.get("env", {}).items():
        env[k] = v

    # Memory allocation
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    env.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

    print(f"\n{'=' * 60}")
    print(f"Config: {config_name} — {config['description']}")
    print(f"  XLA_FLAGS: {config['xla_flags']}")
    if config.get("env"):
        print(f"  Extra env: {config['env']}")
    print(f"  Benchmark: {benchmark}")
    print(f"{'=' * 60}")

    start = time.perf_counter()
    try:
        result = subprocess.run(
            [sys.executable, "-c", BENCHMARK_RUNNER],
            env=env,
            capture_output=True,
            text=True,
            timeout=600,  # 10 min max per config
            cwd=str(project_root),
        )
    except subprocess.TimeoutExpired:
        return {
            "config": config_name,
            "benchmark": benchmark,
            "error": "timeout (600s)",
            "wall_time_s": time.perf_counter() - start,
        }

    wall_time = time.perf_counter() - start

    # Print stderr (progress messages)
    if result.stderr:
        for line in result.stderr.strip().split("\n"):
            print(f"  {line}")

    # Parse JSON from last line of stdout
    if result.returncode != 0:
        print(f"  ERROR: exit code {result.returncode}")
        if result.stderr:
            print(f"  {result.stderr[-500:]}")
        return {
            "config": config_name,
            "benchmark": benchmark,
            "error": f"exit code {result.returncode}",
            "wall_time_s": wall_time,
        }

    try:
        # Find the JSON line (last non-empty line of stdout)
        lines = [l for l in result.stdout.strip().split("\n") if l.strip()]
        data = json.loads(lines[-1])
        data["config"] = config_name
        data["wall_time_s"] = wall_time
        data["description"] = config["description"]
        print(f"  Result: {data['ms_per_step']:.3f} ms/step ({data['n_steps']} steps)")
        return data
    except (json.JSONDecodeError, IndexError) as e:
        print(f"  ERROR parsing output: {e}")
        print(f"  stdout: {result.stdout[-500:]}")
        return {
            "config": config_name,
            "benchmark": benchmark,
            "error": f"parse error: {e}",
            "wall_time_s": wall_time,
        }


def main():
    parser = argparse.ArgumentParser(description="Sweep XLA flag combinations")
    parser.add_argument(
        "--benchmark",
        default="ring",
        help="Comma-separated benchmark names (default: ring)",
    )
    parser.add_argument(
        "--configs",
        default=None,
        help="Comma-separated config names (default: all)",
    )
    parser.add_argument(
        "--include-sparse",
        action="store_true",
        help="Include sparse solver for large circuits",
    )
    parser.add_argument(
        "--force-gpu",
        action="store_true",
        default=True,
        help="Force GPU backend (default: True)",
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Run on CPU instead of GPU",
    )
    parser.add_argument(
        "--n-warmup",
        type=int,
        default=1,
        help="Number of warmup runs (default: 1)",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=3,
        help="Number of timed runs (default: 3)",
    )
    parser.add_argument(
        "--json-output",
        default=None,
        help="Path to write JSON results",
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    benchmarks = [b.strip() for b in args.benchmark.split(",")]
    force_gpu = not args.cpu_only

    if args.configs:
        config_names = [c.strip() for c in args.configs.split(",")]
        configs = {k: FLAG_CONFIGS[k] for k in config_names if k in FLAG_CONFIGS}
    else:
        configs = FLAG_CONFIGS

    all_results = []

    for benchmark in benchmarks:
        # Determine if sparse needed
        use_sparse = args.include_sparse and benchmark in ("c6288", "mul64")

        for config_name, config in configs.items():
            result = run_config(
                config_name,
                config,
                benchmark,
                project_root,
                use_sparse=use_sparse,
                force_gpu=force_gpu,
                n_warmup=args.n_warmup,
                n_runs=args.n_runs,
            )
            all_results.append(result)

    # Print summary table
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print(
        f"{'Config':<25} {'Benchmark':<10} {'ms/step':>10} {'Steps':>8} {'Wall(s)':>10} {'vs base':>10}"
    )
    print("-" * 80)

    # Group by benchmark for relative comparison
    by_benchmark = {}
    for r in all_results:
        bm = r.get("benchmark", "?")
        by_benchmark.setdefault(bm, []).append(r)

    for bm, results in by_benchmark.items():
        baseline_ms = None
        for r in results:
            if r.get("config") == "baseline" and "ms_per_step" in r:
                baseline_ms = r["ms_per_step"]
                break

        for r in results:
            ms = r.get("ms_per_step", None)
            steps = r.get("n_steps", "?")
            wall = r.get("wall_time_s", 0)
            config = r.get("config", "?")
            err = r.get("error", None)

            if err:
                print(f"{config:<25} {bm:<10} {'ERROR':>10} {'':>8} {wall:>10.1f} {err}")
            elif ms is not None:
                ratio_str = ""
                if baseline_ms and baseline_ms > 0:
                    ratio = ms / baseline_ms
                    ratio_str = f"{ratio:.2f}x"
                print(f"{config:<25} {bm:<10} {ms:>10.3f} {steps:>8} {wall:>10.1f} {ratio_str:>10}")

    # Save JSON
    if args.json_output:
        out_path = Path(args.json_output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
