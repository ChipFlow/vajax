#!/usr/bin/env python3
"""Measure benchmark startup and JIT compilation times.

This script measures:
1. Import time for vajax (includes JAX import)
2. Circuit parse time
3. First transient run (JIT compilation)
4. Second transient run (cached JIT)

Usage:
    JAX_PLATFORMS=cpu uv run python scripts/measure_startup.py
    JAX_PLATFORMS=cpu uv run python scripts/measure_startup.py --benchmarks rc,ring
    JAX_PLATFORMS=cpu uv run python scripts/measure_startup.py --runs 3
"""

import argparse
import json
import subprocess
from pathlib import Path


def measure_single_benchmark(
    benchmark: str, max_sim_steps: int = 10, clear_cache: bool = False
) -> dict:
    """Measure timing for a single benchmark via subprocess.

    Uses subprocess to get clean import timing.
    """
    if clear_cache:
        import shutil

        cache_dir = Path.home() / ".cache" / "vajax"
        if cache_dir.exists():
            shutil.rmtree(cache_dir)

    script = f'''
import time
import os

# Measure import time
t0 = time.perf_counter()
import vajax
from vajax import CircuitEngine
t_import = time.perf_counter() - t0

# Get benchmark info
from vajax.benchmarks.registry import get_benchmark
bench = get_benchmark("{benchmark}")
if bench is None:
    print('{{"error": "unknown benchmark"}}')
    exit(1)

# Measure parse time
t0 = time.perf_counter()
engine = CircuitEngine(str(bench.sim_path))
engine.parse()
t_parse = time.perf_counter() - t0

# Limit t_stop for timing (JIT compile time dominates, not step count)
max_sim_steps = {max_sim_steps}
t_stop = bench.t_stop
dt = bench.dt
n_steps = int(t_stop / dt)
if n_steps > max_sim_steps:
    t_stop = dt * max_sim_steps

# Prepare once — auto-computes from t_stop/dt
engine.prepare(t_stop=t_stop, dt=dt)

# First run (includes JIT compilation)
t0 = time.perf_counter()
result1 = engine.run_transient()
t_first_run = time.perf_counter() - t0

# Second run (cached JIT)
t0 = time.perf_counter()
result2 = engine.run_transient()
t_second_run = time.perf_counter() - t0

# Third run to confirm
t0 = time.perf_counter()
result3 = engine.run_transient()
t_third_run = time.perf_counter() - t0

import json
print(json.dumps({{
    "benchmark": "{benchmark}",
    "import_time": t_import,
    "parse_time": t_parse,
    "first_run": t_first_run,
    "second_run": t_second_run,
    "third_run": t_third_run,
    "jit_overhead": t_first_run - t_second_run,
    "n_steps": len(result1.times),
}}))
'''

    env = {"JAX_PLATFORMS": "cpu", "PATH": subprocess.os.environ.get("PATH", "")}
    result = subprocess.run(
        ["uv", "run", "python", "-c", script],
        capture_output=True,
        text=True,
        env=env,
        cwd=Path(__file__).parent.parent,
    )

    if result.returncode != 0:
        return {
            "benchmark": benchmark,
            "error": result.stderr or "Unknown error",
        }

    try:
        # Find JSON line in output (skip any other output)
        for line in result.stdout.strip().split("\n"):
            if line.startswith("{"):
                return json.loads(line)
        return {"benchmark": benchmark, "error": f"No JSON output: {result.stdout}"}
    except json.JSONDecodeError as e:
        return {"benchmark": benchmark, "error": f"JSON decode error: {e}\nOutput: {result.stdout}"}


def main():
    parser = argparse.ArgumentParser(description="Measure benchmark startup times")
    parser.add_argument(
        "--benchmarks", default="rc,ring,graetz", help="Comma-separated list of benchmarks to test"
    )
    parser.add_argument(
        "--runs", type=int, default=1, help="Number of runs per benchmark for averaging"
    )
    parser.add_argument(
        "--max-sim-steps",
        type=int,
        default=10,
        help="Max simulation steps (limits t_stop to keep timing focused on JIT overhead)",
    )
    parser.add_argument(
        "--json", action="store_true", help="Output raw JSON instead of formatted table"
    )
    parser.add_argument(
        "--clear-cache", action="store_true", help="Clear ~/.cache/vajax before each benchmark"
    )
    args = parser.parse_args()

    benchmarks = [b.strip() for b in args.benchmarks.split(",")]
    all_results = []

    print(f"Measuring startup times for: {', '.join(benchmarks)}")
    print(f"Runs per benchmark: {args.runs}")
    print()

    for benchmark in benchmarks:
        runs = []
        for i in range(args.runs):
            if args.runs > 1:
                print(f"  {benchmark} run {i + 1}/{args.runs}...", end=" ", flush=True)
            else:
                print(f"  {benchmark}...", end=" ", flush=True)

            result = measure_single_benchmark(benchmark, args.max_sim_steps, args.clear_cache)
            runs.append(result)

            if "error" in result:
                print(f"ERROR: {result['error']}")
                break
            else:
                print(f"import={result['import_time']:.2f}s, first_run={result['first_run']:.2f}s")

        if runs and "error" not in runs[0]:
            # Average the results
            avg = {
                "benchmark": benchmark,
                "import_time": sum(r["import_time"] for r in runs) / len(runs),
                "parse_time": sum(r["parse_time"] for r in runs) / len(runs),
                "first_run": sum(r["first_run"] for r in runs) / len(runs),
                "second_run": sum(r["second_run"] for r in runs) / len(runs),
                "third_run": sum(r["third_run"] for r in runs) / len(runs),
                "jit_overhead": sum(r["jit_overhead"] for r in runs) / len(runs),
                "n_steps": runs[0]["n_steps"],
            }
            all_results.append(avg)
        else:
            all_results.append(runs[0] if runs else {"benchmark": benchmark, "error": "No runs"})

    print()

    if args.json:
        print(json.dumps(all_results, indent=2))
    else:
        # Print formatted table
        print("=" * 80)
        print(
            f"{'Benchmark':<12} {'Import':>8} {'Parse':>8} {'1st Run':>10} {'2nd Run':>10} {'JIT OH':>10}"
        )
        print("-" * 80)
        for r in all_results:
            if "error" in r:
                print(f"{r['benchmark']:<12} ERROR: {r['error'][:50]}")
            else:
                print(
                    f"{r['benchmark']:<12} "
                    f"{r['import_time']:>7.2f}s "
                    f"{r['parse_time']:>7.2f}s "
                    f"{r['first_run']:>9.2f}s "
                    f"{r['second_run']:>9.2f}s "
                    f"{r['jit_overhead']:>9.2f}s"
                )
        print("=" * 80)

        # Summary stats
        if all_results and "error" not in all_results[0]:
            avg_import = sum(r["import_time"] for r in all_results if "error" not in r) / len(
                all_results
            )
            avg_jit = sum(r["jit_overhead"] for r in all_results if "error" not in r) / len(
                all_results
            )
            print(f"\nAverage import time: {avg_import:.2f}s")
            print(f"Average JIT overhead: {avg_jit:.2f}s")


if __name__ == "__main__":
    main()
