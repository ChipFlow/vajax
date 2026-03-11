#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["jax", "jaxlib"]
# ///
"""Analyze JAX IR for dense benchmark circuits.

Dumps jaxpr, HLO op counts, and cost analysis for the key hot paths:
1. build_system (Jacobian + residual assembly)
2. nr_solve (Newton-Raphson with while_loop)
3. run_while (full transient step with adaptive timestep)
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import jax
import jax.numpy as jnp

from vajax.analysis.engine import CircuitEngine
from vajax.benchmarks.registry import get_benchmark


def count_hlo_ops(hlo_text: str) -> dict[str, int]:
    """Count operation types in HLO text."""
    op_counts: dict[str, int] = {}
    for line in hlo_text.split("\n"):
        if "=" in line and "." in line:
            parts = line.split("=")
            if len(parts) >= 2:
                op_part = parts[1].strip().split()[0] if parts[1].strip() else ""
                if "." in op_part:
                    op_name = op_part.split("(")[0]
                    op_counts[op_name] = op_counts.get(op_name, 0) + 1
    return dict(sorted(op_counts.items(), key=lambda x: -x[1]))


def analyze_function(name: str, fn, args, output_dir: Path):
    """Analyze a single function: jaxpr, HLO, cost."""
    print(f"\n{'=' * 70}")
    print(f"  {name}")
    print(f"{'=' * 70}")

    try:
        if hasattr(fn, "lower"):
            lowered = fn.lower(*args)
        else:
            lowered = jax.jit(fn).lower(*args)

        hlo_text = lowered.as_text()
        hlo_lines = hlo_text.split("\n")
        print(f"  HLO: {len(hlo_lines)} lines")

        ops = count_hlo_ops(hlo_text)
        if ops:
            top = list(ops.items())[:20]
            print("  Top HLO ops:")
            for op, count in top:
                print(f"    {op:40s} {count:6d}")

        compiled = lowered.compile()
        cost = compiled.cost_analysis()
        if cost:
            for i, device_cost in enumerate(cost):
                if device_cost and isinstance(device_cost, dict):
                    print(f"  Cost (device {i}):")
                    for key, val in sorted(device_cost.items()):
                        if isinstance(val, (int, float)):
                            if val > 1e9:
                                print(f"    {key}: {val / 1e9:.2f}G")
                            elif val > 1e6:
                                print(f"    {key}: {val / 1e6:.2f}M")
                            elif val > 1e3:
                                print(f"    {key}: {val / 1e3:.2f}K")
                            else:
                                print(f"    {key}: {val:.2f}")

        # Save HLO
        output_dir.mkdir(parents=True, exist_ok=True)
        hlo_file = output_dir / f"{name}.hlo.txt"
        with open(hlo_file, "w") as f:
            f.write(hlo_text)
        print(f"  Saved: {hlo_file}")

    except Exception as e:
        import traceback

        print(f"  Failed: {e}")
        traceback.print_exc()


def analyze_benchmark(benchmark_name: str, output_dir: Path):
    """Analyze all hot paths for a single benchmark."""
    print(f"\n{'#' * 70}")
    print(f"  Benchmark: {benchmark_name}")
    print(f"{'#' * 70}")

    config = get_benchmark(benchmark_name)
    engine = CircuitEngine(config.sim_path)
    engine.parse()

    # Use short simulation for analysis (just need compilation, not full run)
    num_steps = 100
    engine.prepare(
        t_stop=config.dt * num_steps,
        dt=config.dt,
        use_sparse=False,
    )

    # Get strategy internals
    strategy = engine._strategy
    setup_cache = engine._transient_setup_cache

    n_total = setup_cache["n_total"]
    n_unknowns = setup_cache["n_unknowns"]
    n_vsources = len([d for d in engine.devices if d["model"] == "vsource"])
    n_isources = len([d for d in engine.devices if d["model"] == "isource"])
    n_augmented = n_unknowns + n_vsources

    print(f"  Nodes: {n_total}, Unknowns: {n_unknowns}, Vsources: {n_vsources}")
    print(f"  Augmented system: {n_augmented}x{n_augmented}")

    bench_dir = output_dir / benchmark_name

    # 1. Analyze build_system (Jacobian + residual assembly)
    build_fn = setup_cache.get("build_system_fn")
    device_arrays = engine._device_arrays
    if build_fn is not None and device_arrays is not None:
        X = jnp.zeros(n_total + n_vsources, dtype=jnp.float64)
        vsource_vals = jnp.zeros(n_vsources, dtype=jnp.float64)
        isource_vals = jnp.zeros(max(n_isources, 0), dtype=jnp.float64)
        Q_prev = jnp.zeros(n_unknowns, dtype=jnp.float64)
        integ_c0 = jnp.asarray(1e9, dtype=jnp.float64)  # typical 1/dt
        gmin = jnp.asarray(1e-12, dtype=jnp.float64)
        gshunt = jnp.asarray(0.0, dtype=jnp.float64)
        integ_c1 = jnp.asarray(0.0, dtype=jnp.float64)
        integ_d1 = jnp.asarray(0.0, dtype=jnp.float64)
        dQdt_prev = jnp.zeros(n_unknowns, dtype=jnp.float64)
        integ_c2 = jnp.asarray(0.0, dtype=jnp.float64)
        Q_prev2 = jnp.zeros(n_unknowns, dtype=jnp.float64)
        total_limit_states = setup_cache.get("total_limit_states", 0)
        limit_state = jnp.zeros(total_limit_states, dtype=jnp.float64)
        nr_iter = jnp.asarray(1, dtype=jnp.int32)

        build_args = (
            X,
            vsource_vals,
            isource_vals,
            Q_prev,
            integ_c0,
            device_arrays,
            gmin,
            gshunt,
            integ_c1,
            integ_d1,
            dQdt_prev,
            integ_c2,
            Q_prev2,
            limit_state,
            nr_iter,
        )
        analyze_function("build_system", build_fn, build_args, bench_dir)

    # 2. Analyze nr_solve
    nr_solve = setup_cache.get("nr_solve_fn")
    if nr_solve is not None and device_arrays is not None:
        X_init = jnp.zeros(n_total + n_vsources, dtype=jnp.float64)
        vsource_vals = jnp.zeros(n_vsources, dtype=jnp.float64)
        isource_vals = jnp.zeros(max(n_isources, 0), dtype=jnp.float64)
        Q_prev = jnp.zeros(n_unknowns, dtype=jnp.float64)
        integ_c0 = jnp.asarray(1e9, dtype=jnp.float64)

        nr_args = (X_init, vsource_vals, isource_vals, Q_prev, integ_c0, device_arrays)
        analyze_function("nr_solve", nr_solve, nr_args, bench_dir)

    # 3. Analyze full transient step (run_while)
    run_while = getattr(strategy, "_jit_run_while", None)
    if run_while is None:
        # Try to find it in the cache
        run_while = strategy._jit_run_while_cache.get(
            strategy._get_cache_key() if hasattr(strategy, "_get_cache_key") else None
        )

    if run_while is not None:
        print("\n  Found run_while - analyzing full transient loop")
        # This one is harder to trace without the actual state
        # We'd need to construct a FullMNAState - skip for now
        print("  (skipping run_while - complex state tuple)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze dense benchmark JAX IR")
    parser.add_argument(
        "--benchmark",
        default="rc,graetz,mul,ring",
        help="Comma-separated benchmarks",
    )
    parser.add_argument(
        "--output-dir",
        default="/tmp/claude/jaxpr-analysis",
        help="Output directory for HLO files",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    benchmarks = [b.strip() for b in args.benchmark.split(",")]

    for bench in benchmarks:
        analyze_benchmark(bench, output_dir)
