# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Profile NR iteration phase breakdown using JAX profiling tools.

Instruments the NR solver body with jax.named_scope annotations and
captures a Perfetto trace showing the time split between:
  - build_system: device evaluation + Jacobian/residual assembly
  - linear_solve: sparse or dense linear solve (J*delta = -f)
  - convergence: residual/delta checks, step limiting, solution update

Also uses jax.debug.callback timestamps for a quick text summary
(accurate on CPU since execution is synchronous).

Usage:
    JAX_PLATFORMS=cpu uv run python scripts/profile_nr_phases.py ring
    JAX_PLATFORMS=cpu uv run python scripts/profile_nr_phases.py c6288
    JAX_PLATFORMS=cpu uv run python scripts/profile_nr_phases.py c6288 --trace-dir /tmp/jax_trace
"""

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

# ---------------------------------------------------------------------------
# Phase timing via jax.debug.callback (CPU-accurate)
# ---------------------------------------------------------------------------

phase_timings: list[dict] = []
_phase_clock: dict[str, float] = {}


def _start_phase(phase_name_bytes):
    """Record start time for a phase."""
    phase_name = (
        phase_name_bytes.tobytes().decode()
        if hasattr(phase_name_bytes, "tobytes")
        else str(phase_name_bytes)
    )
    _phase_clock[phase_name] = time.perf_counter_ns()


def _end_phase(phase_name_bytes, iteration):
    """Record end time for a phase."""
    phase_name = (
        phase_name_bytes.tobytes().decode()
        if hasattr(phase_name_bytes, "tobytes")
        else str(phase_name_bytes)
    )
    start = _phase_clock.get(phase_name, 0)
    elapsed_ns = time.perf_counter_ns() - start
    phase_timings.append(
        {
            "phase": phase_name,
            "iteration": int(iteration),
            "elapsed_us": elapsed_ns / 1000,
        }
    )


# ---------------------------------------------------------------------------
# Monkey-patch NR solver to add named scopes + timing callbacks
# ---------------------------------------------------------------------------

import vajax.analysis.solver_factories as sf

_original_make_nr = sf._make_nr_solver_common


def patched_make_nr_solver_common(*, build_system_jit, linear_solve_fn, enforce_noi_fn, **kwargs):
    """Wrap build_system and linear_solve with named scopes and timing."""

    def timed_build_system(*args):
        with jax.named_scope("nr_build_system"):
            return build_system_jit(*args)

    def timed_linear_solve(J_or_data, f):
        with jax.named_scope("nr_linear_solve"):
            return linear_solve_fn(J_or_data, f)

    def timed_enforce_noi(J_or_data, f):
        with jax.named_scope("nr_enforce_noi"):
            return enforce_noi_fn(J_or_data, f)

    return _original_make_nr(
        build_system_jit=timed_build_system,
        linear_solve_fn=timed_linear_solve,
        enforce_noi_fn=timed_enforce_noi,
        **kwargs,
    )


sf._make_nr_solver_common = patched_make_nr_solver_common


# ---------------------------------------------------------------------------
# Also add callback-based timing for text summary
# ---------------------------------------------------------------------------

_original_make_nr2 = sf._make_nr_solver_common  # This is now our patched version


def callback_timed_make_nr(*, build_system_jit, linear_solve_fn, enforce_noi_fn, **kwargs):
    """Add jax.debug.callback timestamps around each phase."""

    def timed_build_system(*args):
        # Extract iteration from args (it's the last positional arg)
        iteration = args[-1] if len(args) > 0 else jnp.array(0)
        build_tag = jnp.array(list(b"build_system"), dtype=jnp.uint8)
        jax.debug.callback(_start_phase, build_tag)
        result = build_system_jit(*args)
        jax.debug.callback(_end_phase, build_tag, iteration)
        return result

    def timed_linear_solve(J_or_data, f):
        solve_tag = jnp.array(list(b"linear_solve"), dtype=jnp.uint8)
        jax.debug.callback(_start_phase, solve_tag)
        result = linear_solve_fn(J_or_data, f)
        jax.debug.callback(_end_phase, solve_tag, jnp.array(-1))
        return result

    return _original_make_nr2(
        build_system_jit=timed_build_system,
        linear_solve_fn=timed_linear_solve,
        enforce_noi_fn=enforce_noi_fn,
        **kwargs,
    )


sf._make_nr_solver_common = callback_timed_make_nr


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    import logging

    from vajax.analysis import CircuitEngine
    from vajax.benchmarks.registry import get_benchmark

    parser = argparse.ArgumentParser(description="Profile NR phase breakdown")
    parser.add_argument("benchmark", help="Benchmark name (e.g. ring, c6288)")
    parser.add_argument(
        "--trace-dir",
        type=Path,
        default=None,
        help="Directory for Perfetto trace (default: /tmp/claude/<benchmark>_trace)",
    )
    parser.add_argument("--t-stop", type=float, default=None, help="Override stop time")
    parser.add_argument(
        "--n-steps", type=int, default=10, help="Number of timesteps to profile (default: 10)"
    )
    args = parser.parse_args()

    logging.getLogger("vajax").setLevel(logging.WARNING)

    info = get_benchmark(args.benchmark)
    assert info is not None, f"Benchmark '{args.benchmark}' not found"

    engine = CircuitEngine(info.sim_path)
    engine.parse()

    use_sparse = info.is_large
    dt = info.dt
    t_stop = args.t_stop or dt * args.n_steps

    print(f"Benchmark: {args.benchmark}")
    print(f"  Nodes: {engine.num_nodes}, Devices: {len(engine.devices)}")
    print(f"  Solver: {'sparse' if use_sparse else 'dense'}")
    print(f"  Profiling {args.n_steps} steps (t_stop={t_stop:.2e}s)")

    trace_dir = args.trace_dir or Path(f"/tmp/claude/{args.benchmark}_trace")
    trace_dir.mkdir(parents=True, exist_ok=True)

    # Prepare (includes JIT warmup)
    print("\nPreparing (JIT warmup)...")
    engine.prepare(t_stop=t_stop, dt=dt, use_sparse=use_sparse)

    # Clear any timing from warmup
    phase_timings.clear()

    # Run with Perfetto trace capture
    print(f"Running with profiler trace -> {trace_dir}")
    jax.profiler.start_trace(str(trace_dir))
    try:
        result = engine.run_transient()
    finally:
        jax.profiler.stop_trace()

    convergence = result.stats.get("convergence_rate", 0) * 100
    print(f"  Steps: {result.num_steps}, convergence: {convergence:.0f}%")
    print(f"  Trace saved to: {trace_dir}")

    # --- Analyze callback timings ---
    if not phase_timings:
        print("\nNo callback timings captured (expected inside lax.while_loop)")
        print(f"View the Perfetto trace at: {trace_dir}")
        print("  Open https://ui.perfetto.dev and load the trace file")
        return

    print(f"\n{'=' * 60}")
    print(f"NR Phase Timing Breakdown ({len(phase_timings)} measurements)")
    print(f"{'=' * 60}")

    # Aggregate by phase
    by_phase: dict[str, list[float]] = {}
    for entry in phase_timings:
        phase = entry["phase"]
        if phase not in by_phase:
            by_phase[phase] = []
        by_phase[phase].append(entry["elapsed_us"])

    total_us = sum(sum(times) for times in by_phase.values())

    print(f"\n{'Phase':<20} {'Count':>6} {'Total (ms)':>12} {'Mean (µs)':>12} {'%':>8}")
    print(f"{'-' * 20} {'-' * 6} {'-' * 12} {'-' * 12} {'-' * 8}")
    for phase, times in sorted(by_phase.items(), key=lambda x: -sum(x[1])):
        total_ms = sum(times) / 1000
        mean_us = np.mean(times)
        pct = sum(times) / total_us * 100 if total_us > 0 else 0
        print(f"{phase:<20} {len(times):>6} {total_ms:>12.2f} {mean_us:>12.1f} {pct:>7.1f}%")

    print(f"\n{'Total':.<20} {'':>6} {total_us / 1000:>12.2f} ms")

    # Per-NR-iteration breakdown (first few)
    build_times = by_phase.get("build_system", [])
    solve_times = by_phase.get("linear_solve", [])

    if build_times and solve_times:
        n_show = min(10, len(build_times))
        print(f"\nPer-iteration breakdown (first {n_show}):")
        print(f"{'Iter':>4} {'Build (µs)':>12} {'Solve (µs)':>12} {'Solve %':>8}")
        print(f"{'-' * 4} {'-' * 12} {'-' * 12} {'-' * 8}")
        for i in range(n_show):
            b = build_times[i]
            s = solve_times[i] if i < len(solve_times) else 0
            total = b + s
            spct = s / total * 100 if total > 0 else 0
            print(f"{i:>4} {b:>12.1f} {s:>12.1f} {spct:>7.1f}%")

    print(f"\nPerfetto trace: {trace_dir}")
    print("  Open https://ui.perfetto.dev and load the .pb or .json.gz file")


if __name__ == "__main__":
    main()
