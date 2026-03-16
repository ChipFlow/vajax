#!/usr/bin/env python3
"""Target script for nsys GPU profiling - runs circuit simulation.

Usage:
    nsys profile -o profile uv run python scripts/nsys_profile_target.py ring 500

Arguments:
    circuit: One of rc, graetz, mul, ring, c6288 (default: ring)
    timesteps: Number of timesteps to simulate (default: 500)

Use 500+ timesteps so JIT warmup overhead is <5% of total profile.
"""

import argparse
import ctypes
import logging
import sys
import time
from pathlib import Path

import jax

sys.path.insert(0, ".")

# Enable INFO logging so solver selection messages are visible
logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")


def timed(label):
    """Context manager that prints elapsed time for a phase."""

    class Timer:
        def __enter__(self):
            self.t0 = time.perf_counter()
            print(f"[{label}] starting...", flush=True)
            return self

        def __exit__(self, *exc):
            dt = time.perf_counter() - self.t0
            print(f"[{label}] done in {dt:.2f}s", flush=True)

    return Timer()


def main():
    parser = argparse.ArgumentParser(description="nsys profiling target for VAJAX")
    parser.add_argument(
        "circuit",
        nargs="?",
        default="ring",
        choices=["rc", "graetz", "mul", "ring", "c6288"],
        help="Circuit to profile (default: ring)",
    )
    parser.add_argument(
        "timesteps",
        nargs="?",
        type=int,
        default=500,
        help="Number of timesteps to simulate (default: 500)",
    )
    parser.add_argument(
        "--backend",
        default="gpu",
        choices=["cpu", "gpu", "auto"],
        help="Backend to use (default: gpu)",
    )
    parser.add_argument(
        "--sparse",
        action="store_true",
        help="Use sparse solver (for large circuits)",
    )
    args = parser.parse_args()

    with timed("JAX init"):
        print(f"JAX backend: {jax.default_backend()}")
        print(f"JAX devices: {jax.devices()}")
    print(f"Circuit: {args.circuit}")
    print(f"Timesteps: {args.timesteps}")

    # Explicit solver availability check
    print()
    print("=== Solver Availability ===")
    with timed("solver imports"):
        try:
            from spineax.cudss.dense_baspacho_solver import is_available

            print("  BaSpaCho dense import: OK")
            print(f"  BaSpaCho dense available: {is_available()}")
        except ImportError as e:
            print(f"  BaSpaCho dense import: FAILED ({e})")
        try:
            from spineax.cudss.solver import CuDSSSolver  # noqa: F401

            print("  cuDSS sparse import: OK")
        except ImportError as e:
            print(f"  cuDSS sparse import: FAILED ({e})")
        try:
            from spineax import baspacho_dense_solve as _mod

            print(f"  baspacho_dense_solve C++ module: OK ({_mod})")
        except ImportError as e:
            print(f"  baspacho_dense_solve C++ module: FAILED ({e})")
    print()

    # Find benchmark .sim file
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    sim_path = repo_root / "vendor" / "VACASK" / "benchmark" / args.circuit / "vacask" / "runme.sim"

    if not sim_path.exists():
        print(f"ERROR: Benchmark file not found: {sim_path}")
        sys.exit(1)

    # Import vajax (auto-configures precision based on backend)
    with timed("vajax import"):
        from vajax.analysis import CircuitEngine

    # Setup circuit using CircuitEngine
    with timed("circuit parse"):
        print(f"Setting up circuit from {sim_path}...")
        engine = CircuitEngine(sim_path)
        engine.parse()

    print(f"Circuit size: {engine.num_nodes} nodes, {len(engine.devices)} devices")
    print()

    # Timestep from analysis params or default
    dt = engine.analysis_params.get("step", 1e-12)
    print(f"Using dt={dt}")
    print()

    # Prepare (includes 1-step JIT warmup)
    with timed("prepare + JIT warmup"):
        print(f"Preparing ({args.timesteps} timesteps, includes JIT warmup)...")
        engine.prepare(
            t_stop=args.timesteps * dt,
            dt=dt,
            use_sparse=args.sparse,
        )
    print("Prepare complete")
    print()

    # NVTX range for nsys --capture-range=nvtx scoping
    try:
        _nvtx = ctypes.CDLL("libnvToolsExt.so")
        _nvtx_push = _nvtx.nvtxRangePushA
        _nvtx_push.argtypes = [ctypes.c_char_p]
        _nvtx_pop = _nvtx.nvtxRangePop
    except OSError:
        _nvtx_push = _nvtx_pop = None

    if _nvtx_push:
        _nvtx_push(b"run_transient")

    with timed("transient simulation"):
        print(f"Starting profiled run ({args.timesteps} timesteps)...", flush=True)
        result = engine.run_transient()

    if _nvtx_pop:
        _nvtx_pop()

    print(flush=True)
    print(f"Completed: {result.num_steps} timesteps", flush=True)
    print(f"Wall time: {result.stats.get('wall_time', 0):.3f}s", flush=True)


if __name__ == "__main__":
    main()
