#!/usr/bin/env python3
"""Target script for nsys GPU profiling - runs circuit simulation.

Uses CUDA profiler API to capture ONLY the simulation run (not warmup/JIT).
Run with nsys --capture-range=cudaProfilerApi to enable selective capture.

Usage:
    nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop \\
        -o profile uv run python scripts/nsys_profile_target.py ring 500

Arguments:
    circuit: One of rc, graetz, mul, ring, c6288 (default: ring)
    timesteps: Number of timesteps to simulate (default: 500)
"""

import argparse
import ctypes
import sys
from pathlib import Path

import jax

sys.path.insert(0, ".")

# Import vajax first to auto-configure precision based on backend
from vajax.analysis import CircuitEngine


def _cuda_profiler_start():
    """Start CUDA profiler capture via cudaProfilerStart()."""
    try:
        libcudart = ctypes.CDLL("libcudart.so")
        libcudart.cudaProfilerStart()
        return True
    except OSError:
        return False


def _cuda_profiler_stop():
    """Stop CUDA profiler capture via cudaProfilerStop()."""
    try:
        libcudart = ctypes.CDLL("libcudart.so")
        libcudart.cudaProfilerStop()
    except OSError:
        pass


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

    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")
    print(f"Circuit: {args.circuit}")
    print(f"Timesteps: {args.timesteps}")
    print()

    # Find benchmark .sim file
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    sim_path = repo_root / "vendor" / "VACASK" / "benchmark" / args.circuit / "vacask" / "runme.sim"

    if not sim_path.exists():
        print(f"ERROR: Benchmark file not found: {sim_path}")
        sys.exit(1)

    # Setup circuit using CircuitEngine
    print(f"Setting up circuit from {sim_path}...")
    engine = CircuitEngine(sim_path)
    engine.parse()

    print(f"Circuit size: {engine.num_nodes} nodes, {len(engine.devices)} devices")
    print()

    # Timestep from analysis params or default
    dt = engine.analysis_params.get("step", 1e-12)
    print(f"Using dt={dt}")
    print()

    # Prepare (includes 1-step JIT warmup) — NOT profiled
    print(f"Preparing ({args.timesteps} timesteps, includes JIT warmup)...")
    engine.prepare(
        t_stop=args.timesteps * dt,
        dt=dt,
        use_sparse=args.sparse,
    )
    print("Prepare complete")
    print()

    # Start CUDA profiler capture — only the simulation run is profiled
    has_profiler = _cuda_profiler_start()
    if has_profiler:
        print("CUDA profiler capture started (warmup excluded)")
    else:
        print("WARNING: cudaProfilerStart() unavailable — profiling entire process")

    print(f"Starting profiled run ({args.timesteps} timesteps)...")
    result = engine.run_transient()

    # Stop CUDA profiler capture
    if has_profiler:
        _cuda_profiler_stop()

    print()
    print(f"Completed: {result.num_steps} timesteps")
    print(f"Wall time: {result.stats.get('wall_time', 0):.3f}s")


if __name__ == "__main__":
    main()
