#!/usr/bin/env python3
"""Target script for nsys-jax profiling - runs circuit simulation.

This script is designed to be wrapped by nsys-jax:
    nsys-jax -o profile.zip python scripts/nsys_profile_target.py [circuit] [timesteps]

nsys-jax automatically handles:
- XLA_FLAGS configuration for HLO metadata dumping
- JAX_TRACEBACK_IN_LOCATIONS_LIMIT for stack traces
- JAX_ENABLE_COMPILATION_CACHE=false for metadata collection

Usage:
    python scripts/nsys_profile_target.py [circuit] [timesteps]

Arguments:
    circuit: One of rc, graetz, mul, ring (default: ring)
    timesteps: Number of timesteps to simulate (default: 50)

Example:
    nsys-jax -o /tmp/profile.zip python scripts/nsys_profile_target.py ring 100
"""

import argparse
import sys
from pathlib import Path

import jax

sys.path.insert(0, ".")

# Import vajax first to auto-configure precision based on backend
from vajax.analysis import CircuitEngine


def main():
    parser = argparse.ArgumentParser(description="nsys-jax profiling target for VA-JAX")
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
        default=50,
        help="Number of timesteps to simulate (default: 50)",
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

    # Prepare (includes 1-step JIT warmup)
    print(f"Preparing ({args.timesteps} timesteps, includes JIT warmup)...")
    engine.prepare(
        t_stop=args.timesteps * dt,
        dt=dt,
        use_sparse=args.sparse,
    )
    print("Prepare complete")
    print()

    # Profiled run - nsys-jax captures this automatically
    print(f"Starting profiled run ({args.timesteps} timesteps)...")
    result = engine.run_transient()

    print()
    print(f"Completed: {result.num_steps} timesteps")
    print(f"Wall time: {result.stats.get('wall_time', 0):.3f}s")


if __name__ == "__main__":
    main()
