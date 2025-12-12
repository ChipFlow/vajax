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

sys.path.insert(0, ".")

import jax

jax.config.update("jax_enable_x64", True)

from jax_spice.analysis.transient import transient_analysis_vectorized
from jax_spice.benchmarks import VACASKBenchmarkRunner


def main():
    parser = argparse.ArgumentParser(description="nsys-jax profiling target for JAX-SPICE")
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
        "--skip-warmup",
        action="store_true",
        help="Skip warmup run (includes JIT compilation in profile)",
    )
    parser.add_argument(
        "--backend",
        default="gpu",
        choices=["cpu", "gpu", "auto"],
        help="Backend to use (default: gpu)",
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

    # Setup circuit using VACASKBenchmarkRunner
    print(f"Setting up circuit from {sim_path}...")
    runner = VACASKBenchmarkRunner(sim_path, verbose=False)
    runner.parse()
    system = runner.to_mna_system()
    system.build_device_groups()

    print(f"Circuit size: {system.num_nodes} nodes, {len(system.devices)} devices")
    print()

    # Timestep from analysis params or default
    dt = runner.analysis_params.get("step", 1e-12)
    print(f"Using dt={dt}")
    print()

    if not args.skip_warmup:
        # Warmup run (JIT compilation happens here, outside profiled region)
        print("Warmup run (JIT compilation)...")
        _, _, _ = transient_analysis_vectorized(
            system,
            t_stop=dt,
            t_step=dt,
            backend=args.backend,
        )
        print("Warmup complete")
        print()

    # Profiled run - nsys-jax captures this automatically
    print(f"Starting profiled run ({args.timesteps} timesteps)...")
    times, solutions, info = transient_analysis_vectorized(
        system,
        t_stop=args.timesteps * dt,
        t_step=dt,
        backend=args.backend,
    )

    print()
    print(f"Completed: {len(times)} timesteps")


if __name__ == "__main__":
    main()
