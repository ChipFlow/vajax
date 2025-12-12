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
    circuit: One of inv_test, nor_test, and_test, c6288_test (default: and_test)
    timesteps: Number of timesteps to simulate (default: 50)

Example:
    nsys-jax -o /tmp/profile.zip python scripts/nsys_profile_target.py and_test 100
"""
import argparse
import sys

sys.path.insert(0, ".")

import jax

jax.config.update("jax_enable_x64", True)

from jax_spice.analysis.transient_gpu import transient_analysis_gpu
from jax_spice.benchmarks.c6288 import C6288Benchmark


def main():
    parser = argparse.ArgumentParser(description="nsys-jax profiling target for JAX-SPICE")
    parser.add_argument(
        "circuit",
        nargs="?",
        default="and_test",
        choices=["inv_test", "nor_test", "and_test", "c6288_test"],
        help="Circuit to profile (default: and_test)",
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
    args = parser.parse_args()

    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")
    print(f"Circuit: {args.circuit}")
    print(f"Timesteps: {args.timesteps}")
    print()

    # Setup circuit
    print("Setting up circuit...")
    bench = C6288Benchmark(verbose=False)
    bench.parse()
    bench.flatten(args.circuit)
    bench.build_system(args.circuit)
    system = bench.system
    system.build_device_groups()

    print(f"Circuit size: {system.num_nodes} nodes, {len(system.devices)} devices")
    print()

    if not args.skip_warmup:
        # Warmup run (JIT compilation happens here, outside profiled region)
        print("Warmup run (JIT compilation)...")
        _, _, _ = transient_analysis_gpu(
            system,
            t_stop=1e-12,
            t_step=1e-12,
            vdd=1.2,
            icmode="uic",
            verbose=False,
        )
        print("Warmup complete")
        print()

    # Profiled run - nsys-jax captures this automatically
    print(f"Starting profiled run ({args.timesteps} timesteps)...")
    times, solutions, info = transient_analysis_gpu(
        system,
        t_stop=args.timesteps * 1e-12,
        t_step=1e-12,
        vdd=1.2,
        icmode="uic",
        verbose=True,
    )

    print()
    print(f"Completed: {len(times)} timesteps, {info['total_iterations']} iterations")


if __name__ == "__main__":
    main()
