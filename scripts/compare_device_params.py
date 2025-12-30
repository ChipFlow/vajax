#!/usr/bin/env python3
"""Compare device parameters between JAX-SPICE and VACASK.

This script traces parameters through all mapping layers to debug
parameter alignment issues between the two simulators.

Usage:
    # Compare parameters for ring benchmark
    uv run scripts/compare_device_params.py --benchmark ring

    # Trace specific instance
    uv run scripts/compare_device_params.py --benchmark ring --instance u1.mp.m

    # Trace specific parameter
    uv run scripts/compare_device_params.py --benchmark ring --instance u1.mp.m --param w

    # Check parameter coverage for all instances
    uv run scripts/compare_device_params.py --benchmark ring --coverage
"""

import argparse
import sys
from pathlib import Path

# Ensure jax-spice is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import jax_spice first to auto-configure precision based on backend
from jax_spice.analysis import CircuitEngine
from jax_spice.analysis.debug import (
    trace_param,
    trace_all_params,
    check_param_coverage,
    format_param_trace,
    format_coverage_chart,
    get_coverage_breakdown,
    format_stats,
    format_devices,
    format_models,
)
from scripts.benchmark_utils import get_vacask_benchmarks, log


def main():
    parser = argparse.ArgumentParser(
        description="Compare device parameters between JAX-SPICE and VACASK"
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="ring",
        help="Benchmark name (default: ring)",
    )
    parser.add_argument(
        "--instance",
        type=str,
        default=None,
        help="Specific instance name to trace (e.g., 'u1.mp.m')",
    )
    parser.add_argument(
        "--param",
        type=str,
        default=None,
        help="Specific parameter to trace (e.g., 'w', 'l', 'type')",
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Show parameter coverage for all instances",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show circuit statistics",
    )
    parser.add_argument(
        "--devices",
        action="store_true",
        help="List all devices",
    )
    parser.add_argument(
        "--models",
        action="store_true",
        help="List all models",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Show visual ASCII chart of parameter coverage",
    )
    parser.add_argument(
        "--sim-file",
        type=str,
        default=None,
        help="Direct path to .sim file (overrides --benchmark)",
    )
    args = parser.parse_args()

    # Get benchmark path
    if args.sim_file:
        sim_path = Path(args.sim_file)
        if not sim_path.exists():
            print(f"Error: File not found: {sim_path}")
            sys.exit(1)
    else:
        benchmarks = get_vacask_benchmarks([args.benchmark])
        if not benchmarks:
            print(f"Error: Benchmark '{args.benchmark}' not found")
            print("Available benchmarks: rc, graetz, mul, ring, c6288")
            sys.exit(1)
        _, sim_path = benchmarks[0]

    print("=" * 70)
    print(f"Device Parameter Comparison: {sim_path.name}")
    print("=" * 70)
    print()

    # Parse circuit
    log("Parsing circuit...", end=" ")
    engine = CircuitEngine(sim_path)
    engine.parse()
    log("done")
    print()

    # Handle simple info commands
    if args.stats:
        print(format_stats(engine.circuit, engine))
        print()
        return

    if args.devices:
        print(format_devices(engine.circuit, engine))
        print()
        return

    if args.models:
        print(format_models(engine.circuit))
        print()
        return

    # Parameter coverage mode
    if args.coverage or args.visualize:
        # Group devices by model type
        by_model = {}
        for dev in engine.devices:
            model = dev.get('model', 'unknown')
            if model not in by_model:
                by_model[model] = []
            by_model[model].append(dev)

        for model, devices in sorted(by_model.items()):
            if model in ('vsource', 'isource'):
                continue  # Skip sources

            # Use first instance as representative
            dev = devices[0]
            name = dev.get('name')

            if args.visualize:
                # Visual ASCII chart
                print(format_coverage_chart(engine, name))
                print()
            else:
                # Simple summary
                print(f"Model: {model} ({len(devices)} instances)")
                coverage = check_param_coverage(engine, name)

                print(f"  Coverage: {coverage['coverage_pct']}")
                print(f"  Mapped: {len(coverage['mapped'])}")
                print(f"  Unmapped: {len(coverage['unmapped'])}")

                if coverage['unmapped']:
                    print(f"  Unmapped params: {', '.join(coverage['unmapped'][:10])}")
                    if len(coverage['unmapped']) > 10:
                        print(f"    ... and {len(coverage['unmapped']) - 10} more")
                print()

        return

    # Trace specific instance
    if args.instance:
        if args.param:
            # Trace single parameter
            print(format_param_trace(engine, args.instance, args.param))
        else:
            # Trace all parameters for instance
            print(format_param_trace(engine, args.instance))
        print()
        return

    # Default: show summary and sample traces
    print(format_stats(engine.circuit, engine))
    print()

    # Show sample device traces
    openvaf_devices = [d for d in engine.devices if d.get('model') not in ('vsource', 'isource')]

    if openvaf_devices:
        print("Sample Device Parameter Traces")
        print("-" * 70)

        # Show first device of each model type
        shown_models = set()
        for dev in openvaf_devices:
            model = dev.get('model')
            if model in shown_models:
                continue
            shown_models.add(model)

            name = dev.get('name')
            print()
            print(format_param_trace(engine, name))
            print()

            if len(shown_models) >= 3:
                remaining = len(set(d.get('model') for d in openvaf_devices)) - len(shown_models)
                if remaining > 0:
                    print(f"... and {remaining} more model types")
                break


if __name__ == '__main__':
    main()
