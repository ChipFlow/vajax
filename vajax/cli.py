"""VA-JAX command-line interface.

Provides ngspice-style CLI for circuit simulation:
    va-jax circuit.sim                    # Run with .control section
    va-jax circuit.sim -o results.raw     # Specify output file
    va-jax circuit.sim --tran 1n 100u     # Override transient params
    va-jax benchmark ring                 # Run benchmark
    va-jax convert input.sp output.sim    # SPICE to VACASK conversion
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

import jax

# Configure JAX before importing vajax
from vajax import CircuitEngine, configure_precision, get_precision_info

logger = logging.getLogger(__name__)


def parse_spice_value(s: str) -> float:
    """Parse a SPICE value with SI suffix.

    Examples:
        '1n' -> 1e-9
        '100u' -> 100e-6
        '1meg' -> 1e6
    """
    s = s.strip().lower()
    suffixes = [
        ("meg", 1e6),
        ("g", 1e9),
        ("t", 1e12),
        ("k", 1e3),
        ("m", 1e-3),
        ("u", 1e-6),
        ("n", 1e-9),
        ("p", 1e-12),
        ("f", 1e-15),
        ("a", 1e-18),
    ]
    for suffix, mult in suffixes:
        if s.endswith(suffix):
            return float(s[: -len(suffix)]) * mult
    return float(s)


def setup_logging(verbose: int = 0, quiet: bool = False) -> None:
    """Configure logging based on verbosity level."""
    if quiet:
        level = logging.WARNING
    elif verbose >= 2:
        level = logging.DEBUG
    elif verbose >= 1:
        level = logging.INFO
    else:
        level = logging.WARNING

    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s",
    )


def cmd_run(args: argparse.Namespace) -> int:
    """Run circuit simulation."""
    circuit_path = Path(args.circuit)
    if not circuit_path.exists():
        print(f"Error: Circuit file not found: {circuit_path}", file=sys.stderr)
        return 1

    # Configure precision
    if args.x64:
        configure_precision(force_x64=True)
    elif args.x32:
        configure_precision(force_x64=False)

    # Force GPU/CPU backend
    if args.gpu:
        import os

        os.environ.setdefault("JAX_PLATFORMS", "cuda,gpu")
    elif args.cpu:
        import os

        os.environ.setdefault("JAX_PLATFORMS", "cpu")

    # Load and parse circuit
    try:
        engine = CircuitEngine(str(circuit_path))
        engine.parse()
    except Exception as e:
        print(f"Error parsing circuit: {e}", file=sys.stderr)
        return 1

    # Determine analysis type
    if args.tran:
        # --tran dt t_stop
        dt = parse_spice_value(args.tran[0])
        t_stop = parse_spice_value(args.tran[1])
        result = _run_transient(engine, dt, t_stop, args)
    elif args.ac:
        # --ac type points fstart fstop
        result = _run_ac(engine, args.ac, args)
    else:
        # Run transient with defaults from circuit or sensible defaults
        result = _run_transient(engine, 1e-9, 1e-6, args)

    if result is None:
        return 1

    # Write output
    output_path = Path(args.output) if args.output else circuit_path.with_suffix(".raw")
    _write_output(result, output_path, args.format)

    print(f"Results written to: {output_path}")
    return 0


def _run_transient(engine: CircuitEngine, dt: float, t_stop: float, args: argparse.Namespace):
    """Run transient analysis."""
    try:
        engine.prepare(t_stop=t_stop, dt=dt, use_sparse=args.sparse)
        result = engine.run_transient()
        print(f"Transient: {len(result.times)} time points, t_stop={t_stop:.2e}s")
        return result
    except Exception as e:
        print(f"Error in transient analysis: {e}", file=sys.stderr)
        logger.exception("Transient analysis failed")
        return None


def _run_ac(engine: CircuitEngine, ac_args: List[str], args: argparse.Namespace):
    """Run AC analysis."""
    # --ac type points fstart fstop
    sweep_type = ac_args[0]
    num_points = int(ac_args[1])
    fstart = parse_spice_value(ac_args[2])
    fstop = parse_spice_value(ac_args[3])

    try:
        result = engine.run_ac(
            freq_start=fstart,
            freq_stop=fstop,
            points=num_points,
            mode=sweep_type,
        )
        print(f"AC: {num_points} frequency points, {fstart:.2e}Hz to {fstop:.2e}Hz")
        return result
    except Exception as e:
        print(f"Error in AC analysis: {e}", file=sys.stderr)
        logger.exception("AC analysis failed")
        return None


def _write_output(result, output_path: Path, fmt: str) -> None:
    """Write simulation results to file."""
    if fmt == "raw":
        from vajax.io.rawfile_writer import write_rawfile

        write_rawfile(result, output_path)
    elif fmt == "csv":
        from vajax.io.csv_writer import write_csv

        write_csv(result, output_path)
    elif fmt == "json":
        import json

        data = {
            "times": result.times.tolist() if hasattr(result, "times") else [],
            "voltages": {k: v.tolist() for k, v in result.voltages.items()},
        }
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
    else:
        print(f"Unknown format: {fmt}", file=sys.stderr)


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Run benchmark circuits."""
    from vajax.benchmarks.registry import get_benchmark, list_benchmarks

    if args.list:
        benchmarks = list_benchmarks()
        print("Available benchmarks:")
        for name in benchmarks:
            print(f"  {name}")
        return 0

    if not args.name:
        print("Error: Benchmark name required (use --list to see available)", file=sys.stderr)
        return 1

    try:
        bench = get_benchmark(args.name)
        if bench is None:
            print(f"Error: Unknown benchmark: {args.name}", file=sys.stderr)
            return 1

        # Configure precision
        if args.x64:
            configure_precision(force_x64=True)
        elif args.x32:
            configure_precision(force_x64=False)

        print(f"Running benchmark: {args.name}")
        print(f"  Circuit: {bench.circuit_path}")

        engine = CircuitEngine(str(bench.circuit_path))
        engine.parse()

        # Run with profiling if requested
        if args.profile:
            import time

            start = time.perf_counter()

        engine.prepare(t_stop=bench.t_stop, dt=bench.dt, use_sparse=args.sparse)
        result = engine.run_transient()

        if args.profile:
            elapsed = time.perf_counter() - start
            print(f"  Time: {elapsed:.3f}s")
            print(f"  Steps: {len(result.times)}")
            print(f"  ms/step: {1000 * elapsed / len(result.times):.2f}")

        print(f"  Result: {'PASS' if result.converged else 'FAIL'}")
        return 0 if result.converged else 1

    except Exception as e:
        print(f"Error running benchmark: {e}", file=sys.stderr)
        logger.exception("Benchmark failed")
        return 1


def cmd_convert(args: argparse.Namespace) -> int:
    """Convert SPICE netlist to VACASK format."""
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        return 1

    try:
        from vajax.netlist_converter import convert_netlist

        convert_netlist(input_path, output_path)
        print(f"Converted: {input_path} -> {output_path}")
        return 0
    except ImportError:
        print("Error: Netlist converter not available", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error converting netlist: {e}", file=sys.stderr)
        return 1


def cmd_info(args: argparse.Namespace) -> int:
    """Show system information."""
    info = get_precision_info()
    print("VA-JAX System Information")
    print("-" * 40)
    print(f"Backend: {info['backend']}")
    print(f"Float64 enabled: {info['x64_enabled']}")
    print(f"Devices: {jax.devices()}")

    try:
        import vajax

        print(f"Version: {getattr(vajax, '__version__', 'unknown')}")
    except Exception:
        pass

    return 0


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        prog="va-jax",
        description="VA-JAX: GPU-accelerated analog circuit simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  va-jax circuit.sim                    Run with defaults
  va-jax circuit.sim -o out.raw         Specify output file
  va-jax circuit.sim --tran 1n 100u     Override transient params
  va-jax circuit.sim --sparse --gpu     Use sparse solver on GPU
  va-jax benchmark ring --profile       Run and profile benchmark
  va-jax convert input.sp output.sim    Convert SPICE to VACASK
        """,
    )

    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")
    parser.add_argument(
        "-v", "--verbose", action="count", default=0, help="Increase verbosity (use -vv for debug)"
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress non-error output")

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Run command (default when circuit file is provided)
    run_parser = subparsers.add_parser("run", help="Run simulation")
    run_parser.add_argument("circuit", help="Circuit file (.sim or SPICE)")
    run_parser.add_argument("-o", "--output", help="Output file path")
    run_parser.add_argument(
        "-f",
        "--format",
        choices=["raw", "csv", "json"],
        default="raw",
        help="Output format (default: raw)",
    )

    # Analysis options
    run_parser.add_argument(
        "--tran", nargs=2, metavar=("DT", "TSTOP"), help="Transient analysis: dt t_stop"
    )
    run_parser.add_argument(
        "--ac",
        nargs=4,
        metavar=("TYPE", "POINTS", "FSTART", "FSTOP"),
        help="AC analysis: dec|lin|oct points fstart fstop",
    )

    # Solver options
    run_parser.add_argument("--sparse", action="store_true", help="Force sparse solver")

    # Backend options
    run_parser.add_argument("--gpu", action="store_true", help="Force GPU backend")
    run_parser.add_argument("--cpu", action="store_true", help="Force CPU backend")
    run_parser.add_argument("--x64", action="store_true", help="Force float64 precision")
    run_parser.add_argument("--x32", action="store_true", help="Force float32 precision")

    run_parser.add_argument("--profile", action="store_true", help="Enable profiling")
    run_parser.set_defaults(func=cmd_run)

    # Benchmark command
    bench_parser = subparsers.add_parser(
        "benchmark", aliases=["bench"], help="Run benchmark circuits"
    )
    bench_parser.add_argument("name", nargs="?", help="Benchmark name")
    bench_parser.add_argument("--list", "-l", action="store_true", help="List available benchmarks")
    bench_parser.add_argument("--sparse", action="store_true", help="Force sparse solver")
    bench_parser.add_argument("--x64", action="store_true", help="Force float64 precision")
    bench_parser.add_argument("--x32", action="store_true", help="Force float32 precision")
    bench_parser.add_argument("--profile", action="store_true", help="Enable profiling")
    bench_parser.set_defaults(func=cmd_benchmark)

    # Convert command
    conv_parser = subparsers.add_parser("convert", help="Convert SPICE to VACASK format")
    conv_parser.add_argument("input", help="Input SPICE file")
    conv_parser.add_argument("output", help="Output VACASK file")
    conv_parser.set_defaults(func=cmd_convert)

    # Info command
    info_parser = subparsers.add_parser("info", help="Show system information")
    info_parser.set_defaults(func=cmd_info)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point."""
    parser = create_parser()

    # Handle case where circuit file is provided without 'run' command
    if argv is None:
        argv = sys.argv[1:]

    # If first arg is a file path (not a subcommand), insert 'run'
    if (
        argv
        and not argv[0].startswith("-")
        and argv[0] not in ("run", "benchmark", "bench", "convert", "info")
    ):
        # Check if it looks like a file path
        if "." in argv[0] or "/" in argv[0]:
            argv = ["run"] + list(argv)

    args = parser.parse_args(argv)

    setup_logging(args.verbose, args.quiet)

    if hasattr(args, "func"):
        return args.func(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
