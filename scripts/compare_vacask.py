#!/usr/bin/env python3
"""Compare VAJAX vs VACASK Performance

Runs both simulators on the same benchmarks with matching parameters
and compares performance.

Usage:
    # Run all benchmarks
    uv run scripts/compare_vacask.py

    # Run specific benchmark
    uv run scripts/compare_vacask.py --benchmark ring

    # Use lax.scan (faster)
    uv run scripts/compare_vacask.py --benchmark ring --use-scan

    # Enable JAX profiling (GPU only, creates Perfetto traces)
    uv run scripts/compare_vacask.py --profile-mode jax --profile-dir /tmp/traces

    # Enable nsys-jax profiling (GPU only, creates .nsys-rep + analysis zip)
    # Note: Requires nsys-jax to be installed (from JAX-Toolbox)
    uv run scripts/compare_vacask.py --profile-mode nsys --profile-dir /tmp/traces

    # Enable both profiling modes (run separately to avoid interference)
    uv run scripts/compare_vacask.py --profile-mode both --profile-dir /tmp/traces
"""

import argparse
import json
import os
import subprocess

# Enable vajax performance logging with memory stats and perf_counter timestamps
# This helps track memory usage and correlate log messages with Perfetto trace timestamps
from vajax._logging import enable_performance_logging

enable_performance_logging(with_memory=True, with_perf_counter=True)
import re
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

# Ensure vajax is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure JAX memory allocation BEFORE importing JAX
# Disable preallocation to avoid grabbing all GPU memory at startup
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
# Use memory growth instead of fixed allocation
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
# Note: Set JAX_PLATFORMS=cpu before running for CPU-only mode

import jax

# Precision is auto-configured by vajax import (imported above via logging)
# Metal/TPU use f32, CPU/CUDA use f64
from vajax.analysis import CircuitEngine
from vajax.benchmarks.registry import (
    BenchmarkInfo,
    get_benchmark,
    list_benchmarks,
)
from vajax.profiling import ProfileConfig

# Note: Benchmark configurations are now auto-discovered from
# vajax.benchmarks.registry. The registry parses .sim files
# to extract dt, t_stop, and device types automatically.

# Published VACASK benchmark times (ms/step) from vendor/VACASK/benchmark/README.md
# Run on AMD Threadripper 7970, single-threaded, KLU solver
# These serve as reference when VACASK binary is not available (e.g., GPU-only CI)
VACASK_REFERENCE_TIMES = {
    # rc: 0.94s / 1005006 steps = 0.935µs/step
    "rc": 0.94 / 1005006 * 1000,  # 0.000935 ms/step
    # graetz: 1.89s / 1000003 steps = 1.89µs/step
    "graetz": 1.89 / 1000003 * 1000,  # 0.00189 ms/step
    # mul: 0.97s / 500056 steps = 1.94µs/step
    "mul": 0.97 / 500056 * 1000,  # 0.00194 ms/step
    # ring: 1.18s / 26066 steps = 45.3µs/step
    "ring": 1.18 / 26066 * 1000,  # 0.0453 ms/step
    # c6288: 57.98s / 1021 steps = 56.8ms/step
    "c6288": 57.98 / 1021 * 1000,  # 56.8 ms/step
}


from vajax.utils import find_vacask_binary

# VA source locations for OSDI compilation
# Maps OSDI filename patterns to VA source paths (relative to project root)
VA_SOURCES = {
    "psp103_ihp": "vendor/VACASK/devices/psp103v4/psp103_nqs.va",
    "psp103v4": "vendor/VACASK/devices/psp103v4/psp103.va",
    "spice/resistor": "vendor/VACASK/devices/spice/resistor.va",
    "spice/capacitor": "vendor/VACASK/devices/spice/capacitor.va",
    "spice/inductor": "vendor/VACASK/devices/spice/inductor.va",
    "spice/diode": "vendor/VACASK/devices/spice/diode.va",
    "resistor": "vendor/VACASK/devices/resistor.va",
    "capacitor": "vendor/VACASK/devices/capacitor.va",
    "diode": "vendor/VACASK/devices/diode.va",
}

# Project root (where vendor/ lives)
PROJECT_ROOT = Path(__file__).parent.parent


def _find_openvaf_r() -> Optional[Path]:
    """Find the openvaf-r compiler binary."""
    # Check OPENVAF_DIR env var first (set in CI)
    openvaf_dir = os.environ.get("OPENVAF_DIR")
    if openvaf_dir:
        binary = Path(openvaf_dir) / "openvaf-r"
        if binary.exists():
            return binary

    # Check common build locations
    for candidate in [
        PROJECT_ROOT / "vendor" / "OpenVAF" / "target" / "release" / "openvaf-r",
    ]:
        if candidate.exists():
            return candidate

    # Check PATH
    import shutil

    path_bin = shutil.which("openvaf-r")
    if path_bin:
        return Path(path_bin)

    return None


def ensure_osdi_files(config: BenchmarkInfo) -> bool:
    """Compile any missing OSDI files needed by a benchmark's sim file.

    Parses the sim file for load directives, checks if each OSDI file exists,
    and compiles missing ones using openvaf-r.

    Returns True if all OSDI files are available, False if compilation failed.
    """
    sim_dir = config.sim_path.parent

    # Parse load directives from sim file
    with open(config.sim_path) as f:
        sim_content = f.read()

    osdi_files = re.findall(r'load\s*"([^"]+\.osdi)"', sim_content)
    if not osdi_files:
        return True  # No OSDI files needed

    # VACASK also searches its module directory (<binary>/../lib/vacask/mod)
    vacask_bin = find_vacask_binary()
    vacask_mod_dir = None
    if vacask_bin:
        vacask_mod_dir = Path(vacask_bin).parent.parent / "lib" / "vacask" / "mod"

    # Check which files are missing from both the sim dir and VACASK mod dir
    missing = []
    for osdi_file in osdi_files:
        osdi_path = sim_dir / osdi_file
        if osdi_path.exists():
            continue
        if vacask_mod_dir and (vacask_mod_dir / osdi_file).exists():
            continue
        missing.append(osdi_file)

    if not missing:
        return True  # All OSDI files present

    # Find openvaf-r compiler
    openvaf_r = _find_openvaf_r()
    if openvaf_r is None:
        print(f"    Warning: openvaf-r not found, cannot compile: {missing}")
        return False

    devices_dir = PROJECT_ROOT / "vendor" / "VACASK" / "devices"

    for osdi_file in missing:
        # Map OSDI filename to VA source
        # Normalize path (strip leading ./)
        osdi_stem = Path(osdi_file).with_suffix("").as_posix()
        if osdi_stem not in VA_SOURCES:
            print(f"    Warning: no VA source mapping for {osdi_file}")
            return False

        va_source = PROJECT_ROOT / VA_SOURCES[osdi_stem]
        if not va_source.exists():
            print(f"    Warning: VA source not found: {va_source}")
            return False

        osdi_path = sim_dir / osdi_file
        # Create parent directory if needed (e.g., spice/)
        osdi_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"    Compiling {va_source.name} -> {osdi_file}...", end=" ", flush=True)
        try:
            result = subprocess.run(
                [
                    str(openvaf_r),
                    "--allow",
                    "variant_const_simparam",
                    f"-I{devices_dir}",
                    str(va_source),
                    "-o",
                    str(osdi_path),
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode != 0:
                print(f"failed (rc={result.returncode})")
                if result.stderr:
                    print(f"    {result.stderr[:200]}")
                return False
            print("done")
        except Exception as e:
            print(f"error: {e}")
            return False

    return True


def run_vacask(config: BenchmarkInfo, num_steps: int) -> Optional[Tuple[float, float]]:
    """Run VACASK and return (time_per_step_ms, wall_time_s).

    Returns None if VACASK is not available or fails.
    """
    vacask_bin = find_vacask_binary()
    if vacask_bin is None:
        return None

    # Create a temporary sim file with modified stop time
    t_stop = config.dt * num_steps
    sim_dir = config.sim_path.parent

    # Read original sim file
    with open(config.sim_path) as f:
        sim_content = f.read()

    # Modify the analysis line to use our t_stop
    # Pattern: analysis <name> tran step=... stop=... maxstep=...
    modified = re.sub(
        r"(analysis\s+\w+\s+tran\s+.*?stop=)[^\s]+", f"\\g<1>{t_stop:.2e}", sim_content
    )

    # Write to temp file
    temp_sim = sim_dir / "compare_temp.sim"
    with open(temp_sim, "w") as f:
        f.write(modified)

    try:
        # Run VACASK with temp sim file
        start = time.perf_counter()
        result = subprocess.run(
            [str(vacask_bin), "compare_temp.sim"],  # Use relative path in cwd
            cwd=sim_dir,
            capture_output=True,
            text=True,
            timeout=300,
        )
        time.perf_counter() - start

        # Parse output - VACASK may output Python errors from post-processing scripts
        # but still succeed. Look for elapsed time in stdout.

        # Check for VACASK elapsed time (most reliable)
        elapsed_match = re.search(r"Elapsed time:\s*([\d.]+)", result.stdout)
        if elapsed_match:
            vacask_elapsed = float(elapsed_match.group(1))
            # Use the number of timesteps we requested for fair comparison
            # NR solver calls != timesteps (multiple NR iters per timestep)
            time_per_step = vacask_elapsed / num_steps * 1000  # ms per timestep
            return time_per_step, vacask_elapsed
        elif result.returncode != 0:
            # Only report failure if no elapsed time found AND returncode is non-zero
            print(f"VACASK failed (rc={result.returncode}): {result.stdout[:200]}")
            return None
        else:
            print("VACASK: could not parse output")
            return None

    except subprocess.TimeoutExpired:
        print("VACASK timed out")
        return None
    except Exception as e:
        print(f"VACASK error: {e}")
        return None
    finally:
        # Clean up temp file
        if temp_sim.exists():
            temp_sim.unlink()


def run_vajax(
    config: BenchmarkInfo,
    num_steps: int,
    use_scan: bool,
    use_sparse: bool = False,
    force_gpu: bool = False,
    profile_config: Optional[ProfileConfig] = None,
    profile_full: bool = False,
    analyze: bool = False,
    analyze_output_dir: Optional[Path] = None,
) -> Tuple[float, float, Dict]:
    """Run VAJAX and return (time_per_step_ms, wall_time_s, stats).

    Args:
        config: Benchmark configuration
        num_steps: Number of timesteps
        use_scan: Use lax.scan for faster execution
        use_sparse: Use sparse solver
        force_gpu: Force GPU backend even for small circuits
        profile_config: If provided, profile the simulation
        profile_full: If True, profile entire run including warmup (helps identify overhead)
        analyze: If True, dump jaxpr and cost analysis for compiled functions
        analyze_output_dir: Directory to save analysis files (optional)
    """
    from vajax.profiling import profile_section

    # Create benchmark-specific profile config if profiling is enabled
    full_profile_config = None
    if profile_config:
        base_dir = Path(profile_config.trace_dir) / f"benchmark_{config.name}"
        if profile_full:
            # Profile entire run - don't also profile simulation to avoid nested profiles
            full_profile_config = ProfileConfig(
                jax=profile_config.jax,
                cuda=profile_config.cuda,
                trace_dir=str(base_dir / "full_run"),
                create_perfetto_link=profile_config.create_perfetto_link,
            )

    def do_run():
        # Use CircuitEngine API
        engine = CircuitEngine(config.sim_path)
        engine.parse()

        t_stop = config.dt * num_steps

        # Determine backend
        backend = "gpu" if force_gpu else None  # None = auto-select

        # Prepare (includes 1-step warmup for JIT compilation)
        # Let prepare() build AdaptiveConfig from netlist options (lte_ratio, etc.)
        startup_start = time.perf_counter()
        engine.prepare(
            t_stop=t_stop,
            dt=config.dt,
            use_sparse=use_sparse,
            backend=backend,
        )
        startup_time = time.perf_counter() - startup_start

        # Run analysis on compiled functions if requested
        if analyze:
            out_dir = analyze_output_dir or Path(f"/tmp/claude/jaxpr-analysis/{config.name}")
            print(f"\n  Dumping jaxpr/HLO analysis to {out_dir} ...")
            engine.dump_jaxpr(out_dir)

        # Timed run - print perf_counter for correlation with Perfetto traces
        # prepare() already called above with same params, strategy is cached
        start = time.perf_counter()
        print(f"TIMED_RUN_START: {start:.6f}")
        result = engine.run_transient()
        after_transient = time.perf_counter()
        print(
            f"AFTER_RUN_TRANSIENT: {after_transient:.6f} (elapsed: {after_transient - start:.6f}s)"
        )
        # Force completion of async JAX operations
        first_node = next(iter(result.voltages))
        _ = float(result.voltages[first_node][0])
        end = time.perf_counter()
        external_elapsed = end - start
        print(
            f"TIMED_RUN_END: {end:.6f} (elapsed: {external_elapsed:.6f}s, sync took: {end - after_transient:.6f}s)"
        )

        # Use wall_time from stats (excludes trace saving overhead)
        # Fall back to external timing if not available
        stats = result.stats
        elapsed = stats.get("wall_time", external_elapsed)
        actual_steps = result.num_steps - 1  # Exclude t=0 initial condition
        time_per_step = elapsed / actual_steps * 1000

        # Add startup time to stats
        stats["startup_time"] = startup_time

        return time_per_step, elapsed, stats

    if full_profile_config:
        with profile_section("full_benchmark", full_profile_config):
            return do_run()
    else:
        return do_run()


def main():
    parser = argparse.ArgumentParser(description="Compare VAJAX vs VACASK")
    parser.add_argument(
        "--benchmark",
        type=str,
        default=None,
        help="Comma-separated list of benchmarks (default: all except c6288)",
    )
    parser.add_argument(
        "--use-scan",
        action="store_true",
        help="Use lax.scan for VAJAX (faster)",
    )
    parser.add_argument(
        "--use-sparse",
        action="store_true",
        help="Use sparse solver (auto-enabled for c6288 unless --force-dense)",
    )
    parser.add_argument(
        "--force-dense",
        action="store_true",
        help="Force dense solver even for large circuits (for GPU performance testing)",
    )
    parser.add_argument(
        "--force-gpu",
        action="store_true",
        help="Force GPU backend even for small circuits (ignores gpu_threshold)",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="[DEPRECATED] Use --profile-mode=jax instead",
    )
    parser.add_argument(
        "--profile-mode",
        type=str,
        choices=["none", "jax", "nsys", "both"],
        default="none",
        help="Profiling mode: none, jax (Perfetto traces), nsys (nsys-jax), or both",
    )
    parser.add_argument(
        "--profile-dir",
        type=str,
        default="/tmp/vajax-traces",
        help="Directory for profiling traces (default: /tmp/vajax-traces)",
    )
    parser.add_argument(
        "--profile-full",
        action="store_true",
        help="Profile entire run including warmup/setup (helps identify overhead)",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Dump JAX cost analysis and jaxpr for compiled functions",
    )
    parser.add_argument(
        "--json-output",
        type=str,
        default=None,
        help="Write benchmark results as JSON to this path",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("VAJAX vs VACASK Performance Comparison")
    print("=" * 70)
    print()

    # Check for VACASK
    vacask_bin = find_vacask_binary()
    if vacask_bin:
        print(f"VACASK binary: {vacask_bin}")
    else:
        print("VACASK binary: NOT FOUND (set VACASK_BIN or build in vendor/VACASK/build)")
    print(f"JAX backend: {jax.default_backend()}")
    print(f"Mode: {'lax.scan' if args.use_scan else 'Python loop'}")
    print(f"Solver: {'sparse' if args.use_sparse else 'dense'}")
    print(f"Backend: {'GPU (forced)' if args.force_gpu else 'auto-select'}")
    print("Steps: full netlist duration")

    # Handle deprecated --profile flag
    profile_mode = args.profile_mode
    if args.profile and profile_mode == "none":
        print("Warning: --profile is deprecated, use --profile-mode=jax instead")
        profile_mode = "jax"

    # Check if we're running under nsys-jax (it sets this env var)
    running_under_nsys = os.environ.get("NSYS_PROFILING_SESSION_ID") is not None

    # For nsys mode, re-exec under nsys-jax if not already running under it
    if profile_mode in ("nsys", "both") and not running_under_nsys:
        nsys_output = Path(args.profile_dir) / "nsys-jax-output"
        nsys_output.mkdir(parents=True, exist_ok=True)

        # Build nsys-jax command
        nsys_cmd = [
            "nsys-jax",
            "-o",
            str(nsys_output / "profile"),
            "--capture-range=cudaProfilerApi",
            "--capture-range-end=stop",
            "python",
            sys.argv[0],
        ]
        # Pass through arguments, but change profile-mode to jax if both
        for i, arg in enumerate(sys.argv[1:]):
            if arg == "--profile-mode":
                continue  # Skip, we'll add our own
            if i > 0 and sys.argv[i] == "--profile-mode":
                continue  # Skip the value too
            nsys_cmd.append(arg)

        # If mode is 'both', run JAX profiling in this process first
        if profile_mode == "both":
            print("Running JAX profiling first, then nsys-jax...")
            nsys_cmd.extend(["--profile-mode", "none"])  # nsys run won't do JAX profiling
        else:
            nsys_cmd.extend(["--profile-mode", "none"])  # Just use CUDA markers

        # Mark that we want CUDA profiler markers
        os.environ["VAJAX_NSYS_MARKERS"] = "1"

        print(f"Re-executing under nsys-jax: {' '.join(nsys_cmd[:5])}...")
        result = subprocess.run(nsys_cmd)
        if result.returncode != 0:
            print(f"nsys-jax failed with code {result.returncode}")
            sys.exit(result.returncode)

        print(f"\nnsys-jax output saved to: {nsys_output}")
        print("To analyze, unzip and open Jupyter notebooks:")
        print(f"  unzip {nsys_output}/profile.zip -d {nsys_output}/")
        print(f"  jupyter notebook {nsys_output}/")

        if profile_mode == "nsys":
            sys.exit(0)  # Done, nsys-jax already ran the benchmarks
        # If 'both', continue to run JAX profiling below

    # Setup JAX profiling if requested
    profile_config = None
    use_cuda_markers = os.environ.get("VAJAX_NSYS_MARKERS") == "1"

    if profile_mode in ("jax", "both") or use_cuda_markers:
        has_gpu = any(d.platform != "cpu" for d in jax.devices())
        # JAX profiling works on both CPU and GPU
        enable_jax = profile_mode in ("jax", "both")
        # CUDA markers only make sense with GPU
        enable_cuda = has_gpu and (use_cuda_markers or running_under_nsys)

        profile_config = ProfileConfig(jax=enable_jax, cuda=enable_cuda, trace_dir=args.profile_dir)
        if enable_jax:
            print(f"JAX Profiling: ENABLED (traces -> {args.profile_dir})")
        if enable_cuda:
            print("CUDA Profiler Markers: ENABLED (for nsys-jax)")
    print()

    # Select benchmarks
    if args.benchmark:
        benchmark_names = [b.strip() for b in args.benchmark.split(",")]
    else:
        # Default: all discovered benchmarks
        benchmark_names = list_benchmarks()

    # Run benchmarks
    results = []

    # Memory profiling setup
    import gc
    import tracemalloc

    if not tracemalloc.is_tracing():
        tracemalloc.start()
    prev_snapshot = tracemalloc.take_snapshot()

    for name in benchmark_names:
        config = get_benchmark(name)
        if config is None:
            print(f"Unknown benchmark: {name}")
            continue

        if config.skip:
            print(f"Skipping {name}: {config.skip_reason}")
            continue

        num_steps = int(config.t_stop / config.dt)
        if config.max_steps is not None and num_steps > config.max_steps:
            num_steps = config.max_steps

        print(f"--- {name} ({num_steps} steps, dt={config.dt:.2e}) ---")

        # Run VAJAX
        # Auto-enable sparse for large circuits unless --force-dense is specified
        use_sparse = args.use_sparse or (config.is_large and not args.force_dense)
        print("  VAJAX warmup...", end=" ", flush=True)
        analyze_dir = Path(args.profile_dir) / "analysis" if args.analyze else None
        jax_ms, jax_wall, stats = run_vajax(
            config,
            num_steps,
            args.use_scan,
            use_sparse=use_sparse,
            force_gpu=args.force_gpu,
            profile_config=profile_config,
            profile_full=args.profile_full,
            analyze=args.analyze,
            analyze_output_dir=analyze_dir,
        )
        startup_time = stats.get("startup_time", 0)
        print("done")
        print(
            f"  VAJAX: {jax_ms:.3f} ms/step ({jax_wall:.3f}s total, startup: {startup_time:.1f}s)"
        )

        # Run VACASK or use reference times
        vacask_ms = None
        vacask_wall = None
        vacask_source = None  # 'run', 'reference', or None
        vacask_note = None  # Reason VACASK result is unavailable
        if vacask_bin:
            # Compile any missing OSDI files before running VACASK
            if not ensure_osdi_files(config):
                vacask_note = "OSDI compile failed"
                print("  VACASK: skipped (OSDI compilation failed)")
            else:
                print("  VACASK...", end=" ", flush=True)
                vacask_result = run_vacask(config, num_steps)
                if vacask_result is not None:
                    vacask_ms, vacask_wall = vacask_result
                    vacask_source = "run"
                    print("done")
                    print(f"  VACASK: {vacask_ms:.3f} ms/step ({vacask_wall:.3f}s total)")
                    ratio = jax_ms / vacask_ms
                    print(f"  Ratio: {ratio:.2f}x {'slower' if ratio > 1 else 'faster'}")
                else:
                    vacask_note = "crashed"
                    print("failed")
        elif name in VACASK_REFERENCE_TIMES:
            # Use published reference times when VACASK binary isn't available
            vacask_ms = VACASK_REFERENCE_TIMES[name]
            vacask_wall = vacask_ms * num_steps / 1000  # Convert to seconds
            vacask_source = "reference"
            print(f"  VACASK (reference): {vacask_ms:.3f} ms/step (AMD Threadripper 7970)")
            ratio = jax_ms / vacask_ms
            print(f"  Ratio: {ratio:.2f}x {'slower' if ratio > 1 else 'faster'}")
        else:
            vacask_note = "no benchmark"
            print("  VACASK: not in VACASK benchmark suite")

        results.append(
            {
                "name": name,
                "steps": num_steps,
                "jax_ms": jax_ms,
                "jax_wall": jax_wall,
                "startup_time": startup_time,
                "vacask_ms": vacask_ms,
                "vacask_wall": vacask_wall,
                "vacask_source": vacask_source,
                "vacask_note": vacask_note,
            }
        )

        # Memory analysis after each benchmark
        gc.collect()  # Force GC before snapshot
        current_snapshot = tracemalloc.take_snapshot()
        current_mem, _ = tracemalloc.get_traced_memory()
        print(f"  Memory: {current_mem / 1024 / 1024:.1f}MB total")

        # Show top memory diffs from previous benchmark
        top_diffs = current_snapshot.compare_to(prev_snapshot, "lineno")
        if top_diffs:
            increases = [d for d in top_diffs if d.size_diff > 0]
            if increases[:3]:
                print("  Top memory increases since last benchmark:")
                for stat in increases[:3]:
                    print(
                        f"    +{stat.size_diff / 1024:.1f}KB: {stat.traceback.format()[0] if stat.traceback else 'unknown'}"
                    )
        prev_snapshot = current_snapshot
        print()

    # Summary table
    has_reference = any(r.get("vacask_source") == "reference" for r in results)
    print("=" * 70)
    print("Summary (per-step timing)")
    print("=" * 70)
    print()
    print("| Benchmark | Steps | VAJAX (ms) | VACASK (ms)   | Ratio |")
    print("|-----------|-------|----------------|---------------|-------|")
    for r in results:
        if r["vacask_ms"]:
            # Add asterisk for reference times
            ref_marker = "*" if r.get("vacask_source") == "reference" else ""
            vacask_str = f"{r['vacask_ms']:.3f}{ref_marker}"
            ratio = r["jax_ms"] / r["vacask_ms"]
            ratio_str = f"{ratio:.2f}x"
        else:
            note = r.get("vacask_note", "N/A")
            vacask_str = note
            ratio_str = ""
        print(
            f"| {r['name']:9} | {r['steps']:5} | {r['jax_ms']:14.3f} | {vacask_str:13} | {ratio_str:5} |"
        )

    print()
    print("=" * 70)
    print("Summary (total wall time)")
    print("=" * 70)
    print()
    print("| Benchmark | Steps | VAJAX (ms) | VACASK (ms)   | Ratio |")
    print("|-----------|-------|----------------|---------------|-------|")
    for r in results:
        # Convert wall time from seconds to milliseconds for better precision
        jax_wall_ms = r["jax_wall"] * 1000
        if r["vacask_wall"]:
            vacask_wall_ms = r["vacask_wall"] * 1000
            ref_marker = "*" if r.get("vacask_source") == "reference" else ""
            vacask_str = f"{vacask_wall_ms:.1f}{ref_marker}"
            ratio = r["jax_wall"] / r["vacask_wall"]
            ratio_str = f"{ratio:.2f}x"
        else:
            note = r.get("vacask_note", "N/A")
            vacask_str = note
            ratio_str = ""
        print(
            f"| {r['name']:9} | {r['steps']:5} | {jax_wall_ms:14.1f} | {vacask_str:13} | {ratio_str:5} |"
        )

    if has_reference:
        print()
        print("* VACASK reference times from AMD Threadripper 7970 (single-threaded)")

    # Startup time summary (shows JIT compilation overhead and breakeven analysis)
    print()
    print("=" * 70)
    print("Startup overhead (JIT compilation)")
    print("=" * 70)
    print()
    print("| Benchmark | Startup (s) | Per-step speedup | Breakeven steps |")
    print("|-----------|-------------|------------------|-----------------|")
    for r in results:
        startup_s = r.get("startup_time", 0)
        if r["vacask_ms"] and r["jax_ms"] < r["vacask_ms"]:
            # JAX is faster per-step, calculate breakeven
            # breakeven = startup / (vacask_time - jax_time) per step
            speedup_per_step_ms = r["vacask_ms"] - r["jax_ms"]
            breakeven = (
                int(startup_s * 1000 / speedup_per_step_ms)
                if speedup_per_step_ms > 0
                else float("inf")
            )
            speedup_str = f"{r['vacask_ms'] / r['jax_ms']:.1f}x faster"
            breakeven_str = f"{breakeven:,}"
        elif r["vacask_ms"]:
            # JAX is slower per-step
            speedup_str = f"{r['jax_ms'] / r['vacask_ms']:.1f}x slower"
            breakeven_str = "N/A (slower)"
        else:
            note = r.get("vacask_note", "N/A")
            speedup_str = note
            breakeven_str = ""
        print(f"| {r['name']:9} | {startup_s:11.1f} | {speedup_str:16} | {breakeven_str:15} |")

    print()

    # Write JSON output if requested
    if args.json_output:
        json_results = []
        for r in results:
            ratio = r["jax_ms"] / r["vacask_ms"] if r["vacask_ms"] else None
            json_results.append(
                {
                    "name": r["name"],
                    "steps": r["steps"],
                    "jax_ms_per_step": round(r["jax_ms"], 6),
                    "vacask_ms_per_step": round(r["vacask_ms"], 6) if r["vacask_ms"] else None,
                    "ratio": round(ratio, 2) if ratio else None,
                    "vacask_source": r.get("vacask_source"),
                    "vacask_note": r.get("vacask_note"),
                    "startup_s": round(r.get("startup_time", 0), 1),
                    "backend": str(jax.default_backend()),
                }
            )
        json_path = Path(args.json_output)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(json_results, f, indent=2)
        print(f"\nJSON results written to: {json_path}")

    # Report profiling traces location
    if profile_config and profile_config.jax:
        trace_dir = Path(args.profile_dir)
        if trace_dir.exists():
            # Find JAX traces (directories with .xplane.pb files)
            jax_traces = list(trace_dir.glob("benchmark_*"))
            if jax_traces:
                print(f"JAX profiling traces saved to: {trace_dir}")
                print("  To view in Perfetto: https://ui.perfetto.dev/")
                for t in jax_traces[:5]:  # Show first 5
                    print(f"    - {t.name}/")
                if len(jax_traces) > 5:
                    print(f"    ... and {len(jax_traces) - 5} more")

            # Check for nsys-jax output
            nsys_output = trace_dir / "nsys-jax-output"
            if nsys_output.exists():
                nsys_zips = list(nsys_output.glob("*.zip"))
                if nsys_zips:
                    print(f"\nnsys-jax traces saved to: {nsys_output}")
                    print("  To analyze:")
                    print(f"    unzip {nsys_zips[0]} -d {nsys_output}/analysis")
                    print(f"    jupyter notebook {nsys_output}/analysis/")
    print("=" * 70)


if __name__ == "__main__":
    main()
