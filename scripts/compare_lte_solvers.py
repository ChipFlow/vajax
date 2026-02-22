"""Compare per-step LTE diagnostics between UMFPACK (CPU) and cuDSS (GPU) solvers.

This script captures per-step transient simulation data (time, dt, NR iterations,
LTE norm, accept/reject decisions) and saves them for comparison. This helps
diagnose why the GPU cuDSS solver produces different timestep behavior from
the CPU UMFPACK solver on the same circuit.

Usage:
    # Run locally (CPU/UMFPACK) and save trace:
    JAX_PLATFORMS=cpu JAX_SPICE_NO_PROGRESS=1 uv run python scripts/compare_lte_solvers.py \
        --benchmark ring --output /tmp/lte_trace_umfpack.json

    # Run on GPU (cuDSS) and save trace:
    JAX_PLATFORMS=cuda JAX_SPICE_NO_PROGRESS=1 uv run python scripts/compare_lte_solvers.py \
        --benchmark ring --output /tmp/lte_trace_cudss.json

    # Compare two traces:
    uv run python scripts/compare_lte_solvers.py \
        --compare /tmp/lte_trace_umfpack.json /tmp/lte_trace_cudss.json

    # Run locally with both dense and sparse (UMFPACK) and compare:
    JAX_PLATFORMS=cpu JAX_SPICE_NO_PROGRESS=1 uv run python scripts/compare_lte_solvers.py \
        --benchmark ring --compare-local
"""

import argparse
import json
import logging
import re
import sys
import time
from io import StringIO
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
    stream=sys.stderr,  # All logging to stderr, keep stdout for debug_steps
)
logger = logging.getLogger(__name__)


class TeeWriter:
    """Write to both the original stream and a capture buffer."""

    def __init__(self, original):
        self.original = original
        self.captured = StringIO()

    def write(self, text):
        self.original.write(text)
        self.captured.write(text)
        return len(text)

    def flush(self):
        self.original.flush()

    def get_captured(self) -> str:
        return self.captured.getvalue()


def parse_debug_output(text: str) -> list[dict]:
    """Parse debug_steps output into structured per-step records.

    Expected format from body_fn debug callback:
        Step   3: t=   1.000ps dt=  0.5000ps NR= 4 res=1.23e-13 LTE/tol=0.45       dt_lte=  1.2000ps [accept]
        Step   4: t=   1.500ps dt=  0.5000ps NR= 7 res=4.56e-10 LTE/tol=3.2 → REJECT dt_lte=  0.2000ps [REJECT]
    """
    records = []
    pattern = re.compile(
        r"Step\s+(\d+):\s+"
        r"t=\s*([\d.]+)ps\s+"
        r"dt=\s*([\d.]+)ps\s+"
        r"NR=\s*(\d+)\s+"
        r"res=([\d.eE+-]+)\s+"
        r"(.*?)\s+"
        r"dt_lte=\s*([\d.]+)ps\s+"
        r"\[(accept|REJECT)\]"
        r"(.*)"
    )
    # Fallback pattern for old format without dt_lte
    pattern_old = re.compile(
        r"Step\s+(\d+):\s+"
        r"t=\s*([\d.]+)ps\s+"
        r"dt=\s*([\d.]+)ps\s+"
        r"NR=\s*(\d+)\s+"
        r"res=([\d.eE+-]+)\s+"
        r"(.*?)\s*"
        r"\[(accept|REJECT)\]"
    )

    for line in text.split("\n"):
        m = pattern.search(line)
        dt_lte_ps = None
        nr_failed = False

        if m:
            step = int(m.group(1))
            t_ps = float(m.group(2))
            dt_ps = float(m.group(3))
            nr_iters = int(m.group(4))
            residual = float(m.group(5))
            lte_info = m.group(6).strip()
            dt_lte_ps = float(m.group(7))
            status = m.group(8)
            extra = m.group(9).strip()
            nr_failed = "NR_FAIL" in extra
        else:
            m = pattern_old.search(line)
            if not m:
                continue
            step = int(m.group(1))
            t_ps = float(m.group(2))
            dt_ps = float(m.group(3))
            nr_iters = int(m.group(4))
            residual = float(m.group(5))
            lte_info = m.group(6).strip()
            status = m.group(7)

        # Parse LTE value
        lte_norm = None
        lte_reject = False
        if "Cannot estimate" in lte_info:
            lte_norm = None
        else:
            lte_match = re.search(r"LTE/tol=([\d.]+)", lte_info)
            if lte_match:
                lte_norm = float(lte_match.group(1))
            if "REJECT" in lte_info:
                lte_reject = True

        records.append({
            "step": step,
            "t_ps": t_ps,
            "dt_ps": dt_ps,
            "dt_lte_ps": dt_lte_ps,
            "nr_iters": nr_iters,
            "residual": residual,
            "lte_norm": lte_norm,
            "lte_reject": lte_reject,
            "nr_failed": nr_failed,
            "accepted": status == "accept",
        })

    return records


def run_simulation(benchmark_name: str, use_sparse: bool) -> tuple[list[dict], dict]:
    """Run a benchmark simulation with debug_steps enabled and capture per-step data.

    Uses sys.stdout tee to capture jax.debug.callback print() output while
    still showing it on screen.

    Args:
        benchmark_name: Name of the benchmark (e.g., "ring", "c6288")
        use_sparse: Whether to use sparse solver

    Returns:
        Tuple of (per-step records, summary stats)
    """
    import jax

    from jax_spice.analysis.engine import CircuitEngine
    from jax_spice.analysis.transient.adaptive import AdaptiveConfig
    from jax_spice.benchmarks.registry import get_benchmark

    info = get_benchmark(benchmark_name)
    if info is None:
        logger.error(f"Unknown benchmark: {benchmark_name}")
        sys.exit(1)

    backend = jax.default_backend()
    solver_name = "unknown"
    if use_sparse:
        if backend in ("cuda", "gpu"):
            solver_name = "cuDSS/Spineax"
        else:
            solver_name = "UMFPACK"
    else:
        solver_name = "dense"

    logger.info(f"Running '{benchmark_name}' with {solver_name} solver (backend={backend})")

    engine = CircuitEngine(info.sim_path)
    engine.parse()

    # Enable debug_steps to get per-step callback output
    config = AdaptiveConfig(debug_steps=True)

    engine.prepare(use_sparse=use_sparse, adaptive_config=config)

    # Tee stdout so we capture jax.debug.callback's print() output
    # while still showing it on screen (useful for monitoring long runs)
    tee = TeeWriter(sys.stdout)
    original_stdout = sys.stdout
    sys.stdout = tee

    t_start = time.perf_counter()
    try:
        result = engine.run_transient()
    finally:
        sys.stdout = original_stdout

    wall_time = time.perf_counter() - t_start

    # Parse captured debug output
    captured_text = tee.get_captured()
    records = parse_debug_output(captured_text)

    logger.info(f"Captured {len(records)} step records from debug output")

    # Build summary
    accepted = [r for r in records if r["accepted"]]
    rejected = [r for r in records if not r["accepted"]]
    lte_values = [r["lte_norm"] for r in records if r["lte_norm"] is not None]

    summary = {
        "benchmark": benchmark_name,
        "solver": solver_name,
        "backend": str(backend),
        "use_sparse": use_sparse,
        "wall_time_s": wall_time,
        "total_steps_attempted": len(records),
        "accepted_steps": len(accepted),
        "rejected_steps": len(rejected),
        "num_steps_result": result.num_steps,
        "lte_stats": {
            "count": len(lte_values),
            "min": min(lte_values) if lte_values else None,
            "max": max(lte_values) if lte_values else None,
            "mean": sum(lte_values) / len(lte_values) if lte_values else None,
        },
        "dt_stats": {
            "min_ps": min(r["dt_ps"] for r in records) if records else None,
            "max_ps": max(r["dt_ps"] for r in records) if records else None,
        },
        "nr_stats": {
            "min_iters": min(r["nr_iters"] for r in records) if records else None,
            "max_iters": max(r["nr_iters"] for r in records) if records else None,
            "mean_iters": (
                sum(r["nr_iters"] for r in records) / len(records) if records else None
            ),
        },
    }

    return records, summary


def save_trace(records: list[dict], summary: dict, output_path: str) -> None:
    """Save per-step trace to JSON file."""
    data = {
        "summary": summary,
        "steps": records,
    }
    Path(output_path).write_text(json.dumps(data, indent=2, default=str))
    logger.info(f"Saved {len(records)} step records to {output_path}")


def load_trace(path: str) -> tuple[list[dict], dict]:
    """Load per-step trace from JSON file."""
    data = json.loads(Path(path).read_text())
    return data["steps"], data["summary"]


def print_summary(summary: dict) -> None:
    """Print summary of a simulation run to stderr."""
    logger.info(
        f"  {summary['benchmark']} ({summary['solver']}, {summary['backend']}): "
        f"{summary['accepted_steps']} accepted, {summary['rejected_steps']} rejected, "
        f"{summary['wall_time_s']:.1f}s"
    )
    if summary["lte_stats"]["mean"] is not None:
        logger.info(
            f"  LTE norm: min={summary['lte_stats']['min']:.4f}, "
            f"max={summary['lte_stats']['max']:.4f}, "
            f"mean={summary['lte_stats']['mean']:.4f}"
        )
    if summary["dt_stats"]["min_ps"] is not None:
        logger.info(
            f"  dt range: [{summary['dt_stats']['min_ps']:.6f}, "
            f"{summary['dt_stats']['max_ps']:.6f}] ps"
        )


def compare_traces(path_a: str, path_b: str) -> None:
    """Compare two per-step LTE traces and report divergence points."""
    records_a, summary_a = load_trace(path_a)
    records_b, summary_b = load_trace(path_b)

    label_a = f"{summary_a['solver']} ({summary_a['backend']})"
    label_b = f"{summary_b['solver']} ({summary_b['backend']})"

    print(f"\n{'='*80}")
    print(f"LTE Trace Comparison: {summary_a['benchmark']}")
    print(f"{'='*80}")
    print(f"  A: {label_a} - {path_a}")
    print(f"  B: {label_b} - {path_b}")
    print()

    # Summary comparison
    print(f"{'Metric':<30} {'A':>15} {'B':>15} {'Diff':>15}")
    print(f"{'-'*75}")

    for key in ["total_steps_attempted", "accepted_steps", "rejected_steps", "num_steps_result"]:
        va = summary_a.get(key, "N/A")
        vb = summary_b.get(key, "N/A")
        diff = ""
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            diff = f"{vb - va:+d}" if isinstance(va, int) else f"{vb - va:+.4f}"
        print(f"{key:<30} {str(va):>15} {str(vb):>15} {diff:>15}")

    print(f"\n{'LTE Stats':<30} {'A':>15} {'B':>15}")
    print(f"{'-'*60}")
    for key in ["min", "max", "mean"]:
        va = summary_a["lte_stats"].get(key)
        vb = summary_b["lte_stats"].get(key)
        va_str = f"{va:.4f}" if va is not None else "N/A"
        vb_str = f"{vb:.4f}" if vb is not None else "N/A"
        print(f"  lte_{key:<26} {va_str:>15} {vb_str:>15}")

    print(f"\n{'dt Stats (ps)':<30} {'A':>15} {'B':>15}")
    print(f"{'-'*60}")
    for key in ["min_ps", "max_ps"]:
        va = summary_a["dt_stats"].get(key)
        vb = summary_b["dt_stats"].get(key)
        va_str = f"{va:.6f}" if va is not None else "N/A"
        vb_str = f"{vb:.6f}" if vb is not None else "N/A"
        print(f"  {key:<28} {va_str:>15} {vb_str:>15}")

    # Find accepted steps in each trace (these are the ones that advance time)
    accepted_a = [r for r in records_a if r["accepted"]]
    accepted_b = [r for r in records_b if r["accepted"]]

    # Compare accepted steps side-by-side
    print(f"\n{'='*80}")
    print("Step-by-step comparison (accepted steps only)")
    print(f"{'='*80}")
    print(
        f"{'Step':>5} {'t_A (ps)':>12} {'t_B (ps)':>12} {'dt_A (ps)':>12} {'dt_B (ps)':>12} "
        f"{'LTE_A':>8} {'LTE_B':>8} {'NR_A':>5} {'NR_B':>5} {'t_diff':>12}"
    )
    print(f"{'-'*100}")

    # Walk through accepted steps and find divergence
    max_compare = min(len(accepted_a), len(accepted_b))
    first_divergence = None
    DIVERGENCE_THRESHOLD = 0.01  # 1% relative difference in time

    for i in range(min(max_compare, 200)):  # Show first 200 steps
        ra = accepted_a[i]
        rb = accepted_b[i]

        t_diff = rb["t_ps"] - ra["t_ps"]
        t_rel_diff = abs(t_diff) / max(abs(ra["t_ps"]), 1e-30)

        lte_a = f"{ra['lte_norm']:.2f}" if ra["lte_norm"] is not None else "N/A"
        lte_b = f"{rb['lte_norm']:.2f}" if rb["lte_norm"] is not None else "N/A"

        # Flag divergence
        marker = ""
        if t_rel_diff > DIVERGENCE_THRESHOLD and first_divergence is None:
            first_divergence = i
            marker = " <-- DIVERGENCE"

        # Only print every 10th step, or divergence points, or first/last 5
        show = (
            i < 5
            or i >= max_compare - 5
            or i % 10 == 0
            or (first_divergence is not None and i - first_divergence < 10)
        )
        if show:
            print(
                f"{i:5d} {ra['t_ps']:12.4f} {rb['t_ps']:12.4f} "
                f"{ra['dt_ps']:12.6f} {rb['dt_ps']:12.6f} "
                f"{lte_a:>8} {lte_b:>8} "
                f"{ra['nr_iters']:5d} {rb['nr_iters']:5d} "
                f"{t_diff:12.6f}{marker}"
            )

    if first_divergence is not None:
        print(f"\nFirst significant divergence at accepted step {first_divergence}")
        print(
            f"  A: t={accepted_a[first_divergence]['t_ps']:.6f}ps, "
            f"dt={accepted_a[first_divergence]['dt_ps']:.6f}ps"
        )
        print(
            f"  B: t={accepted_b[first_divergence]['t_ps']:.6f}ps, "
            f"dt={accepted_b[first_divergence]['dt_ps']:.6f}ps"
        )
    else:
        print(f"\nNo significant divergence found in {max_compare} compared steps")

    # Rejection analysis
    print(f"\n{'='*80}")
    print("Rejection analysis")
    print(f"{'='*80}")

    rejected_a = [r for r in records_a if not r["accepted"]]
    rejected_b = [r for r in records_b if not r["accepted"]]

    print(
        f"  A rejections: {len(rejected_a)} "
        f"(LTE: {sum(1 for r in rejected_a if r.get('lte_reject'))}, "
        f"NR: {sum(1 for r in rejected_a if not r.get('lte_reject'))})"
    )
    print(
        f"  B rejections: {len(rejected_b)} "
        f"(LTE: {sum(1 for r in rejected_b if r.get('lte_reject'))}, "
        f"NR: {sum(1 for r in rejected_b if not r.get('lte_reject'))})"
    )

    if rejected_b:
        print("\n  First 10 B rejections:")
        for r in rejected_b[:10]:
            lte_str = f"LTE={r['lte_norm']:.2f}" if r["lte_norm"] is not None else "LTE=N/A"
            print(
                f"    Step {r['step']:4d}: t={r['t_ps']:.4f}ps dt={r['dt_ps']:.6f}ps "
                f"NR={r['nr_iters']} res={r['residual']:.2e} {lte_str}"
            )


def run_compare_local(benchmark_name: str) -> None:
    """Run benchmark with both dense and sparse (UMFPACK) solvers locally and compare."""
    logger.info(f"Running local comparison for '{benchmark_name}'")

    # Run dense
    logger.info("--- Running DENSE solver ---")
    records_dense, summary_dense = run_simulation(benchmark_name, use_sparse=False)
    print_summary(summary_dense)

    # Run sparse (UMFPACK on CPU)
    logger.info("--- Running SPARSE (UMFPACK) solver ---")
    records_sparse, summary_sparse = run_simulation(benchmark_name, use_sparse=True)
    print_summary(summary_sparse)

    # Save both traces
    dense_path = f"/tmp/lte_trace_{benchmark_name}_dense.json"
    sparse_path = f"/tmp/lte_trace_{benchmark_name}_umfpack.json"

    save_trace(records_dense, summary_dense, dense_path)
    save_trace(records_sparse, summary_sparse, sparse_path)

    # Compare
    compare_traces(dense_path, sparse_path)


def main():
    parser = argparse.ArgumentParser(description="Compare per-step LTE between solvers")
    parser.add_argument(
        "--benchmark", "-b", default="ring", help="Benchmark name (default: ring)"
    )
    parser.add_argument("--output", "-o", help="Output JSON path for trace data")
    parser.add_argument(
        "--sparse",
        action="store_true",
        default=None,
        help="Use sparse solver (auto-detected if not specified)",
    )
    parser.add_argument("--dense", action="store_true", help="Force dense solver")
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("TRACE_A", "TRACE_B"),
        help="Compare two saved traces",
    )
    parser.add_argument(
        "--compare-local",
        action="store_true",
        help="Run locally with both dense and sparse, then compare",
    )
    args = parser.parse_args()

    if args.compare:
        compare_traces(args.compare[0], args.compare[1])
        return

    if args.compare_local:
        run_compare_local(args.benchmark)
        return

    # Single run mode
    use_sparse = True  # default to sparse for meaningful comparison
    if args.dense:
        use_sparse = False
    elif args.sparse is not None:
        use_sparse = args.sparse

    records, summary = run_simulation(args.benchmark, use_sparse)
    print_summary(summary)

    # Save trace
    if args.output:
        output_path = args.output
    else:
        import jax

        backend = jax.default_backend()
        solver = "sparse" if use_sparse else "dense"
        output_path = f"/tmp/lte_trace_{args.benchmark}_{solver}_{backend}.json"

    save_trace(records, summary, output_path)


if __name__ == "__main__":
    main()
