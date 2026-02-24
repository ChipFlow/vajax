"""Transient simulation diagnostics for VA-JAX.

Parse and analyze per-step debug output from transient simulations,
including step acceptance/rejection, LTE behaviour, NR convergence,
and VACASK step-level comparison.

Usage:
    from vajax.debug import parse_debug_output, capture_step_trace

    # Parse existing debug output text
    records = parse_debug_output(debug_text)

    # Run a benchmark with debug_steps and get parsed trace
    records, summary = capture_step_trace("ring", use_sparse=True)
"""

from __future__ import annotations

import logging
import re
import sys
import time
from dataclasses import dataclass
from io import StringIO

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class StepRecord:
    """One parsed step from VA-JAX debug_steps output."""

    step: int
    t_ps: float
    dt_ps: float
    dt_lte_ps: float | None
    nr_iters: int
    residual: float
    lte_norm: float | None
    lte_reject: bool
    nr_failed: bool
    accepted: bool


@dataclass
class StepTraceSummary:
    """Aggregated statistics over a transient step trace."""

    benchmark: str
    solver: str
    backend: str
    use_sparse: bool
    wall_time_s: float
    total_steps_attempted: int
    accepted_steps: int
    rejected_steps: int
    num_steps_result: int
    lte_min: float | None = None
    lte_max: float | None = None
    lte_mean: float | None = None
    dt_min_ps: float | None = None
    dt_max_ps: float | None = None
    nr_min_iters: int | None = None
    nr_max_iters: int | None = None
    nr_mean_iters: float | None = None


@dataclass
class VACASKStepRecord:
    """One parsed step from VACASK tran_debug=1 output."""

    t: float
    dt: float
    status: str  # "accept" or "reject"
    order: int | None = None
    hk_ratio: float | None = None
    point: int | None = None


@dataclass
class ConvergenceSweepResult:
    """Convergence stats for a single t_stop run."""

    t_stop: float
    num_steps: int
    convergence_rate: float
    rejected_steps: int
    dt_min: float
    dt_max: float


# ---------------------------------------------------------------------------
# TeeWriter — capture stdout while still printing
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

# Main pattern with dt_lte field
_STEP_PATTERN = re.compile(
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
_STEP_PATTERN_OLD = re.compile(
    r"Step\s+(\d+):\s+"
    r"t=\s*([\d.]+)ps\s+"
    r"dt=\s*([\d.]+)ps\s+"
    r"NR=\s*(\d+)\s+"
    r"res=([\d.eE+-]+)\s+"
    r"(.*?)\s*"
    r"\[(accept|REJECT)\]"
)


def parse_debug_output(text: str) -> list[StepRecord]:
    """Parse debug_steps output into structured per-step records.

    Expected format from body_fn debug callback::

        Step   3: t=   1.000ps dt=  0.5000ps NR= 4 res=1.23e-13 LTE/tol=0.45       dt_lte=  1.2000ps [accept]
        Step   4: t=   1.500ps dt=  0.5000ps NR= 7 res=4.56e-10 LTE/tol=3.2 → REJECT dt_lte=  0.2000ps [REJECT]
    """
    records: list[StepRecord] = []

    for line in text.split("\n"):
        m = _STEP_PATTERN.search(line)
        dt_lte_ps: float | None = None
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
            m = _STEP_PATTERN_OLD.search(line)
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
        lte_norm: float | None = None
        lte_reject = False
        if "Cannot estimate" not in lte_info:
            lte_match = re.search(r"LTE/tol=([\d.]+)", lte_info)
            if lte_match:
                lte_norm = float(lte_match.group(1))
            if "REJECT" in lte_info:
                lte_reject = True

        records.append(
            StepRecord(
                step=step,
                t_ps=t_ps,
                dt_ps=dt_ps,
                dt_lte_ps=dt_lte_ps,
                nr_iters=nr_iters,
                residual=residual,
                lte_norm=lte_norm,
                lte_reject=lte_reject,
                nr_failed=nr_failed,
                accepted=status == "accept",
            )
        )

    return records


_VACASK_ACCEPT = re.compile(r"Point #(\d+) accepted at t=([\d.e+-]+), dt=([\d.e+-]+), order=(\d+)")
_VACASK_HK = re.compile(r"hk/hknew=(\d+(?:\.\d+)?(?:e[+-]?\d+)?)")
_VACASK_REJECT = re.compile(r"Point rejected at t=([\d.e+-]+), dt=([\d.e+-]+)")


def parse_vacask_debug_output(text: str) -> list[VACASKStepRecord]:
    """Parse VACASK tran_debug=1 output into structured records.

    Handles both accepted and rejected steps, preserving their
    original interleaved order.
    """
    records: list[VACASKStepRecord] = []
    pending_hk_ratio: float | None = None

    for line in text.splitlines():
        m = _VACASK_ACCEPT.search(line)
        if m:
            records.append(
                VACASKStepRecord(
                    point=int(m.group(1)),
                    t=float(m.group(2)),
                    dt=float(m.group(3)),
                    order=int(m.group(4)),
                    status="accept",
                )
            )
            continue

        m = _VACASK_HK.search(line)
        if m:
            pending_hk_ratio = float(m.group(1))
            continue

        m = _VACASK_REJECT.search(line)
        if m:
            records.append(
                VACASKStepRecord(
                    t=float(m.group(1)),
                    dt=float(m.group(2)),
                    status="reject",
                    hk_ratio=pending_hk_ratio,
                )
            )
            pending_hk_ratio = None

    return records


# ---------------------------------------------------------------------------
# Capture and summarise
# ---------------------------------------------------------------------------


def _build_summary(
    records: list[StepRecord],
    *,
    benchmark: str,
    solver: str,
    backend: str,
    use_sparse: bool,
    wall_time_s: float,
    num_steps_result: int,
) -> StepTraceSummary:
    accepted = [r for r in records if r.accepted]
    rejected = [r for r in records if not r.accepted]
    lte_values = [r.lte_norm for r in records if r.lte_norm is not None]

    return StepTraceSummary(
        benchmark=benchmark,
        solver=solver,
        backend=backend,
        use_sparse=use_sparse,
        wall_time_s=wall_time_s,
        total_steps_attempted=len(records),
        accepted_steps=len(accepted),
        rejected_steps=len(rejected),
        num_steps_result=num_steps_result,
        lte_min=min(lte_values) if lte_values else None,
        lte_max=max(lte_values) if lte_values else None,
        lte_mean=sum(lte_values) / len(lte_values) if lte_values else None,
        dt_min_ps=min(r.dt_ps for r in records) if records else None,
        dt_max_ps=max(r.dt_ps for r in records) if records else None,
        nr_min_iters=min(r.nr_iters for r in records) if records else None,
        nr_max_iters=max(r.nr_iters for r in records) if records else None,
        nr_mean_iters=(sum(r.nr_iters for r in records) / len(records) if records else None),
    )


def capture_step_trace(
    benchmark_name: str,
    *,
    use_sparse: bool = True,
) -> tuple[list[StepRecord], StepTraceSummary]:
    """Run a benchmark simulation with debug_steps and return parsed trace.

    Args:
        benchmark_name: Name of the benchmark (e.g. "ring", "c6288").
        use_sparse: Whether to use sparse solver.

    Returns:
        Tuple of (per-step records, summary statistics).
    """
    import jax

    from vajax.analysis.engine import CircuitEngine
    from vajax.analysis.transient.adaptive import AdaptiveConfig
    from vajax.benchmarks.registry import get_benchmark

    info = get_benchmark(benchmark_name)
    assert info is not None, f"Unknown benchmark: {benchmark_name}"

    backend = str(jax.default_backend())
    if use_sparse:
        solver = "cuDSS/Spineax" if backend in ("cuda", "gpu") else "UMFPACK"
    else:
        solver = "dense"

    logger.info("Running '%s' with %s solver (backend=%s)", benchmark_name, solver, backend)

    engine = CircuitEngine(info.sim_path)
    engine.parse()
    config = AdaptiveConfig(debug_steps=True)
    engine.prepare(use_sparse=use_sparse, adaptive_config=config)

    tee = TeeWriter(sys.stdout)
    original_stdout = sys.stdout
    sys.stdout = tee

    t_start = time.perf_counter()
    try:
        result = engine.run_transient()
    finally:
        sys.stdout = original_stdout

    wall_time = time.perf_counter() - t_start
    records = parse_debug_output(tee.get_captured())

    logger.info("Captured %d step records from debug output", len(records))

    summary = _build_summary(
        records,
        benchmark=benchmark_name,
        solver=solver,
        backend=backend,
        use_sparse=use_sparse,
        wall_time_s=wall_time,
        num_steps_result=result.num_steps,
    )
    return records, summary


def convergence_sweep(
    benchmark_name: str,
    t_stops: list[float],
    *,
    dt: float = 1e-6,
) -> list[ConvergenceSweepResult]:
    """Run a benchmark at multiple t_stop values and collect convergence stats.

    Args:
        benchmark_name: Name of the benchmark (e.g. "graetz").
        t_stops: List of t_stop values in seconds.
        dt: Initial timestep in seconds.

    Returns:
        List of per-t_stop convergence results.
    """
    from vajax import CircuitEngine
    from vajax.benchmarks.registry import get_benchmark

    info = get_benchmark(benchmark_name)
    assert info is not None, f"Unknown benchmark: {benchmark_name}"

    results: list[ConvergenceSweepResult] = []
    for t_stop in t_stops:
        engine = CircuitEngine(info.sim_path)
        engine.parse()
        engine.prepare(t_stop=t_stop, dt=dt)
        r = engine.run_transient()
        stats = r.stats or {}
        results.append(
            ConvergenceSweepResult(
                t_stop=t_stop,
                num_steps=r.num_steps,
                convergence_rate=stats.get("convergence_rate", 0.0),
                rejected_steps=stats.get("rejected_steps", 0),
                dt_min=stats.get("min_dt", 0.0),
                dt_max=stats.get("max_dt", 0.0),
            )
        )
    return results


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------


def print_step_summary(
    records: list[StepRecord],
    summary: StepTraceSummary,
) -> None:
    """Print a formatted summary table to stderr."""
    logger.info(
        "  %s (%s, %s): %d accepted, %d rejected, %.1fs",
        summary.benchmark,
        summary.solver,
        summary.backend,
        summary.accepted_steps,
        summary.rejected_steps,
        summary.wall_time_s,
    )
    if summary.lte_mean is not None:
        logger.info(
            "  LTE norm: min=%.4f, max=%.4f, mean=%.4f",
            summary.lte_min,
            summary.lte_max,
            summary.lte_mean,
        )
    if summary.dt_min_ps is not None:
        logger.info(
            "  dt range: [%.6f, %.6f] ps",
            summary.dt_min_ps,
            summary.dt_max_ps,
        )
    if summary.nr_mean_iters is not None:
        logger.info(
            "  NR iters: min=%d, max=%d, mean=%.1f",
            summary.nr_min_iters,
            summary.nr_max_iters,
            summary.nr_mean_iters,
        )
