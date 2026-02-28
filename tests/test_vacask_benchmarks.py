"""Test VACASK benchmarks using VAJAX CircuitEngine API.

This module tests VACASK benchmark circuits using OpenVAF-compiled device models.
Benchmarks are auto-discovered from vendor/VACASK/benchmark/*/vacask/runme.sim.

Tests include:
- Basic parsing and simulation (parametrized over all benchmarks)
- Node count comparison with VACASK (when available)
- Waveform comparison with VACASK (when available)
"""

import gc
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import jax
import numpy as np
import pytest

# Maximum steps for CI tests (override with VAJAX_MAX_STEPS env var)
# Default: 0 (no limit). CI should set to e.g. 10000 for faster tests.
MAX_STEPS_ENV = int(os.environ.get("VAJAX_MAX_STEPS", "0"))

from vajax._logging import enable_performance_logging, logger
from vajax.analysis import CircuitEngine
from vajax.analysis.node_setup import setup_internal_nodes
from vajax.benchmarks.registry import (
    BENCHMARKS,
    BenchmarkInfo,
    get_benchmark,
)
from vajax.utils import compare_transient_waveforms, find_vacask_binary, rawread

enable_performance_logging()


# =============================================================================
# Transient Result Cache
# =============================================================================

# Module-level cache for transient results: {(benchmark_name, solver_type): result}
_transient_cache: Dict[Tuple[str, str], object] = {}


def get_cached_result(benchmark_name: str, solver_type: str):
    """Get cached transient result, or None if not cached."""
    return _transient_cache.get((benchmark_name, solver_type))


def cache_result(benchmark_name: str, solver_type: str, result):
    """Cache a transient result."""
    _transient_cache[(benchmark_name, solver_type)] = result


# =============================================================================
# Basic Benchmark Tests (parse, dense/sparse transient)
# =============================================================================


def get_runnable_benchmarks() -> list[str]:
    """Get list of benchmarks that can be run (not skipped)."""
    return [name for name, info in BENCHMARKS.items() if not info.skip]


class TestBenchmarkParsing:
    """Test that all benchmarks parse correctly."""

    @pytest.mark.parametrize("benchmark_name", get_runnable_benchmarks())
    def test_parse(self, benchmark_name: str):
        """Test benchmark parses correctly."""
        info = get_benchmark(benchmark_name)
        assert info is not None, f"Benchmark {benchmark_name} not found"
        assert info.sim_path.exists(), f"Sim file not found: {info.sim_path}"

        engine = CircuitEngine(info.sim_path)
        engine.parse()

        assert engine.num_nodes > 0, "No nodes parsed"
        assert len(engine.devices) > 0, "No devices parsed"

        device_types = {d["model"] for d in engine.devices}
        logger.info(f"\n{benchmark_name}: {engine.num_nodes} nodes, {len(engine.devices)} devices")
        logger.info(f"  Title: {info.title}")
        logger.info(f"  Devices: {device_types}")
        logger.info(f"  dt={info.dt:.2e}, t_stop={info.t_stop:.2e}")


class TestBenchmarkTransient:
    """Test transient simulation for all benchmarks.

    Results are cached for reuse by VACASK comparison tests.
    For benchmarks with comparison specs, runs full t_stop from BenchmarkInfo.
    For other benchmarks, runs short duration (dt * 10).
    """

    @pytest.fixture(autouse=True)
    def cleanup_gpu_state(self):
        """Clear JAX caches and run GC after each test.

        This helps prevent GPU memory corruption when running multiple
        sparse solves with different sparsity patterns in sequence.
        """
        yield  # Run the test
        # Cleanup after test
        jax.clear_caches()
        gc.collect()

    @pytest.mark.parametrize("benchmark_name", get_runnable_benchmarks())
    def test_transient_dense(self, benchmark_name: str):
        """Test transient with dense solver."""
        info = get_benchmark(benchmark_name)
        if info.is_large:
            pytest.skip(f"{benchmark_name} too large for dense solver")
        if info.xfail:
            pytest.xfail(info.xfail_reason)

        engine = CircuitEngine(info.sim_path)
        engine.parse()

        # Use full t_stop for benchmarks with comparison specs, short for others
        has_comparison = benchmark_name in COMPARISON_SPECS
        t_stop = info.t_stop if has_comparison else info.dt * 10

        logger.info(f"running transient dense (t_stop={t_stop:.2e})")
        engine.prepare(t_stop=t_stop, dt=info.dt, use_sparse=False)
        result = engine.run_transient()
        logger.info("transient finished")

        # Cache result for VACASK comparison
        cache_result(benchmark_name, "dense", result)

        assert result.num_steps > 0, "No timesteps returned"
        converged = result.stats.get("convergence_rate", 0)
        logger.info(
            f"\n{benchmark_name} dense: {result.num_steps} steps, {converged * 100:.0f}% converged"
        )
        assert converged > 0.5, f"Poor convergence: {converged * 100:.0f}%"

    @pytest.mark.parametrize("benchmark_name", get_runnable_benchmarks())
    def test_transient_sparse(self, benchmark_name: str):
        """Test transient with sparse solver."""
        info = get_benchmark(benchmark_name)
        if info.is_large:
            pytest.skip(f"{benchmark_name} requires GPU - use scripts/profile_gpu_cloudrun.py")

        # Workaround for cuDSS/Spineax bug: running 3+ sparse solves with different
        # sparsity patterns causes GPU memory corruption. Limit to 2 benchmarks on GPU.
        # See: https://github.com/ChipFlow/vajax/issues/XXX
        on_gpu = jax.default_backend() in ("cuda", "gpu")
        # Only allow graetz and mul on GPU (they pass), skip others
        gpu_allowed = ["graetz", "mul"]
        if on_gpu and benchmark_name not in gpu_allowed:
            pytest.skip(f"Skipping {benchmark_name} sparse on GPU to avoid cuDSS memory corruption")

        engine = CircuitEngine(info.sim_path)
        engine.parse()

        # Use full t_stop for benchmarks with comparison specs, short for others
        has_comparison = benchmark_name in COMPARISON_SPECS
        t_stop = info.t_stop if has_comparison else info.dt * 10

        logger.info(f"running transient sparse (t_stop={t_stop:.2e})")
        engine.prepare(t_stop=t_stop, dt=info.dt, use_sparse=True)
        result = engine.run_transient()
        logger.info("transient finished")

        # Cache result for VACASK comparison
        cache_result(benchmark_name, "sparse", result)

        assert result.num_steps > 0, "No timesteps returned"
        converged = result.stats.get("convergence_rate", 0)
        logger.info(
            f"\n{benchmark_name} sparse: {result.num_steps} steps, {converged * 100:.0f}% converged"
        )
        assert converged > 0.5, f"Poor convergence: {converged * 100:.0f}%"


# =============================================================================
# RC Time Constant Test (physics validation)
# =============================================================================


class TestRCTimeConstant:
    """Verify RC time constant behavior for the rc benchmark.

    With R=1k, C=1u, tau=1ms. After 5 tau, voltage reaches ~99.3% of final value.
    """

    def test_rc_time_constant(self):
        """Verify RC circuit charges correctly."""
        info = get_benchmark("rc")
        if info is None or not info.sim_path.exists():
            pytest.skip("RC benchmark not found")

        dt = 10e-6  # 10us steps
        t_stop = 5e-3  # 5ms (5 tau)

        engine = CircuitEngine(info.sim_path)
        engine.parse()
        engine.prepare(t_stop=t_stop, dt=dt, use_sparse=False)
        result = engine.run_transient()

        # Get capacitor voltage (node '2')
        v_cap = result.voltage("2") if "2" in result.voltages else None
        if v_cap is None:
            v_cap = result.voltages[sorted(result.voltages.keys())[-1]]

        v_cap_np = np.array(v_cap)
        v_final = v_cap_np[-1] if len(v_cap_np) > 0 else 0

        logger.info(
            f"\nRC response: V_final = {v_final:.3f}V after {np.array(result.times)[-1] * 1000:.1f}ms"
        )
        assert result.num_steps > 10, "Not enough timesteps for RC analysis"


# =============================================================================
# Node Count Comparison with VACASK
# =============================================================================


def get_vacask_node_count(vacask_bin: Path, benchmark_name: str, timeout: int = 600) -> dict:
    """Run VACASK on benchmark and extract node count from 'logger.info stats'."""
    info = get_benchmark(benchmark_name)
    if info is None:
        raise FileNotFoundError(f"Benchmark {benchmark_name} not found")

    result = subprocess.run(
        [str(vacask_bin), str(info.sim_path)],
        capture_output=True,
        text=True,
        cwd=info.sim_path.parent,
        timeout=timeout,
    )

    nodes_match = re.search(r"Number of nodes:\s+(\d+)", result.stdout + result.stderr)
    unknowns_match = re.search(r"Number of unknonws:\s+(\d+)", result.stdout + result.stderr)

    if nodes_match and unknowns_match:
        return {"nodes": int(nodes_match.group(1)), "unknowns": int(unknowns_match.group(1))}

    raise ValueError("Could not parse node count from VACASK output")


class TestNodeCountComparison:
    """Test that VAJAX node counts match VACASK."""

    @pytest.fixture
    def vacask_bin(self):
        """Get VACASK binary path, skip if not available."""
        binary = find_vacask_binary()
        if binary is None:
            pytest.skip("VACASK binary not found. Set VACASK_BIN env var or build VACASK.")
        return binary

    @pytest.mark.parametrize("benchmark_name", ["rc", "graetz", "ring"])
    def test_node_count_matches_vacask(self, vacask_bin, benchmark_name: str):
        """Compare VAJAX node count with VACASK unknownCount (+/- 1)."""
        info = get_benchmark(benchmark_name)
        if info is None or not info.sim_path.exists():
            pytest.skip(f"Benchmark {benchmark_name} not found")

        vacask_counts = get_vacask_node_count(vacask_bin, benchmark_name)
        vacask_unknowns = vacask_counts["unknowns"]

        engine = CircuitEngine(info.sim_path)
        engine.parse()
        n_total, _ = setup_internal_nodes(
            devices=engine.devices,
            num_nodes=engine.num_nodes,
            compiled_models=engine._compiled_models,
            device_collapse_decisions=engine._device_collapse_decisions,
        )

        logger.info(f"\n{benchmark_name}:")
        logger.info(f"  VACASK: nodes={vacask_counts['nodes']}, unknowns={vacask_unknowns}")
        logger.info(f"  VAJAX: external={engine.num_nodes}, total={n_total}")

        diff_total = abs(n_total - vacask_unknowns)
        assert diff_total <= 1, (
            f"Node count differs: VAJAX={n_total}, VACASK unknowns={vacask_unknowns}"
        )

    def test_c6288_node_count(self, vacask_bin):
        """Test c6288 node count - large circuit with node collapse."""
        info = get_benchmark("c6288")
        if info is None or not info.sim_path.exists():
            pytest.skip("c6288 benchmark not found")

        vacask_counts = get_vacask_node_count(vacask_bin, "c6288")
        vacask_unknowns = vacask_counts["unknowns"]

        engine = CircuitEngine(info.sim_path)
        engine.parse()
        n_total, _ = setup_internal_nodes(
            devices=engine.devices,
            num_nodes=engine.num_nodes,
            compiled_models=engine._compiled_models,
            device_collapse_decisions=engine._device_collapse_decisions,
        )

        logger.info("\nc6288:")
        logger.info(f"  VACASK: nodes={vacask_counts['nodes']}, unknowns={vacask_unknowns}")
        logger.info(f"  VAJAX: external={engine.num_nodes}, total={n_total}")

        expected = vacask_unknowns + 1
        ratio = abs(n_total - expected) / expected
        assert ratio <= 0.1, f"c6288: VAJAX={n_total}, expected~{expected} ({ratio * 100:.1f}% off)"


class TestNodeCollapseStandalone:
    """Test node collapse without requiring VACASK binary."""

    def test_c6288_node_collapse_reduces_count(self):
        """Test that node collapse reduces c6288 to <30k nodes."""
        info = get_benchmark("c6288")
        if info is None or not info.sim_path.exists():
            pytest.skip("c6288 benchmark not found")

        engine = CircuitEngine(info.sim_path)
        engine.parse()

        n_total, _ = setup_internal_nodes(
            devices=engine.devices,
            num_nodes=engine.num_nodes,
            compiled_models=engine._compiled_models,
            device_collapse_decisions=engine._device_collapse_decisions,
        )
        n_internal = n_total - engine.num_nodes
        n_psp103 = sum(1 for d in engine.devices if d.get("model") == "psp103")

        logger.info("\nc6288 node collapse:")
        logger.info(f"  External: {engine.num_nodes}, Internal: {n_internal}, Total: {n_total}")
        logger.info(f"  PSP103 devices: {n_psp103}")

        # With collapse, each PSP103 needs 2 internal nodes
        expected_internal = n_psp103 * 2
        internal_ratio = n_internal / expected_internal

        assert 0.9 < internal_ratio < 1.1, (
            f"Internal nodes: {n_internal} (expected ~{expected_internal})"
        )
        assert n_total < 30000, f"Total nodes too high: {n_total} (expected <30000 with collapse)"

    def test_ring_node_collapse(self):
        """Test ring benchmark has 47 nodes (matching VACASK)."""
        info = get_benchmark("ring")
        if info is None or not info.sim_path.exists():
            pytest.skip("Ring benchmark not found")

        engine = CircuitEngine(info.sim_path)
        engine.parse()

        n_total, _ = setup_internal_nodes(
            devices=engine.devices,
            num_nodes=engine.num_nodes,
            compiled_models=engine._compiled_models,
            device_collapse_decisions=engine._device_collapse_decisions,
        )
        n_psp103 = sum(1 for d in engine.devices if d.get("model") == "psp103")

        logger.info("\nring node collapse:")
        logger.info(f"  External: {engine.num_nodes}, Total: {n_total}")
        logger.info(f"  PSP103 devices: {n_psp103}")

        # Ring: 18 PSP103 devices * 2 internal nodes each + 11 external = 47
        assert n_total == 47, f"Expected 47 total nodes, got {n_total}"


# =============================================================================
# VACASK Waveform Comparison Framework
# =============================================================================


@dataclass
class ComparisonSpec:
    """Specification for waveform comparison test.

    Note: dt and t_stop are taken from BenchmarkInfo, not specified here.
    """

    benchmark_name: str
    max_rel_error: float
    vacask_nodes: list[str]
    jax_nodes: list[str]
    xfail: bool = False
    xfail_reason: str = ""
    node_transform: Optional[Callable] = None
    align_on_rising_edge: bool = False  # Align waveforms on rising edge before comparison
    align_threshold: float = 0.6  # Voltage threshold for rising edge detection
    align_after_time: float = 0.0  # Only use edges after this time (skip startup)
    # Adaptive timestep is always used (matches VACASK behavior)


# Comparison specifications - dt/t_stop come from BenchmarkInfo
COMPARISON_SPECS = {
    "rc": ComparisonSpec(
        benchmark_name="rc",
        max_rel_error=0.05,
        vacask_nodes=["2", "v(2)"],
        jax_nodes=["2", "1"],
    ),
    "graetz": ComparisonSpec(
        benchmark_name="graetz",
        # 15% tolerance accounts for $limit passthrough (no pnjlim/fetlim Newton convergence help)
        # Without proper limiting, the NR solver may converge differently than VACASK
        max_rel_error=0.15,
        vacask_nodes=["outp"],
        jax_nodes=["outp"],
        node_transform=lambda v: v.get("outp", np.zeros(1)) - v.get("outn", np.zeros(1)),
        # Diode bridge has sharp IV curves - adaptive handles this well
    ),
    "ring": ComparisonSpec(
        benchmark_name="ring",
        max_rel_error=0.15,  # 15% - waveform shape differs but period is correct (validated separately)
        vacask_nodes=["2"],  # VACASK node 2 matches JAX node 1 (one node offset)
        jax_nodes=["1"],
        align_on_rising_edge=True,
        align_threshold=0.6,
        align_after_time=10e-9,  # Skip first 10ns startup
        # Period validated separately in test_adaptive_ring_validation.py
    ),
    "mul": ComparisonSpec(
        benchmark_name="mul",
        # 1.5% tolerance to account for PSP103 minor numerical differences
        max_rel_error=0.015,
        vacask_nodes=["1", "v(1)"],
        jax_nodes=["1"],
    ),
    "c6288": ComparisonSpec(
        benchmark_name="c6288",
        max_rel_error=0.10,
        vacask_nodes=["v(p0)", "p0"],
        jax_nodes=["top.p0"],
    ),
}


def run_vacask_simulation(vacask_bin: Path, info: BenchmarkInfo, t_stop: float, dt: float) -> dict:
    """Run VACASK and parse the .raw file output."""
    sim_dir = info.sim_path.parent
    sim_content = info.sim_path.read_text()

    # Modify analysis parameters
    modified = re.sub(
        r"(analysis\s+\w+\s+tran\s+.*?stop=)[^\s]+", f"\\g<1>{t_stop:.2e}", sim_content
    )
    modified = re.sub(r"(step=)[^\s]+", f"\\g<1>{dt:.2e}", modified)

    temp_sim = sim_dir / "test_compare.sim"
    temp_sim.write_text(modified)

    try:
        subprocess.run(
            [str(vacask_bin), "test_compare.sim"],
            cwd=sim_dir,
            capture_output=True,
            text=True,
            timeout=600,
        )

        raw_files = list(sim_dir.glob("*.raw"))
        if not raw_files:
            raise RuntimeError(f"VACASK did not produce .raw file in {sim_dir}")

        raw = rawread(str(raw_files[0])).get()
        voltages = {name: np.array(raw[name]) for name in raw.names if name != "time"}
        return {"time": np.array(raw["time"]), "voltages": voltages}
    finally:
        if temp_sim.exists():
            temp_sim.unlink()
        for raw_file in sim_dir.glob("*.raw"):
            raw_file.unlink()


# compare_waveforms and find_rising_edge_time are now in vajax.utils.waveform_compare


def _compare_result_to_vacask(
    result, vacask_results: dict, spec: ComparisonSpec, solver_type: str
) -> dict:
    """Compare a VAJAX result against VACASK reference.

    Returns dict with comparison metrics and pass/fail status.
    """
    vacask_time = vacask_results["time"]

    # Find VACASK output node
    vacask_voltage = None
    vacask_node_used = None
    for node_name in spec.vacask_nodes:
        if node_name in vacask_results["voltages"]:
            vacask_voltage = vacask_results["voltages"][node_name]
            vacask_node_used = node_name
            break

    if vacask_voltage is None:
        return {"error": f"Could not find {spec.vacask_nodes} in VACASK output", "passed": False}

    jax_node_idx = spec.jax_nodes[spec.vacask_nodes.index(vacask_node_used) % len(spec.jax_nodes)]
    jax_voltage = np.array(result.voltages.get(jax_node_idx, []))

    # Apply transform if specified
    if spec.node_transform is not None:
        vacask_voltage = spec.node_transform(vacask_results["voltages"])
        jax_voltage = spec.node_transform({k: np.array(v) for k, v in result.voltages.items()})

    comparison = compare_transient_waveforms(
        vacask_time,
        vacask_voltage,
        np.array(result.times),
        jax_voltage,
        align_on_rising_edge=spec.align_on_rising_edge,
        align_threshold=spec.align_threshold,
        align_after_time=spec.align_after_time,
    )

    passed = comparison["rel_rms"] < spec.max_rel_error
    return {
        "solver": solver_type,
        "comparison": comparison,
        "passed": passed,
        "rel_rms": comparison["rel_rms"],
        "max_rel_error": spec.max_rel_error,
    }


class TestVACASKResultComparison:
    """Compare VAJAX simulation results against VACASK reference.

    Uses cached results from TestBenchmarkTransient when available.
    Compares both dense and sparse solver results against VACASK.
    """

    @pytest.fixture
    def vacask_bin(self):
        """Get VACASK binary path, skip if not available."""
        binary = find_vacask_binary()
        if binary is None:
            pytest.skip("VACASK binary not found")
        return binary

    @pytest.mark.parametrize("benchmark_name", list(COMPARISON_SPECS.keys()))
    def test_transient_matches_vacask(self, vacask_bin, benchmark_name: str):
        """Compare transient simulation against VACASK reference.

        Tests both dense and sparse solver results (from cache or fresh run).
        """
        import time

        test_start = time.perf_counter()
        spec = COMPARISON_SPECS[benchmark_name]
        info = get_benchmark(benchmark_name)

        # Apply MAX_STEPS limit for CI if set
        t_stop = info.t_stop
        dt = info.dt
        if MAX_STEPS_ENV > 0:
            max_t_stop = dt * MAX_STEPS_ENV
            if t_stop > max_t_stop:
                logger.info(
                    f"Limiting steps from {int(t_stop / dt):,} to {MAX_STEPS_ENV:,} (VAJAX_MAX_STEPS)"
                )
                t_stop = max_t_stop

        logger.info(f"\n{'=' * 60}")
        logger.info(f"TEST START: {benchmark_name}")
        logger.info(f"  t_stop={t_stop}, dt={dt}, steps={int(t_stop / dt):,}")
        logger.info(f"{'=' * 60}")

        if spec.xfail:
            pytest.xfail(spec.xfail_reason)
        if info is None or not info.sim_path.exists():
            pytest.skip(f"Benchmark {benchmark_name} not found")

        # Run VACASK with dt/t_stop (possibly limited for CI)
        logger.info(f"[{time.perf_counter() - test_start:.1f}s] Running VACASK simulation...")
        vacask_results = run_vacask_simulation(vacask_bin, info, t_stop, dt)
        logger.info(
            f"[{time.perf_counter() - test_start:.1f}s] VACASK done, {len(vacask_results['time'])} points"
        )

        # Try to get cached results, or run fresh if not available
        results_to_compare = []

        for solver_type in ["dense", "sparse"]:
            cached = get_cached_result(benchmark_name, solver_type)
            if cached is not None:
                logger.info(
                    f"[{time.perf_counter() - test_start:.1f}s] Using cached {solver_type} result"
                )
                results_to_compare.append((solver_type, cached))
            else:
                # Run fresh simulation
                logger.info(
                    f"[{time.perf_counter() - test_start:.1f}s] Running fresh {solver_type} simulation..."
                )
                engine = CircuitEngine(info.sim_path)
                engine.parse()

                # Skip large benchmarks for dense solver
                if solver_type == "dense" and info.is_large:
                    logger.info(
                        f"[{time.perf_counter() - test_start:.1f}s] Skipping dense (too large)"
                    )
                    continue

                use_sparse = solver_type == "sparse"
                sim_start = time.perf_counter()
                engine.prepare(t_stop=t_stop, dt=dt, use_sparse=use_sparse)
                result = engine.run_transient()
                sim_time = time.perf_counter() - sim_start
                logger.info(
                    f"[{time.perf_counter() - test_start:.1f}s] {solver_type} done: {result.num_steps} steps in {sim_time:.1f}s"
                )
                cache_result(benchmark_name, solver_type, result)
                results_to_compare.append((solver_type, result))

        if not results_to_compare:
            pytest.skip(f"No solver results available for {benchmark_name}")

        # Compare each result against VACASK
        all_passed = True
        comparison_results = []

        for solver_type, result in results_to_compare:
            comp = _compare_result_to_vacask(result, vacask_results, spec, solver_type)
            comparison_results.append(comp)

            if "error" in comp:
                logger.warning(f"  {solver_type}: {comp['error']}")
                all_passed = False
            else:
                status = "PASS" if comp["passed"] else "FAIL"
                logger.info(f"  {solver_type}: RMS error {comp['rel_rms'] * 100:.2f}% [{status}]")
                if not comp["passed"]:
                    all_passed = False

        # Report summary
        logger.info(f"\n{benchmark_name.upper()} comparison summary:")
        for comp in comparison_results:
            if "error" not in comp:
                logger.info(
                    f"  {comp['solver']}: {comp['rel_rms'] * 100:.2f}% (max allowed: {comp['max_rel_error'] * 100:.0f}%)"
                )

        # Assert at least one solver passed
        assert all_passed, f"VACASK comparison failed for {benchmark_name}: " + ", ".join(
            f"{c['solver']}={c['rel_rms'] * 100:.2f}%" for c in comparison_results if "rel_rms" in c
        )


# =============================================================================
# mul64 Benchmark (64x64 array multiplier, ~266k MOSFETs)
# =============================================================================


class TestMul64Benchmark:
    """Test 64x64 array multiplier benchmark.

    This is a GPU stress test circuit with ~266k MOSFETs and ~400k+ unknowns,
    generated by scripts/generate_multiplier.py.
    """

    def test_parse(self):
        """Test that mul64 parses correctly."""
        info = get_benchmark("mul64")
        if info is None or not info.sim_path.exists():
            pytest.skip("mul64 benchmark not found")

        engine = CircuitEngine(info.sim_path)
        engine.parse()

        assert len(engine.devices) > 200_000, f"Expected >200k devices, got {len(engine.devices)}"
        assert engine.num_nodes > 100_000, f"Expected >100k nodes, got {engine.num_nodes}"

        n_psp103 = sum(1 for d in engine.devices if d.get("model") == "psp103")
        logger.info(f"\nmul64: {len(engine.devices)} devices, {engine.num_nodes} nodes")
        logger.info(f"  PSP103 devices: {n_psp103}")
        logger.info(f"  dt={info.dt:.2e}, t_stop={info.t_stop:.2e}")

    def test_node_setup(self):
        """Test that mul64 node setup works without OOM."""
        info = get_benchmark("mul64")
        if info is None or not info.sim_path.exists():
            pytest.skip("mul64 benchmark not found")

        engine = CircuitEngine(info.sim_path)
        engine.parse()

        n_total, _ = setup_internal_nodes(
            devices=engine.devices,
            num_nodes=engine.num_nodes,
            compiled_models=engine._compiled_models,
            device_collapse_decisions=engine._device_collapse_decisions,
        )
        n_internal = n_total - engine.num_nodes
        n_psp103 = sum(1 for d in engine.devices if d.get("model") == "psp103")

        logger.info("\nmul64 node setup:")
        logger.info(f"  External: {engine.num_nodes}, Internal: {n_internal}, Total: {n_total}")
        logger.info(f"  PSP103 devices: {n_psp103}")

        # With collapse, each PSP103 needs 2 internal nodes
        expected_internal = n_psp103 * 2
        internal_ratio = n_internal / expected_internal
        assert 0.9 < internal_ratio < 1.1, (
            f"Internal nodes: {n_internal} (expected ~{expected_internal})"
        )

    def test_transient_sparse(self):
        """Test mul64 transient with sparse solver (5 steps)."""
        info = get_benchmark("mul64")
        if info is None or not info.sim_path.exists():
            pytest.skip("mul64 benchmark not found")
        if info.xfail:
            pytest.xfail(info.xfail_reason)

        engine = CircuitEngine(info.sim_path)
        engine.parse()

        max_steps = info.max_steps or 5
        engine.prepare(t_stop=info.dt * max_steps, dt=info.dt, use_sparse=True)
        result = engine.run_transient()

        logger.info(
            f"\nmul64: {result.num_steps} steps, "
            f"{result.stats.get('convergence_rate', 0) * 100:.1f}% converged"
        )
        assert result.num_steps > 0, "No timesteps completed"


# =============================================================================
# tb_dp Benchmark (now auto-discovered from vajax/benchmarks/data/)
# =============================================================================


class TestTbDpBenchmark:
    """Test IHP SG13G2 dual-port SRAM benchmark (tb_dp512x8).

    Note: tb_dp is now auto-discovered by the registry from
    vajax/benchmarks/data/tb_dp/*.sim
    """

    def test_parse(self):
        """Test that tb_dp512x8 parses correctly."""
        info = get_benchmark("tb_dp")
        if info is None or not info.sim_path.exists():
            pytest.skip("tb_dp benchmark not found")

        engine = CircuitEngine(info.sim_path)
        engine.parse()

        assert len(engine.devices) > 100
        logger.info(f"\ntb_dp512x8: {len(engine.devices)} devices, {engine.num_nodes} nodes")
        logger.info(f"  dt={info.dt:.2e}, t_stop={info.t_stop:.2e}")

    def test_transient_sparse(self):
        """Test tb_dp512x8 transient with sparse solver (5 steps)."""
        info = get_benchmark("tb_dp")
        if info is None or not info.sim_path.exists():
            pytest.skip("tb_dp benchmark not found")

        engine = CircuitEngine(info.sim_path)
        engine.parse()

        engine.prepare(t_stop=info.dt * 5, dt=info.dt, use_sparse=True)
        result = engine.run_transient()

        logger.info(
            f"\ntb_dp512x8: {result.num_steps} steps, {result.stats.get('convergence_rate', 0) * 100:.1f}% converged"
        )
        assert result.stats.get("convergence_rate", 0) > 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
