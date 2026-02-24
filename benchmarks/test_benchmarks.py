"""VACASK Benchmark Tests

Runs all VACASK benchmark circuits and outputs results to GitHub step summary.
Tests for correctness (no NaN/Inf) and reports timing.
"""

import os
import time
from pathlib import Path
from typing import List

import jax
import numpy as np
import pytest

# Precision is auto-configured by vajax import based on backend capabilities
from vajax.analysis import CircuitEngine

# Benchmark configuration
BENCHMARK_ROOT = Path(__file__).parent.parent / "vendor" / "VACASK" / "benchmark"

# All benchmarks with their configurations
BENCHMARKS = {
    "rc": {"use_sparse": False},
    "graetz": {"use_sparse": False},
    "mul": {"use_sparse": False},
    "ring": {"use_sparse": False},
    "c6288": {"use_sparse": True},  # Sparse required for 86k nodes
}


def get_benchmark_sim_file(name: str) -> Path:
    """Get the sim file path for a benchmark."""
    return BENCHMARK_ROOT / name / "vacask" / "runme.sim"


def write_github_summary(content: str):
    """Write content to GitHub step summary if available."""
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a") as f:
            f.write(content)


class BenchmarkResults:
    """Collects and formats benchmark results."""

    def __init__(self):
        self.results: List[dict] = []

    def add(
        self,
        name: str,
        nodes: int,
        devices: int,
        openvaf: int,
        timesteps: int,
        time_s: float,
        solver: str,
        passed: bool,
        error: str = None,
    ):
        self.results.append(
            {
                "name": name,
                "nodes": nodes,
                "devices": devices,
                "openvaf": openvaf,
                "timesteps": timesteps,
                "time_s": time_s,
                "time_per_step_ms": (time_s / timesteps * 1000) if timesteps > 0 else 0,
                "solver": solver,
                "passed": passed,
                "error": error,
            }
        )

    def to_markdown(self) -> str:
        """Generate markdown report."""
        lines = []
        lines.append("# VACASK Benchmark Results\n")

        # Environment
        lines.append("## Environment\n")
        lines.append(f"- **JAX Backend**: `{jax.default_backend()}`")
        lines.append(f"- **Devices**: `{[str(d) for d in jax.devices()]}`")
        lines.append("")

        # Results table
        lines.append("## Results\n")
        lines.append(
            "| Benchmark | Nodes | Devices | OpenVAF | Solver | Steps | Time (s) | ms/step | Status |"
        )
        lines.append(
            "|-----------|-------|---------|---------|--------|-------|----------|---------|--------|"
        )

        for r in self.results:
            status = "PASS" if r["passed"] else f"FAIL: {r['error']}"
            if len(status) > 20:
                status = status[:17] + "..."
            lines.append(
                f"| {r['name']} | {r['nodes']} | {r['devices']} | {r['openvaf']} | "
                f"{r['solver']} | {r['timesteps']} | {r['time_s']:.2f} | "
                f"{r['time_per_step_ms']:.1f} | {status} |"
            )

        lines.append("")
        return "\n".join(lines)


# Shared results collector
_results = BenchmarkResults()


@pytest.fixture(scope="module")
def benchmark_results():
    """Fixture to collect and report benchmark results."""
    yield _results

    # After all tests, write summary
    report = _results.to_markdown()
    print("\n" + report)
    write_github_summary(report)


class TestVACASKBenchmarks:
    """Test all VACASK benchmark circuits."""

    @pytest.mark.parametrize("benchmark_name", list(BENCHMARKS.keys()))
    def test_benchmark(self, benchmark_name, benchmark_results):
        """Run a VACASK benchmark and verify correctness."""
        sim_file = get_benchmark_sim_file(benchmark_name)

        if not sim_file.exists():
            pytest.skip(f"Benchmark {benchmark_name} not found at {sim_file}")

        config = BENCHMARKS[benchmark_name]
        use_sparse = config["use_sparse"]

        print(f"\n{'=' * 60}")
        print(f"Running {benchmark_name} benchmark")
        print(f"{'=' * 60}")

        try:
            # Parse circuit
            engine = CircuitEngine(sim_file)
            engine.parse()

            nodes = engine.num_nodes
            devices = len(engine.devices)
            openvaf = sum(1 for d in engine.devices if d.get("is_openvaf"))
            solver = "sparse" if use_sparse else "dense"

            print(f"Circuit: {nodes} nodes, {devices} devices ({openvaf} OpenVAF)")

            # Run transient analysis — prepare() reads t_stop/dt from netlist
            start = time.perf_counter()
            engine.prepare(use_sparse=use_sparse)
            result = engine.run_transient()
            elapsed = time.perf_counter() - start

            timesteps = result.num_steps
            print(
                f"Completed: {timesteps} timesteps in {elapsed:.2f}s ({elapsed / timesteps * 1000:.1f}ms/step)"
            )

            # Verify correctness
            all_voltages = np.concatenate([v for v in result.voltages.values()])
            has_nan = np.any(np.isnan(all_voltages))
            has_inf = np.any(np.isinf(all_voltages))

            if has_nan:
                raise ValueError("NaN values in output")
            if has_inf:
                raise ValueError("Inf values in output")

            # Record success
            benchmark_results.add(
                name=benchmark_name,
                nodes=nodes,
                devices=devices,
                openvaf=openvaf,
                timesteps=timesteps,
                time_s=elapsed,
                solver=solver,
                passed=True,
            )

            print(f"  PASS: {benchmark_name}")

        except Exception as e:
            # Record failure
            benchmark_results.add(
                name=benchmark_name,
                nodes=nodes if "nodes" in dir() else 0,
                devices=devices if "devices" in dir() else 0,
                openvaf=openvaf if "openvaf" in dir() else 0,
                timesteps=0,
                time_s=0,
                solver=solver if "solver" in dir() else "unknown",
                passed=False,
                error=str(e),
            )
            raise
