"""Benchmark runner utilities for JAX-SPICE.

Provides conveniences for running circuit benchmarks:
- Warmup runs for accurate timing
- Multiple run averaging
- Memory cleanup between circuits

For core circuit simulation, use CircuitEngine from jax_spice.analysis.engine.
"""

from pathlib import Path
from typing import Any, Dict, Optional

# Re-export CircuitEngine for convenience
from jax_spice.analysis.engine import CircuitEngine


class BenchmarkRunner:
    """Runner for circuit benchmarks with timing and comparison features.

    Wraps CircuitEngine with benchmark-specific conveniences:
    - Warmup runs for accurate timing
    - Multiple run averaging
    - Memory cleanup between circuits
    - Result comparison utilities

    Example:
        runner = BenchmarkRunner()
        results = runner.run_benchmark(
            "vendor/VACASK/benchmark/ring/vacask/runme.sim",
            t_stop=1e-9, dt=1e-12, warmup_runs=1, timed_runs=3
        )
        print(f"Average time: {results['avg_time']:.3f}s")
    """

    def __init__(self):
        self._engine: Optional[CircuitEngine] = None

    def run_benchmark(
        self,
        circuit_path: Path,
        t_stop: float,
        dt: float,
        *,
        warmup_runs: int = 1,
        timed_runs: int = 1,
        use_sparse: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Run a benchmark with warmup and timing.

        Args:
            circuit_path: Path to .sim file
            t_stop: Simulation stop time
            dt: Timestep
            warmup_runs: Number of warmup runs (for JIT compilation)
            timed_runs: Number of timed runs to average
            use_sparse: Use sparse solver (auto-detect if None)

        Returns:
            Dict with:
                - times: Time array from last run
                - voltages: Voltage dict from last run
                - stats: Stats from last run
                - run_times: List of wall times for each timed run
                - avg_time: Average wall time
                - num_nodes: Circuit node count
        """
        import time

        # Parse circuit
        self._engine = CircuitEngine(Path(circuit_path))
        self._engine.parse()

        # Prepare once — max_steps auto-computed with 10% headroom
        self._engine.prepare(t_stop=t_stop, dt=dt, use_sparse=use_sparse)

        # Warmup runs
        for _ in range(warmup_runs):
            self._engine.run_transient()

        # Timed runs
        run_times = []
        result = None
        for _ in range(timed_runs):
            start = time.perf_counter()
            result = self._engine.run_transient()
            elapsed = time.perf_counter() - start
            run_times.append(elapsed)

        return {
            "times": result.times,
            "voltages": result.voltages,
            "stats": result.stats,
            "run_times": run_times,
            "avg_time": sum(run_times) / len(run_times),
            "num_nodes": self._engine.num_nodes,
        }

    def clear(self):
        """Clear cached data and free memory."""
        if self._engine is not None:
            self._engine.clear_cache()
            self._engine = None
