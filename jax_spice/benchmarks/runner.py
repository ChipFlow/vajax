"""Benchmark runner utilities for JAX-SPICE.

Provides conveniences for running circuit benchmarks:
- Warmup runs for accurate timing
- Multiple run averaging
- Memory cleanup between circuits

For core circuit simulation, use CircuitEngine from jax_spice.analysis.engine.
"""

from pathlib import Path
from typing import Any, Dict, Optional

from jax_spice.profiling import ProfileConfig

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
        use_scan: bool = True,
        profile_config: Optional[ProfileConfig] = None,
    ) -> Dict[str, Any]:
        """Run a benchmark with warmup and timing.

        Args:
            circuit_path: Path to .sim file
            t_stop: Simulation stop time
            dt: Timestep
            warmup_runs: Number of warmup runs (for JIT compilation)
            timed_runs: Number of timed runs to average
            use_sparse: Use sparse solver (auto-detect if None)
            use_scan: Use lax.scan solver
            profile_config: Optional profiling configuration

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

        max_steps = int(t_stop / dt) + 10

        # Warmup runs
        for _ in range(warmup_runs):
            self._engine.run_transient(
                t_stop=t_stop, dt=dt, max_steps=max_steps,
                use_sparse=use_sparse, use_while_loop=use_scan
            )

        # Timed runs
        run_times = []
        for _ in range(timed_runs):
            start = time.perf_counter()
            times, voltages, stats = self._engine.run_transient(
                t_stop=t_stop, dt=dt, max_steps=max_steps,
                use_sparse=use_sparse, use_while_loop=use_scan,
                profile_config=profile_config
            )
            elapsed = time.perf_counter() - start
            run_times.append(elapsed)

        return {
            'times': times,
            'voltages': voltages,
            'stats': stats,
            'run_times': run_times,
            'avg_time': sum(run_times) / len(run_times),
            'num_nodes': self._engine.num_nodes,
        }

    def clear(self):
        """Clear cached data and free memory."""
        if self._engine is not None:
            self._engine.clear_cache()
            self._engine = None
