"""Python loop transient analysis strategy.

This strategy uses a traditional Python for-loop over timesteps with a
JIT-compiled Newton-Raphson solver for each step. It's the most flexible
approach and provides full convergence tracking.

Performance: ~0.5ms/step on CPU (ring benchmark)

Advantages:
- Full convergence tracking per timestep
- Detailed logging and debugging info
- Works with all circuit types
- Easy to profile with standard Python tools

Disadvantages:
- Python loop overhead (~5x slower than lax.scan)
- Not fully JIT-compiled

Use this strategy when:
- Debugging convergence issues
- Need detailed per-step statistics
- First time running a circuit
"""

import time as time_module
from typing import Any, Dict, List, Tuple

import jax
import jax.numpy as jnp

from .base import TransientStrategy
from jax_spice.logging import logger
from jax_spice.analysis.gpu_backend import get_default_dtype


# Newton-Raphson solver constants
MAX_NR_ITERATIONS = 100


class PythonLoopStrategy(TransientStrategy):
    """Transient analysis using Python for-loop with JIT-compiled NR solver.

    This is the reference implementation that provides full convergence
    tracking and debugging capability at the cost of Python loop overhead.

    Example:
        runner = VACASKBenchmarkRunner(sim_path)
        runner.parse()

        strategy = PythonLoopStrategy(runner, use_sparse=False)

        # Warmup (JIT compilation)
        _ = strategy.run(t_stop=1e-9, dt=1e-12)

        # Timed run
        times, voltages, stats = strategy.run(t_stop=1e-6, dt=1e-9)
        print(f"Performance: {stats['time_per_step_ms']:.2f}ms/step")
    """

    def run(self, t_stop: float, dt: float,
            max_steps: int = 10000) -> Tuple[jax.Array, Dict[int, jax.Array], Dict]:
        """Run transient analysis with Python for-loop.

        Args:
            t_stop: Simulation stop time in seconds
            dt: Time step in seconds
            max_steps: Maximum number of time steps

        Returns:
            Tuple of (times, voltages, stats) where:
            - times: JAX array of time points
            - voltages: Dict mapping node index to voltage array
            - stats: Dict with detailed convergence info
        """
        # Ensure setup and solver are ready
        setup = self.ensure_setup()
        nr_solve = self.ensure_solver()

        n_external = setup.n_external
        n_total = setup.n_total
        source_fn = setup.source_fn

        # Initialize state
        V = jnp.zeros(n_total, dtype=get_default_dtype())
        times_list: List[float] = []
        voltages_dict: Dict[int, List[float]] = {i: [] for i in range(n_external)}

        total_nr_iters = 0
        non_converged_steps: List[Tuple[float, float]] = []

        # Compute number of timesteps (using integer-based iteration)
        num_timesteps = self._compute_num_timesteps(t_stop, dt)
        if num_timesteps > max_steps:
            num_timesteps = max_steps
            dt = t_stop / (max_steps - 1) if max_steps > 1 else t_stop
            logger.info(f"{self.name}: Limiting to {max_steps} steps, dt={dt:.2e}s")

        logger.info(f"{self.name}: Starting simulation ({num_timesteps} timesteps, "
                   f"{n_total} nodes, {'sparse' if self.use_sparse else 'dense'} solver)")
        t_start = time_module.perf_counter()

        for step_idx in range(num_timesteps):
            t = step_idx * dt

            # Evaluate sources at time t
            source_values = source_fn(t)
            vsource_vals, isource_vals = self._build_source_arrays(source_values)

            # JIT-compiled NR solve
            V_new, iterations, converged, max_f = nr_solve(V, vsource_vals, isource_vals)

            # Extract Python values for tracking
            nr_iters = int(iterations)
            is_converged = bool(converged)
            residual = float(max_f)

            V = V_new
            total_nr_iters += nr_iters

            if not is_converged:
                non_converged_steps.append((t, residual))
                if nr_iters >= MAX_NR_ITERATIONS:
                    logger.warning(f"t={t:.2e}s hit max iterations ({MAX_NR_ITERATIONS}), "
                                 f"max_f={residual:.2e}")

            # Record state
            times_list.append(t)
            for i in range(n_external):
                voltages_dict[i].append(float(V[i]))

        wall_time = time_module.perf_counter() - t_start

        # Build results
        times = jnp.array(times_list)
        voltages = {i: jnp.array(v) for i, v in voltages_dict.items()}

        stats = {
            'total_timesteps': len(times_list),
            'total_nr_iterations': total_nr_iters,
            'avg_nr_iterations': total_nr_iters / max(len(times_list), 1),
            'non_converged_count': len(non_converged_steps),
            'non_converged_steps': non_converged_steps,
            'convergence_rate': 1.0 - len(non_converged_steps) / max(len(times_list), 1),
            'wall_time': wall_time,
            'time_per_step_ms': wall_time / len(times_list) * 1000 if times_list else 0,
            'strategy': 'python_loop',
            'solver': 'sparse' if self.use_sparse else 'dense',
        }

        logger.info(f"{self.name}: Completed {len(times_list)} steps in {wall_time:.3f}s "
                   f"({stats['time_per_step_ms']:.2f}ms/step, "
                   f"{total_nr_iters} NR iters, "
                   f"{len(non_converged_steps)} non-converged)")

        return times, voltages, stats
