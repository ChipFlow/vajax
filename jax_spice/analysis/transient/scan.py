"""Lax.scan transient analysis strategy.

This strategy uses JAX's lax.scan for a fully JIT-compiled simulation loop.
It provides the best performance but with less debugging information.

Performance: ~0.1ms/step on CPU (ring benchmark) - 5x faster than Python loop

Advantages:
- Fully JIT-compiled (no Python loop overhead)
- Best performance for production runs
- Efficient GPU utilization

Disadvantages:
- Less per-step debugging info
- Longer initial compilation time (~20s vs ~13s)
- Requires same timestep count for warmup and timed run

IMPORTANT: JAX traces lax.scan based on array shapes. If you use different
timestep counts between warmup and timed run, JAX will re-trace (re-compile),
causing apparent "slow" performance. Always use the same timestep count for
both warmup and timed run.

Use this strategy when:
- Running production benchmarks
- Need maximum performance
- Circuit is already validated
"""

import time as time_module
from typing import Any, Callable, Dict, List, Tuple, Optional

import jax
import jax.numpy as jnp

from .base import TransientStrategy
from jax_spice._logging import logger


class ScanStrategy(TransientStrategy):
    """Transient analysis using lax.scan for fully JIT-compiled simulation.

    This strategy pre-computes all source values and uses lax.scan to
    eliminate Python loop overhead entirely, achieving 5x+ speedup.

    Example:
        runner = CircuitEngine(sim_path)
        runner.parse()

        strategy = ScanStrategy(runner, use_sparse=False)

        # Warmup (JIT compilation) - MUST use same timesteps as timed run!
        num_steps = 100
        dt = 1e-12
        _ = strategy.run(t_stop=num_steps*dt, dt=dt)

        # Timed run (same timesteps!)
        times, voltages, stats = strategy.run(t_stop=num_steps*dt, dt=dt)
        print(f"Performance: {stats['time_per_step_ms']:.2f}ms/step")
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Cache for the compiled scan function
        self._cached_scan_fn: Optional[Callable] = None
        self._cached_scan_key: Optional[Tuple] = None

    def run(self, t_stop: float, dt: float,
            max_steps: int = 10000) -> Tuple[jax.Array, Dict[int, jax.Array], Dict]:
        """Run transient analysis with lax.scan.

        Args:
            t_stop: Simulation stop time in seconds
            dt: Time step in seconds
            max_steps: Maximum number of time steps

        Returns:
            Tuple of (times, voltages, stats) where:
            - times: JAX array of time points
            - voltages: Dict mapping node index to voltage array
            - stats: Dict with basic statistics
        """
        # Ensure setup and solver are ready
        setup = self.ensure_setup()
        nr_solve = self.ensure_solver()

        n_external = setup.n_external
        n_total = setup.n_total
        n_unknowns = setup.n_unknowns
        source_device_data = setup.source_device_data

        n_vsources = len(source_device_data.get('vsource', {}).get('names', []))
        n_isources = len(source_device_data.get('isource', {}).get('names', []))
        n_nodes = n_unknowns + 1

        # Compute number of timesteps
        num_timesteps = self._compute_num_timesteps(t_stop, dt)
        if num_timesteps > max_steps:
            num_timesteps = max_steps
            dt = t_stop / (max_steps - 1) if max_steps > 1 else t_stop
            logger.info(f"{self.name}: Limiting to {max_steps} steps, dt={dt:.2e}s")

        logger.info(f"{self.name}: Starting simulation ({num_timesteps} timesteps, "
                   f"{n_total} nodes, {'sparse' if self.use_sparse else 'dense'} solver)")

        # Generate time array for source pre-computation
        times = jnp.linspace(0.0, t_stop, num_timesteps)

        # Pre-compute source values for ALL timesteps
        # This is required because lax.scan needs static input shapes
        logger.debug(f"{self.name}: Pre-computing source values...")
        t_precompute = time_module.perf_counter()

        all_vsource_vals = self._precompute_sources(
            'vsource', source_device_data, times, num_timesteps
        )
        all_isource_vals = self._precompute_sources(
            'isource', source_device_data, times, num_timesteps
        )

        logger.debug(f"{self.name}: Source pre-computation: "
                    f"{time_module.perf_counter() - t_precompute:.3f}s")

        # Initial state
        V0 = jnp.zeros(n_nodes, dtype=jnp.float64)

        # Get or create the cached scan function
        # Cache key does NOT include num_timesteps - JAX handles variable-length
        scan_cache_key = (n_nodes, n_vsources, n_isources, n_external, self.use_dense)

        if self._cached_scan_fn is not None and self._cached_scan_key == scan_cache_key:
            run_simulation = self._cached_scan_fn
            logger.debug(f"{self.name}: Reusing cached lax.scan function")
        else:
            run_simulation = self._make_scan_fn(nr_solve, n_external)
            self._cached_scan_fn = run_simulation
            self._cached_scan_key = scan_cache_key
            logger.info(f"{self.name}: Created and cached lax.scan function")

        # Run the simulation
        logger.info(f"{self.name}: Running lax.scan simulation...")
        t0 = time_module.perf_counter()
        all_V, all_iters, all_converged = run_simulation(V0, all_vsource_vals, all_isource_vals)
        jax.block_until_ready(all_V)  # Ensure computation is complete
        wall_time = time_module.perf_counter() - t0

        # Build results
        total_iters = int(jnp.sum(all_iters))
        non_converged = int(jnp.sum(~all_converged))

        stats = {
            'total_timesteps': num_timesteps,
            'total_nr_iterations': total_iters,
            'avg_nr_iterations': total_iters / max(num_timesteps, 1),
            'non_converged_count': non_converged,
            'non_converged_steps': [],  # Not tracked in scan mode
            'convergence_rate': 1.0 - non_converged / max(num_timesteps, 1),
            'wall_time': wall_time,
            'time_per_step_ms': wall_time / num_timesteps * 1000,
            'strategy': 'lax_scan',
            'solver': 'sparse' if self.use_sparse else 'dense',
        }

        logger.info(f"{self.name}: Completed {num_timesteps} steps in {wall_time:.3f}s "
                   f"({stats['time_per_step_ms']:.2f}ms/step, "
                   f"{total_iters} NR iters, {non_converged} non-converged)")

        # Convert to dict format - use string names only
        voltages: Dict[str, jax.Array] = {}
        for name, idx in self.runner.node_names.items():
            if idx > 0 and idx < n_external:  # Skip ground (0), only external nodes
                voltages[name] = all_V[:, idx]

        return times, voltages, stats

    def _precompute_sources(self, source_type: str,
                           source_device_data: Dict,
                           times: jax.Array,
                           num_timesteps: int) -> jax.Array:
        """Pre-compute source values for all timesteps.

        Args:
            source_type: 'vsource' or 'isource'
            source_device_data: Dict with source device info
            times: Array of time points
            num_timesteps: Number of timesteps

        Returns:
            Array of shape [num_timesteps, n_sources]
        """
        if source_type not in source_device_data:
            return jnp.zeros((num_timesteps, 0))

        source_names = source_device_data[source_type]['names']
        if not source_names:
            return jnp.zeros((num_timesteps, 0))

        all_values = []
        for name in source_names:
            dev = next((d for d in self.runner.devices if d['name'] == name), None)
            if dev:
                src_fn = self.runner._get_source_fn_for_device(dev)
                if src_fn is not None:
                    # Time-varying source - evaluate at all times
                    vals = jax.vmap(src_fn)(times)
                else:
                    # DC source
                    dc_val = dev['params'].get('dc', 0.0)
                    vals = jnp.full(num_timesteps, float(dc_val))
                all_values.append(vals)

        return jnp.stack(all_values, axis=1) if all_values else jnp.zeros((num_timesteps, 0))

    def _make_scan_fn(self, nr_solve: Callable, n_external: int) -> Callable:
        """Create a JIT-compiled scan function.

        The function is created in a factory to ensure proper closure capture
        and avoid re-tracing issues.

        Args:
            nr_solve: JIT-compiled Newton-Raphson solver
            n_external: Number of external nodes

        Returns:
            JIT-compiled function that runs the full simulation
        """
        @jax.jit
        def run_simulation_with_outputs(V_init, all_vsource, all_isource):
            """Run simulation with time-varying sources using lax.scan.

            Args:
                V_init: Initial voltage vector
                all_vsource: Pre-computed vsource values [num_timesteps, n_vsources]
                all_isource: Pre-computed isource values [num_timesteps, n_isources]

            Returns:
                Tuple of (all_V, all_iters, all_converged)
            """
            def step_fn(V, source_vals):
                vsource_vals, isource_vals = source_vals
                V_new, iterations, converged, max_f = nr_solve(V, vsource_vals, isource_vals)
                return V_new, (V_new[:n_external], iterations, converged)

            # Stack source arrays for scan input
            source_inputs = (all_vsource, all_isource)
            _, (all_V, all_iters, all_converged) = jax.lax.scan(
                step_fn, V_init, source_inputs
            )
            return all_V, all_iters, all_converged

        return run_simulation_with_outputs
