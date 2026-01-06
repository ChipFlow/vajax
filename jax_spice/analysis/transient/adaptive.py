"""Adaptive transient analysis with LTE-based timestep control.

This strategy implements Local Truncation Error (LTE) based timestep control
similar to VACASK/SPICE3f5. It uses a predictor-corrector approach to estimate
the local truncation error and adjust the timestep accordingly.

Key features:
- Predictor-corrector LTE estimation
- Timestep acceptance/rejection based on LTE
- Automatic timestep adjustment
- Compatible with trapezoidal integration

VACASK defaults (from options.cpp):
- tran_lteratio = 3.5 (LTE safety factor - larger = looser tolerance)
- tran_redofactor = 2.5 (rejection threshold - reject if hk/hk_new > this)
- reltol = 1e-3 (relative tolerance for LTE check)

References:
- VACASK coretran.cpp lines 1070-1223
- SPICE3f5 ckttran.c
"""

import time as time_module
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp

from .base import TransientStrategy
from jax_spice._logging import logger


@dataclass
class AdaptiveConfig:
    """Configuration for LTE-based adaptive timestep control."""

    # LTE control parameters (VACASK defaults)
    lte_ratio: float = 3.5  # tran_lteratio - safety factor for LTE
    redo_factor: float = 2.5  # tran_redofactor - reject if hk/hk_new > this
    reltol: float = 1e-3  # Relative tolerance for LTE
    abstol: float = 1e-12  # Absolute tolerance for LTE

    # Timestep limits
    min_step_factor: float = 0.125  # tran_fmin - minimum step reduction
    max_step_factor: float = 2.0  # tran_fmax - maximum step increase
    initial_step_factor: float = 0.25  # tran_fs - initial step scaling

    # NR failure handling
    step_on_failure: float = 0.25  # tran_ft - step reduction on NR failure

    # Predictor
    use_predictor: bool = False  # tran_predictor - use predicted value for NR start

    # Integration order
    order: int = 1  # Backward Euler (order 1)

    # Debug
    debug: int = 0  # Debug verbosity


class AdaptiveStrategy(TransientStrategy):
    """Transient analysis with LTE-based adaptive timestep control.

    This strategy uses predictor-corrector LTE estimation to automatically
    adjust the timestep size based on accuracy requirements.

    Example:
        runner = CircuitEngine(sim_path)
        runner.parse()

        config = AdaptiveConfig(
            lte_ratio=3.5,
            redo_factor=2.5,
            reltol=1e-3,
        )
        strategy = AdaptiveStrategy(runner, config=config)

        times, voltages, stats = strategy.run(t_stop=1e-6, dt=1e-9)
        print(f"Accepted: {stats['accepted']}, Rejected: {stats['rejected']}")
    """

    def __init__(self, *args, config: Optional[AdaptiveConfig] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config or AdaptiveConfig()

    def run(self, t_stop: float, dt: float,
            max_steps: int = 100000) -> Tuple[jax.Array, Dict[int, jax.Array], Dict]:
        """Run transient analysis with adaptive timestep control.

        Args:
            t_stop: Simulation stop time in seconds
            dt: Initial/target time step in seconds
            max_steps: Maximum number of time steps (accepted + rejected)

        Returns:
            Tuple of (times, voltages, stats)
        """
        cfg = self.config

        # Ensure setup and solver are ready
        setup = self.ensure_setup()
        nr_solve = self.ensure_solver()
        device_arrays = self.get_device_arrays()

        n_external = setup.n_external
        n_total = setup.n_total
        n_unknowns = setup.n_unknowns
        source_fn = setup.source_fn

        # Initialize state
        V = jnp.zeros(n_total, dtype=jnp.float64)
        Q_prev = self.get_initial_Q()
        # Use backward Euler for stability (no dQdt tracking needed)

        # History for predictor (need at least 2 past points for order-1 predictor)
        # V_history[0] = current (t_k), V_history[1] = t_{k-1}, etc.
        V_history: List[jax.Array] = [V]
        dt_history: List[float] = []  # Past timesteps

        # Results storage
        times_list: List[float] = []
        voltages_dict: Dict[int, List[float]] = {i: [] for i in range(n_external)}

        # Statistics
        total_nr_iters = 0
        accepted_count = 0
        rejected_count = 0
        non_converged_steps: List[Tuple[float, float]] = []

        # Current time and timestep
        t = 0.0
        hk = dt * cfg.initial_step_factor  # Start with smaller step

        # Record initial point
        times_list.append(t)
        for i in range(n_external):
            voltages_dict[i].append(float(V[i]))

        logger.info(f"{self.name}: Starting adaptive simulation (t_stop={t_stop:.2e}s, "
                   f"initial dt={hk:.2e}s, {n_total} nodes)")
        t_start = time_module.perf_counter()

        step_count = 0
        while t < t_stop and step_count < max_steps:
            step_count += 1

            # Target time for this step
            t_new = t + hk
            if t_new > t_stop:
                hk = t_stop - t
                t_new = t_stop

            # Compute inverse timestep for this step
            inv_dt = 1.0 / hk

            # Predict solution at t_new using polynomial extrapolation
            V_predicted = self._predict(V_history, dt_history, hk, cfg.order)

            # Use predicted value as NR starting point if enabled
            V_start = V_predicted if cfg.use_predictor else V

            # Evaluate sources at t_new
            source_values = source_fn(t_new)
            vsource_vals, isource_vals = self._build_source_arrays(source_values)

            # Run NR solver with backward Euler (uses default integ coefficients)
            V_new, iterations, converged, max_f, Q = nr_solve(
                V_start, vsource_vals, isource_vals, Q_prev, inv_dt, device_arrays
            )

            nr_iters = int(iterations)
            is_converged = bool(converged)
            residual = float(max_f)
            total_nr_iters += nr_iters

            if not is_converged:
                # NR failed - reduce timestep and retry
                rejected_count += 1
                non_converged_steps.append((t_new, residual))

                hk_new = hk * cfg.step_on_failure
                if cfg.debug > 0:
                    logger.debug(f"t={t_new:.2e}s: NR failed ({nr_iters} iters), "
                               f"reducing dt {hk:.2e} -> {hk_new:.2e}")

                # Check for timestep too small
                if hk_new < t_new * 1e-15:
                    logger.error(f"Timestep too small at t={t_new:.2e}s")
                    break

                hk = hk_new
                continue

            # NR converged - check LTE
            accept, hk_new = self._check_lte_and_adjust_step(
                V_new, V_predicted, V_history, dt_history, hk, cfg
            )

            if not accept:
                # LTE too large - reject and retry with smaller step
                rejected_count += 1

                if cfg.debug > 0:
                    logger.debug(f"t={t_new:.2e}s: LTE too large, "
                               f"reducing dt {hk:.2e} -> {hk_new:.2e}")

                hk = hk_new
                continue

            # Accept the point
            accepted_count += 1
            t = t_new
            V = V_new

            # Update charge state for backward Euler (just Q_prev)
            Q_prev = Q

            # Update history
            V_history.insert(0, V)
            if len(V_history) > 4:  # Keep at most 4 past points
                V_history.pop()
            dt_history.insert(0, hk)
            if len(dt_history) > 3:
                dt_history.pop()

            # Record state
            times_list.append(t)
            for i in range(n_external):
                voltages_dict[i].append(float(V[i]))

            if cfg.debug > 1:
                logger.debug(f"t={t:.2e}s: accepted, next dt={hk_new:.2e}")

            # Update timestep for next iteration
            hk = hk_new

            # Don't exceed max step
            max_step = dt * cfg.max_step_factor
            if hk > max_step:
                hk = max_step

        wall_time = time_module.perf_counter() - t_start

        # Build results
        times = jnp.array(times_list)
        idx_to_voltage = {i: jnp.array(v) for i, v in voltages_dict.items()}
        voltages: Dict[str, jax.Array] = {}
        for name, idx in self.runner.node_names.items():
            if idx > 0 and idx < n_external:
                voltages[name] = idx_to_voltage[idx]

        stats = {
            'total_timesteps': len(times_list),
            'accepted': accepted_count,
            'rejected': rejected_count,
            'total_nr_iterations': total_nr_iters,
            'avg_nr_iterations': total_nr_iters / max(step_count, 1),
            'non_converged_count': len(non_converged_steps),
            'non_converged_steps': non_converged_steps[:10],  # First 10
            'convergence_rate': 1.0 - len(non_converged_steps) / max(step_count, 1),
            'wall_time': wall_time,
            'time_per_step_ms': wall_time / len(times_list) * 1000 if times_list else 0,
            'strategy': 'adaptive',
            'solver': 'sparse' if self.use_sparse else 'dense',
        }

        logger.info(f"{self.name}: Completed {accepted_count} accepted, {rejected_count} rejected "
                   f"in {wall_time:.3f}s ({stats['time_per_step_ms']:.2f}ms/step)")

        return times, voltages, stats

    def _predict(self, V_history: List[jax.Array], dt_history: List[float],
                 hk: float, order: int) -> jax.Array:
        """Predict solution at t + hk using polynomial extrapolation.

        For order 1 (linear extrapolation):
            V_pred = V[0] + (V[0] - V[1]) * hk / dt[0]

        Args:
            V_history: Past solutions [V(t_k), V(t_{k-1}), ...]
            dt_history: Past timesteps [dt_{k-1}, dt_{k-2}, ...]
            hk: Current timestep
            order: Prediction order (1 or 2)

        Returns:
            Predicted solution at t + hk
        """
        if len(V_history) < 2 or len(dt_history) < 1:
            # Not enough history - use current value
            return V_history[0]

        V0 = V_history[0]  # V(t_k)
        V1 = V_history[1]  # V(t_{k-1})
        dt0 = dt_history[0]  # dt_{k-1}

        if order == 1 or len(V_history) < 3 or len(dt_history) < 2:
            # Linear extrapolation
            return V0 + (V0 - V1) * (hk / dt0)

        # Quadratic extrapolation (order 2)
        V2 = V_history[2]  # V(t_{k-2})
        dt1 = dt_history[1]  # dt_{k-2}

        # Coefficients for polynomial through (0, V0), (-dt0, V1), (-(dt0+dt1), V2)
        # evaluated at hk
        t1 = -dt0
        t2 = -(dt0 + dt1)

        # Lagrange interpolation
        L0 = ((hk - t1) * (hk - t2)) / ((0 - t1) * (0 - t2))
        L1 = ((hk - 0) * (hk - t2)) / ((t1 - 0) * (t1 - t2))
        L2 = ((hk - 0) * (hk - t1)) / ((t2 - 0) * (t2 - t1))

        return L0 * V0 + L1 * V1 + L2 * V2

    def _check_lte_and_adjust_step(
        self,
        V_new: jax.Array,
        V_predicted: jax.Array,
        V_history: List[jax.Array],
        dt_history: List[float],
        hk: float,
        cfg: AdaptiveConfig
    ) -> Tuple[bool, float]:
        """Check LTE and compute adjusted timestep.

        LTE estimation:
            lte = factor * (V_new - V_predicted)

        where factor = errorCoeff_integrator / (errorCoeff_integrator - errorCoeff_predictor)

        For trapezoidal (order 2), factor ≈ 0.5

        Args:
            V_new: Converged NR solution
            V_predicted: Predicted solution
            V_history: Past solutions
            dt_history: Past timesteps
            hk: Current timestep
            cfg: Adaptive configuration

        Returns:
            Tuple of (accept, hk_new) where:
            - accept: True if timestep should be accepted
            - hk_new: Suggested next timestep
        """
        if len(V_history) < 2 or len(dt_history) < 1:
            # Not enough history for LTE check - accept
            return True, min(hk * cfg.max_step_factor, hk * 2)

        # LTE factor for backward Euler + linear predictor
        # Backward Euler error coeff: 1/2 (for order 1)
        # Linear predictor error coeff: -1/2 (for order 1)
        # factor = (1/2) / ((1/2) - (-1/2)) = (1/2) / 1 = 0.5
        factor = 0.5

        # Compute LTE estimate
        lte = factor * (V_new - V_predicted)

        # Compute tolerance for each unknown
        # tol = max(|V_new| * reltol, abstol)
        V_abs = jnp.abs(V_new)
        tol = jnp.maximum(V_abs * cfg.reltol, cfg.abstol)

        # Scaled LTE ratio = |lte| / (tol * lte_ratio)
        # We want max(scaled_lte) to be < 1 for acceptance
        scaled_lte = jnp.abs(lte) / (tol * cfg.lte_ratio)
        max_ratio = float(jnp.max(scaled_lte))

        # Compute optimal timestep factor
        # LTE ∝ h^(order+1), so h_new = h * (1/ratio)^(1/(order+1))
        if max_ratio > 0:
            order = cfg.order
            hk_factor = max_ratio ** (-1.0 / (order + 1))
            hk_factor = max(hk_factor, cfg.min_step_factor)
            hk_factor = min(hk_factor, cfg.max_step_factor)
            hk_new = hk * hk_factor
        else:
            # LTE is zero - can increase step
            hk_new = hk * cfg.max_step_factor

        # Check if we should reject
        hk_ratio = hk / hk_new
        if cfg.redo_factor > 0 and hk_ratio > cfg.redo_factor:
            # LTE too large - reject
            return False, hk_new

        # Accept
        return True, hk_new
