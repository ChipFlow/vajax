"""Adaptive timestep transient analysis strategy.

This module implements LTE-based adaptive timestep control for transient
analysis, following the predictor-corrector algorithm used by VACASK.

The algorithm:
1. Use polynomial extrapolation to predict solution at next timestep
2. Solve with Newton-Raphson (corrector step)
3. Estimate Local Truncation Error (LTE) from predictor-corrector difference
4. Adjust timestep based on LTE and tolerance requirements
5. Accept or reject the step based on error threshold

Key benefits:
- Automatically uses smaller steps during fast transients
- Uses larger steps during slow evolution
- Matches VACASK behavior for validation

Reference: VACASK coretran.cpp lines 1070-1220
"""

import time as time_module
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp

from jax_spice._logging import logger
from jax_spice.analysis.integration import IntegrationMethod, compute_coefficients

from .base import TransientStrategy
from .predictor import (
    PredictorCoeffs,
    compute_new_timestep,
    compute_predictor_coeffs,
    estimate_lte,
    predict,
)

if TYPE_CHECKING:
    from jax_spice.analysis.engine import CircuitEngine


@dataclass
class AdaptiveConfig:
    """Configuration for adaptive timestep control.

    Attributes:
        lte_ratio: LTE tolerance multiplier (tran_lteratio). Higher values
            allow larger LTE before reducing timestep. Default 3.5.
        redo_factor: Rejection threshold (tran_redofactor). If current_dt /
            new_dt exceeds this, the timestep is rejected. Default 2.5.
        reltol: Relative tolerance for LTE comparison. Default 1e-3.
        abstol: Absolute tolerance for LTE comparison. Default 1e-12.
        min_dt: Minimum allowed timestep. Default 1e-18 seconds.
        max_dt: Maximum allowed timestep. Default infinity (no limit).
        warmup_steps: Number of fixed-dt steps before enabling LTE control.
            Need at least 2 past solutions for linear extrapolation. Default 2.
        max_order: Maximum order for polynomial predictor. Default 2 (quadratic).
        grow_factor: Maximum factor by which timestep can grow per step. Default 2.0.
    """

    lte_ratio: float = 3.5
    redo_factor: float = 2.5
    reltol: float = 1e-3
    abstol: float = 1e-12
    min_dt: float = 1e-18
    max_dt: float = float("inf")
    warmup_steps: int = 2
    max_order: int = 2
    grow_factor: float = 2.0


@dataclass
class SolutionHistory:
    """Circular buffer for past solutions and timesteps.

    Stores the most recent solutions for polynomial extrapolation predictor.
    Elements are stored most-recent-first: V[0] = V_n, V[1] = V_{n-1}, etc.

    Attributes:
        V: List of past voltage vectors, most recent first
        Q: List of past charge vectors, most recent first
        dt: List of past timesteps, most recent first
        max_depth: Maximum number of history entries to keep
    """

    V: List[jax.Array] = field(default_factory=list)
    Q: List[jax.Array] = field(default_factory=list)
    dt: List[float] = field(default_factory=list)
    max_depth: int = 4

    def push(self, V_new: jax.Array, Q_new: jax.Array, dt_new: float) -> None:
        """Add new solution to history, evicting oldest if necessary.

        Args:
            V_new: Voltage vector for accepted timestep
            Q_new: Charge vector for accepted timestep
            dt_new: Timestep that was used
        """
        self.V.insert(0, V_new)
        self.Q.insert(0, Q_new)
        self.dt.insert(0, dt_new)

        # Maintain max depth
        if len(self.V) > self.max_depth:
            self.V.pop()
            self.Q.pop()
            self.dt.pop()

    def clear(self) -> None:
        """Clear all history."""
        self.V.clear()
        self.Q.clear()
        self.dt.clear()

    @property
    def depth(self) -> int:
        """Number of valid history entries."""
        return len(self.V)


@dataclass
class AdaptiveStats:
    """Statistics from adaptive timestep simulation.

    Attributes:
        total_timesteps: Total number of accepted timesteps
        accepted_steps: Number of accepted steps (same as total_timesteps)
        rejected_steps: Number of rejected steps (LTE too large)
        total_nr_iterations: Total Newton-Raphson iterations across all steps
        min_dt_used: Smallest timestep that was used
        max_dt_used: Largest timestep that was used
        wall_time: Total wall clock time for simulation
    """

    total_timesteps: int = 0
    accepted_steps: int = 0
    rejected_steps: int = 0
    total_nr_iterations: int = 0
    min_dt_used: float = float("inf")
    max_dt_used: float = 0.0
    wall_time: float = 0.0


class AdaptiveStrategy(TransientStrategy):
    """Transient analysis with LTE-based adaptive timestep control.

    This strategy implements a predictor-corrector scheme where:
    1. The predictor uses polynomial extrapolation from past solutions
    2. The corrector uses Newton-Raphson with the implicit integration method
    3. LTE is estimated from the predictor-corrector difference
    4. Timestep is adjusted based on LTE magnitude vs tolerance

    Example:
        config = AdaptiveConfig(lte_ratio=3.5, redo_factor=2.5)
        strategy = AdaptiveStrategy(runner, config=config)
        times, voltages, stats = strategy.run(t_stop=1e-6, dt=1e-9)
        print(f"Accepted: {stats['accepted_steps']}, Rejected: {stats['rejected_steps']}")
    """

    def __init__(
        self,
        runner: "CircuitEngine",
        use_sparse: bool = False,
        backend: str = "cpu",
        config: Optional[AdaptiveConfig] = None,
    ):
        """Initialize adaptive transient strategy.

        Args:
            runner: CircuitEngine instance with parsed circuit
            use_sparse: If True, use sparse solver; if False, use dense solver
            backend: 'cpu' or 'gpu' for device evaluation
            config: Adaptive timestep configuration. If None, uses defaults.
        """
        super().__init__(runner, use_sparse, backend)
        self.config = config or AdaptiveConfig()

    def run(
        self, t_stop: float, dt: float, max_steps: int = 1000000
    ) -> Tuple[jax.Array, Dict[str, jax.Array], Dict[str, jax.Array], Dict]:
        """Run transient analysis with adaptive timestep control.

        Args:
            t_stop: Simulation stop time in seconds
            dt: Initial timestep in seconds (will be adapted)
            max_steps: Maximum number of timesteps (safety limit)

        Returns:
            Tuple of (times, voltages, currents, stats) where:
            - times: JAX array of accepted time points
            - voltages: Dict mapping node name to voltage array
            - currents: Dict mapping vsource name to current array
            - stats: Dict with adaptive timestep statistics
        """
        # Ensure setup and solver are ready
        setup = self.ensure_setup()
        nr_solve = self.ensure_solver()

        n_external = setup.n_external
        n_total = setup.n_total
        n_unknowns = setup.n_unknowns
        source_fn = setup.source_fn
        config = self.config

        # Get integration method from runner's analysis_params
        tran_method = self.runner.analysis_params.get(
            "tran_method", IntegrationMethod.BACKWARD_EULER
        )

        # Initialize state
        V = jnp.zeros(n_total, dtype=jnp.float64)
        history = SolutionHistory(max_depth=config.max_order + 2)
        stats = AdaptiveStats()

        # Result accumulators (Python lists for dynamic sizing)
        times_list: List[float] = []
        voltages_dict: Dict[int, List[float]] = {i: [] for i in range(n_external)}

        # Get vsource names for current tracking
        vsource_names = setup.source_device_data.get('vsource', {}).get('names', [])
        current_history: List[jax.Array] = []

        # Initialize charge state from DC operating point
        Q_prev = jnp.zeros(n_unknowns, dtype=jnp.float64)
        if hasattr(self.runner, "_cached_build_system") and hasattr(
            self.runner, "_device_arrays"
        ):
            vsource_dc = setup.source_device_data.get("vsource", {}).get(
                "dc", jnp.array([])
            )
            isource_dc = setup.source_device_data.get("isource", {}).get(
                "dc", jnp.array([])
            )
            if vsource_dc is not None and len(vsource_dc) > 0:
                vsource_dc = jnp.array(vsource_dc)
            else:
                vsource_dc = jnp.array([])
            if isource_dc is not None and len(isource_dc) > 0:
                isource_dc = jnp.array(isource_dc)
            else:
                isource_dc = jnp.array([])
            _, _, Q_prev, _ = self.runner._cached_build_system(
                V,
                vsource_dc,
                isource_dc,
                jnp.zeros(n_unknowns),
                0.0,
                self.runner._device_arrays,
            )
            Q_prev.block_until_ready()

        # Initialize dQdt_prev for trapezoidal method
        integ_coeffs = compute_coefficients(tran_method, dt)
        dQdt_prev = (
            jnp.zeros(n_unknowns, dtype=jnp.float64)
            if integ_coeffs.needs_dqdt_history
            else None
        )

        # Initialize Q_prev2 for Gear2 method
        Q_prev2 = (
            jnp.zeros(n_unknowns, dtype=jnp.float64)
            if integ_coeffs.history_depth >= 2
            else None
        )

        # Get device_arrays for nr_solve
        device_arrays = (
            self.runner._device_arrays
            if hasattr(self.runner, "_device_arrays")
            else {}
        )

        # Apply max_dt constraint from config
        current_dt = min(dt, config.max_dt)
        current_dt = max(current_dt, config.min_dt)

        logger.info(
            f"{self.name}: Starting adaptive simulation to t={t_stop:.2e}s, "
            f"initial dt={current_dt:.2e}s, method={tran_method.value}"
        )
        t_start = time_module.perf_counter()

        # Record initial state at t=0
        times_list.append(0.0)
        for i in range(n_external):
            voltages_dict[i].append(float(V[i]))

        # Record initial current (zeros at t=0 before simulation starts)
        if len(vsource_names) > 0:
            current_history.append(jnp.zeros(len(vsource_names), dtype=jnp.float64))

        # Main simulation loop
        t = 0.0
        step_count = 0
        warmup_complete = False

        while t < t_stop and step_count < max_steps:
            step_count += 1

            # Recompute integration coefficients for current timestep
            integ_coeffs = compute_coefficients(tran_method, current_dt)

            # Get source values at next time
            t_next = t + current_dt
            source_values = source_fn(t_next)
            vsource_vals, isource_vals = self._build_source_arrays(source_values)

            # Prediction step (if we have enough history)
            V_init = V  # Default: use previous solution
            pred_coeffs: Optional[PredictorCoeffs] = None

            if warmup_complete and history.depth >= 2:
                # Compute predictor coefficients
                order = min(history.depth - 1, config.max_order)
                try:
                    pred_coeffs = compute_predictor_coeffs(
                        history.dt, current_dt, order
                    )
                    # Predict solution
                    V_init = predict(pred_coeffs, history.V)
                except Exception as e:
                    logger.warning(f"Predictor failed: {e}, using previous solution")
                    pred_coeffs = None

            # Corrector step: Newton-Raphson solve
            V_new, iterations, converged, max_f, Q, dQdt, I_vsource = nr_solve(
                V_init,
                vsource_vals,
                isource_vals,
                Q_prev,
                integ_coeffs.c0,
                device_arrays,
                1e-12,
                0.0,  # gmin, gshunt
                integ_coeffs.c1,
                integ_coeffs.d1,
                dQdt_prev,
                integ_coeffs.c2,
                Q_prev2,
            )

            nr_iters = int(iterations)
            is_converged = bool(converged)
            stats.total_nr_iterations += nr_iters

            # Handle NR convergence failure
            if not is_converged:
                # Halve timestep and retry
                old_dt = current_dt
                current_dt = max(current_dt / 2, config.min_dt)
                stats.rejected_steps += 1
                logger.debug(
                    f"t={t:.2e}s: NR failed after {nr_iters} iters, "
                    f"reducing dt {old_dt:.2e} -> {current_dt:.2e}"
                )
                if current_dt <= config.min_dt:
                    logger.warning(f"t={t:.2e}s: dt at minimum, accepting anyway")
                else:
                    continue  # Retry with smaller timestep

            # LTE estimation and timestep adjustment (only after warmup)
            dt_new = current_dt
            accept_step = True

            if warmup_complete and pred_coeffs is not None:
                # Estimate LTE from predictor-corrector difference
                lte = estimate_lte(
                    V_init,  # V_predicted
                    V_new,  # V_corrected
                    pred_coeffs.error_coeff,
                    integ_coeffs.error_coeff,
                )

                # Compute new timestep based on LTE
                dt_new, max_lte_ratio = compute_new_timestep(
                    lte,
                    V_new,
                    config.reltol,
                    config.abstol,
                    config.lte_ratio,
                    current_dt,
                    pred_coeffs.order,
                    config.min_dt,
                    config.max_dt,
                )

                # Accept/reject decision
                if current_dt / dt_new > config.redo_factor:
                    # LTE too large - reject and retry with smaller timestep
                    stats.rejected_steps += 1
                    current_dt = dt_new
                    logger.debug(
                        f"t={t:.2e}s: LTE too large (ratio={max_lte_ratio:.2f}), "
                        f"reducing dt to {current_dt:.2e}"
                    )
                    continue  # Retry with smaller timestep

            # Step accepted - update state
            stats.accepted_steps += 1
            stats.min_dt_used = min(stats.min_dt_used, current_dt)
            stats.max_dt_used = max(stats.max_dt_used, current_dt)

            # Update history
            history.push(V_new, Q, current_dt)

            # Update charge state for next step
            if integ_coeffs.history_depth >= 2:
                Q_prev2 = Q_prev
            Q_prev = Q
            if integ_coeffs.needs_dqdt_history:
                dQdt_prev = dQdt

            # Update voltage
            V = V_new

            # Record accepted state
            t = t_next
            times_list.append(t)
            for i in range(n_external):
                voltages_dict[i].append(float(V[i]))

            # Record vsource currents (computed from KCL in build_system)
            if len(vsource_names) > 0 and I_vsource.size > 0:
                current_history.append(I_vsource)

            # Check if warmup is complete
            if not warmup_complete and history.depth >= config.warmup_steps:
                warmup_complete = True
                logger.debug(f"Warmup complete at t={t:.2e}s, enabling LTE control")

            # Adjust timestep for next iteration (grow slowly)
            if warmup_complete:
                # Limit growth to grow_factor
                dt_new = min(dt_new, current_dt * config.grow_factor)
                # Apply bounds
                dt_new = max(config.min_dt, min(config.max_dt, dt_new))
                # Don't overshoot t_stop
                if t + dt_new > t_stop:
                    dt_new = t_stop - t
                current_dt = dt_new

        wall_time = time_module.perf_counter() - t_start
        stats.total_timesteps = len(times_list)
        stats.wall_time = wall_time

        # Build results with string node names
        times = jnp.array(times_list)
        idx_to_voltage = {i: jnp.array(v) for i, v in voltages_dict.items()}
        voltages: Dict[str, jax.Array] = {}
        for name, idx in self.runner.node_names.items():
            if 0 < idx < n_external:
                voltages[name] = idx_to_voltage[idx]

        # Build currents dictionary from current history
        currents: Dict[str, jax.Array] = {}
        if current_history and len(vsource_names) > 0:
            I_stacked = jnp.stack(current_history)  # Shape: (n_timesteps, n_vsources)
            for i, name in enumerate(vsource_names):
                currents[name] = I_stacked[:, i]

        # Build stats dict
        stats_dict = {
            "total_timesteps": stats.total_timesteps,
            "accepted_steps": stats.accepted_steps,
            "rejected_steps": stats.rejected_steps,
            "total_nr_iterations": stats.total_nr_iterations,
            "avg_nr_iterations": (
                stats.total_nr_iterations / max(stats.accepted_steps, 1)
            ),
            "min_dt_used": stats.min_dt_used if stats.min_dt_used != float("inf") else 0,
            "max_dt_used": stats.max_dt_used,
            "wall_time": wall_time,
            "time_per_step_ms": wall_time / max(len(times_list), 1) * 1000,
            "strategy": "adaptive",
            "solver": "sparse" if self.use_sparse else "dense",
            "integration_method": tran_method.value,
            "lte_ratio": config.lte_ratio,
            "redo_factor": config.redo_factor,
        }

        logger.info(
            f"{self.name}: Completed {stats.total_timesteps} steps in {wall_time:.3f}s "
            f"({stats_dict['time_per_step_ms']:.2f}ms/step, "
            f"{stats.rejected_steps} rejected, "
            f"dt range [{stats.min_dt_used:.2e}, {stats.max_dt_used:.2e}])"
        )

        return times, voltages, currents, stats_dict
