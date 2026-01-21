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
import numpy as np

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
    """Circular buffer for past solutions and timesteps using JAX arrays.

    Uses fixed-size arrays with a write index for efficient updates without
    causing JAX recompilation. Elements are accessed most-recent-first via
    get_V(), get_dt() methods that handle the circular indexing.

    Attributes:
        V_buffer: Fixed-size buffer for voltage vectors (max_depth, n_total)
        Q_buffer: Fixed-size buffer for charge vectors (max_depth, n_unknowns)
        dt_buffer: Fixed-size buffer for timesteps (max_depth,)
        write_idx: Next write position in circular buffer
        count: Number of valid entries (0 to max_depth)
        max_depth: Maximum number of history entries to keep
    """

    V_buffer: jax.Array  # Shape: (max_depth, n_total)
    Q_buffer: jax.Array  # Shape: (max_depth, n_unknowns)
    dt_buffer: jax.Array  # Shape: (max_depth,)
    write_idx: int = 0  # Next write position
    count: int = 0  # Number of valid entries
    max_depth: int = 4

    @classmethod
    def create(
        cls, n_total: int, n_unknowns: int, max_depth: int = 4
    ) -> "SolutionHistory":
        """Create a new empty history buffer.

        Args:
            n_total: Size of voltage vectors (all nodes including ground)
            n_unknowns: Size of charge vectors (non-ground nodes)
            max_depth: Maximum history entries to keep

        Returns:
            Initialized SolutionHistory with zero-filled buffers
        """
        return cls(
            V_buffer=jnp.zeros((max_depth, n_total), dtype=jnp.float64),
            Q_buffer=jnp.zeros((max_depth, n_unknowns), dtype=jnp.float64),
            dt_buffer=jnp.zeros(max_depth, dtype=jnp.float64),
            write_idx=0,
            count=0,
            max_depth=max_depth,
        )

    def push(self, V_new: jax.Array, Q_new: jax.Array, dt_new: float) -> "SolutionHistory":
        """Add new solution to history, returning updated history.

        Args:
            V_new: Voltage vector for accepted timestep
            Q_new: Charge vector for accepted timestep
            dt_new: Timestep that was used

        Returns:
            New SolutionHistory with updated buffers
        """
        # Write to current position
        new_V = self.V_buffer.at[self.write_idx].set(V_new)
        new_Q = self.Q_buffer.at[self.write_idx].set(Q_new)
        new_dt = self.dt_buffer.at[self.write_idx].set(dt_new)

        # Advance write index (circular)
        new_write_idx = (self.write_idx + 1) % self.max_depth
        new_count = min(self.count + 1, self.max_depth)

        return SolutionHistory(
            V_buffer=new_V,
            Q_buffer=new_Q,
            dt_buffer=new_dt,
            write_idx=new_write_idx,
            count=new_count,
            max_depth=self.max_depth,
        )

    def get_V(self, i: int) -> jax.Array:
        """Get voltage vector at history position i (0 = most recent).

        Args:
            i: History index (0 = most recent, 1 = second most recent, etc.)

        Returns:
            Voltage vector at position i
        """
        # Most recent is at (write_idx - 1), second most recent at (write_idx - 2), etc.
        idx = (self.write_idx - 1 - i) % self.max_depth
        return self.V_buffer[idx]

    def get_dt(self, i: int) -> jax.Array:
        """Get timestep at history position i (0 = most recent).

        Args:
            i: History index (0 = most recent, 1 = second most recent, etc.)

        Returns:
            Timestep value at position i
        """
        idx = (self.write_idx - 1 - i) % self.max_depth
        return self.dt_buffer[idx]

    def get_V_list(self, n: int) -> List[jax.Array]:
        """Get list of n most recent voltage vectors (for predictor compatibility).

        Args:
            n: Number of entries to retrieve

        Returns:
            List of voltage vectors, most recent first
        """
        n = min(n, self.count)
        return [self.get_V(i) for i in range(n)]

    def get_dt_list(self, n: int) -> List[float]:
        """Get list of n most recent timesteps (for predictor compatibility).

        Args:
            n: Number of entries to retrieve

        Returns:
            List of timesteps, most recent first
        """
        n = min(n, self.count)
        return [float(self.get_dt(i)) for i in range(n)]

    @property
    def depth(self) -> int:
        """Number of valid history entries."""
        return len(self.V)


# =============================================================================
# JAX-Compatible Solution History (for JIT-compiled loops)
# =============================================================================


class SolutionHistoryJAX(NamedTuple):
    """JAX-compatible solution history using fixed-size arrays.

    Uses a "shift and write" approach where new entries are written to index 0
    and old entries are shifted down using jnp.roll.

    This is a NamedTuple so it's a valid JAX pytree and can be passed through
    JIT boundaries without issues.

    Attributes:
        V: Past voltage vectors, shape (max_depth, n_unknowns), most recent first
        Q: Past charge vectors, shape (max_depth, n_unknowns), most recent first
        dt: Past timesteps, shape (max_depth,), most recent first
        depth: Number of valid history entries (0 to max_depth)
    """

    V: Array  # (max_depth, n_unknowns)
    Q: Array  # (max_depth, n_unknowns)
    dt: Array  # (max_depth,)
    depth: Array  # Scalar int


def create_solution_history_jax(
    max_depth: int, n_voltages: int, n_charges: int = None, dtype=jnp.float64
) -> SolutionHistoryJAX:
    """Create an empty JAX-compatible solution history.

    Args:
        max_depth: Maximum number of history entries
        n_voltages: Number of voltage values (typically n_total)
        n_charges: Number of charge values (typically n_unknowns).
            If None, defaults to n_voltages.

    Returns:
        Empty SolutionHistoryJAX
    """
    if n_charges is None:
        n_charges = n_voltages
    return SolutionHistoryJAX(
        V=jnp.zeros((max_depth, n_voltages), dtype=dtype),
        Q=jnp.zeros((max_depth, n_charges), dtype=dtype),
        dt=jnp.zeros(max_depth, dtype=dtype),
        depth=jnp.array(0, dtype=jnp.int32),
    )


def push_solution_history_jax(
    history: SolutionHistoryJAX,
    V_new: Array,
    Q_new: Array,
    dt_new: Array,
) -> SolutionHistoryJAX:
    """Add new solution to history, shifting old entries.

    This function is JIT-compatible.

    Args:
        history: Current history state
        V_new: New voltage vector
        Q_new: New charge vector
        dt_new: New timestep (as JAX scalar)

    Returns:
        Updated SolutionHistoryJAX with new entry at index 0
    """
    max_depth = history.V.shape[0]

    # Shift old entries down by rolling and then overwriting index 0
    V_shifted = jnp.roll(history.V, 1, axis=0).at[0].set(V_new)
    Q_shifted = jnp.roll(history.Q, 1, axis=0).at[0].set(Q_new)
    dt_shifted = jnp.roll(history.dt, 1).at[0].set(dt_new)

    # Increment depth up to max_depth
    new_depth = jnp.minimum(history.depth + 1, max_depth)

    return SolutionHistoryJAX(
        V=V_shifted,
        Q=Q_shifted,
        dt=dt_shifted,
        depth=new_depth,
    )


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
        max_history = config.max_order + 2
        # Use JAX-compatible history for JIT-friendly state updates
        # V is (n_total,), Q is (n_unknowns,) - different sizes
        history = create_solution_history_jax(
            max_history, n_voltages=n_total, n_charges=n_unknowns, dtype=jnp.float64
        )
        stats = AdaptiveStats()

        # Result accumulators - use Python lists (faster than JAX .at[].set() in loop)
        times_list: List[float] = []
        voltages_list: List[jax.Array] = []  # Store V arrays directly, convert at end

        # Get vsource names for current tracking
        vsource_names = setup.source_device_data.get('vsource', {}).get('names', [])
        n_vsources = len(vsource_names)

        # Pre-allocate currents buffer (vsource currents need JAX buffer for .at[].set())
        currents_buffer = jnp.zeros((max_steps + 1, max(n_vsources, 1)), dtype=jnp.float64)
        current_write_idx = 0

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
            # Pass all arguments to match nr_solve's internal call signature
            # This prevents JAX from recompiling due to different pytree structures
            # nr_solve converts None to jnp.zeros, so we must pass arrays here too
            _, _, Q_prev, _ = self.runner._cached_build_system(
                V,
                vsource_dc,
                isource_dc,
                jnp.zeros(n_unknowns),
                0.0,  # integ_c0
                self.runner._device_arrays,
                1e-12,  # gmin
                0.0,    # gshunt
                0.0,    # integ_c1
                0.0,    # integ_d1
                jnp.zeros(n_unknowns, dtype=jnp.float64),  # dQdt_prev
                0.0,    # integ_c2
                jnp.zeros(n_unknowns, dtype=jnp.float64),  # Q_prev2
            )
            Q_prev.block_until_ready()

        # Initialize dQdt_prev for tran_method
        # Always use JAX arrays (not None) for JIT-compatible jnp.where
        integ_coeffs = compute_coefficients(tran_method, dt)
        dQdt_prev = jnp.zeros(n_unknowns, dtype=jnp.float64)
        needs_dqdt = integ_coeffs.needs_dqdt_history

        # Initialize Q_prev2 for Gear2 method
        Q_prev2 = jnp.zeros(n_unknowns, dtype=jnp.float64)
        needs_q_prev2 = integ_coeffs.history_depth >= 2

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
        voltages_list.append(V[:n_external])
        # currents_buffer[0] is already zeros from initialization
        current_write_idx = 1  # Next write position after initial zeros

        # Main simulation loop
        # =====================================================================
        # Phase 2 optimization: Only 1 JAX↔Python crossing per iteration
        # The loop condition is the ONLY crossing; all state updates use jnp.where
        # =====================================================================
        step_count = 0
        warmup_count_jax = jnp.array(0, dtype=jnp.int32)
        cfg_warmup_steps = jnp.array(config.warmup_steps, dtype=jnp.int32)

        while t < t_stop and step_count < max_steps:
            step_count += 1

            # Recompute integration coefficients for current timestep
            integ_coeffs = compute_coefficients(tran_method, current_dt)

            # Get source values at next time
            t_next = t + current_dt
            source_values = source_fn(t_next)
            vsource_vals, isource_vals = self._build_source_arrays(source_values)

            # ================================================================
            # Prediction step (unconditional - use jnp.where for selection)
            # ================================================================
            warmup_complete_jax = warmup_count_jax >= cfg_warmup_steps
            can_predict = warmup_complete_jax & (history.depth >= 2)

            # Use JAX predictor with fixed-size history arrays
            # compute_predictor_coeffs_jax expects order to be static, so we use
            # a fixed order and the predictor handles variable history internally
            pred_order = config.max_order
            pred_coeffs_jax = compute_predictor_coeffs_jax(
                history.dt, dt_jax, pred_order
            )
            V_pred = predict_jax(pred_coeffs_jax.a, history.V, pred_order + 1)

            # Select V_init based on whether we can predict
            V_init = jnp.where(can_predict, V_pred, V)

            # ================================================================
            # Corrector step: Newton-Raphson solve (unconditional)
            # ================================================================
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

            # ================================================================
            # LTE estimation (unconditional - use jnp.where for selection)
            # ================================================================
            # Compute LTE even if we won't use it (cheaper than Python branching)
            lte = (V_new - V_pred) * (
                jit_coeffs.error_coeff
                / (jit_coeffs.error_coeff - pred_coeffs_jax.error_coeff + 1e-30)
            )

            # Compute new timestep from LTE
            dt_lte, max_lte_ratio = compute_new_timestep_jax(
                lte, V_new,
                cfg_reltol, cfg_abstol, cfg_lte_ratio,
                dt_jax, pred_order,
                cfg_min_dt, cfg_max_dt,
            )

            # ================================================================
            # Accept/reject decision (all in JAX - no Python control flow)
            # ================================================================
            # NR convergence check: accept if converged OR dt at minimum
            at_min_dt = dt_jax <= cfg_min_dt
            nr_ok = converged | at_min_dt

            # LTE check: accept if not in warmup OR ratio is acceptable
            lte_ratio_ok = (dt_jax / dt_lte) <= cfg_redo_factor
            lte_ok = ~warmup_complete_jax | ~can_predict | lte_ratio_ok

            # Combined accept condition
            accept = nr_ok & lte_ok

            # ================================================================
            # State updates using jnp.where (no Python branching)
            # ================================================================

            # Compute candidate values for accepted step
            t_accepted = t_jax + dt_jax

            # Compute dt for next iteration based on outcome:
            # - If NR failed: halve dt
            # - If LTE rejected: use dt_lte
            # - If accepted during warmup: grow dt
            # - If accepted after warmup: use min(dt_lte, dt * grow_factor)
            dt_nr_failed = jnp.maximum(dt_jax / 2, cfg_min_dt)
            dt_warmup_grow = jnp.minimum(dt_jax * cfg_grow_factor, cfg_max_dt)
            dt_lte_capped = jnp.minimum(dt_lte, dt_jax * cfg_grow_factor)

            # Select dt_next based on outcome
            dt_next = jnp.where(
                ~converged & ~at_min_dt,  # NR failed, not at min
                dt_nr_failed,
                jnp.where(
                    ~lte_ok,  # LTE rejected
                    dt_lte,
                    jnp.where(
                        warmup_complete_jax,  # After warmup
                        dt_lte_capped,
                        dt_warmup_grow,  # During warmup
                    )
                )
            )

            # Update time (only on accept)
            t_jax = jnp.where(accept, t_accepted, t_jax)

            # Update voltage state (only on accept)
            V = jnp.where(accept, V_new, V)

            # Update charge states (only on accept)
            # Q_prev2 = old Q_prev, Q_prev = Q (shift history)
            Q_prev2_new = jnp.where(needs_q_prev2 & accept, Q_prev, Q_prev2)
            Q_prev_new = jnp.where(accept, Q, Q_prev)
            dQdt_prev_new = jnp.where(needs_dqdt & accept, dQdt, dQdt_prev)
            Q_prev2 = Q_prev2_new
            Q_prev = Q_prev_new
            dQdt_prev = dQdt_prev_new

            # Update history (only on accept)
            history_new = push_solution_history_jax(history, V_new, Q, dt_jax)
            history = SolutionHistoryJAX(
                V=jnp.where(accept, history_new.V, history.V),
                Q=jnp.where(accept, history_new.Q, history.Q),
                dt=jnp.where(accept, history_new.dt, history.dt),
                depth=jnp.where(accept, history_new.depth, history.depth),
            )

            # Update output arrays (only on accept, and only if space available)
            idx = out_idx
            can_write = accept & (idx < est_max_steps)
            times_out = jnp.where(
                can_write,
                times_out.at[idx].set(t_accepted),
                times_out,
            )
            V_out = jnp.where(
                can_write,
                V_out.at[idx].set(V_new[:n_external]),
                V_out,
            )
            if n_vsources_out > 0:
                I_out = jnp.where(
                    can_write,
                    I_out.at[idx].set(I_vsource[:n_vsources_out]),
                    I_out,
                )
            out_idx = jnp.where(accept & (idx < est_max_steps), idx + 1, idx)

            # LTE check: accept if can't predict (no history) OR ratio is acceptable
            lte_ratio_ok = (dt_jax / dt_lte) <= cfg_redo_factor
            lte_ok = ~can_predict | lte_ratio_ok

            # Combined accept condition
            accept = nr_ok & lte_ok

            # ================================================================
            # State updates using jnp.where (no Python branching)
            # ================================================================

            # Compute candidate values for accepted step
            t_accepted = t_jax + dt_jax

            # Compute dt for next iteration based on outcome:
            # - If NR failed: halve dt
            # - If LTE rejected: use dt_lte
            # - If accepted: use min(dt_lte, dt * grow_factor)
            dt_nr_failed = jnp.maximum(dt_jax / 2, cfg_min_dt)
            dt_lte_capped = jnp.minimum(dt_lte, dt_jax * cfg_grow_factor)

            # Select dt_next based on outcome
            dt_next = jnp.where(
                ~converged & ~at_min_dt,  # NR failed, not at min
                dt_nr_failed,
                jnp.where(
                    ~lte_ok,  # LTE rejected
                    dt_lte,
                    dt_lte_capped,  # Accepted: use LTE-based dt capped by grow_factor
                )
            )

            # Update stats (only on accept for some, always for rejected_steps)
            accepted_steps = jnp.where(accept, accepted_steps + 1, accepted_steps)
            rejected_steps = jnp.where(accept, rejected_steps, rejected_steps + 1)
            min_dt_used = jnp.where(accept, jnp.minimum(min_dt_used, dt_jax), min_dt_used)
            max_dt_used = jnp.where(accept, jnp.maximum(max_dt_used, dt_jax), max_dt_used)

            # Update warmup counter (only on accept)
            warmup_count_jax = jnp.where(accept, warmup_count_jax + 1, warmup_count_jax)

            # Update dt for next iteration, clipped to not overshoot t_stop
            t_remaining = jnp.array(t_stop, dtype=jnp.float64) - t_jax
            dt_jax = jnp.clip(dt_next, cfg_min_dt, t_remaining)

        wall_time = time_module.perf_counter() - t_start
        stats.total_timesteps = len(times_list)
        stats.wall_time = wall_time

        # Build results - stack voltage arrays and convert to numpy for slicing
        times = jnp.asarray(np.array(times_list))
        # Stack all voltage snapshots and convert to numpy for indexing
        V_stacked = np.asarray(jnp.stack(voltages_list))  # (n_timesteps, n_external)
        voltages: Dict[str, jax.Array] = {}
        for name, idx in self.runner.node_names.items():
            if 0 < idx < n_external:
                voltages[name] = jnp.asarray(V_stacked[:, idx])

        # Build currents dictionary from pre-allocated buffer
        currents: Dict[str, jax.Array] = {}
        if n_vsources > 0 and current_write_idx > 0:
            I_np = np.asarray(currents_buffer)[:current_write_idx]
            for i, name in enumerate(vsource_names):
                currents[name] = jnp.asarray(I_np[:, i])

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


# =============================================================================
# Scan-based adaptive strategy (fully JIT-compiled)
# =============================================================================

from typing import NamedTuple, Callable
from jax import lax

# Use jnp.float64 directly (JAX-SPICE uses 64-bit precision on CPU)


def _make_jit_source_evaluator(
    source_fn: Callable,
    vsource_names: list,
    vsource_dc: jax.Array,
    isource_names: list,
    isource_dc: jax.Array,
) -> Callable:
    """Create a JIT-compatible source evaluator that returns arrays.

    Instead of returning a dict that requires Python loops to convert to arrays,
    this returns (vsource_vals, isource_vals) directly using JAX operations.

    Args:
        source_fn: Original source function returning dict
        vsource_names: List of voltage source names
        vsource_dc: DC values for voltage sources (fallback)
        isource_names: List of current source names
        isource_dc: DC values for current sources (fallback)

    Returns:
        Function t -> (vsource_vals, isource_vals) that is JIT-compatible
    """
    n_vsources = len(vsource_names)
    n_isources = len(isource_names)
    dtype = jnp.float64

    # Build vectorized evaluator by calling each source function
    # Note: source_fn internally calls jnp.where, so individual evaluations are JIT-safe

    def jit_source_eval(t: jax.Array) -> tuple:
        source_values = source_fn(t)  # Returns dict - OK as long as inner fns are JAX

        # Build vsource array using pure JAX operations
        if n_vsources == 0:
            vsource_vals = jnp.array([], dtype=dtype)
        elif n_vsources == 1:
            val = source_values.get(vsource_names[0], vsource_dc[0])
            vsource_vals = jnp.array([val], dtype=dtype)
        else:
            # For multiple sources, we need to stack values
            # This works because source_values dict values are already JAX arrays/scalars
            vals = [source_values.get(name, dc)
                    for name, dc in zip(vsource_names, vsource_dc)]
            vsource_vals = jnp.stack(vals)

        if n_isources == 0:
            isource_vals = jnp.array([], dtype=dtype)
        elif n_isources == 1:
            val = source_values.get(isource_names[0], isource_dc[0])
            isource_vals = jnp.array([val], dtype=dtype)
        else:
            vals = [source_values.get(name, dc)
                    for name, dc in zip(isource_names, isource_dc)]
            isource_vals = jnp.stack(vals)

        return vsource_vals, isource_vals

    return jit_source_eval


class AdaptiveLoopState(NamedTuple):
    """State for scan-based adaptive loop. All arrays are fixed size."""
    # Current simulation state
    t: jax.Array                    # Scalar: current time
    dt: jax.Array                   # Scalar: current timestep
    V: jax.Array                    # (n_total,): current voltage
    Q_prev: jax.Array               # (n_unknowns,): previous charge
    dQdt_prev: jax.Array            # (n_unknowns,): previous dQ/dt
    Q_prev2: jax.Array              # (n_unknowns,): charge 2 steps ago

    # History buffer (fixed size ring buffer)
    V_history: jax.Array            # (max_history, n_total): past voltages
    dt_history: jax.Array           # (max_history,): past timesteps
    history_count: jax.Array        # Scalar: number of valid entries (0 to max_history)

    # Loop control
    step_idx: jax.Array             # Scalar: current output index
    done: jax.Array                 # Scalar bool: early termination flag

    # Statistics
    total_nr_iters: jax.Array       # Scalar: total NR iterations
    rejected_steps: jax.Array       # Scalar: number of rejected steps
    min_dt_used: jax.Array          # Scalar: minimum dt used
    max_dt_used: jax.Array          # Scalar: maximum dt used


class AdaptiveLoopOutputs(NamedTuple):
    """Per-step outputs from scan loop."""
    t: jax.Array                    # Scalar: time at this step
    V: jax.Array                    # (n_external,): voltages at external nodes
    I_vsource: jax.Array            # (n_vsources,): vsource currents
    accepted: jax.Array             # Scalar bool: was this step accepted


def _predict_fixed_order(
    V_history: jax.Array,           # (max_history, n_total)
    dt_history: jax.Array,          # (max_history,)
    history_count: jax.Array,       # Scalar
    new_dt: jax.Array,              # Scalar
    max_order: int,
) -> Tuple[jax.Array, jax.Array]:
    """Compute predictor using fixed-size arrays.

    Returns:
        Tuple of (V_predicted, error_coeff)
    """
    # Compute actual order to use (min of history_count-1 and max_order)
    order = jnp.minimum(history_count - 1, max_order)
    order = jnp.maximum(order, 0)  # At least 0

    # Compute normalized timepoints: tau_k = (t_{n-k} - t_{n+1}) / h_n
    # t_n is at -new_dt, t_{n-1} is at -(new_dt + dt_history[0]), etc.
    cumsum_dt = jnp.cumsum(dt_history)  # Cumulative sum of past timesteps
    tau = -(new_dt + jnp.concatenate([jnp.array([0.0]), cumsum_dt[:-1]])) / new_dt

    # Build Vandermonde-like matrix for polynomial interpolation
    # For order p, we need p+1 points and solve for coefficients
    # Use order=2 (quadratic) as max, pad smaller orders with zeros

    # Simple approach: compute coefficients for each possible order
    # and select based on actual order

    # Order 0 (constant): a = [1]
    a0 = jnp.array([1.0, 0.0, 0.0])
    err0 = 1.0

    # Order 1 (linear): V_pred = V_n + (V_n - V_{n-1}) * new_dt / dt_{n-1}
    # Coefficients: a_0 = 1 + new_dt/dt_0, a_1 = -new_dt/dt_0
    ratio1 = new_dt / jnp.maximum(dt_history[0], 1e-30)
    a1 = jnp.array([1.0 + ratio1, -ratio1, 0.0])
    err1 = -ratio1 * (1 + ratio1) / 2  # Approximate error coefficient

    # Order 2 (quadratic): more complex, use Lagrange interpolation
    # tau values for 3 points
    tau0 = -1.0  # t_n relative to t_{n+1}, normalized
    tau1 = tau[0]  # t_{n-1}
    tau2 = tau[1]  # t_{n-2}

    # Lagrange coefficients for interpolating to tau=0 (t_{n+1})
    denom0 = (tau0 - tau1) * (tau0 - tau2)
    denom1 = (tau1 - tau0) * (tau1 - tau2)
    denom2 = (tau2 - tau0) * (tau2 - tau1)

    # Avoid division by zero
    denom0 = jnp.where(jnp.abs(denom0) < 1e-30, 1e-30, denom0)
    denom1 = jnp.where(jnp.abs(denom1) < 1e-30, 1e-30, denom1)
    denom2 = jnp.where(jnp.abs(denom2) < 1e-30, 1e-30, denom2)

    l0 = (0 - tau1) * (0 - tau2) / denom0  # For V_n
    l1 = (0 - tau0) * (0 - tau2) / denom1  # For V_{n-1}
    l2 = (0 - tau0) * (0 - tau1) / denom2  # For V_{n-2}

    a2 = jnp.array([l0, l1, l2])
    err2 = -tau0 * tau1 * tau2 / 6  # Third derivative contribution

    # Select coefficients based on order
    a = jnp.where(order == 0, a0, jnp.where(order == 1, a1, a2))
    err_coeff = jnp.where(order == 0, err0, jnp.where(order == 1, err1, err2))

    # Compute prediction: V_pred = sum_i a_i * V_history[i]
    # Only use valid entries based on order
    mask = jnp.arange(3) <= order
    a_masked = jnp.where(mask, a, 0.0)

    # V_history is (max_history, n_total), we need first 3 rows
    V_pred = jnp.einsum('i,ij->j', a_masked[:3], V_history[:3])

    return V_pred, err_coeff


def _make_adaptive_scan_fn(
    nr_solve,
    jit_source_eval: Callable,
    device_arrays,
    config: AdaptiveConfig,
    n_total: int,
    n_unknowns: int,
    n_external: int,
    n_vsources: int,
    t_stop: float,
    method_idx: int,
    dtype,
):
    """Create the scan body function for adaptive timestep.

    This function is JIT-compiled and handles one potential timestep.

    Args:
        nr_solve: Newton-Raphson solver function
        jit_source_eval: JIT-compatible source evaluator: t -> (vsource_vals, isource_vals)
        device_arrays: Device parameter arrays
        config: Adaptive timestep configuration
        n_total: Total number of nodes
        n_unknowns: Number of unknown node voltages
        n_external: Number of external (user-visible) nodes
        n_vsources: Number of voltage sources
        t_stop: Simulation stop time
        warmup_steps: Number of warmup steps before enabling LTE control
        tran_method: Integration method
        method_idx: Integration method index (0=BE, 1=TRAP, 2=GEAR2)
        dtype: Float dtype to use (float32 or float64)
    """

    max_history = config.max_order + 2  # Need order+1 points for predictor

    def scan_body(carry: AdaptiveLoopState, _) -> Tuple[AdaptiveLoopState, AdaptiveLoopOutputs]:
        """One iteration of adaptive timestep loop."""

        # Unpack state
        t = carry.t
        dt = carry.dt
        V = carry.V
        Q_prev = carry.Q_prev
        dQdt_prev = carry.dQdt_prev
        Q_prev2 = carry.Q_prev2
        V_history = carry.V_history
        dt_history = carry.dt_history
        history_count = carry.history_count
        step_idx = carry.step_idx
        done = carry.done
        total_nr_iters = carry.total_nr_iters
        rejected_steps = carry.rejected_steps
        min_dt_used = carry.min_dt_used
        max_dt_used = carry.max_dt_used

        # If already done, return no-op
        def done_branch():
            return carry, AdaptiveLoopOutputs(
                t=t,
                V=V[:n_external],
                I_vsource=jnp.zeros(n_vsources, dtype=dtype),
                accepted=jnp.array(False),
            )

        def continue_branch():
            # Compute integration coefficients
            c0 = 1.0 / dt
            c1 = -1.0 / dt
            d1 = 0.0
            c2 = 0.0
            error_coeff_integ = -0.5  # BE error coefficient

            # Get source values at next time using JIT-compatible evaluator
            t_next = t + dt
            vsource_vals, isource_vals = jit_source_eval(t_next)

            # Prediction step - can predict when we have at least 2 history points
            can_predict = history_count >= 2

            V_pred, pred_err_coeff = _predict_fixed_order(
                V_history, dt_history, history_count, dt, config.max_order
            )
            V_init = jnp.where(can_predict, V_pred, V)

            # Newton-Raphson solve
            V_new, iterations, converged, max_f, Q, dQdt_out, I_vsource = nr_solve(
                V_init,
                vsource_vals,
                isource_vals,
                Q_prev,
                c0,
                device_arrays,
                1e-12,  # gmin
                0.0,    # gshunt
                c1,
                d1,
                dQdt_prev,
                c2,
                Q_prev2,
            )

            # Update NR iteration count (ensure int32 to match state dtype)
            new_total_nr_iters = total_nr_iters + jnp.int32(iterations)

            # Handle NR convergence failure
            def nr_failed():
                new_dt = jnp.maximum(dt / 2, config.min_dt)
                at_min = dt <= config.min_dt
                # If at minimum dt, accept anyway; otherwise retry
                # Use jnp.where for all selections to be JIT-compatible
                return (
                    new_dt,
                    rejected_steps + jnp.where(at_min, 0, 1),
                    at_min,  # accept_step
                    jnp.where(at_min, V_new, V),
                    jnp.where(at_min, Q, Q_prev),
                    jnp.where(at_min, dQdt_out, dQdt_prev),
                )

            def nr_succeeded():
                return (
                    dt,
                    rejected_steps,
                    jnp.array(True),
                    V_new,
                    Q,
                    dQdt_out,
                )

            dt_after_nr, new_rejected, nr_accept, V_after_nr, Q_after_nr, dQdt_after_nr = lax.cond(
                converged, nr_succeeded, nr_failed
            )

            # LTE estimation (only if predictor was used)
            def compute_lte_dt():
                # Estimate LTE
                lte = (V_new - V_pred) * (error_coeff_integ / (error_coeff_integ - pred_err_coeff))
                lte_norm = jnp.max(jnp.abs(lte) / (config.reltol * jnp.abs(V_new) + config.abstol))

                # Compute new timestep
                # dt_new = dt * (lte_ratio / lte_norm)^(1/(order+1))
                order = jnp.minimum(history_count - 1, config.max_order)
                factor = jnp.power(config.lte_ratio / jnp.maximum(lte_norm, 1e-10), 1.0 / (order + 2))
                factor = jnp.clip(factor, 0.1, config.grow_factor)
                return dt * factor, lte_norm

            def no_lte_dt():
                return dt, jnp.array(0.0)

            dt_lte, lte_ratio = lax.cond(
                can_predict & nr_accept,
                compute_lte_dt,
                no_lte_dt,
            )

            # Check if LTE requires step rejection
            lte_reject = (dt / dt_lte > config.redo_factor) & can_predict & nr_accept

            def lte_rejected():
                return (
                    dt_lte,
                    new_rejected + 1,
                    jnp.array(False),
                )

            def lte_accepted():
                return dt_lte, new_rejected, nr_accept

            final_dt, final_rejected, accept_step = lax.cond(
                lte_reject, lte_rejected, lte_accepted
            )

            # Update state for accepted step
            def accepted_update():
                # Update history (shift and insert new values)
                new_V_history = jnp.roll(V_history, 1, axis=0)
                new_V_history = new_V_history.at[0].set(V_after_nr)

                new_dt_history = jnp.roll(dt_history, 1)
                new_dt_history = new_dt_history.at[0].set(dt)

                new_history_count = jnp.minimum(history_count + 1, max_history)

                # Clip dt to not overshoot
                next_dt = jnp.minimum(final_dt, t_stop - t_next)
                next_dt = jnp.clip(next_dt, config.min_dt, config.max_dt)

                new_done = t_next >= t_stop

                return AdaptiveLoopState(
                    t=t_next,
                    dt=next_dt,
                    V=V_after_nr,
                    Q_prev=Q_after_nr,
                    dQdt_prev=dQdt_after_nr,
                    Q_prev2=Q_prev,  # Shift Q history
                    V_history=new_V_history,
                    dt_history=new_dt_history,
                    history_count=new_history_count,
                    step_idx=step_idx + 1,
                    done=new_done,
                    total_nr_iters=new_total_nr_iters,
                    rejected_steps=final_rejected,
                    min_dt_used=jnp.minimum(min_dt_used, dt),
                    max_dt_used=jnp.maximum(max_dt_used, dt),
                )

            def rejected_update():
                # Just update dt for retry, don't advance time
                next_dt = jnp.clip(final_dt, config.min_dt, config.max_dt)
                return AdaptiveLoopState(
                    t=t,
                    dt=next_dt,
                    V=V,
                    Q_prev=Q_prev,
                    dQdt_prev=dQdt_prev,
                    Q_prev2=Q_prev2,
                    V_history=V_history,
                    dt_history=dt_history,
                    history_count=history_count,
                    step_idx=step_idx,
                    done=done,
                    total_nr_iters=new_total_nr_iters,
                    rejected_steps=final_rejected,
                    min_dt_used=min_dt_used,
                    max_dt_used=max_dt_used,
                )

            new_state = lax.cond(accept_step, accepted_update, rejected_update)

            outputs = AdaptiveLoopOutputs(
                t=jnp.where(accept_step, t_next, t),
                V=V_after_nr[:n_external],
                I_vsource=I_vsource if n_vsources > 0 else jnp.zeros(max(n_vsources, 1), dtype=dtype),
                accepted=accept_step,
            )

            return new_state, outputs

        return lax.cond(done, done_branch, continue_branch)

    return scan_body


class AdaptiveScanStrategy(TransientStrategy):
    """Adaptive timestep strategy using lax.scan for full JIT compilation.

    This is a faster version of AdaptiveStrategy that uses fixed-size arrays
    and lax.scan instead of a Python loop. This enables full JIT compilation
    of the entire simulation loop.
    """

    name = "adaptive_scan"

    def __init__(
        self,
        runner: 'CircuitEngine',
        use_sparse: bool = False,
        backend: str = "cpu",
        config: Optional[AdaptiveConfig] = None,
    ):
        super().__init__(runner, use_sparse=use_sparse, backend=backend)
        self.config = config or AdaptiveConfig()

    def run(
        self, t_stop: float, dt: float, max_steps: int = 100000
    ) -> Tuple[jax.Array, Dict[str, jax.Array], Dict[str, jax.Array], Dict]:
        """Run adaptive transient analysis using lax.scan.

        Args:
            t_stop: Simulation stop time in seconds
            dt: Initial timestep in seconds (will be adapted)
            max_steps: Maximum number of iterations (including rejected steps)

        Returns:
            Tuple of (times, voltages, currents, stats)
        """
        setup = self.ensure_setup()
        nr_solve = self.ensure_solver()

        n_total = setup.n_total
        n_unknowns = setup.n_unknowns
        n_external = setup.n_external
        source_fn = setup.source_fn
        config = self.config

        # Get integration method
        tran_method = self.runner.analysis_params.get(
            "tran_method", IntegrationMethod.BACKWARD_EULER
        )

        # Get dtype based on x64 configuration
        dtype = jnp.float64

        # Get source info for JIT-compatible evaluator
        vsource_data = setup.source_device_data.get('vsource', {})
        isource_data = setup.source_device_data.get('isource', {})
        vsource_names = vsource_data.get('names', [])
        isource_names = isource_data.get('names', [])
        vsource_dc = jnp.asarray(vsource_data.get('dc', []), dtype=dtype)
        isource_dc = jnp.asarray(isource_data.get('dc', []), dtype=dtype)
        n_vsources = len(vsource_names)

        device_arrays = self.runner._device_arrays

        max_history = config.max_order + 2

        # Create JIT-compatible source evaluator
        jit_source_eval = _make_jit_source_evaluator(
            source_fn=source_fn,
            vsource_names=vsource_names,
            vsource_dc=vsource_dc,
            isource_names=isource_names,
            isource_dc=isource_dc,
        )

        # Initialize state
        V = jnp.zeros(n_total, dtype=dtype)

        # Initialize Q from build_system at DC
        Q_init = jnp.zeros(n_unknowns, dtype=dtype)
        if hasattr(self.runner, "_cached_build_system"):
            vsource_dc_init = vsource_dc if len(vsource_dc) > 0 else jnp.array([], dtype=dtype)
            isource_dc_init = isource_dc if len(isource_dc) > 0 else jnp.array([], dtype=dtype)
            _, _, Q_init, _ = self.runner._cached_build_system(
                V, vsource_dc_init, isource_dc_init,
                jnp.zeros(n_unknowns, dtype=dtype),
                0.0,
                device_arrays,
                1e-12, 0.0, 0.0, 0.0,
                jnp.zeros(n_unknowns, dtype=dtype),
                0.0,
                jnp.zeros(n_unknowns, dtype=dtype),
            )

        initial_dt = jnp.minimum(jnp.array(dt), jnp.array(config.max_dt))
        initial_dt = jnp.maximum(initial_dt, jnp.array(config.min_dt))

        init_state = AdaptiveLoopState(
            t=jnp.array(0.0, dtype=dtype),
            dt=initial_dt,
            V=V,
            Q_prev=Q_init,
            dQdt_prev=jnp.zeros(n_unknowns, dtype=dtype),
            Q_prev2=jnp.zeros(n_unknowns, dtype=dtype),
            V_history=jnp.zeros((max_history, n_total), dtype=dtype),
            dt_history=jnp.full(max_history, dt, dtype=dtype),
            history_count=jnp.array(0, dtype=jnp.int32),
            step_idx=jnp.array(0, dtype=jnp.int32),
            done=jnp.array(False, dtype=jnp.bool_),
            total_nr_iters=jnp.array(0, dtype=jnp.int32),
            rejected_steps=jnp.array(0, dtype=jnp.int32),
            min_dt_used=jnp.array(jnp.inf, dtype=dtype),
            max_dt_used=jnp.array(0.0, dtype=dtype),
        )

        # Create scan function
        scan_fn = _make_adaptive_scan_fn(
            nr_solve=nr_solve,
            jit_source_eval=jit_source_eval,
            device_arrays=device_arrays,
            config=config,
            n_total=n_total,
            n_unknowns=n_unknowns,
            n_external=n_external,
            n_vsources=max(n_vsources, 1),  # At least 1 to avoid empty arrays
            t_stop=t_stop,
            method_idx=method_to_index(tran_method),
            dtype=dtype,
        )

        logger.info(
            f"{self.name}: Starting scan-based adaptive simulation to t={t_stop:.2e}s, "
            f"initial dt={dt:.2e}s, max_steps={max_steps}"
        )
        t_start = time_module.perf_counter()

        # Run scan
        final_state, outputs = lax.scan(scan_fn, init_state, None, length=max_steps)

        # Wait for computation
        jax.block_until_ready(final_state)

        wall_time = time_module.perf_counter() - t_start

        # Extract accepted steps
        accepted_mask = outputs.accepted
        n_accepted = int(jnp.sum(accepted_mask))

        # Filter outputs to only accepted steps
        times = outputs.t[accepted_mask]
        V_out = outputs.V[accepted_mask]  # (n_accepted, n_external)
        I_out = outputs.I_vsource[accepted_mask]  # (n_accepted, n_vsources)

        # Build voltage dict with node names
        voltages: Dict[str, jax.Array] = {}
        for name, idx in self.runner.node_names.items():
            if 0 < idx < n_external:
                voltages[name] = V_out[:, idx]

        # Build current dict
        currents: Dict[str, jax.Array] = {}
        for i, name in enumerate(vsource_names):
            currents[name] = I_out[:, i]

        stats_dict = {
            "total_timesteps": n_accepted,
            "accepted_steps": n_accepted,
            "rejected_steps": int(final_state.rejected_steps),
            "total_nr_iterations": int(final_state.total_nr_iters),
            "avg_nr_iterations": float(final_state.total_nr_iters) / max(n_accepted, 1),
            "min_dt_used": float(final_state.min_dt_used),
            "max_dt_used": float(final_state.max_dt_used),
            "wall_time": wall_time,
            "time_per_step_ms": wall_time / max(n_accepted, 1) * 1000,
            "strategy": "adaptive_scan",
            "solver": "sparse" if self.use_sparse else "dense",
            "convergence_rate": n_accepted / max(n_accepted + int(final_state.rejected_steps), 1),
        }

        logger.info(
            f"{self.name}: Completed {n_accepted} steps in {wall_time:.3f}s "
            f"({stats_dict['time_per_step_ms']:.2f}ms/step, "
            f"{int(final_state.rejected_steps)} rejected)"
        )

        return times, voltages, currents, stats_dict


# =============================================================================
# While-loop based adaptive strategy (JIT-compiled with early termination)
# =============================================================================

class WhileLoopState(NamedTuple):
    """State for while-loop based adaptive timestep."""
    # Current simulation state
    t: jax.Array                    # Scalar: current time
    dt: jax.Array                   # Scalar: current timestep
    V: jax.Array                    # (n_total,): current voltage
    Q_prev: jax.Array               # (n_unknowns,): previous charge
    dQdt_prev: jax.Array            # (n_unknowns,): previous dQ/dt
    Q_prev2: jax.Array              # (n_unknowns,): charge 2 steps ago

    # History buffer (fixed size ring buffer)
    V_history: jax.Array            # (max_history, n_total): past voltages
    dt_history: jax.Array           # (max_history,): past timesteps
    history_count: jax.Array        # Scalar: number of valid entries

    # Output arrays (pre-allocated, filled up to step_idx)
    times_out: jax.Array            # (max_steps,): output times
    V_out: jax.Array                # (max_steps, n_external): output voltages
    I_out: jax.Array                # (max_steps, n_vsources): output currents

    # Loop control
    step_idx: jax.Array             # Scalar: current output index

    # Statistics
    total_nr_iters: jax.Array       # Scalar: total NR iterations
    rejected_steps: jax.Array       # Scalar: number of rejected steps
    min_dt_used: jax.Array          # Scalar: minimum dt used
    max_dt_used: jax.Array          # Scalar: maximum dt used


def _make_while_loop_fns(
    nr_solve,
    jit_source_eval: Callable,
    device_arrays,
    config: AdaptiveConfig,
    n_total: int,
    n_unknowns: int,
    n_external: int,
    n_vsources: int,
    max_steps: int,
    t_stop: float,
    method_idx: int,
    dtype,
):
    """Create cond and body functions for while_loop adaptive timestep."""

    max_history = config.max_order + 2

    def cond_fn(state: WhileLoopState) -> jax.Array:
        """Continue while t < t_stop and step_idx < max_steps."""
        return (state.t < t_stop) & (state.step_idx < max_steps)

    def body_fn(state: WhileLoopState) -> WhileLoopState:
        """One iteration of adaptive timestep loop."""
        t = state.t
        dt = state.dt
        V = state.V
        Q_prev = state.Q_prev
        dQdt_prev = state.dQdt_prev
        Q_prev2 = state.Q_prev2
        V_history = state.V_history
        dt_history = state.dt_history
        history_count = state.history_count
        step_idx = state.step_idx

        # Integration coefficients (Backward Euler)
        c0 = 1.0 / dt
        c1 = -1.0 / dt
        error_coeff_integ = -0.5

        # Get source values at next time
        t_next = t + dt
        vsource_vals, isource_vals = jit_source_eval(t_next)

        # Prediction step - can predict when we have at least 2 history points
        can_predict = history_count >= 2

        V_pred, pred_err_coeff = _predict_fixed_order(
            V_history, dt_history, history_count, dt, config.max_order
        )
        V_init = jnp.where(can_predict, V_pred, V)

        # Newton-Raphson solve
        V_new, iterations, converged, max_f, Q, dQdt_out, I_vsource = nr_solve(
            V_init, vsource_vals, isource_vals, Q_prev, c0, device_arrays,
            1e-12, 0.0, c1, 0.0, dQdt_prev, 0.0, Q_prev2,
        )

        new_total_nr_iters = state.total_nr_iters + jnp.int32(iterations)

        # Handle NR failure - match Python loop logic exactly
        at_min_dt = dt <= config.min_dt
        nr_ok = converged | at_min_dt

        # LTE estimation
        lte = (V_new - V_pred) * (error_coeff_integ / (error_coeff_integ - pred_err_coeff + 1e-30))
        lte_norm = jnp.max(jnp.abs(lte) / (config.reltol * jnp.abs(V_new) + config.abstol))
        order = jnp.minimum(history_count - 1, config.max_order)
        factor = jnp.power(config.lte_ratio / jnp.maximum(lte_norm, 1e-10), 1.0 / (order + 2))
        factor = jnp.clip(factor, 0.1, config.grow_factor)
        dt_lte = dt * factor

        # Decision: accept or reject - simplified logic (no warmup)
        # lte_ok = ~can_predict | (dt/dt_lte <= redo_factor)
        # accept = nr_ok & lte_ok
        lte_ratio_ok = (dt / dt_lte) <= config.redo_factor
        lte_ok = ~can_predict | lte_ratio_ok
        accept_step = nr_ok & lte_ok

        # Compute new dt - simplified logic (no warmup)
        # If NR failed and not at min: halve dt
        # If LTE rejected: use dt_lte
        # If accepted: use min(dt_lte, dt * grow_factor)
        nr_failed = ~converged
        dt_nr_failed = jnp.maximum(dt / 2, config.min_dt)
        dt_lte_capped = jnp.minimum(dt_lte, dt * config.grow_factor)

        new_dt = jnp.where(
            nr_failed & ~at_min_dt,  # NR failed, not at min
            dt_nr_failed,
            jnp.where(
                ~lte_ok,  # LTE rejected
                dt_lte,
                dt_lte_capped,  # Accepted: use LTE-based dt capped by grow_factor
            )
        )
        # Clip and don't overshoot
        t_remaining = t_stop - t_next
        new_dt = jnp.clip(new_dt, config.min_dt, jnp.maximum(t_remaining, config.min_dt))

        # Update state based on accept/reject
        new_t = jnp.where(accept_step, t_next, t)
        new_V = jnp.where(accept_step, V_new, V)
        new_Q_prev = jnp.where(accept_step, Q, Q_prev)
        new_dQdt_prev = jnp.where(accept_step, dQdt_out, dQdt_prev)
        new_Q_prev2 = jnp.where(accept_step, Q_prev, Q_prev2)

        # Update history on accept
        new_V_history = jnp.where(
            accept_step,
            jnp.roll(V_history, 1, axis=0).at[0].set(V_new),
            V_history
        )
        new_dt_history = jnp.where(
            accept_step,
            jnp.roll(dt_history, 1).at[0].set(dt),
            dt_history
        )
        new_history_count = jnp.where(
            accept_step,
            jnp.minimum(history_count + 1, max_history),
            history_count
        )

        # Update output arrays on accept
        new_times_out = jnp.where(
            accept_step,
            state.times_out.at[step_idx].set(t_next),
            state.times_out
        )
        new_V_out = jnp.where(
            accept_step,
            state.V_out.at[step_idx].set(V_new[:n_external]),
            state.V_out
        )
        new_I_out = jnp.where(
            accept_step,
            state.I_out.at[step_idx].set(I_vsource[:n_vsources] if n_vsources > 0 else jnp.zeros(1, dtype=dtype)),
            state.I_out
        )
        new_step_idx = jnp.where(accept_step, step_idx + 1, step_idx)

        # Update statistics
        new_rejected = state.rejected_steps + jnp.where(~accept_step, 1, 0)
        new_min_dt = jnp.where(accept_step, jnp.minimum(state.min_dt_used, dt), state.min_dt_used)
        new_max_dt = jnp.where(accept_step, jnp.maximum(state.max_dt_used, dt), state.max_dt_used)

        return WhileLoopState(
            t=new_t,
            dt=new_dt,
            V=new_V,
            Q_prev=new_Q_prev,
            dQdt_prev=new_dQdt_prev,
            Q_prev2=new_Q_prev2,
            V_history=new_V_history,
            dt_history=new_dt_history,
            history_count=new_history_count,
            times_out=new_times_out,
            V_out=new_V_out,
            I_out=new_I_out,
            step_idx=new_step_idx,
            total_nr_iters=new_total_nr_iters,
            rejected_steps=new_rejected,
            min_dt_used=new_min_dt,
            max_dt_used=new_max_dt,
        )

    return cond_fn, body_fn


class AdaptiveWhileLoopStrategy(TransientStrategy):
    """Adaptive timestep using lax.while_loop for early termination.

    This strategy uses while_loop instead of scan, which allows genuine
    early termination when t >= t_stop, rather than running all max_steps
    iterations with a done flag.
    """

    name = "adaptive_while"

    def __init__(
        self,
        runner: 'CircuitEngine',
        use_sparse: bool = False,
        backend: str = "cpu",
        config: Optional[AdaptiveConfig] = None,
    ):
        super().__init__(runner, use_sparse=use_sparse, backend=backend)
        self.config = config or AdaptiveConfig()

    def run(
        self, t_stop: float, dt: float, max_steps: int = 100000
    ) -> Tuple[jax.Array, Dict[str, jax.Array], Dict[str, jax.Array], Dict]:
        """Run adaptive transient analysis using lax.while_loop."""
        setup = self.ensure_setup()
        nr_solve = self.ensure_solver()

        n_total = setup.n_total
        n_unknowns = setup.n_unknowns
        n_external = setup.n_external
        source_fn = setup.source_fn
        config = self.config

        tran_method = self.runner.analysis_params.get(
            "tran_method", IntegrationMethod.BACKWARD_EULER
        )

        dtype = jnp.float64

        vsource_data = setup.source_device_data.get('vsource', {})
        isource_data = setup.source_device_data.get('isource', {})
        vsource_names = vsource_data.get('names', [])
        isource_names = isource_data.get('names', [])
        vsource_dc = jnp.asarray(vsource_data.get('dc', []), dtype=dtype)
        isource_dc = jnp.asarray(isource_data.get('dc', []), dtype=dtype)
        n_vsources = max(len(vsource_names), 1)

        device_arrays = self.runner._device_arrays
        max_history = config.max_order + 2

        jit_source_eval = _make_jit_source_evaluator(
            source_fn, vsource_names, vsource_dc, isource_names, isource_dc
        )

        # Initialize state
        V = jnp.zeros(n_total, dtype=dtype)
        Q_init = jnp.zeros(n_unknowns, dtype=dtype)
        if hasattr(self.runner, "_cached_build_system"):
            vsource_dc_init = vsource_dc if len(vsource_dc) > 0 else jnp.array([], dtype=dtype)
            isource_dc_init = isource_dc if len(isource_dc) > 0 else jnp.array([], dtype=dtype)
            _, _, Q_init, _ = self.runner._cached_build_system(
                V, vsource_dc_init, isource_dc_init,
                jnp.zeros(n_unknowns, dtype=dtype), 0.0, device_arrays,
                1e-12, 0.0, 0.0, 0.0, jnp.zeros(n_unknowns, dtype=dtype),
                0.0, jnp.zeros(n_unknowns, dtype=dtype),
            )

        initial_dt = jnp.clip(jnp.array(dt, dtype=dtype), config.min_dt, config.max_dt)

        init_state = WhileLoopState(
            t=jnp.array(0.0, dtype=dtype),
            dt=initial_dt,
            V=V,
            Q_prev=Q_init,
            dQdt_prev=jnp.zeros(n_unknowns, dtype=dtype),
            Q_prev2=jnp.zeros(n_unknowns, dtype=dtype),
            V_history=jnp.zeros((max_history, n_total), dtype=dtype),
            dt_history=jnp.full(max_history, dt, dtype=dtype),
            history_count=jnp.array(0, dtype=jnp.int32),
            times_out=jnp.zeros(max_steps, dtype=dtype),
            V_out=jnp.zeros((max_steps, n_external), dtype=dtype),
            I_out=jnp.zeros((max_steps, n_vsources), dtype=dtype),
            step_idx=jnp.array(0, dtype=jnp.int32),
            total_nr_iters=jnp.array(0, dtype=jnp.int32),
            rejected_steps=jnp.array(0, dtype=jnp.int32),
            min_dt_used=jnp.array(jnp.inf, dtype=dtype),
            max_dt_used=jnp.array(0.0, dtype=dtype),
        )

        cond_fn, body_fn = _make_while_loop_fns(
            nr_solve, jit_source_eval, device_arrays, config,
            n_total, n_unknowns, n_external, n_vsources, max_steps,
            t_stop, method_to_index(tran_method), dtype,
        )

        logger.info(
            f"{self.name}: Starting while-loop adaptive simulation to t={t_stop:.2e}s, "
            f"initial dt={dt:.2e}s, max_steps={max_steps}"
        )
        t_start = time_module.perf_counter()

        # Run while loop
        final_state = lax.while_loop(cond_fn, body_fn, init_state)
        jax.block_until_ready(final_state)

        wall_time = time_module.perf_counter() - t_start

        # Extract valid outputs
        n_steps = int(final_state.step_idx)
        times = final_state.times_out[:n_steps]
        V_out = final_state.V_out[:n_steps]
        I_out = final_state.I_out[:n_steps]

        # Build voltage dict
        voltages: Dict[str, jax.Array] = {}
        for name, idx in self.runner.node_names.items():
            if 0 < idx < n_external:
                voltages[name] = V_out[:, idx]

        # Build current dict
        currents: Dict[str, jax.Array] = {}
        for i, name in enumerate(vsource_names):
            currents[name] = I_out[:, i]

        stats_dict = {
            "total_timesteps": n_steps,
            "accepted_steps": n_steps,
            "rejected_steps": int(final_state.rejected_steps),
            "total_nr_iterations": int(final_state.total_nr_iters),
            "avg_nr_iterations": float(final_state.total_nr_iters) / max(n_steps, 1),
            "min_dt_used": float(final_state.min_dt_used),
            "max_dt_used": float(final_state.max_dt_used),
            "wall_time": wall_time,
            "time_per_step_ms": wall_time / max(n_steps, 1) * 1000,
            "strategy": "adaptive_while",
            "solver": "sparse" if self.use_sparse else "dense",
        }

        logger.info(
            f"{self.name}: Completed {n_steps} steps in {wall_time:.3f}s "
            f"({stats_dict['time_per_step_ms']:.2f}ms/step, "
            f"{int(final_state.rejected_steps)} rejected)"
        )

        return times, voltages, currents, stats_dict
