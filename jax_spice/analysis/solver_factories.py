"""Newton-Raphson solver factory functions for circuit simulation.

This module provides factory functions that create JIT-compiled NR solvers
using full MNA (Modified Nodal Analysis) with branch currents as explicit
unknowns.

Available solvers:
- Dense full MNA solver (JAX scipy.linalg.solve) - for small/medium circuits
- Sparse full MNA solver (JAX spsolve) - for large circuits
- Spineax full MNA solver (cuDSS on GPU) - for GPU acceleration
- UMFPACK full MNA solver (CPU sparse) - for CPU sparse with cached factorization
- UMFPACK FFI full MNA solver (CPU sparse) - zero callback overhead variant

Full MNA solvers use an augmented system X = [V; I] where V are node voltages
and I are branch currents for voltage sources. This provides more accurate
current extraction than the high-G approximation.

Integration coefficient meanings:
    - integ_c0: Coefficient for Q_new (leading coefficient). 0 for DC.
    - integ_c1: Coefficient for Q_prev. Default 0.
    - integ_d1: Coefficient for dQdt_prev. Default 0 (only trap uses -1).
    - dQdt_prev: Previous dQ/dt vector for trapezoidal method.
"""

from typing import TYPE_CHECKING, Callable, Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, lax

from jax_spice._logging import logger

if TYPE_CHECKING:
    from jax_spice.analysis.options import SimulationOptions


def _compute_noi_masks(
    noi_indices: Optional[Array],
    n_nodes: int,
    bcsr_indptr: Optional[Array] = None,
    bcsr_indices: Optional[Array] = None,
) -> Dict:
    """Pre-compute masks for NOI node constraint enforcement.

    NOI (noise correlation) nodes have extremely high conductance (1e40) which
    causes numerical instability. We enforce delta[noi] = 0 by modifying the
    linear system before solving.

    Args:
        noi_indices: Array of NOI node indices (in full V vector)
        n_nodes: Total node count including ground
        bcsr_indptr: CSR row pointers (for sparse solvers)
        bcsr_indices: CSR column indices (for sparse solvers)

    Returns:
        Dict with pre-computed masks:
        - noi_res_idx: NOI residual indices (noi_indices - 1)
        - residual_mask: Boolean mask for convergence check
        - noi_row_mask: CSR indices for NOI rows (sparse only)
        - noi_col_mask: CSR indices for NOI columns (sparse only)
        - noi_diag_indices: CSR indices for NOI diagonals (sparse only)
        - noi_res_indices_arr: Sorted NOI residual indices (sparse only)
    """
    result = {
        "noi_res_idx": None,
        "residual_mask": None,
        "noi_row_mask": None,
        "noi_col_mask": None,
        "noi_diag_indices": None,
        "noi_res_indices_arr": None,
    }

    if noi_indices is None or len(noi_indices) == 0:
        return result

    n_unknowns = n_nodes - 1
    noi_res_idx = noi_indices - 1  # Convert to residual indices

    # Residual mask for convergence check
    residual_mask = jnp.ones(n_unknowns, dtype=jnp.bool_)
    residual_mask = residual_mask.at[noi_res_idx].set(False)

    result["noi_res_idx"] = noi_res_idx
    result["residual_mask"] = residual_mask

    # CSR masks for sparse solvers
    if bcsr_indptr is not None and bcsr_indices is not None:
        row_mask_list = []
        col_mask_list = []
        diag_list = []
        noi_set = set(int(x) for x in np.asarray(noi_res_idx))

        indptr_np = np.asarray(bcsr_indptr)
        indices_np = np.asarray(bcsr_indices)

        # Find NOI row entries and diagonals
        for noi_idx in noi_set:
            row_start, row_end = int(indptr_np[noi_idx]), int(indptr_np[noi_idx + 1])
            row_mask_list.extend(range(row_start, row_end))
            for j in range(row_start, row_end):
                if indices_np[j] == noi_idx:
                    diag_list.append(j)

        # Find NOI column entries
        for row in range(n_unknowns):
            row_start, row_end = int(indptr_np[row]), int(indptr_np[row + 1])
            for j in range(row_start, row_end):
                if int(indices_np[j]) in noi_set:
                    col_mask_list.append(j)

        result["noi_row_mask"] = jnp.array(row_mask_list, dtype=jnp.int32)
        result["noi_col_mask"] = jnp.array(col_mask_list, dtype=jnp.int32)
        result["noi_diag_indices"] = jnp.array(diag_list, dtype=jnp.int32)
        result["noi_res_indices_arr"] = jnp.array(sorted(noi_set), dtype=jnp.int32)

    return result


def make_dense_full_mna_solver(
    build_system_jit: Callable,
    n_nodes: int,
    n_vsources: int,
    noi_indices: Optional[Array] = None,
    max_iterations: int = 100,
    abstol: float = 1e-12,
    max_step: float = 1.0,
    total_limit_states: int = 0,
    options: Optional["SimulationOptions"] = None,
) -> Callable:
    """Create a JIT-compiled dense NR solver for full MNA formulation.

    This solver handles the augmented system where voltage source currents
    are explicit unknowns:

        X = [V_ground, V_1, ..., V_n, I_vs1, ..., I_vsm]

    The Jacobian has size (n_unknowns + n_vsources) × (n_unknowns + n_vsources).

    Args:
        build_system_jit: JIT-wrapped full MNA function
            (X, vsource_vals, isource_vals, Q_prev, integ_c0, device_arrays, ...)
            -> (J_augmented, f_augmented, Q, I_vsource)
        n_nodes: Total node count including ground
        n_vsources: Number of voltage sources (branch currents)
        noi_indices: Optional array of NOI node indices to constrain to 0V
        max_iterations: Maximum NR iterations
        abstol: Absolute tolerance for convergence
        max_step: Maximum voltage/current step per iteration
        options: SimulationOptions for NR damping and other solver parameters.
                 If None, uses defaults (nr_damping=1.0, nr_convtol=0.01).

    Returns:
        JIT-compiled solver function with signature:
            (X, vsource_vals, isource_vals, Q_prev, integ_c0, device_arrays, ...)
            -> (X, iterations, converged, max_f, Q, dQdt, I_vsource)
    """
    # Extract options (captured at trace time, not JAX values)
    nr_damping = options.nr_damping if options is not None else 1.0

    # Extract convergence tolerances from options
    vntol = options.vntol if options is not None else 1e-6
    reltol = options.reltol if options is not None else 1e-3
    n_unknowns = n_nodes - 1
    n_total = n_nodes  # Size of voltage part of X

    # Per-equation absolute tolerance for residual convergence check.
    # Uses vntol (1e-6) as a uniform floor for all equations. The residual check
    # is a safety net — the delta check (with per-unknown tolerances) is the
    # primary VACASK-style convergence criterion. Using raw abstol (1e-12) for
    # KCL equations is too tight for stiff circuits (large capacitors create
    # condition numbers ~1e11 that limit achievable residual precision).
    residual_abs_tol = jnp.concatenate([
        jnp.full(n_unknowns, vntol, dtype=jnp.float64),
        jnp.full(n_vsources, vntol, dtype=jnp.float64),
    ])

    # Per-unknown absolute tolerance for VACASK-style delta convergence check.
    # Voltage unknowns use vntol, branch current unknowns use abstol.
    delta_abs_tol = jnp.concatenate([
        jnp.full(n_unknowns, vntol, dtype=jnp.float64),
        jnp.full(n_vsources, abstol, dtype=jnp.float64),
    ])

    # Compute NOI masks for node equations only
    masks = _compute_noi_masks(noi_indices, n_nodes)
    noi_res_idx = masks["noi_res_idx"]

    # Create augmented residual mask (node equations + branch equations)
    if masks["residual_mask"] is not None:
        # NOI nodes should be masked in residual convergence check
        # Branch equations (vsource voltages) are always checked
        residual_mask = jnp.concatenate(
            [masks["residual_mask"], jnp.ones(n_vsources, dtype=jnp.bool_)]
        )
    else:
        residual_mask = None

    # Define cond_fn and body_fn at factory level to enable JAX tracing cache.
    # These are created once per solver instance, not per solve call.
    # The varying parameters are passed through the state tuple.
    def cond_fn(state):
        # State: (X, iteration, converged, max_f, max_delta, Q, limit_state, <solver_params...>)
        (
            _,
            iteration,
            converged,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
        ) = state
        return jnp.logical_and(~converged, iteration < max_iterations)

    def body_fn(state):
        # Unpack state - includes both iteration state and solver parameters
        (
            X,
            iteration,
            _,
            _,
            _,
            _,
            limit_state,
            vsource_vals,
            isource_vals,
            Q_prev,
            integ_c0,
            device_arrays_arg,
            gmin,
            gshunt,
            integ_c1,
            integ_d1,
            _dQdt_prev,
            integ_c2,
            _Q_prev2,
        ) = state

        J, f, Q, _, limit_state_out = build_system_jit(
            X,
            vsource_vals,
            isource_vals,
            Q_prev,
            integ_c0,
            device_arrays_arg,
            gmin,
            gshunt,
            integ_c1,
            integ_d1,
            _dQdt_prev,
            integ_c2,
            _Q_prev2,
            limit_state,
            iteration,  # NR iteration for iniLim/iteration simparams
        )

        # Check residual convergence with per-equation tolerances (mask NOI nodes)
        if residual_mask is not None:
            f_check = jnp.where(residual_mask, f, 0.0)
        else:
            f_check = f
        max_f = jnp.max(jnp.abs(f_check))
        residual_converged = jnp.all(jnp.abs(f_check) < residual_abs_tol)

        # Enforce NOI constraints on node equations only
        if noi_res_idx is not None:
            # Zero out NOI rows and columns in the node block
            J = J.at[noi_res_idx, :].set(0.0)
            J = J.at[:, noi_res_idx].set(0.0)
            J = J.at[noi_res_idx, noi_res_idx].set(1.0)
            f = f.at[noi_res_idx].set(0.0)

        # Add Tikhonov regularization for numerical stability on GPU
        # JAX's scipy.linalg.solve raises hard errors on singular matrices
        reg = 1e-14 * jnp.eye(J.shape[0], dtype=J.dtype)
        J_reg = J + reg

        # Solve linear system J @ delta = -f
        delta = jax.scipy.linalg.solve(J_reg, -f)

        # VACASK-style delta convergence (before step limiting)
        # Check the damped correction that would actually be applied
        conv_delta = jnp.concatenate([
            delta[:n_unknowns] * nr_damping,
            delta[n_unknowns:],
        ])
        X_ref = jnp.concatenate([X[1:n_total], X[n_total:]])
        tol = jnp.maximum(jnp.abs(X_ref) * reltol, delta_abs_tol)
        if residual_mask is not None:
            conv_delta = jnp.where(residual_mask, conv_delta, 0.0)
        delta_converged = jnp.all(jnp.abs(conv_delta) < tol)

        # Step limiting
        max_delta = jnp.max(jnp.abs(delta))
        scale = jnp.where(max_delta > max_step, max_step / max_delta, 1.0)
        delta = delta * scale

        # Update X: skip ground (index 0), update node voltages and branch currents
        # VACASK system-level damping: simple scalar multiply (nrsolver.cpp:363-365)
        V_damped = X[1:n_total] + delta[:n_unknowns] * nr_damping
        X_new = X.at[1:n_total].set(V_damped)
        X_new = X_new.at[n_total:].add(delta[n_unknowns:])

        # Clamp NOI nodes to 0V
        if noi_indices is not None and len(noi_indices) > 0:
            X_new = X_new.at[noi_indices].set(0.0)

        converged = jnp.logical_or(residual_converged, delta_converged)

        # Return updated state with same solver params (unchanged)
        return (
            X_new,
            iteration + 1,
            converged,
            max_f,
            max_delta,
            Q,
            limit_state_out,
            vsource_vals,
            isource_vals,
            Q_prev,
            integ_c0,
            device_arrays_arg,
            gmin,
            gshunt,
            integ_c1,
            integ_d1,
            _dQdt_prev,
            integ_c2,
            _Q_prev2,
        )

    def nr_solve(
        X_init: Array,
        vsource_vals: Array,
        isource_vals: Array,
        Q_prev: Array,
        integ_c0: float | Array,
        device_arrays_arg: Dict[str, Array],
        gmin: float | Array = 1e-12,
        gshunt: float | Array = 0.0,
        integ_c1: float | Array = 0.0,
        integ_d1: float | Array = 0.0,
        dQdt_prev: Array | None = None,
        integ_c2: float | Array = 0.0,
        Q_prev2: Array | None = None,
        limit_state_in: Array | None = None,
    ):
        # Ensure dQdt_prev is a proper array for JIT tracing
        _dQdt_prev = (
            dQdt_prev if dQdt_prev is not None else jnp.zeros(n_unknowns, dtype=jnp.float64)
        )
        # Ensure Q_prev2 is a proper array for JIT tracing (Gear2 method)
        _Q_prev2 = Q_prev2 if Q_prev2 is not None else jnp.zeros(n_unknowns, dtype=jnp.float64)
        # Ensure limit_state_in is a proper array for JIT tracing
        # Use total_limit_states (captured at trace time) for correct size
        _limit_state = (
            limit_state_in
            if limit_state_in is not None
            else jnp.zeros(total_limit_states, dtype=jnp.float64)
        )

        # Convert scalar parameters to JAX arrays to avoid weak_type retracing
        # Python floats have weak_type=True, explicit arrays have weak_type=False
        _integ_c0 = jnp.asarray(integ_c0, dtype=jnp.float64)
        _gmin = jnp.asarray(gmin, dtype=jnp.float64)
        _gshunt = jnp.asarray(gshunt, dtype=jnp.float64)
        _integ_c1 = jnp.asarray(integ_c1, dtype=jnp.float64)
        _integ_d1 = jnp.asarray(integ_d1, dtype=jnp.float64)
        _integ_c2 = jnp.asarray(integ_c2, dtype=jnp.float64)

        init_Q = jnp.zeros(n_unknowns, dtype=jnp.float64)
        # State includes both iteration state and solver parameters
        init_state = (
            X_init,
            jnp.array(0, dtype=jnp.int32),
            jnp.array(False),
            jnp.array(jnp.inf),
            jnp.array(jnp.inf),
            init_Q,
            _limit_state,
            vsource_vals,
            isource_vals,
            Q_prev,
            _integ_c0,
            device_arrays_arg,
            _gmin,
            _gshunt,
            _integ_c1,
            _integ_d1,
            _dQdt_prev,
            _integ_c2,
            _Q_prev2,
        )

        result_state = lax.while_loop(cond_fn, body_fn, init_state)
        (
            X_final,
            iterations,
            converged,
            max_f,
            _,
            _,
            limit_state_final,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
        ) = result_state

        # Recompute Q and I_vsource from converged solution
        _, _, Q_final, I_vsource, _ = build_system_jit(
            X_final,
            vsource_vals,
            isource_vals,
            Q_prev,
            _integ_c0,
            device_arrays_arg,
            _gmin,
            _gshunt,
            _integ_c1,
            _integ_d1,
            _dQdt_prev,
            _integ_c2,
            _Q_prev2,
            limit_state_final,
            iterations,  # Use final iteration count (iniLim=0 since > 1)
        )

        # Compute dQdt for next timestep (needed for trapezoidal and Gear2 methods)
        dQdt_final = (
            _integ_c0 * Q_final + _integ_c1 * Q_prev + _integ_d1 * _dQdt_prev + _integ_c2 * _Q_prev2
        )

        return (
            X_final,
            iterations,
            converged,
            max_f,
            Q_final,
            dQdt_final,
            I_vsource,
            limit_state_final,
        )

    logger.info(
        f"Creating dense full MNA solver: V({n_nodes}) + I({n_vsources}), NOI: {noi_indices is not None}"
    )
    return jax.jit(nr_solve)


def make_sparse_full_mna_solver(
    build_system_jit: Callable,
    n_nodes: int,
    n_vsources: int,
    nse: int,
    noi_indices: Optional[Array] = None,
    max_iterations: int = 100,
    abstol: float = 1e-12,
    max_step: float = 1.0,
    coo_sort_perm: Optional[Array] = None,
    csr_segment_ids: Optional[Array] = None,
    bcsr_indices: Optional[Array] = None,
    bcsr_indptr: Optional[Array] = None,
    total_limit_states: int = 0,
    options: Optional["SimulationOptions"] = None,
) -> Callable:
    """Create a JIT-compiled sparse NR solver for full MNA formulation.

    This solver handles the augmented system where voltage source currents
    are explicit unknowns, using sparse matrix operations.

        X = [V_ground, V_1, ..., V_n, I_vs1, ..., I_vsm]

    The Jacobian has size (n_unknowns + n_vsources) × (n_unknowns + n_vsources).

    Args:
        build_system_jit: JIT-wrapped full MNA function returning (J_bcoo, f, Q, I_vsource)
        n_nodes: Total node count including ground
        n_vsources: Number of voltage sources (branch currents)
        nse: Number of stored elements after summing duplicates
        noi_indices: Optional array of NOI node indices to constrain to 0V
        max_iterations: Maximum NR iterations
        abstol: Absolute tolerance for convergence
        max_step: Maximum voltage/current step per iteration
        coo_sort_perm: Pre-computed COO→CSR permutation
        csr_segment_ids: Pre-computed segment IDs for duplicate summing
        bcsr_indices: Pre-computed CSR column indices
        bcsr_indptr: Pre-computed CSR row pointers

    Returns:
        JIT-compiled solver function with same signature as make_dense_full_mna_solver
    """
    # Extract options (captured at trace time, not JAX values)
    nr_damping = options.nr_damping if options is not None else 1.0

    # Extract convergence tolerances from options
    vntol = options.vntol if options is not None else 1e-6
    reltol = options.reltol if options is not None else 1e-3
    from jax.experimental.sparse import BCSR
    from jax.experimental.sparse.linalg import spsolve

    n_unknowns = n_nodes - 1
    n_total = n_nodes  # Size of voltage part of X

    # Per-equation absolute tolerance for residual convergence check.
    # Uses vntol (1e-6) as a uniform floor for all equations. The residual check
    # is a safety net — the delta check (with per-unknown tolerances) is the
    # primary VACASK-style convergence criterion. Using raw abstol (1e-12) for
    # KCL equations is too tight for stiff circuits (large capacitors create
    # condition numbers ~1e11 that limit achievable residual precision).
    residual_abs_tol = jnp.concatenate([
        jnp.full(n_unknowns, vntol, dtype=jnp.float64),
        jnp.full(n_vsources, vntol, dtype=jnp.float64),
    ])

    # Per-unknown absolute tolerance for VACASK-style delta convergence check.
    # Voltage unknowns use vntol, branch current unknowns use abstol.
    delta_abs_tol = jnp.concatenate([
        jnp.full(n_unknowns, vntol, dtype=jnp.float64),
        jnp.full(n_vsources, abstol, dtype=jnp.float64),
    ])

    use_precomputed = (
        coo_sort_perm is not None
        and csr_segment_ids is not None
        and bcsr_indices is not None
        and bcsr_indptr is not None
    )

    # Compute NOI masks for node equations only (branch equations are not masked)
    masks = _compute_noi_masks(noi_indices, n_nodes, bcsr_indptr, bcsr_indices)
    noi_row_mask = masks["noi_row_mask"]
    noi_col_mask = masks["noi_col_mask"]
    noi_diag_indices = masks["noi_diag_indices"]
    noi_res_indices_arr = masks["noi_res_indices_arr"]

    # Create augmented residual mask (node equations + branch equations)
    if masks["residual_mask"] is not None:
        # NOI nodes should be masked in residual convergence check
        # Branch equations (vsource voltages) are always checked
        residual_mask = jnp.concatenate(
            [masks["residual_mask"], jnp.ones(n_vsources, dtype=jnp.bool_)]
        )
    else:
        residual_mask = None

    # Define cond_fn and body_fn at factory level to enable JAX tracing cache.
    def cond_fn(state):
        # State: (X, iteration, converged, max_f, max_delta, Q, <solver_params...>)
        (
            _,
            iteration,
            converged,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
        ) = state
        return jnp.logical_and(~converged, iteration < max_iterations)

    def body_fn(state):
        # Unpack state - includes both iteration state and solver parameters
        (
            X,
            iteration,
            _,
            _,
            _,
            _,
            vsource_vals,
            isource_vals,
            Q_prev,
            integ_c0,
            device_arrays_arg,
            gmin,
            gshunt,
            integ_c1,
            integ_d1,
            integ_c2,
            _dQdt_prev,
            _Q_prev2,
        ) = state

        J_bcoo, f, Q, _, _ = build_system_jit(
            X,
            vsource_vals,
            isource_vals,
            Q_prev,
            integ_c0,
            device_arrays_arg,
            gmin,
            gshunt,
            integ_c1,
            integ_d1,
            _dQdt_prev,
            integ_c2,
            _Q_prev2,
            None,
            iteration,  # NR iteration for iniLim/iteration simparams
        )

        # Check residual convergence with per-equation tolerances (mask NOI nodes)
        if residual_mask is not None:
            f_check = jnp.where(residual_mask, f, 0.0)
        else:
            f_check = f
        max_f = jnp.max(jnp.abs(f_check))
        residual_converged = jnp.all(jnp.abs(f_check) < residual_abs_tol)

        if use_precomputed:
            # Fast path: pre-computed COO→CSR
            coo_vals = J_bcoo.data
            sorted_vals = coo_vals[coo_sort_perm]
            csr_data = jax.ops.segment_sum(sorted_vals, csr_segment_ids, num_segments=nse)

            f_solve = f
            if noi_row_mask is not None:
                csr_data = csr_data.at[noi_row_mask].set(0.0)
                csr_data = csr_data.at[noi_col_mask].set(0.0)
                csr_data = csr_data.at[noi_diag_indices].set(1.0)
                f_solve = f.at[noi_res_indices_arr].set(0.0)

            delta = spsolve(csr_data, bcsr_indices, bcsr_indptr, -f_solve, tol=1e-6)
        else:
            # Fallback: sort each iteration
            J_bcoo_dedup = J_bcoo.sum_duplicates(nse=nse)
            J_bcsr = BCSR.from_bcoo(J_bcoo_dedup)

            data = J_bcsr.data
            f_solve = f
            if noi_row_mask is not None:
                data = data.at[noi_row_mask].set(0.0)
                data = data.at[noi_col_mask].set(0.0)
                data = data.at[noi_diag_indices].set(1.0)
                f_solve = f.at[noi_res_indices_arr].set(0.0)

            delta = spsolve(data, J_bcsr.indices, J_bcsr.indptr, -f_solve, tol=1e-6)

        # VACASK-style delta convergence (before step limiting)
        conv_delta = jnp.concatenate([
            delta[:n_unknowns] * nr_damping,
            delta[n_unknowns:],
        ])
        X_ref = jnp.concatenate([X[1:n_total], X[n_total:]])
        tol = jnp.maximum(jnp.abs(X_ref) * reltol, delta_abs_tol)
        if residual_mask is not None:
            conv_delta = jnp.where(residual_mask, conv_delta, 0.0)
        delta_converged = jnp.all(jnp.abs(conv_delta) < tol)

        # Step limiting
        max_delta = jnp.max(jnp.abs(delta))
        scale = jnp.where(max_delta > max_step, max_step / max_delta, 1.0)
        delta = delta * scale

        # Update X: VACASK system-level damping: simple scalar multiply
        V_damped = X[1:n_total] + delta[:n_unknowns] * nr_damping
        X_new = X.at[1:n_total].set(V_damped)
        X_new = X_new.at[n_total:].add(delta[n_unknowns:])

        # Clamp NOI nodes to 0V
        if noi_indices is not None and len(noi_indices) > 0:
            X_new = X_new.at[noi_indices].set(0.0)

        converged = jnp.logical_or(residual_converged, delta_converged)

        # Return updated state with same solver params (unchanged)
        return (
            X_new,
            iteration + 1,
            converged,
            max_f,
            max_delta,
            Q,
            vsource_vals,
            isource_vals,
            Q_prev,
            integ_c0,
            device_arrays_arg,
            gmin,
            gshunt,
            integ_c1,
            integ_d1,
            integ_c2,
            _dQdt_prev,
            _Q_prev2,
        )

    def nr_solve(
        X_init: Array,
        vsource_vals: Array,
        isource_vals: Array,
        Q_prev: Array,
        integ_c0: float | Array,
        device_arrays_arg: Dict[str, Array],
        gmin: float | Array = 1e-12,
        gshunt: float | Array = 0.0,
        integ_c1: float | Array = 0.0,
        integ_d1: float | Array = 0.0,
        dQdt_prev: Array | None = None,
        integ_c2: float | Array = 0.0,
        Q_prev2: Array | None = None,
    ):
        # Ensure arrays are proper for JIT tracing
        _dQdt_prev = (
            dQdt_prev if dQdt_prev is not None else jnp.zeros(n_unknowns, dtype=jnp.float64)
        )
        _Q_prev2 = Q_prev2 if Q_prev2 is not None else jnp.zeros(n_unknowns, dtype=jnp.float64)

        # Convert scalar parameters to JAX arrays to avoid weak_type retracing
        _integ_c0 = jnp.asarray(integ_c0, dtype=jnp.float64)
        _gmin = jnp.asarray(gmin, dtype=jnp.float64)
        _gshunt = jnp.asarray(gshunt, dtype=jnp.float64)
        _integ_c1 = jnp.asarray(integ_c1, dtype=jnp.float64)
        _integ_d1 = jnp.asarray(integ_d1, dtype=jnp.float64)
        _integ_c2 = jnp.asarray(integ_c2, dtype=jnp.float64)

        init_Q = jnp.zeros(n_unknowns, dtype=jnp.float64)
        # State includes both iteration state and solver parameters
        init_state = (
            X_init,
            jnp.array(0, dtype=jnp.int32),
            jnp.array(False),
            jnp.array(jnp.inf),
            jnp.array(jnp.inf),
            init_Q,
            vsource_vals,
            isource_vals,
            Q_prev,
            _integ_c0,
            device_arrays_arg,
            _gmin,
            _gshunt,
            _integ_c1,
            _integ_d1,
            _integ_c2,
            _dQdt_prev,
            _Q_prev2,
        )

        result_state = lax.while_loop(cond_fn, body_fn, init_state)
        (
            X_final,
            iterations,
            converged,
            max_f,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
        ) = result_state

        # Recompute Q and I_vsource from converged solution
        _, _, Q_final, I_vsource, limit_state_final = build_system_jit(
            X_final,
            vsource_vals,
            isource_vals,
            Q_prev,
            _integ_c0,
            device_arrays_arg,
            _gmin,
            _gshunt,
            _integ_c1,
            _integ_d1,
            _dQdt_prev,
            _integ_c2,
            _Q_prev2,
            None,
            iterations,  # Use final iteration count (iniLim=0)
        )

        # Compute dQdt for next timestep
        dQdt_final = (
            _integ_c0 * Q_final + _integ_c1 * Q_prev + _integ_d1 * _dQdt_prev + _integ_c2 * _Q_prev2
        )

        return (
            X_final,
            iterations,
            converged,
            max_f,
            Q_final,
            dQdt_final,
            I_vsource,
            limit_state_final,
        )

    logger.info(
        f"Creating sparse full MNA solver: V({n_nodes}) + I({n_vsources}), nse={nse}, NOI: {noi_indices is not None}"
    )
    return jax.jit(nr_solve)


def make_umfpack_full_mna_solver(
    build_system_jit: Callable,
    n_nodes: int,
    n_vsources: int,
    nse: int,
    bcsr_indptr: Array,
    bcsr_indices: Array,
    noi_indices: Optional[Array] = None,
    max_iterations: int = 100,
    abstol: float = 1e-12,
    max_step: float = 1.0,
    coo_sort_perm: Optional[Array] = None,
    csr_segment_ids: Optional[Array] = None,
    total_limit_states: int = 0,
    options: Optional["SimulationOptions"] = None,
) -> Callable:
    """Create a JIT-compiled UMFPACK NR solver for full MNA formulation.

    Uses UMFPACK with cached symbolic factorization for fast CPU solving
    with the augmented system where voltage source currents are explicit unknowns.

    Args:
        build_system_jit: JIT-wrapped full MNA function returning (J_bcoo, f, Q, I_vsource)
        n_nodes: Total node count including ground
        n_vsources: Number of voltage sources (branch currents)
        nse: Number of stored elements after summing duplicates
        bcsr_indptr: Pre-computed BCSR row pointers
        bcsr_indices: Pre-computed BCSR column indices
        noi_indices: Optional array of NOI node indices to constrain to 0V
        max_iterations: Maximum NR iterations
        abstol: Absolute tolerance for convergence
        max_step: Maximum voltage/current step per iteration
        coo_sort_perm: Pre-computed COO→CSR permutation
        csr_segment_ids: Pre-computed segment IDs
        options: SimulationOptions for NR damping and other solver parameters.
                 If None, uses defaults (nr_damping=1.0, nr_convtol=0.01).

    Returns:
        JIT-compiled UMFPACK solver function with full MNA augmented system
    """
    # Extract options (captured at trace time, not JAX values)
    nr_damping = options.nr_damping if options is not None else 1.0

    # Extract convergence tolerances from options
    vntol = options.vntol if options is not None else 1e-6
    reltol = options.reltol if options is not None else 1e-3

    from jax.experimental.sparse import BCSR

    from jax_spice.analysis.umfpack_solver import UMFPACKSolver

    n_unknowns = n_nodes - 1
    n_total = n_nodes

    # Per-equation absolute tolerance for residual convergence check.
    # Uses vntol (1e-6) as a uniform floor for all equations. The residual check
    # is a safety net — the delta check (with per-unknown tolerances) is the
    # primary VACASK-style convergence criterion. Using raw abstol (1e-12) for
    # KCL equations is too tight for stiff circuits (large capacitors create
    # condition numbers ~1e11 that limit achievable residual precision).
    residual_abs_tol = jnp.concatenate([
        jnp.full(n_unknowns, vntol, dtype=jnp.float64),
        jnp.full(n_vsources, vntol, dtype=jnp.float64),
    ])

    # Per-unknown absolute tolerance for VACASK-style delta convergence check.
    # Voltage unknowns use vntol, branch current unknowns use abstol.
    delta_abs_tol = jnp.concatenate([
        jnp.full(n_unknowns, vntol, dtype=jnp.float64),
        jnp.full(n_vsources, abstol, dtype=jnp.float64),
    ])

    use_precomputed = coo_sort_perm is not None and csr_segment_ids is not None

    # Compute NOI masks for node equations only
    masks = _compute_noi_masks(noi_indices, n_nodes, bcsr_indptr, bcsr_indices)
    noi_row_mask = masks["noi_row_mask"]
    noi_col_mask = masks["noi_col_mask"]
    noi_diag_indices = masks["noi_diag_indices"]
    noi_res_indices_arr = masks["noi_res_indices_arr"]

    # Augmented residual mask
    if masks["residual_mask"] is not None:
        residual_mask = jnp.concatenate(
            [masks["residual_mask"], jnp.ones(n_vsources, dtype=jnp.bool_)]
        )
    else:
        residual_mask = None

    # Create UMFPACK solver with cached symbolic factorization
    umfpack_solver = UMFPACKSolver(bcsr_indptr, bcsr_indices)
    logger.info("Created UMFPACK full MNA solver with cached symbolic factorization")

    # Define cond_fn and body_fn at factory level to enable JAX tracing cache.
    def cond_fn(state):
        (
            _,
            iteration,
            converged,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
        ) = state
        return jnp.logical_and(~converged, iteration < max_iterations)

    def body_fn(state):
        (
            X,
            iteration,
            _,
            _,
            _,
            _,
            vsource_vals,
            isource_vals,
            Q_prev,
            integ_c0,
            device_arrays_arg,
            gmin,
            gshunt,
            integ_c1,
            integ_d1,
            integ_c2,
            _dQdt_prev,
            _Q_prev2,
        ) = state

        J_bcoo, f, Q, _, _ = build_system_jit(
            X,
            vsource_vals,
            isource_vals,
            Q_prev,
            integ_c0,
            device_arrays_arg,
            gmin,
            gshunt,
            integ_c1,
            integ_d1,
            _dQdt_prev,
            integ_c2,
            _Q_prev2,
            None,
            iteration,  # NR iteration for iniLim/iteration simparams
        )

        # Check residual convergence with per-equation tolerances (mask NOI nodes)
        if residual_mask is not None:
            f_check = jnp.where(residual_mask, f, 0.0)
        else:
            f_check = f
        max_f = jnp.max(jnp.abs(f_check))
        residual_converged = jnp.all(jnp.abs(f_check) < residual_abs_tol)

        if use_precomputed:
            coo_vals = J_bcoo.data
            sorted_vals = coo_vals[coo_sort_perm]
            csr_data = jax.ops.segment_sum(sorted_vals, csr_segment_ids, num_segments=nse)
        else:
            J_bcoo_dedup = J_bcoo.sum_duplicates(nse=nse)
            J_bcsr = BCSR.from_bcoo(J_bcoo_dedup)
            csr_data = J_bcsr.data

        f_solve = f
        if noi_row_mask is not None:
            csr_data = csr_data.at[noi_row_mask].set(0.0)
            csr_data = csr_data.at[noi_col_mask].set(0.0)
            csr_data = csr_data.at[noi_diag_indices].set(1.0)
            f_solve = f.at[noi_res_indices_arr].set(0.0)

        delta, _info = umfpack_solver(-f_solve, csr_data)

        # VACASK-style delta convergence (before step limiting)
        conv_delta = jnp.concatenate([
            delta[:n_unknowns] * nr_damping,
            delta[n_unknowns:],
        ])
        X_ref = jnp.concatenate([X[1:n_total], X[n_total:]])
        tol = jnp.maximum(jnp.abs(X_ref) * reltol, delta_abs_tol)
        if residual_mask is not None:
            conv_delta = jnp.where(residual_mask, conv_delta, 0.0)
        delta_converged = jnp.all(jnp.abs(conv_delta) < tol)

        # Step limiting
        max_delta = jnp.max(jnp.abs(delta))
        scale = jnp.where(max_delta > max_step, max_step / max_delta, 1.0)
        delta = delta * scale

        # Update X: VACASK system-level damping: simple scalar multiply
        V_damped = X[1:n_total] + delta[:n_unknowns] * nr_damping
        X_new = X.at[1:n_total].set(V_damped)
        X_new = X_new.at[n_total:].add(delta[n_unknowns:])

        if noi_indices is not None and len(noi_indices) > 0:
            X_new = X_new.at[noi_indices].set(0.0)

        converged = jnp.logical_or(residual_converged, delta_converged)

        return (
            X_new,
            iteration + 1,
            converged,
            max_f,
            max_delta,
            Q,
            vsource_vals,
            isource_vals,
            Q_prev,
            integ_c0,
            device_arrays_arg,
            gmin,
            gshunt,
            integ_c1,
            integ_d1,
            integ_c2,
            _dQdt_prev,
            _Q_prev2,
        )

    def nr_solve(
        X_init: Array,
        vsource_vals: Array,
        isource_vals: Array,
        Q_prev: Array,
        integ_c0: float | Array,
        device_arrays_arg: Dict[str, Array],
        gmin: float | Array = 1e-12,
        gshunt: float | Array = 0.0,
        integ_c1: float | Array = 0.0,
        integ_d1: float | Array = 0.0,
        dQdt_prev: Array | None = None,
        integ_c2: float | Array = 0.0,
        Q_prev2: Array | None = None,
    ):
        _dQdt_prev = (
            dQdt_prev if dQdt_prev is not None else jnp.zeros(n_unknowns, dtype=jnp.float64)
        )
        _Q_prev2 = Q_prev2 if Q_prev2 is not None else jnp.zeros(n_unknowns, dtype=jnp.float64)

        # Convert scalar parameters to JAX arrays to avoid weak_type retracing
        _integ_c0 = jnp.asarray(integ_c0, dtype=jnp.float64)
        _gmin = jnp.asarray(gmin, dtype=jnp.float64)
        _gshunt = jnp.asarray(gshunt, dtype=jnp.float64)
        _integ_c1 = jnp.asarray(integ_c1, dtype=jnp.float64)
        _integ_d1 = jnp.asarray(integ_d1, dtype=jnp.float64)
        _integ_c2 = jnp.asarray(integ_c2, dtype=jnp.float64)

        init_Q = jnp.zeros(n_unknowns, dtype=jnp.float64)
        init_state = (
            X_init,
            jnp.array(0, dtype=jnp.int32),
            jnp.array(False),
            jnp.array(jnp.inf),
            jnp.array(jnp.inf),
            init_Q,
            vsource_vals,
            isource_vals,
            Q_prev,
            _integ_c0,
            device_arrays_arg,
            _gmin,
            _gshunt,
            _integ_c1,
            _integ_d1,
            _integ_c2,
            _dQdt_prev,
            _Q_prev2,
        )

        result_state = lax.while_loop(cond_fn, body_fn, init_state)
        (
            X_final,
            iterations,
            converged,
            max_f,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
        ) = result_state

        _, _, Q_final, I_vsource, limit_state_final = build_system_jit(
            X_final,
            vsource_vals,
            isource_vals,
            Q_prev,
            _integ_c0,
            device_arrays_arg,
            _gmin,
            _gshunt,
            _integ_c1,
            _integ_d1,
            _dQdt_prev,
            _integ_c2,
            _Q_prev2,
            None,
            iterations,  # Use final iteration count (iniLim=0)
        )

        dQdt_final = (
            _integ_c0 * Q_final + _integ_c1 * Q_prev + _integ_d1 * _dQdt_prev + _integ_c2 * _Q_prev2
        )

        return (
            X_final,
            iterations,
            converged,
            max_f,
            Q_final,
            dQdt_final,
            I_vsource,
            limit_state_final,
        )

    logger.info(f"Creating UMFPACK full MNA solver: V({n_nodes}) + I({n_vsources})")
    return jax.jit(nr_solve)


def make_spineax_full_mna_solver(
    build_system_jit: Callable,
    n_nodes: int,
    n_vsources: int,
    nse: int,
    bcsr_indptr: Array,
    bcsr_indices: Array,
    noi_indices: Optional[Array] = None,
    max_iterations: int = 100,
    abstol: float = 1e-12,
    max_step: float = 1.0,
    coo_sort_perm: Optional[Array] = None,
    csr_segment_ids: Optional[Array] = None,
    total_limit_states: int = 0,
    options: Optional["SimulationOptions"] = None,
) -> Callable:
    """Create a JIT-compiled sparse NR solver using Spineax/cuDSS for full MNA.

    Uses Spineax's cuDSS wrapper with cached symbolic factorization.
    Supports full MNA formulation with branch currents as explicit unknowns.

    Args:
        build_system_jit: JIT-wrapped full MNA function returning (J_bcoo, f, Q, I_vsource, limit_state_out)
        n_nodes: Total node count including ground
        n_vsources: Number of voltage sources (branch currents)
        nse: Number of stored elements after summing duplicates
        bcsr_indptr: Pre-computed BCSR row pointers
        bcsr_indices: Pre-computed BCSR column indices
        noi_indices: Optional array of NOI node indices to constrain to 0V
        max_iterations: Maximum NR iterations
        abstol: Absolute tolerance for convergence
        max_step: Maximum voltage/current step per iteration
        coo_sort_perm: Pre-computed COO→CSR permutation
        csr_segment_ids: Pre-computed segment IDs
        options: SimulationOptions for NR damping and other solver parameters.
                 If None, uses defaults (nr_damping=1.0, nr_convtol=0.01).

    Returns:
        JIT-compiled Spineax solver function for full MNA
    """
    # Extract options (captured at trace time, not JAX values)
    nr_damping = options.nr_damping if options is not None else 1.0

    # Extract convergence tolerances from options
    vntol = options.vntol if options is not None else 1e-6
    reltol = options.reltol if options is not None else 1e-3

    from jax.experimental.sparse import BCSR
    from spineax.cudss.solver import CuDSSSolver

    n_unknowns = n_nodes - 1
    n_total = n_nodes

    # Per-equation absolute tolerance for residual convergence check.
    # Uses vntol (1e-6) as a uniform floor for all equations. The residual check
    # is a safety net — the delta check (with per-unknown tolerances) is the
    # primary VACASK-style convergence criterion. Using raw abstol (1e-12) for
    # KCL equations is too tight for stiff circuits (large capacitors create
    # condition numbers ~1e11 that limit achievable residual precision).
    residual_abs_tol = jnp.concatenate([
        jnp.full(n_unknowns, vntol, dtype=jnp.float64),
        jnp.full(n_vsources, vntol, dtype=jnp.float64),
    ])

    # Per-unknown absolute tolerance for VACASK-style delta convergence check.
    # Voltage unknowns use vntol, branch current unknowns use abstol.
    delta_abs_tol = jnp.concatenate([
        jnp.full(n_unknowns, vntol, dtype=jnp.float64),
        jnp.full(n_vsources, abstol, dtype=jnp.float64),
    ])
    use_precomputed = coo_sort_perm is not None and csr_segment_ids is not None

    masks = _compute_noi_masks(noi_indices, n_nodes, bcsr_indptr, bcsr_indices)
    noi_row_mask = masks["noi_row_mask"]
    noi_col_mask = masks["noi_col_mask"]
    noi_diag_indices = masks["noi_diag_indices"]
    noi_res_indices_arr = masks["noi_res_indices_arr"]

    # Extend residual mask to include branch current rows
    if masks["residual_mask"] is not None:
        residual_mask = jnp.concatenate(
            [masks["residual_mask"], jnp.ones(n_vsources, dtype=jnp.bool_)]
        )
    else:
        residual_mask = None

    # Create Spineax solver with cached symbolic factorization
    spineax_solver = CuDSSSolver(
        bcsr_indptr,
        bcsr_indices,
        device_id=0,
        mtype_id=1,
        mview_id=0,
    )
    logger.info("Created Spineax full MNA solver with cached symbolic factorization")

    # Define cond_fn and body_fn at factory level to enable JAX tracing cache.
    def cond_fn(state):
        (
            _,
            iteration,
            converged,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
        ) = state
        return jnp.logical_and(~converged, iteration < max_iterations)

    def body_fn(state):
        (
            X,
            iteration,
            _,
            _,
            _,
            _,
            vsource_vals,
            isource_vals,
            Q_prev,
            integ_c0,
            device_arrays_arg,
            gmin,
            gshunt,
            integ_c1,
            integ_d1,
            _dQdt_prev,
            integ_c2,
            _Q_prev2,
        ) = state

        J_bcoo, f, Q, _, _ = build_system_jit(
            X,
            vsource_vals,
            isource_vals,
            Q_prev,
            integ_c0,
            device_arrays_arg,
            gmin,
            gshunt,
            integ_c1,
            integ_d1,
            _dQdt_prev,
            integ_c2,
            _Q_prev2,
            None,  # limit_state_in
            iteration,  # NR iteration for iniLim/iteration simparams
        )

        # Check residual convergence with per-equation tolerances (mask NOI nodes)
        if residual_mask is not None:
            f_check = jnp.where(residual_mask, f, 0.0)
        else:
            f_check = f
        max_f = jnp.max(jnp.abs(f_check))
        residual_converged = jnp.all(jnp.abs(f_check) < residual_abs_tol)

        if use_precomputed:
            coo_vals = J_bcoo.data
            sorted_vals = coo_vals[coo_sort_perm]
            csr_data = jax.ops.segment_sum(sorted_vals, csr_segment_ids, num_segments=nse)
        else:
            J_bcoo_dedup = J_bcoo.sum_duplicates(nse=nse)
            J_bcsr = BCSR.from_bcoo(J_bcoo_dedup)
            csr_data = J_bcsr.data

        f_solve = f
        if noi_row_mask is not None:
            csr_data = csr_data.at[noi_row_mask].set(0.0)
            csr_data = csr_data.at[noi_col_mask].set(0.0)
            csr_data = csr_data.at[noi_diag_indices].set(1.0)
            f_solve = f.at[noi_res_indices_arr].set(0.0)

        delta, _info = spineax_solver(-f_solve, csr_data)

        # VACASK-style delta convergence (before step limiting)
        conv_delta = jnp.concatenate([
            delta[:n_unknowns] * nr_damping,
            delta[n_unknowns:],
        ])
        X_ref = jnp.concatenate([X[1:n_total], X[n_total:]])
        tol = jnp.maximum(jnp.abs(X_ref) * reltol, delta_abs_tol)
        if residual_mask is not None:
            conv_delta = jnp.where(residual_mask, conv_delta, 0.0)
        delta_converged = jnp.all(jnp.abs(conv_delta) < tol)

        # Step limiting
        max_delta = jnp.max(jnp.abs(delta))
        scale = jnp.where(max_delta > max_step, max_step / max_delta, 1.0)
        delta = delta * scale

        # Update X: VACASK system-level damping: simple scalar multiply
        V_damped = X[1:n_total] + delta[:n_unknowns] * nr_damping
        X_new = X.at[1:n_total].set(V_damped)
        X_new = X_new.at[n_total:].add(delta[n_unknowns:])

        if noi_indices is not None and len(noi_indices) > 0:
            X_new = X_new.at[noi_indices].set(0.0)

        converged = jnp.logical_or(residual_converged, delta_converged)

        return (
            X_new,
            iteration + 1,
            converged,
            max_f,
            max_delta,
            Q,
            vsource_vals,
            isource_vals,
            Q_prev,
            integ_c0,
            device_arrays_arg,
            gmin,
            gshunt,
            integ_c1,
            integ_d1,
            _dQdt_prev,
            integ_c2,
            _Q_prev2,
        )

    def nr_solve(
        X_init: Array,
        vsource_vals: Array,
        isource_vals: Array,
        Q_prev: Array,
        integ_c0: float | Array,
        device_arrays_arg: Dict[str, Array],
        gmin: float | Array = 1e-12,
        gshunt: float | Array = 0.0,
        integ_c1: float | Array = 0.0,
        integ_d1: float | Array = 0.0,
        dQdt_prev: Array | None = None,
        integ_c2: float | Array = 0.0,
        Q_prev2: Array | None = None,
        limit_state_in: Array | None = None,
    ):
        _dQdt_prev = (
            dQdt_prev if dQdt_prev is not None else jnp.zeros(n_unknowns, dtype=jnp.float64)
        )
        _Q_prev2 = Q_prev2 if Q_prev2 is not None else jnp.zeros(n_unknowns, dtype=jnp.float64)

        # Convert scalar parameters to JAX arrays to avoid weak_type retracing
        _integ_c0 = jnp.asarray(integ_c0, dtype=jnp.float64)
        _gmin = jnp.asarray(gmin, dtype=jnp.float64)
        _gshunt = jnp.asarray(gshunt, dtype=jnp.float64)
        _integ_c1 = jnp.asarray(integ_c1, dtype=jnp.float64)
        _integ_d1 = jnp.asarray(integ_d1, dtype=jnp.float64)
        _integ_c2 = jnp.asarray(integ_c2, dtype=jnp.float64)

        init_Q = jnp.zeros(n_unknowns, dtype=jnp.float64)
        init_state = (
            X_init,
            jnp.array(0, dtype=jnp.int32),
            jnp.array(False),
            jnp.array(jnp.inf),
            jnp.array(jnp.inf),
            init_Q,
            vsource_vals,
            isource_vals,
            Q_prev,
            _integ_c0,
            device_arrays_arg,
            _gmin,
            _gshunt,
            _integ_c1,
            _integ_d1,
            _dQdt_prev,
            _integ_c2,
            _Q_prev2,
        )

        result_state = lax.while_loop(cond_fn, body_fn, init_state)
        (
            X_final,
            iterations,
            converged,
            max_f,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
        ) = result_state

        _, _, Q_final, I_vsource, limit_state_final = build_system_jit(
            X_final,
            vsource_vals,
            isource_vals,
            Q_prev,
            _integ_c0,
            device_arrays_arg,
            _gmin,
            _gshunt,
            _integ_c1,
            _integ_d1,
            _dQdt_prev,
            _integ_c2,
            _Q_prev2,
            None,  # limit_state_in
            iterations,  # Use final iteration count (iniLim=0)
        )

        # Compute dQdt for next timestep
        dQdt_final = (
            _integ_c0 * Q_final + _integ_c1 * Q_prev + _integ_d1 * _dQdt_prev + _integ_c2 * _Q_prev2
        )

        return (
            X_final,
            iterations,
            converged,
            max_f,
            Q_final,
            dQdt_final,
            I_vsource,
            limit_state_final,
        )

    logger.info(f"Creating Spineax full MNA NR solver: V({n_nodes}) + I({n_vsources})")
    return jax.jit(nr_solve)


def is_umfpack_ffi_available() -> bool:
    """Check if the UMFPACK FFI extension is available.

    The FFI version eliminates the ~100ms pure_callback overhead per solve,
    reducing solve time from ~117ms to ~17ms for large circuits.

    Returns:
        True if the FFI extension is installed and working.
    """
    try:
        from jax_spice.sparse import umfpack_jax

        return umfpack_jax.is_available()
    except ImportError:
        return False


def make_umfpack_ffi_full_mna_solver(
    build_system_jit: Callable,
    n_nodes: int,
    n_vsources: int,
    nse: int,
    bcsr_indptr: Array,
    bcsr_indices: Array,
    noi_indices: Optional[Array] = None,
    max_iterations: int = 100,
    abstol: float = 1e-12,
    max_step: float = 1.0,
    coo_sort_perm: Optional[Array] = None,
    csr_segment_ids: Optional[Array] = None,
    total_limit_states: int = 0,
    options: Optional["SimulationOptions"] = None,
) -> Callable:
    """Create a JIT-compiled UMFPACK FFI NR solver for full MNA formulation.

    This is the FFI-based version that eliminates the ~100ms pure_callback overhead.
    Uses UMFPACK directly via XLA FFI for fast CPU solving.

    Args:
        build_system_jit: JIT-wrapped full MNA function returning (J_bcoo, f, Q, I_vsource, limit_state_out)
        n_nodes: Total node count including ground
        n_vsources: Number of voltage sources (branch currents)
        nse: Number of stored elements after summing duplicates
        bcsr_indptr: Pre-computed BCSR row pointers
        bcsr_indices: Pre-computed BCSR column indices
        noi_indices: Optional array of NOI node indices to constrain to 0V
        max_iterations: Maximum NR iterations
        abstol: Absolute tolerance for convergence
        max_step: Maximum voltage/current step per iteration
        coo_sort_perm: Pre-computed COO→CSR permutation
        csr_segment_ids: Pre-computed segment IDs
        total_limit_states: Total number of limit states across all device models
        options: SimulationOptions for NR damping and other solver parameters.
                 If None, uses defaults (nr_damping=1.0, nr_convtol=0.01).

    Returns:
        JIT-compiled solver function
    """
    # Extract options (captured at trace time, not JAX values)
    nr_damping = options.nr_damping if options is not None else 1.0

    # Extract convergence tolerances from options
    vntol = options.vntol if options is not None else 1e-6
    reltol = options.reltol if options is not None else 1e-3

    from jax.experimental.sparse import BCSR

    from jax_spice.sparse import umfpack_jax

    if not umfpack_jax.is_available():
        raise RuntimeError(
            "UMFPACK FFI extension not available. "
            "Install with: cd jax_spice/sparse && pip install ."
        )

    n_unknowns = n_nodes - 1
    n_total = n_nodes

    # Per-equation absolute tolerance for residual convergence check.
    # Uses vntol (1e-6) as a uniform floor for all equations. The residual check
    # is a safety net — the delta check (with per-unknown tolerances) is the
    # primary VACASK-style convergence criterion. Using raw abstol (1e-12) for
    # KCL equations is too tight for stiff circuits (large capacitors create
    # condition numbers ~1e11 that limit achievable residual precision).
    residual_abs_tol = jnp.concatenate([
        jnp.full(n_unknowns, vntol, dtype=jnp.float64),
        jnp.full(n_vsources, vntol, dtype=jnp.float64),
    ])

    # Per-unknown absolute tolerance for VACASK-style delta convergence check.
    # Voltage unknowns use vntol, branch current unknowns use abstol.
    delta_abs_tol = jnp.concatenate([
        jnp.full(n_unknowns, vntol, dtype=jnp.float64),
        jnp.full(n_vsources, abstol, dtype=jnp.float64),
    ])

    use_precomputed = coo_sort_perm is not None and csr_segment_ids is not None

    masks = _compute_noi_masks(noi_indices, n_nodes, bcsr_indptr, bcsr_indices)
    noi_row_mask = masks["noi_row_mask"]
    noi_col_mask = masks["noi_col_mask"]
    noi_diag_indices = masks["noi_diag_indices"]
    noi_res_indices_arr = masks["noi_res_indices_arr"]

    if masks["residual_mask"] is not None:
        residual_mask = jnp.concatenate(
            [masks["residual_mask"], jnp.ones(n_vsources, dtype=jnp.bool_)]
        )
    else:
        residual_mask = None

    logger.info("Creating UMFPACK FFI full MNA solver with zero callback overhead")

    # Define body_fn and cond_fn at factory level to enable JAX tracing cache.
    # These are created once per solver instance, not per solve call.
    # The varying parameters are passed through the state tuple.
    def cond_fn(state):
        # State: (X, iteration, converged, max_f, max_delta, Q, limit_state, <solver_params...>)
        _, iteration, converged, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = state
        return jnp.logical_and(~converged, iteration < max_iterations)

    def body_fn(state):
        # Unpack state - includes both iteration state and solver parameters
        (
            X,
            iteration,
            _,
            _,
            _,
            _,
            limit_state,
            vsource_vals,
            isource_vals,
            Q_prev,
            integ_c0,
            device_arrays_arg,
            gmin,
            gshunt,
            integ_c1,
            integ_d1,
            _dQdt_prev,
            integ_c2,
            _Q_prev2,
        ) = state

        J_bcoo, f, Q, _, limit_state_out = build_system_jit(
            X,
            vsource_vals,
            isource_vals,
            Q_prev,
            integ_c0,
            device_arrays_arg,
            gmin,
            gshunt,
            integ_c1,
            integ_d1,
            _dQdt_prev,
            integ_c2,
            _Q_prev2,
            limit_state,
            iteration,  # NR iteration for iniLim/iteration simparams
        )

        # Check residual convergence with per-equation tolerances (mask NOI nodes)
        if residual_mask is not None:
            f_check = jnp.where(residual_mask, f, 0.0)
        else:
            f_check = f
        max_f = jnp.max(jnp.abs(f_check))
        residual_converged = jnp.all(jnp.abs(f_check) < residual_abs_tol)

        if use_precomputed:
            coo_vals = J_bcoo.data
            sorted_vals = coo_vals[coo_sort_perm]
            csr_data = jax.ops.segment_sum(sorted_vals, csr_segment_ids, num_segments=nse)
        else:
            J_bcoo_dedup = J_bcoo.sum_duplicates(nse=nse)
            J_bcsr = BCSR.from_bcoo(J_bcoo_dedup)
            csr_data = J_bcsr.data

        f_solve = f
        if noi_row_mask is not None:
            csr_data = csr_data.at[noi_row_mask].set(0.0)
            csr_data = csr_data.at[noi_col_mask].set(0.0)
            csr_data = csr_data.at[noi_diag_indices].set(1.0)
            f_solve = f.at[noi_res_indices_arr].set(0.0)

        # Use FFI-based UMFPACK solve (no pure_callback overhead)
        delta = umfpack_jax.solve(bcsr_indptr, bcsr_indices, csr_data, -f_solve)

        # VACASK-style delta convergence (before step limiting)
        conv_delta = jnp.concatenate([
            delta[:n_unknowns] * nr_damping,
            delta[n_unknowns:],
        ])
        X_ref = jnp.concatenate([X[1:n_total], X[n_total:]])
        tol = jnp.maximum(jnp.abs(X_ref) * reltol, delta_abs_tol)
        if residual_mask is not None:
            conv_delta = jnp.where(residual_mask, conv_delta, 0.0)
        delta_converged = jnp.all(jnp.abs(conv_delta) < tol)

        # Step limiting
        max_delta = jnp.max(jnp.abs(delta))
        scale = jnp.where(max_delta > max_step, max_step / max_delta, 1.0)
        delta = delta * scale

        # Update X: VACASK system-level damping: simple scalar multiply
        V_damped = X[1:n_total] + delta[:n_unknowns] * nr_damping
        X_new = X.at[1:n_total].set(V_damped)
        X_new = X_new.at[n_total:].add(delta[n_unknowns:])

        if noi_indices is not None and len(noi_indices) > 0:
            X_new = X_new.at[noi_indices].set(0.0)

        converged = jnp.logical_or(residual_converged, delta_converged)

        # Return updated state with same solver params (unchanged)
        return (
            X_new,
            iteration + 1,
            converged,
            max_f,
            max_delta,
            Q,
            limit_state_out,
            vsource_vals,
            isource_vals,
            Q_prev,
            integ_c0,
            device_arrays_arg,
            gmin,
            gshunt,
            integ_c1,
            integ_d1,
            _dQdt_prev,
            integ_c2,
            _Q_prev2,
        )

    def nr_solve(
        X_init: Array,
        vsource_vals: Array,
        isource_vals: Array,
        Q_prev: Array,
        integ_c0: float | Array,
        device_arrays_arg: Dict[str, Array],
        gmin: float | Array = 1e-12,
        gshunt: float | Array = 0.0,
        integ_c1: float | Array = 0.0,
        integ_d1: float | Array = 0.0,
        dQdt_prev: Array | None = None,
        integ_c2: float | Array = 0.0,
        Q_prev2: Array | None = None,
        limit_state_in: Array | None = None,
    ):
        _dQdt_prev = (
            dQdt_prev if dQdt_prev is not None else jnp.zeros(n_unknowns, dtype=jnp.float64)
        )
        _Q_prev2 = Q_prev2 if Q_prev2 is not None else jnp.zeros(n_unknowns, dtype=jnp.float64)
        # Ensure limit_state_in is a proper array for JIT tracing
        _limit_state = (
            limit_state_in
            if limit_state_in is not None
            else jnp.zeros(total_limit_states, dtype=jnp.float64)
        )

        # Convert scalar parameters to JAX arrays to avoid weak_type retracing
        _integ_c0 = jnp.asarray(integ_c0, dtype=jnp.float64)
        _gmin = jnp.asarray(gmin, dtype=jnp.float64)
        _gshunt = jnp.asarray(gshunt, dtype=jnp.float64)
        _integ_c1 = jnp.asarray(integ_c1, dtype=jnp.float64)
        _integ_d1 = jnp.asarray(integ_d1, dtype=jnp.float64)
        _integ_c2 = jnp.asarray(integ_c2, dtype=jnp.float64)

        init_Q = jnp.zeros(n_unknowns, dtype=jnp.float64)
        # State includes both iteration state and solver parameters
        init_state = (
            X_init,
            jnp.array(0, dtype=jnp.int32),
            jnp.array(False),
            jnp.array(jnp.inf),
            jnp.array(jnp.inf),
            init_Q,
            _limit_state,
            vsource_vals,
            isource_vals,
            Q_prev,
            _integ_c0,
            device_arrays_arg,
            _gmin,
            _gshunt,
            _integ_c1,
            _integ_d1,
            _dQdt_prev,
            _integ_c2,
            _Q_prev2,
        )

        result_state = lax.while_loop(cond_fn, body_fn, init_state)
        (
            X_final,
            iterations,
            converged,
            max_f,
            _,
            _,
            limit_state_final,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
        ) = result_state

        # Recompute Q and I_vsource from converged solution
        _, _, Q_final, I_vsource, _ = build_system_jit(
            X_final,
            vsource_vals,
            isource_vals,
            Q_prev,
            _integ_c0,
            device_arrays_arg,
            _gmin,
            _gshunt,
            _integ_c1,
            _integ_d1,
            _dQdt_prev,
            _integ_c2,
            _Q_prev2,
            limit_state_final,
            iterations,  # Use final iteration count (iniLim=0)
        )

        dQdt_final = (
            _integ_c0 * Q_final + _integ_c1 * Q_prev + _integ_d1 * _dQdt_prev + _integ_c2 * _Q_prev2
        )

        return (
            X_final,
            iterations,
            converged,
            max_f,
            Q_final,
            dQdt_final,
            I_vsource,
            limit_state_final,
        )

    logger.info(f"Creating UMFPACK FFI full MNA solver: V({n_nodes}) + I({n_vsources})")
    return jax.jit(nr_solve)
