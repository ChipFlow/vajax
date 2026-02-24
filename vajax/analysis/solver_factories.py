"""Newton-Raphson solver factory functions for circuit simulation.

This module provides factory functions that create JIT-compiled NR solvers
using full MNA (Modified Nodal Analysis) with branch currents as explicit
unknowns.

Available solvers:
- Dense full MNA solver (JAX scipy.linalg.solve) - for small/medium circuits
- Spineax full MNA solver (cuDSS on GPU) - for GPU acceleration
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

from vajax._logging import logger

if TYPE_CHECKING:
    from vajax.analysis.options import SimulationOptions


def _compute_noi_masks(
    noi_indices: Optional[Array],
    n_nodes: int,
    bcsr_indptr: Optional[Array] = None,
    bcsr_indices: Optional[Array] = None,
    internal_device_indices: Optional[Array] = None,
) -> Dict:
    """Pre-compute masks for NOI node constraint enforcement and internal device nodes.

    NOI (noise correlation) nodes have extremely high conductance (1e40) which
    causes numerical instability. We enforce delta[noi] = 0 by modifying the
    linear system before solving.

    Internal device nodes are skipped in the residual convergence check
    (VACASK coreopnr.cpp:903) because they have only one device contributing.
    Near convergence, maxResidualContribution approaches zero, making the
    tolerance fall to abstol which prevents convergence.

    Args:
        noi_indices: Array of NOI node indices (in full V vector)
        n_nodes: Total node count including ground
        bcsr_indptr: CSR row pointers (for sparse solvers)
        bcsr_indices: CSR column indices (for sparse solvers)
        internal_device_indices: Array of ALL internal device node indices
            (in full V vector). Used to build residual_conv_mask.

    Returns:
        Dict with pre-computed masks:
        - noi_res_idx: NOI residual indices (noi_indices - 1)
        - residual_mask: Boolean mask for NOI convergence (delta check)
        - residual_conv_mask: Boolean mask excluding ALL internal device nodes
          (residual convergence check only, VACASK-style)
        - noi_row_mask: CSR indices for NOI rows (sparse only)
        - noi_col_mask: CSR indices for NOI columns (sparse only)
        - noi_diag_indices: CSR indices for NOI diagonals (sparse only)
        - noi_res_indices_arr: Sorted NOI residual indices (sparse only)
    """
    n_unknowns = n_nodes - 1

    result = {
        "noi_res_idx": None,
        "residual_mask": None,
        "residual_conv_mask": None,
        "noi_row_mask": None,
        "noi_col_mask": None,
        "noi_diag_indices": None,
        "noi_res_indices_arr": None,
    }

    # Build residual convergence mask that skips ALL internal device nodes
    # (VACASK coreopnr.cpp:903: skip InternalDeviceNode in checkResidual)
    if internal_device_indices is not None and len(internal_device_indices) > 0:
        residual_conv_mask = jnp.ones(n_unknowns, dtype=jnp.bool_)
        internal_res_idx = jnp.array(internal_device_indices) - 1
        residual_conv_mask = residual_conv_mask.at[internal_res_idx].set(False)
        result["residual_conv_mask"] = residual_conv_mask

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
    internal_device_indices: Optional[Array] = None,
    max_iterations: int = 100,
    abstol: float = 1e-12,
    total_limit_states: int = 0,
    options: Optional["SimulationOptions"] = None,
    max_step: float = 1e30,
) -> Callable:
    """Create a JIT-compiled dense NR solver for full MNA formulation.

    This solver handles the augmented system where voltage source currents
    are explicit unknowns:

        X = [V_ground, V_1, ..., V_n, I_vs1, ..., I_vsm]

    The Jacobian has size (n_unknowns + n_vsources) × (n_unknowns + n_vsources).

    Voltage-only step limiting caps max voltage change per NR iteration to
    max_step volts. Branch currents receive full Newton steps.

    Args:
        build_system_jit: JIT-wrapped full MNA function
            (X, vsource_vals, isource_vals, Q_prev, integ_c0, device_arrays, ...)
            -> (J_augmented, f_augmented, Q, I_vsource)
        n_nodes: Total node count including ground
        n_vsources: Number of voltage sources (branch currents)
        noi_indices: Optional array of NOI node indices to constrain to 0V
        max_iterations: Maximum NR iterations
        abstol: Absolute tolerance for convergence
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

    # Per-unknown absolute tolerance for VACASK-style delta convergence check.
    # Voltage unknowns use vntol, branch current unknowns use abstol.
    delta_abs_tol = jnp.concatenate(
        [
            jnp.full(n_unknowns, vntol, dtype=jnp.float64),
            jnp.full(n_vsources, abstol, dtype=jnp.float64),
        ]
    )

    # Compute NOI masks for node equations only
    masks = _compute_noi_masks(
        noi_indices, n_nodes, internal_device_indices=internal_device_indices
    )
    noi_res_idx = masks["noi_res_idx"]

    # Create augmented residual mask for delta check (NOI-only)
    if masks["residual_mask"] is not None:
        residual_mask = jnp.concatenate(
            [masks["residual_mask"], jnp.ones(n_vsources, dtype=jnp.bool_)]
        )
    else:
        residual_mask = None

    # Broader mask for residual convergence: skip ALL internal device nodes
    # (VACASK coreopnr.cpp:903: InternalDeviceNode skipped in checkResidual)
    if masks["residual_conv_mask"] is not None:
        residual_conv_mask = jnp.concatenate(
            [masks["residual_conv_mask"], jnp.ones(n_vsources, dtype=jnp.bool_)]
        )
    elif residual_mask is not None:
        residual_conv_mask = residual_mask
    else:
        residual_conv_mask = None

    # Define cond_fn and body_fn at factory level to enable JAX tracing cache.
    # These are created once per solver instance, not per solve call.
    # The varying parameters are passed through the state tuple.
    def cond_fn(state):
        # State: (X, iteration, converged, max_f, max_delta, Q, limit_state, <solver_params...>, res_tol_floor)
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
            res_tol_floor,
        ) = state

        J, f, Q, _, limit_state_out, max_res_contrib = build_system_jit(
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

        # VACASK-style residual tolerance (coreopnr.cpp:929):
        #   tol[i] = max(|maxResContrib[i]| * reltol, residual_abstol[i])
        # With historic floor (coreopnr.cpp:921, relref=alllocal):
        #   tolref = max(maxResContrib[i], historicMaxResContrib[i])
        # This prevents tolerance collapse at zero crossings where device
        # currents are momentarily small.
        res_tol_nodes = jnp.maximum(max_res_contrib * reltol, res_tol_floor)
        res_tol = jnp.concatenate(
            [res_tol_nodes, jnp.full(n_vsources, vntol, dtype=res_tol_nodes.dtype)]
        )
        if residual_conv_mask is not None:
            f_check = jnp.where(residual_conv_mask, f, 0.0)
        else:
            f_check = f
        max_f = jnp.max(jnp.abs(f_check))
        residual_converged = jnp.all(jnp.abs(f_check) < res_tol)

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
        conv_delta = jnp.concatenate(
            [
                delta[:n_unknowns] * nr_damping,
                delta[n_unknowns:],
            ]
        )
        X_ref = jnp.concatenate([X[1:n_total], X[n_total:]])
        tol = jnp.maximum(jnp.abs(X_ref) * reltol, delta_abs_tol)
        if residual_mask is not None:
            conv_delta = jnp.where(residual_mask, conv_delta, 0.0)
        # VACASK skips delta check at iteration 0 (coreopnr.cpp: if(iteration>1))
        # First NR delta is always large since predictor guess is far from solution
        delta_converged = (iteration == 0) | jnp.all(jnp.abs(conv_delta) < tol)

        # Track max delta for diagnostics
        max_delta = jnp.max(jnp.abs(delta))

        # Voltage-only step limiting: cap max voltage change per iteration.
        V_delta = delta[:n_unknowns]
        max_V_delta = jnp.max(jnp.abs(V_delta))
        V_scale = jnp.minimum(1.0, max_step / jnp.maximum(max_V_delta, 1e-30))
        V_damped = V_delta * V_scale * nr_damping
        X_new = X.at[1:n_total].add(V_damped)
        X_new = X_new.at[n_total:].add(delta[n_unknowns:])

        # Clamp NOI nodes to 0V
        if noi_indices is not None and len(noi_indices) > 0:
            X_new = X_new.at[noi_indices].set(0.0)

        # VACASK-style AND convergence (nrsolver.h:226).
        # Both solution delta and KCL residual must be below tolerance.
        converged = jnp.logical_and(residual_converged, delta_converged)

        # VACASK preventedConvergence (nrsolver.cpp:326, coreopnr.cpp:778):
        # When device limiting (pnjlim/fetlim) is active, block convergence.
        # Detected by checking if limit_state has settled between iterations.
        # When pnjlim modifies a voltage, the stored limited voltage changes;
        # until it stabilizes (device sees consistent voltages), the solution
        # is not self-consistent and convergence must be prevented.
        if total_limit_states > 0:
            limit_delta = jnp.max(jnp.abs(limit_state_out - limit_state))
            limit_ref = jnp.maximum(jnp.max(jnp.abs(limit_state)) * reltol, vntol)
            limit_settled = limit_delta < limit_ref
            # Also require at least iteration 1 (iteration 0 always has
            # unsettled limit_state since it starts from initial/previous step)
            converged = converged & limit_settled & (iteration >= 1)

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
            res_tol_floor,
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
        res_tol_floor: Array | None = None,
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
        # Ensure res_tol_floor is a proper array
        # Default: abstol everywhere (no historic floor)
        _res_tol_floor = (
            res_tol_floor
            if res_tol_floor is not None
            else jnp.full(n_unknowns, abstol, dtype=jnp.float64)
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
            _res_tol_floor,
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
            _,
        ) = result_state

        # Recompute Q and I_vsource from converged solution
        _, _, Q_final, I_vsource, _, max_res_contrib_final = build_system_jit(
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
            max_res_contrib_final,
        )

    logger.info(
        f"Creating dense full MNA solver: V({n_nodes}) + I({n_vsources}), NOI: {noi_indices is not None}"
    )
    return nr_solve


def make_spineax_full_mna_solver(
    build_system_jit: Callable,
    n_nodes: int,
    n_vsources: int,
    nse: int,
    bcsr_indptr: Array,
    bcsr_indices: Array,
    noi_indices: Optional[Array] = None,
    internal_device_indices: Optional[Array] = None,
    max_iterations: int = 100,
    abstol: float = 1e-12,
    coo_sort_perm: Optional[Array] = None,
    csr_segment_ids: Optional[Array] = None,
    total_limit_states: int = 0,
    options: Optional["SimulationOptions"] = None,
    max_step: float = 1e30,
) -> Callable:
    """Create a JIT-compiled sparse NR solver using Spineax/cuDSS for full MNA.

    Uses Spineax's cuDSS wrapper with cached symbolic factorization.
    Supports full MNA formulation with branch currents as explicit unknowns.

    Voltage-only step limiting caps max voltage change per NR iteration to
    max_step volts. Branch currents receive full Newton steps.

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

    # Per-unknown absolute tolerance for VACASK-style delta convergence check.
    # Voltage unknowns use vntol, branch current unknowns use abstol.
    delta_abs_tol = jnp.concatenate(
        [
            jnp.full(n_unknowns, vntol, dtype=jnp.float64),
            jnp.full(n_vsources, abstol, dtype=jnp.float64),
        ]
    )
    use_precomputed = coo_sort_perm is not None and csr_segment_ids is not None

    masks = _compute_noi_masks(
        noi_indices,
        n_nodes,
        bcsr_indptr,
        bcsr_indices,
        internal_device_indices=internal_device_indices,
    )
    noi_row_mask = masks["noi_row_mask"]
    noi_col_mask = masks["noi_col_mask"]
    noi_diag_indices = masks["noi_diag_indices"]
    noi_res_indices_arr = masks["noi_res_indices_arr"]

    # Create augmented residual mask for delta check (NOI-only)
    if masks["residual_mask"] is not None:
        residual_mask = jnp.concatenate(
            [masks["residual_mask"], jnp.ones(n_vsources, dtype=jnp.bool_)]
        )
    else:
        residual_mask = None

    # Broader mask for residual convergence: skip ALL internal device nodes
    # (VACASK coreopnr.cpp:903: InternalDeviceNode skipped in checkResidual)
    if masks["residual_conv_mask"] is not None:
        residual_conv_mask = jnp.concatenate(
            [masks["residual_conv_mask"], jnp.ones(n_vsources, dtype=jnp.bool_)]
        )
    elif residual_mask is not None:
        residual_conv_mask = residual_mask
    else:
        residual_conv_mask = None

    # Create Spineax solver with cached symbolic factorization
    # mtype_id=0 (general) because MNA Jacobians are non-symmetric
    # (transconductance terms create asymmetric off-diagonals).
    # mview_id=0 (full) to pass the complete matrix to cuDSS.
    spineax_solver = CuDSSSolver(
        bcsr_indptr,
        bcsr_indices,
        device_id=0,
        mtype_id=0,
        mview_id=0,
    )
    logger.info("Created Spineax full MNA solver with cached symbolic factorization")

    # Define cond_fn and body_fn at factory level to enable JAX tracing cache.
    def cond_fn(state):
        # State: (X, iteration, converged, max_f, max_delta, Q, limit_state, <solver_params...>, res_tol_floor)
        _, iteration, converged, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = state
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
            res_tol_floor,
        ) = state

        J_bcoo, f, Q, _, limit_state_out, max_res_contrib = build_system_jit(
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

        # VACASK-style residual tolerance (coreopnr.cpp:929):
        #   tol[i] = max(|maxResContrib[i]| * reltol, residual_abstol[i])
        # With historic floor (coreopnr.cpp:921, relref=alllocal):
        #   tolref = max(maxResContrib[i], historicMaxResContrib[i])
        # This prevents tolerance collapse at zero crossings where device
        # currents are momentarily small.
        res_tol_nodes = jnp.maximum(max_res_contrib * reltol, res_tol_floor)
        res_tol = jnp.concatenate(
            [res_tol_nodes, jnp.full(n_vsources, vntol, dtype=res_tol_nodes.dtype)]
        )
        if residual_conv_mask is not None:
            f_check = jnp.where(residual_conv_mask, f, 0.0)
        else:
            f_check = f
        max_f = jnp.max(jnp.abs(f_check))
        residual_converged = jnp.all(jnp.abs(f_check) < res_tol)

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
        conv_delta = jnp.concatenate(
            [
                delta[:n_unknowns] * nr_damping,
                delta[n_unknowns:],
            ]
        )
        X_ref = jnp.concatenate([X[1:n_total], X[n_total:]])
        tol = jnp.maximum(jnp.abs(X_ref) * reltol, delta_abs_tol)
        if residual_mask is not None:
            conv_delta = jnp.where(residual_mask, conv_delta, 0.0)
        # VACASK skips delta check at iteration 0 (coreopnr.cpp: if(iteration>1))
        delta_converged = (iteration == 0) | jnp.all(jnp.abs(conv_delta) < tol)

        # Track max delta for diagnostics
        max_delta = jnp.max(jnp.abs(delta))

        # Voltage-only step limiting
        V_delta = delta[:n_unknowns]
        max_V_delta = jnp.max(jnp.abs(V_delta))
        V_scale = jnp.minimum(1.0, max_step / jnp.maximum(max_V_delta, 1e-30))
        V_damped = V_delta * V_scale * nr_damping
        X_new = X.at[1:n_total].add(V_damped)
        X_new = X_new.at[n_total:].add(delta[n_unknowns:])

        if noi_indices is not None and len(noi_indices) > 0:
            X_new = X_new.at[noi_indices].set(0.0)

        # VACASK-style AND convergence (nrsolver.h:226).
        converged = jnp.logical_and(residual_converged, delta_converged)

        # VACASK preventedConvergence (nrsolver.cpp:326, coreopnr.cpp:778):
        # When device limiting (pnjlim/fetlim) is active, block convergence.
        # Detected by checking if limit_state has settled between iterations.
        if total_limit_states > 0:
            limit_delta = jnp.max(jnp.abs(limit_state_out - limit_state))
            limit_ref = jnp.maximum(jnp.max(jnp.abs(limit_state)) * reltol, vntol)
            limit_settled = limit_delta < limit_ref
            converged = converged & limit_settled & (iteration >= 1)

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
            res_tol_floor,
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
        res_tol_floor: Array | None = None,
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
        # Ensure res_tol_floor is a proper array
        # Default: abstol everywhere (no historic floor)
        _res_tol_floor = (
            res_tol_floor
            if res_tol_floor is not None
            else jnp.full(n_unknowns, abstol, dtype=jnp.float64)
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
            _res_tol_floor,
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
            _,
        ) = result_state

        # Recompute Q and I_vsource from converged solution
        _, _, Q_final, I_vsource, _, max_res_contrib_final = build_system_jit(
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
            max_res_contrib_final,
        )

    logger.info(f"Creating Spineax full MNA NR solver: V({n_nodes}) + I({n_vsources})")
    return nr_solve


def is_umfpack_ffi_available() -> bool:
    """Check if the UMFPACK FFI extension is available.

    The FFI version eliminates the ~100ms pure_callback overhead per solve,
    reducing solve time from ~117ms to ~17ms for large circuits.

    Returns:
        True if the FFI extension is installed and working.
    """
    try:
        from vajax.sparse import umfpack_jax

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
    internal_device_indices: Optional[Array] = None,
    max_iterations: int = 100,
    abstol: float = 1e-12,
    coo_sort_perm: Optional[Array] = None,
    csr_segment_ids: Optional[Array] = None,
    total_limit_states: int = 0,
    options: Optional["SimulationOptions"] = None,
    max_step: float = 1e30,
) -> Callable:
    """Create a JIT-compiled UMFPACK FFI NR solver for full MNA formulation.

    This is the FFI-based version that eliminates the ~100ms pure_callback overhead.
    Uses UMFPACK directly via XLA FFI for fast CPU solving.

    Voltage-only step limiting caps max voltage change per NR iteration to
    max_step volts. Branch currents receive full Newton steps.

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

    from vajax.sparse import umfpack_jax

    if not umfpack_jax.is_available():
        raise RuntimeError(
            "UMFPACK FFI extension not available. "
            "Install with: cd vajax/sparse && pip install ."
        )

    n_unknowns = n_nodes - 1
    n_total = n_nodes

    # Per-unknown absolute tolerance for VACASK-style delta convergence check.
    # Voltage unknowns use vntol, branch current unknowns use abstol.
    delta_abs_tol = jnp.concatenate(
        [
            jnp.full(n_unknowns, vntol, dtype=jnp.float64),
            jnp.full(n_vsources, abstol, dtype=jnp.float64),
        ]
    )

    use_precomputed = coo_sort_perm is not None and csr_segment_ids is not None

    masks = _compute_noi_masks(
        noi_indices,
        n_nodes,
        bcsr_indptr,
        bcsr_indices,
        internal_device_indices=internal_device_indices,
    )
    noi_row_mask = masks["noi_row_mask"]
    noi_col_mask = masks["noi_col_mask"]
    noi_diag_indices = masks["noi_diag_indices"]
    noi_res_indices_arr = masks["noi_res_indices_arr"]

    # Create augmented residual mask for delta check (NOI-only)
    if masks["residual_mask"] is not None:
        residual_mask = jnp.concatenate(
            [masks["residual_mask"], jnp.ones(n_vsources, dtype=jnp.bool_)]
        )
    else:
        residual_mask = None

    # Broader mask for residual convergence: skip ALL internal device nodes
    # (VACASK coreopnr.cpp:903: InternalDeviceNode skipped in checkResidual)
    if masks["residual_conv_mask"] is not None:
        residual_conv_mask = jnp.concatenate(
            [masks["residual_conv_mask"], jnp.ones(n_vsources, dtype=jnp.bool_)]
        )
    elif residual_mask is not None:
        residual_conv_mask = residual_mask
    else:
        residual_conv_mask = None

    logger.info("Creating UMFPACK FFI full MNA solver with zero callback overhead")

    # Define body_fn and cond_fn at factory level to enable JAX tracing cache.
    # These are created once per solver instance, not per solve call.
    # The varying parameters are passed through the state tuple.
    def cond_fn(state):
        # State: (X, iteration, converged, max_f, max_delta, Q, limit_state, <solver_params...>, res_tol_floor)
        _, iteration, converged, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = state
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
            res_tol_floor,
        ) = state

        J_bcoo, f, Q, _, limit_state_out, max_res_contrib = build_system_jit(
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

        # VACASK-style residual tolerance (coreopnr.cpp:929):
        #   tol[i] = max(|maxResContrib[i]| * reltol, residual_abstol[i])
        # With historic floor (coreopnr.cpp:921, relref=alllocal):
        #   tolref = max(maxResContrib[i], historicMaxResContrib[i])
        # This prevents tolerance collapse at zero crossings where device
        # currents are momentarily small.
        res_tol_nodes = jnp.maximum(max_res_contrib * reltol, res_tol_floor)
        res_tol = jnp.concatenate(
            [res_tol_nodes, jnp.full(n_vsources, vntol, dtype=res_tol_nodes.dtype)]
        )
        if residual_conv_mask is not None:
            f_check = jnp.where(residual_conv_mask, f, 0.0)
        else:
            f_check = f
        max_f = jnp.max(jnp.abs(f_check))
        residual_converged = jnp.all(jnp.abs(f_check) < res_tol)

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
        conv_delta = jnp.concatenate(
            [
                delta[:n_unknowns] * nr_damping,
                delta[n_unknowns:],
            ]
        )
        X_ref = jnp.concatenate([X[1:n_total], X[n_total:]])
        tol = jnp.maximum(jnp.abs(X_ref) * reltol, delta_abs_tol)
        if residual_mask is not None:
            conv_delta = jnp.where(residual_mask, conv_delta, 0.0)
        # VACASK skips delta check at iteration 0 (coreopnr.cpp: if(iteration>1))
        delta_converged = (iteration == 0) | jnp.all(jnp.abs(conv_delta) < tol)

        # Track max delta for diagnostics
        max_delta = jnp.max(jnp.abs(delta))

        # Voltage-only step limiting
        V_delta = delta[:n_unknowns]
        max_V_delta = jnp.max(jnp.abs(V_delta))
        V_scale = jnp.minimum(1.0, max_step / jnp.maximum(max_V_delta, 1e-30))
        V_damped = V_delta * V_scale * nr_damping
        X_new = X.at[1:n_total].add(V_damped)
        X_new = X_new.at[n_total:].add(delta[n_unknowns:])

        if noi_indices is not None and len(noi_indices) > 0:
            X_new = X_new.at[noi_indices].set(0.0)

        # VACASK-style AND convergence (nrsolver.h:226).
        converged = jnp.logical_and(residual_converged, delta_converged)

        # VACASK preventedConvergence (nrsolver.cpp:326, coreopnr.cpp:778):
        # When device limiting (pnjlim/fetlim) is active, block convergence.
        # Detected by checking if limit_state has settled between iterations.
        if total_limit_states > 0:
            limit_delta = jnp.max(jnp.abs(limit_state_out - limit_state))
            limit_ref = jnp.maximum(jnp.max(jnp.abs(limit_state)) * reltol, vntol)
            limit_settled = limit_delta < limit_ref
            converged = converged & limit_settled & (iteration >= 1)

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
            res_tol_floor,
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
        res_tol_floor: Array | None = None,
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
        # Ensure res_tol_floor is a proper array
        # Default: abstol everywhere (no historic floor)
        _res_tol_floor = (
            res_tol_floor
            if res_tol_floor is not None
            else jnp.full(n_unknowns, abstol, dtype=jnp.float64)
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
            _res_tol_floor,
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
            _,
        ) = result_state

        # Recompute Q and I_vsource from converged solution
        _, _, Q_final, I_vsource, _, max_res_contrib_final = build_system_jit(
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
            max_res_contrib_final,
        )

    logger.info(f"Creating UMFPACK FFI full MNA solver: V({n_nodes}) + I({n_vsources})")
    return nr_solve
