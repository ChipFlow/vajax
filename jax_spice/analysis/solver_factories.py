"""Newton-Raphson solver factory functions for circuit simulation.

This module provides factory functions that create JIT-compiled NR solvers
for different backends:
- Dense solver (JAX scipy.linalg.solve)
- Sparse solver (JAX spsolve)
- Spineax solver (cuDSS on GPU)
- UMFPACK solver (CPU sparse)

All factories return a JIT-compiled function with signature:
    (V, vsource_vals, isource_vals, Q_prev, integ_c0, device_arrays, gmin, gshunt,
     integ_c1, integ_d1, dQdt_prev)
    -> (V_final, iterations, converged, max_residual, Q_final, dQdt_final, I_vsource)

The I_vsource return value is the vsource current computed from KCL (device residuals)
instead of the high-G formula I = G * (Vp - Vn - Vtarget). This provides smoother
current derivatives (dI/dt) that match reference simulators like VACASK.

Integration coefficient meanings:
    - integ_c0: Coefficient for Q_new (leading coefficient). 0 for DC.
    - integ_c1: Coefficient for Q_prev. Default 0.
    - integ_d1: Coefficient for dQdt_prev. Default 0 (only trap uses -1).
    - dQdt_prev: Previous dQ/dt vector for trapezoidal method.
"""

from typing import Callable, Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, lax

from jax_spice._logging import logger
from jax_spice.analysis.limiting import apply_voltage_damping, DEFAULT_NR_DAMPING

# Newton-Raphson solver constants
MAX_NR_ITERATIONS = 100
DEFAULT_ABSTOL = 1e4  # Corresponds to ~10nV voltage accuracy with G=1e12

# Voltage damping constants (pnjlim-style)
# These values are based on SPICE3 pnjlim defaults
DAMPING_VT = 0.026  # Thermal voltage at 300K
DAMPING_VCRIT = 0.6  # Critical voltage for PN junctions

# Global NR damping factor (1.0 = no damping, <1.0 = reduced step size)
# This can be overridden by the engine via set_nr_damping()
_NR_DAMPING = DEFAULT_NR_DAMPING


def set_nr_damping(value: float) -> None:
    """Set the global NR damping factor.

    Args:
        value: Damping factor (1.0 = no damping, 0.5 = half steps, etc.)
               Must be > 0 and <= 1.0
    """
    global _NR_DAMPING
    if value <= 0 or value > 1.0:
        raise ValueError(f"nr_damping must be in (0, 1], got {value}")
    _NR_DAMPING = value


def get_nr_damping() -> float:
    """Get the current global NR damping factor."""
    return _NR_DAMPING


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
        'noi_res_idx': None,
        'residual_mask': None,
        'noi_row_mask': None,
        'noi_col_mask': None,
        'noi_diag_indices': None,
        'noi_res_indices_arr': None,
    }

    if noi_indices is None or len(noi_indices) == 0:
        return result

    n_unknowns = n_nodes - 1
    noi_res_idx = noi_indices - 1  # Convert to residual indices

    # Residual mask for convergence check
    residual_mask = jnp.ones(n_unknowns, dtype=jnp.bool_)
    residual_mask = residual_mask.at[noi_res_idx].set(False)

    result['noi_res_idx'] = noi_res_idx
    result['residual_mask'] = residual_mask

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

        result['noi_row_mask'] = jnp.array(row_mask_list, dtype=jnp.int32)
        result['noi_col_mask'] = jnp.array(col_mask_list, dtype=jnp.int32)
        result['noi_diag_indices'] = jnp.array(diag_list, dtype=jnp.int32)
        result['noi_res_indices_arr'] = jnp.array(sorted(noi_set), dtype=jnp.int32)

    return result


def make_dense_full_mna_solver(
    build_system_jit: Callable,
    n_nodes: int,
    n_vsources: int,
    noi_indices: Optional[Array] = None,
    max_iterations: int = MAX_NR_ITERATIONS,
    abstol: float = DEFAULT_ABSTOL,
    nr_convtol: float = 1.0,
    max_step: float = 1.0,
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
        nr_convtol: NR convergence tolerance factor (multiplier on abstol). Default 1.0.
        max_step: Maximum voltage/current step per iteration

    Returns:
        JIT-compiled solver function with signature:
            (X, vsource_vals, isource_vals, Q_prev, integ_c0, device_arrays, ...)
            -> (X, iterations, converged, max_f, Q, dQdt, I_vsource)
    """
    # Apply tolerance factor
    effective_abstol = abstol * nr_convtol
    n_unknowns = n_nodes - 1
    n_augmented = n_unknowns + n_vsources
    n_total = n_nodes  # Size of voltage part of X

    # Compute NOI masks for node equations only
    masks = _compute_noi_masks(noi_indices, n_nodes)
    noi_res_idx = masks['noi_res_idx']

    # Create augmented residual mask (node equations + branch equations)
    if masks['residual_mask'] is not None:
        # NOI nodes should be masked in residual convergence check
        # Branch equations (vsource voltages) are always checked
        residual_mask = jnp.concatenate([
            masks['residual_mask'],
            jnp.ones(n_vsources, dtype=jnp.bool_)
        ])
    else:
        residual_mask = None

    def nr_solve(X_init: Array, vsource_vals: Array, isource_vals: Array,
                 Q_prev: Array, integ_c0: float | Array,
                 device_arrays_arg: Dict[str, Array],
                 gmin: float | Array = 1e-12, gshunt: float | Array = 0.0,
                 integ_c1: float | Array = 0.0, integ_d1: float | Array = 0.0,
                 dQdt_prev: Array | None = None,
                 integ_c2: float | Array = 0.0, Q_prev2: Array | None = None):

        # Ensure dQdt_prev is a proper array for JIT tracing
        _dQdt_prev = dQdt_prev if dQdt_prev is not None else jnp.zeros(n_unknowns, dtype=jnp.float64)
        # Ensure Q_prev2 is a proper array for JIT tracing (Gear2 method)
        _Q_prev2 = Q_prev2 if Q_prev2 is not None else jnp.zeros(n_unknowns, dtype=jnp.float64)

        init_Q = jnp.zeros(n_unknowns, dtype=jnp.float64)
        init_state = (
            X_init,
            jnp.array(0, dtype=jnp.int32),
            jnp.array(False),
            jnp.array(jnp.inf),
            jnp.array(jnp.inf),
            init_Q,
        )

        def cond_fn(state):
            X, iteration, converged, max_f, max_delta, Q = state
            return jnp.logical_and(~converged, iteration < max_iterations)

        def body_fn(state):
            X, iteration, _, _, _, _ = state

            J, f, Q, _ = build_system_jit(X, vsource_vals, isource_vals, Q_prev, integ_c0,
                                          device_arrays_arg, gmin, gshunt,
                                          integ_c1, integ_d1, _dQdt_prev, integ_c2, _Q_prev2)

            # Check residual convergence (mask NOI nodes)
            if residual_mask is not None:
                f_masked = jnp.where(residual_mask, f, 0.0)
                max_f = jnp.max(jnp.abs(f_masked))
            else:
                max_f = jnp.max(jnp.abs(f))
            residual_converged = max_f < effective_abstol

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

            # Step limiting
            max_delta = jnp.max(jnp.abs(delta))
            scale = jnp.where(max_delta > max_step, max_step / max_delta, 1.0)
            delta = delta * scale

            # Update X: skip ground (index 0), update node voltages and branch currents
            # X has structure: [V_ground, V_1, ..., V_n, I_vs1, ..., I_vsm]
            # delta has structure: [delta_V_1, ..., delta_V_n, delta_I_vs1, ..., delta_I_vsm]
            V_candidate = X[1:n_total] + delta[:n_unknowns]
            V_damped = apply_voltage_damping(V_candidate, X[1:n_total], DAMPING_VT, DAMPING_VCRIT, nr_damping=_NR_DAMPING)
            X_new = X.at[1:n_total].set(V_damped)  # Update node voltages with damping
            X_new = X_new.at[n_total:].add(delta[n_unknowns:])  # Update branch currents

            # Clamp NOI nodes to 0V
            if noi_indices is not None and len(noi_indices) > 0:
                X_new = X_new.at[noi_indices].set(0.0)

            delta_converged = max_delta < 1e-12
            converged = jnp.logical_or(residual_converged, delta_converged)

            return (X_new, iteration + 1, converged, max_f, max_delta, Q)

        X_final, iterations, converged, max_f, max_delta, _ = lax.while_loop(
            cond_fn, body_fn, init_state
        )

        # Recompute Q and I_vsource from converged solution
        _, _, Q_final, I_vsource = build_system_jit(X_final, vsource_vals, isource_vals, Q_prev,
                                                    integ_c0, device_arrays_arg, gmin, gshunt,
                                                    integ_c1, integ_d1, _dQdt_prev, integ_c2, _Q_prev2)

        # Compute dQdt for next timestep (needed for trapezoidal and Gear2 methods)
        dQdt_final = integ_c0 * Q_final + integ_c1 * Q_prev + integ_d1 * _dQdt_prev + integ_c2 * _Q_prev2

        return X_final, iterations, converged, max_f, Q_final, dQdt_final, I_vsource

    logger.info(f"Creating dense full MNA solver: V({n_nodes}) + I({n_vsources}), NOI: {noi_indices is not None}")
    return jax.jit(nr_solve)


def make_sparse_full_mna_solver(
    build_system_jit: Callable,
    n_nodes: int,
    n_vsources: int,
    nse: int,
    noi_indices: Optional[Array] = None,
    max_iterations: int = MAX_NR_ITERATIONS,
    abstol: float = DEFAULT_ABSTOL,
    nr_convtol: float = 1.0,
    max_step: float = 1.0,
    coo_sort_perm: Optional[Array] = None,
    csr_segment_ids: Optional[Array] = None,
    bcsr_indices: Optional[Array] = None,
    bcsr_indptr: Optional[Array] = None,
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
        nr_convtol: NR convergence tolerance factor (multiplier on abstol). Default 1.0.
        max_step: Maximum voltage/current step per iteration
        coo_sort_perm: Pre-computed COO→CSR permutation
        csr_segment_ids: Pre-computed segment IDs for duplicate summing
        bcsr_indices: Pre-computed CSR column indices
        bcsr_indptr: Pre-computed CSR row pointers

    Returns:
        JIT-compiled solver function with same signature as make_dense_full_mna_solver
    """
    # Apply tolerance factor
    effective_abstol = abstol * nr_convtol
    from jax.experimental.sparse import BCSR
    from jax.experimental.sparse.linalg import spsolve

    n_unknowns = n_nodes - 1
    n_augmented = n_unknowns + n_vsources
    n_total = n_nodes  # Size of voltage part of X

    use_precomputed = (coo_sort_perm is not None and csr_segment_ids is not None
                       and bcsr_indices is not None and bcsr_indptr is not None)

    # Compute NOI masks for node equations only (branch equations are not masked)
    masks = _compute_noi_masks(noi_indices, n_nodes, bcsr_indptr, bcsr_indices)
    noi_row_mask = masks['noi_row_mask']
    noi_col_mask = masks['noi_col_mask']
    noi_diag_indices = masks['noi_diag_indices']
    noi_res_indices_arr = masks['noi_res_indices_arr']

    # Create augmented residual mask (node equations + branch equations)
    if masks['residual_mask'] is not None:
        # NOI nodes should be masked in residual convergence check
        # Branch equations (vsource voltages) are always checked
        residual_mask = jnp.concatenate([
            masks['residual_mask'],
            jnp.ones(n_vsources, dtype=jnp.bool_)
        ])
    else:
        residual_mask = None

    def nr_solve(X_init: Array, vsource_vals: Array, isource_vals: Array,
                 Q_prev: Array, integ_c0: float | Array,
                 device_arrays_arg: Dict[str, Array],
                 gmin: float | Array = 1e-12, gshunt: float | Array = 0.0,
                 integ_c1: float | Array = 0.0, integ_d1: float | Array = 0.0,
                 dQdt_prev: Array | None = None,
                 integ_c2: float | Array = 0.0, Q_prev2: Array | None = None):

        # Ensure dQdt_prev is a proper array for JIT tracing
        _dQdt_prev = dQdt_prev if dQdt_prev is not None else jnp.zeros(n_unknowns, dtype=jnp.float64)
        _Q_prev2 = Q_prev2 if Q_prev2 is not None else jnp.zeros(n_unknowns, dtype=jnp.float64)

        init_Q = jnp.zeros(n_unknowns, dtype=jnp.float64)
        init_state = (
            X_init,
            jnp.array(0, dtype=jnp.int32),
            jnp.array(False),
            jnp.array(jnp.inf),
            jnp.array(jnp.inf),
            init_Q,
        )

        def cond_fn(state):
            X, iteration, converged, max_f, max_delta, Q = state
            return jnp.logical_and(~converged, iteration < max_iterations)

        def body_fn(state):
            X, iteration, _, _, _, _ = state

            J_bcoo, f, Q, _ = build_system_jit(X, vsource_vals, isource_vals, Q_prev,
                                                integ_c0, device_arrays_arg, gmin, gshunt,
                                                integ_c1, integ_d1, _dQdt_prev, integ_c2, _Q_prev2)

            # Check residual convergence
            if residual_mask is not None:
                f_masked = jnp.where(residual_mask, f, 0.0)
                max_f = jnp.max(jnp.abs(f_masked))
            else:
                max_f = jnp.max(jnp.abs(f))
            residual_converged = max_f < effective_abstol

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

            # Step limiting
            max_delta = jnp.max(jnp.abs(delta))
            scale = jnp.where(max_delta > max_step, max_step / max_delta, 1.0)
            delta = delta * scale

            # Update X: skip ground (index 0), update node voltages and branch currents
            # X has structure: [V_ground, V_1, ..., V_n, I_vs1, ..., I_vsm]
            # delta has structure: [delta_V_1, ..., delta_V_n, delta_I_vs1, ..., delta_I_vsm]
            V_candidate = X[1:n_total] + delta[:n_unknowns]
            V_damped = apply_voltage_damping(V_candidate, X[1:n_total], DAMPING_VT, DAMPING_VCRIT, nr_damping=_NR_DAMPING)
            X_new = X.at[1:n_total].set(V_damped)  # Update node voltages with damping
            X_new = X_new.at[n_total:].add(delta[n_unknowns:])  # Update branch currents

            # Clamp NOI nodes to 0V
            if noi_indices is not None and len(noi_indices) > 0:
                X_new = X_new.at[noi_indices].set(0.0)

            delta_converged = max_delta < 1e-12
            converged = jnp.logical_or(residual_converged, delta_converged)

            return (X_new, iteration + 1, converged, max_f, max_delta, Q)

        X_final, iterations, converged, max_f, max_delta, _ = lax.while_loop(
            cond_fn, body_fn, init_state
        )

        # Recompute Q and I_vsource from converged solution
        _, _, Q_final, I_vsource = build_system_jit(X_final, vsource_vals, isource_vals, Q_prev,
                                                    integ_c0, device_arrays_arg, gmin, gshunt,
                                                    integ_c1, integ_d1, _dQdt_prev, integ_c2, _Q_prev2)

        # Compute dQdt for next timestep
        dQdt_final = integ_c0 * Q_final + integ_c1 * Q_prev + integ_d1 * _dQdt_prev + integ_c2 * _Q_prev2

        return X_final, iterations, converged, max_f, Q_final, dQdt_final, I_vsource

    logger.info(f"Creating sparse full MNA solver: V({n_nodes}) + I({n_vsources}), nse={nse}, NOI: {noi_indices is not None}")
    return jax.jit(nr_solve)


def make_umfpack_full_mna_solver(
    build_system_jit: Callable,
    n_nodes: int,
    n_vsources: int,
    nse: int,
    bcsr_indptr: Array,
    bcsr_indices: Array,
    noi_indices: Optional[Array] = None,
    max_iterations: int = MAX_NR_ITERATIONS,
    abstol: float = DEFAULT_ABSTOL,
    max_step: float = 1.0,
    coo_sort_perm: Optional[Array] = None,
    csr_segment_ids: Optional[Array] = None,
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

    Returns:
        JIT-compiled UMFPACK solver function with full MNA augmented system
    """
    from jax.experimental.sparse import BCSR

    from jax_spice.analysis.umfpack_solver import UMFPACKSolver

    n_unknowns = n_nodes - 1
    n_augmented = n_unknowns + n_vsources
    n_total = n_nodes

    use_precomputed = coo_sort_perm is not None and csr_segment_ids is not None

    # Compute NOI masks for node equations only
    masks = _compute_noi_masks(noi_indices, n_nodes, bcsr_indptr, bcsr_indices)
    noi_row_mask = masks['noi_row_mask']
    noi_col_mask = masks['noi_col_mask']
    noi_diag_indices = masks['noi_diag_indices']
    noi_res_indices_arr = masks['noi_res_indices_arr']

    # Augmented residual mask
    if masks['residual_mask'] is not None:
        residual_mask = jnp.concatenate([
            masks['residual_mask'],
            jnp.ones(n_vsources, dtype=jnp.bool_)
        ])
    else:
        residual_mask = None

    # Create UMFPACK solver with cached symbolic factorization
    umfpack_solver = UMFPACKSolver(bcsr_indptr, bcsr_indices)
    logger.info("Created UMFPACK full MNA solver with cached symbolic factorization")

    def nr_solve(X_init: Array, vsource_vals: Array, isource_vals: Array,
                 Q_prev: Array, integ_c0: float | Array,
                 device_arrays_arg: Dict[str, Array],
                 gmin: float | Array = 1e-12, gshunt: float | Array = 0.0,
                 integ_c1: float | Array = 0.0, integ_d1: float | Array = 0.0,
                 dQdt_prev: Array | None = None,
                 integ_c2: float | Array = 0.0, Q_prev2: Array | None = None):

        _dQdt_prev = dQdt_prev if dQdt_prev is not None else jnp.zeros(n_unknowns, dtype=jnp.float64)
        _Q_prev2 = Q_prev2 if Q_prev2 is not None else jnp.zeros(n_unknowns, dtype=jnp.float64)

        init_Q = jnp.zeros(n_unknowns, dtype=jnp.float64)
        init_state = (
            X_init,
            jnp.array(0, dtype=jnp.int32),
            jnp.array(False),
            jnp.array(jnp.inf),
            jnp.array(jnp.inf),
            init_Q,
        )

        def cond_fn(state):
            X, iteration, converged, max_f, max_delta, Q = state
            return jnp.logical_and(~converged, iteration < max_iterations)

        def body_fn(state):
            X, iteration, _, _, _, _ = state

            J_bcoo, f, Q, _ = build_system_jit(X, vsource_vals, isource_vals, Q_prev,
                                                integ_c0, device_arrays_arg, gmin, gshunt,
                                                integ_c1, integ_d1, _dQdt_prev, integ_c2, _Q_prev2)

            # Check residual convergence
            if residual_mask is not None:
                f_masked = jnp.where(residual_mask, f, 0.0)
                max_f = jnp.max(jnp.abs(f_masked))
            else:
                max_f = jnp.max(jnp.abs(f))
            residual_converged = max_f < abstol

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

            # Step limiting
            max_delta = jnp.max(jnp.abs(delta))
            scale = jnp.where(max_delta > max_step, max_step / max_delta, 1.0)
            delta = delta * scale

            # Update X: node voltages and branch currents with damping
            V_candidate = X[1:n_total] + delta[:n_unknowns]
            V_damped = apply_voltage_damping(V_candidate, X[1:n_total], DAMPING_VT, DAMPING_VCRIT, nr_damping=_NR_DAMPING)
            X_new = X.at[1:n_total].set(V_damped)
            X_new = X_new.at[n_total:].add(delta[n_unknowns:])

            if noi_indices is not None and len(noi_indices) > 0:
                X_new = X_new.at[noi_indices].set(0.0)

            delta_converged = max_delta < 1e-12
            converged = jnp.logical_or(residual_converged, delta_converged)

            return (X_new, iteration + 1, converged, max_f, max_delta, Q)

        X_final, iterations, converged, max_f, max_delta, _ = lax.while_loop(
            cond_fn, body_fn, init_state
        )

        _, _, Q_final, I_vsource = build_system_jit(X_final, vsource_vals, isource_vals, Q_prev,
                                                    integ_c0, device_arrays_arg, gmin, gshunt,
                                                    integ_c1, integ_d1, _dQdt_prev, integ_c2, _Q_prev2)

        dQdt_final = integ_c0 * Q_final + integ_c1 * Q_prev + integ_d1 * _dQdt_prev + integ_c2 * _Q_prev2

        return X_final, iterations, converged, max_f, Q_final, dQdt_final, I_vsource

    logger.info(f"Creating UMFPACK full MNA solver: V({n_nodes}) + I({n_vsources})")
    return jax.jit(nr_solve)


def make_dense_solver(
    build_system_jit: Callable,
    n_nodes: int,
    noi_indices: Optional[Array] = None,
    max_iterations: int = MAX_NR_ITERATIONS,
    abstol: float = DEFAULT_ABSTOL,
    max_step: float = 1.0,
) -> Callable:
    """Create a JIT-compiled dense NR solver.

    Uses jax.scipy.linalg.solve for the linear system. Suitable for
    small to medium circuits (<1000 nodes).

    Args:
        build_system_jit: JIT-wrapped function
            (V, vsource_vals, isource_vals, Q_prev, integ_c0, device_arrays, gmin, gshunt,
             integ_c1, integ_d1, dQdt_prev, integ_c2, Q_prev2) -> (J, f, Q, I_vsource)
        n_nodes: Total node count including ground
        noi_indices: Optional array of NOI node indices to constrain to 0V
        max_iterations: Maximum NR iterations
        abstol: Absolute tolerance for convergence
        max_step: Maximum voltage step per iteration

    Returns:
        JIT-compiled solver function with signature:
            (V, vsource_vals, isource_vals, Q_prev, integ_c0, device_arrays, gmin, gshunt,
             integ_c1, integ_d1, dQdt_prev, integ_c2, Q_prev2) -> (V, iterations, converged, max_f, Q, dQdt, I_vsource)
    """
    n_unknowns = n_nodes - 1
    masks = _compute_noi_masks(noi_indices, n_nodes)
    noi_res_idx = masks['noi_res_idx']
    residual_mask = masks['residual_mask']

    def nr_solve(V_init: Array, vsource_vals: Array, isource_vals: Array,
                 Q_prev: Array, integ_c0: float | Array,
                 device_arrays_arg: Dict[str, Array],
                 gmin: float | Array = 1e-12, gshunt: float | Array = 0.0,
                 integ_c1: float | Array = 0.0, integ_d1: float | Array = 0.0,
                 dQdt_prev: Array | None = None,
                 integ_c2: float | Array = 0.0, Q_prev2: Array | None = None):

        # Ensure dQdt_prev is a proper array for JIT tracing
        _dQdt_prev = dQdt_prev if dQdt_prev is not None else jnp.zeros(n_unknowns, dtype=jnp.float64)
        # Ensure Q_prev2 is a proper array for JIT tracing (Gear2 method)
        _Q_prev2 = Q_prev2 if Q_prev2 is not None else jnp.zeros(n_unknowns, dtype=jnp.float64)

        init_Q = jnp.zeros(n_unknowns, dtype=jnp.float64)
        init_state = (
            V_init,
            jnp.array(0, dtype=jnp.int32),
            jnp.array(False),
            jnp.array(jnp.inf),
            jnp.array(jnp.inf),
            init_Q,
        )

        def cond_fn(state):
            V, iteration, converged, max_f, max_delta, Q = state
            return jnp.logical_and(~converged, iteration < max_iterations)

        def body_fn(state):
            V, iteration, _, _, _, _ = state

            J, f, Q, _ = build_system_jit(V, vsource_vals, isource_vals, Q_prev, integ_c0,
                                          device_arrays_arg, gmin, gshunt,
                                          integ_c1, integ_d1, _dQdt_prev, integ_c2, _Q_prev2)

            # Check residual convergence (mask NOI nodes)
            if residual_mask is not None:
                f_masked = jnp.where(residual_mask, f, 0.0)
                max_f = jnp.max(jnp.abs(f_masked))
            else:
                max_f = jnp.max(jnp.abs(f))
            residual_converged = max_f < abstol

            # Enforce NOI constraints
            if noi_res_idx is not None:
                J = J.at[noi_res_idx, :].set(0.0)
                J = J.at[:, noi_res_idx].set(0.0)
                J = J.at[noi_res_idx, noi_res_idx].set(1.0)
                f = f.at[noi_res_idx].set(0.0)

            # Add Tikhonov regularization for numerical stability on GPU
            # JAX's scipy.linalg.solve raises hard errors on singular matrices
            reg = 1e-14 * jnp.eye(J.shape[0], dtype=J.dtype)
            J_reg = J + reg

            # Solve linear system
            delta = jax.scipy.linalg.solve(J_reg, -f)

            # Step limiting
            max_delta = jnp.max(jnp.abs(delta))
            scale = jnp.where(max_delta > max_step, max_step / max_delta, 1.0)
            delta = delta * scale

            # Update V (ground stays fixed) with damping
            V_candidate = V[1:] + delta
            V_damped = apply_voltage_damping(V_candidate, V[1:], DAMPING_VT, DAMPING_VCRIT, nr_damping=_NR_DAMPING)
            V_new = V.at[1:].set(V_damped)

            # Clamp NOI nodes to 0V
            if noi_indices is not None and len(noi_indices) > 0:
                V_new = V_new.at[noi_indices].set(0.0)

            delta_converged = max_delta < 1e-12
            converged = jnp.logical_or(residual_converged, delta_converged)

            return (V_new, iteration + 1, converged, max_f, max_delta, Q)

        V_final, iterations, converged, max_f, max_delta, _ = lax.while_loop(
            cond_fn, body_fn, init_state
        )

        # Recompute Q and I_vsource from converged voltage
        _, _, Q_final, I_vsource = build_system_jit(V_final, vsource_vals, isource_vals, Q_prev,
                                                    integ_c0, device_arrays_arg, gmin, gshunt,
                                                    integ_c1, integ_d1, _dQdt_prev, integ_c2, _Q_prev2)

        # Compute dQdt for next timestep (needed for trapezoidal and Gear2 methods)
        # dQdt = c0 * Q + c1 * Q_prev + d1 * dQdt_prev + c2 * Q_prev2
        dQdt_final = integ_c0 * Q_final + integ_c1 * Q_prev + integ_d1 * _dQdt_prev + integ_c2 * _Q_prev2

        return V_final, iterations, converged, max_f, Q_final, dQdt_final, I_vsource

    logger.info(f"Creating dense NR solver: V({n_nodes}), NOI: {noi_indices is not None}")
    return jax.jit(nr_solve)


def make_sparse_solver(
    build_system_jit: Callable,
    n_nodes: int,
    nse: int,
    noi_indices: Optional[Array] = None,
    max_iterations: int = MAX_NR_ITERATIONS,
    abstol: float = DEFAULT_ABSTOL,
    max_step: float = 1.0,
    coo_sort_perm: Optional[Array] = None,
    csr_segment_ids: Optional[Array] = None,
    bcsr_indices: Optional[Array] = None,
    bcsr_indptr: Optional[Array] = None,
) -> Callable:
    """Create a JIT-compiled sparse NR solver using JAX spsolve.

    Uses JAX's sparse direct solver (QR factorization) for large circuits.

    Args:
        build_system_jit: JIT-wrapped function returning (J_bcoo, f, Q)
        n_nodes: Total node count including ground
        nse: Number of stored elements after summing duplicates
        noi_indices: Optional array of NOI node indices
        max_iterations: Maximum NR iterations
        abstol: Absolute tolerance for convergence
        max_step: Maximum voltage step per iteration
        coo_sort_perm: Pre-computed COO→CSR permutation
        csr_segment_ids: Pre-computed segment IDs for duplicate summing
        bcsr_indices: Pre-computed CSR column indices
        bcsr_indptr: Pre-computed CSR row pointers

    Returns:
        JIT-compiled sparse solver function
    """
    from jax.experimental.sparse import BCSR
    from jax.experimental.sparse.linalg import spsolve

    n_unknowns = n_nodes - 1
    use_precomputed = (coo_sort_perm is not None and csr_segment_ids is not None
                       and bcsr_indices is not None and bcsr_indptr is not None)

    masks = _compute_noi_masks(noi_indices, n_nodes, bcsr_indptr, bcsr_indices)
    residual_mask = masks['residual_mask']
    noi_row_mask = masks['noi_row_mask']
    noi_col_mask = masks['noi_col_mask']
    noi_diag_indices = masks['noi_diag_indices']
    noi_res_indices_arr = masks['noi_res_indices_arr']

    def nr_solve(V_init: Array, vsource_vals: Array, isource_vals: Array,
                 Q_prev: Array, integ_c0: float | Array,
                 device_arrays_arg: Dict[str, Array],
                 gmin: float | Array = 1e-12, gshunt: float | Array = 0.0,
                 integ_c1: float | Array = 0.0, integ_d1: float | Array = 0.0,
                 dQdt_prev: Array | None = None,
                 integ_c2: float | Array = 0.0, Q_prev2: Array | None = None):

        # Ensure dQdt_prev is a proper array for JIT tracing
        _dQdt_prev = dQdt_prev if dQdt_prev is not None else jnp.zeros(n_unknowns, dtype=jnp.float64)
        # Ensure Q_prev2 is a proper array for JIT tracing (Gear2 method)
        _Q_prev2 = Q_prev2 if Q_prev2 is not None else jnp.zeros(n_unknowns, dtype=jnp.float64)

        init_Q = jnp.zeros(n_unknowns, dtype=jnp.float64)
        init_state = (
            V_init,
            jnp.array(0, dtype=jnp.int32),
            jnp.array(False),
            jnp.array(jnp.inf),
            jnp.array(jnp.inf),
            init_Q,
        )

        def cond_fn(state):
            V, iteration, converged, max_f, max_delta, Q = state
            return jnp.logical_and(~converged, iteration < max_iterations)

        def body_fn(state):
            V, iteration, _, _, _, _ = state

            J_bcoo, f, Q, _ = build_system_jit(V, vsource_vals, isource_vals, Q_prev,
                                               integ_c0, device_arrays_arg, gmin, gshunt,
                                               integ_c1, integ_d1, _dQdt_prev, integ_c2, _Q_prev2)

            # Check residual convergence
            if residual_mask is not None:
                f_masked = jnp.where(residual_mask, f, 0.0)
                max_f = jnp.max(jnp.abs(f_masked))
            else:
                max_f = jnp.max(jnp.abs(f))
            residual_converged = max_f < abstol

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

            # Step limiting
            max_delta = jnp.max(jnp.abs(delta))
            scale = jnp.where(max_delta > max_step, max_step / max_delta, 1.0)
            delta = delta * scale

            # Update V with damping
            V_candidate = V[1:] + delta
            V_damped = apply_voltage_damping(V_candidate, V[1:], DAMPING_VT, DAMPING_VCRIT, nr_damping=_NR_DAMPING)
            V_new = V.at[1:].set(V_damped)

            if noi_indices is not None and len(noi_indices) > 0:
                V_new = V_new.at[noi_indices].set(0.0)

            delta_converged = max_delta < 1e-12
            converged = jnp.logical_or(residual_converged, delta_converged)

            return (V_new, iteration + 1, converged, max_f, max_delta, Q)

        V_final, iterations, converged, max_f, max_delta, _ = lax.while_loop(
            cond_fn, body_fn, init_state
        )

        _, _, Q_final, I_vsource = build_system_jit(V_final, vsource_vals, isource_vals, Q_prev,
                                                    integ_c0, device_arrays_arg, gmin, gshunt,
                                                    integ_c1, integ_d1, _dQdt_prev, integ_c2, _Q_prev2)

        # Compute dQdt for next timestep
        dQdt_final = integ_c0 * Q_final + integ_c1 * Q_prev + integ_d1 * _dQdt_prev + integ_c2 * _Q_prev2

        return V_final, iterations, converged, max_f, Q_final, dQdt_final, I_vsource

    logger.info(f"Creating sparse NR solver: V({n_nodes}), precomputed={use_precomputed}")
    return jax.jit(nr_solve)


def make_spineax_solver(
    build_system_jit: Callable,
    n_nodes: int,
    nse: int,
    bcsr_indptr: Array,
    bcsr_indices: Array,
    noi_indices: Optional[Array] = None,
    max_iterations: int = MAX_NR_ITERATIONS,
    abstol: float = DEFAULT_ABSTOL,
    max_step: float = 1.0,
    coo_sort_perm: Optional[Array] = None,
    csr_segment_ids: Optional[Array] = None,
) -> Callable:
    """Create a JIT-compiled sparse NR solver using Spineax/cuDSS.

    Uses Spineax's cuDSS wrapper with cached symbolic factorization.
    The symbolic analysis is done once when the solver is created.

    Args:
        build_system_jit: JIT-wrapped function returning (J_bcoo, f, Q)
        n_nodes: Total node count including ground
        nse: Number of stored elements after summing duplicates
        bcsr_indptr: Pre-computed BCSR row pointers
        bcsr_indices: Pre-computed BCSR column indices
        noi_indices: Optional array of NOI node indices
        max_iterations: Maximum NR iterations
        abstol: Absolute tolerance for convergence
        max_step: Maximum voltage step per iteration
        coo_sort_perm: Pre-computed COO→CSR permutation
        csr_segment_ids: Pre-computed segment IDs

    Returns:
        JIT-compiled Spineax solver function
    """
    from jax.experimental.sparse import BCSR
    from spineax.cudss.solver import CuDSSSolver

    n_unknowns = n_nodes - 1
    use_precomputed = coo_sort_perm is not None and csr_segment_ids is not None

    masks = _compute_noi_masks(noi_indices, n_nodes, bcsr_indptr, bcsr_indices)
    residual_mask = masks['residual_mask']
    noi_row_mask = masks['noi_row_mask']
    noi_col_mask = masks['noi_col_mask']
    noi_diag_indices = masks['noi_diag_indices']
    noi_res_indices_arr = masks['noi_res_indices_arr']

    # Create Spineax solver with cached symbolic factorization
    spineax_solver = CuDSSSolver(
        bcsr_indptr,
        bcsr_indices,
        device_id=0,
        mtype_id=1,
        mview_id=0,
    )
    logger.info("Created Spineax solver with cached symbolic factorization")

    def nr_solve(V_init: Array, vsource_vals: Array, isource_vals: Array,
                 Q_prev: Array, integ_c0: float | Array,
                 device_arrays_arg: Dict[str, Array],
                 gmin: float | Array = 1e-12, gshunt: float | Array = 0.0,
                 integ_c1: float | Array = 0.0, integ_d1: float | Array = 0.0,
                 dQdt_prev: Array | None = None,
                 integ_c2: float | Array = 0.0, Q_prev2: Array | None = None):

        # Ensure dQdt_prev is a proper array for JIT tracing
        _dQdt_prev = dQdt_prev if dQdt_prev is not None else jnp.zeros(n_unknowns, dtype=jnp.float64)
        # Ensure Q_prev2 is a proper array for JIT tracing (Gear2 method)
        _Q_prev2 = Q_prev2 if Q_prev2 is not None else jnp.zeros(n_unknowns, dtype=jnp.float64)

        init_Q = jnp.zeros(n_unknowns, dtype=jnp.float64)
        init_state = (
            V_init,
            jnp.array(0, dtype=jnp.int32),
            jnp.array(False),
            jnp.array(jnp.inf),
            jnp.array(jnp.inf),
            init_Q,
        )

        def cond_fn(state):
            V, iteration, converged, max_f, max_delta, Q = state
            return jnp.logical_and(~converged, iteration < max_iterations)

        def body_fn(state):
            V, iteration, _, _, _, _ = state

            J_bcoo, f, Q, _ = build_system_jit(V, vsource_vals, isource_vals, Q_prev,
                                              integ_c0, device_arrays_arg, gmin, gshunt,
                                              integ_c1, integ_d1, _dQdt_prev, integ_c2, _Q_prev2)

            if residual_mask is not None:
                f_masked = jnp.where(residual_mask, f, 0.0)
                max_f = jnp.max(jnp.abs(f_masked))
            else:
                max_f = jnp.max(jnp.abs(f))
            residual_converged = max_f < abstol

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

            max_delta = jnp.max(jnp.abs(delta))
            scale = jnp.where(max_delta > max_step, max_step / max_delta, 1.0)
            delta = delta * scale

            # Update V with damping
            V_candidate = V[1:] + delta
            V_damped = apply_voltage_damping(V_candidate, V[1:], DAMPING_VT, DAMPING_VCRIT, nr_damping=_NR_DAMPING)
            V_new = V.at[1:].set(V_damped)

            if noi_indices is not None and len(noi_indices) > 0:
                V_new = V_new.at[noi_indices].set(0.0)

            delta_converged = max_delta < 1e-12
            converged = jnp.logical_or(residual_converged, delta_converged)

            return (V_new, iteration + 1, converged, max_f, max_delta, Q)

        V_final, iterations, converged, max_f, max_delta, _ = lax.while_loop(
            cond_fn, body_fn, init_state
        )

        _, _, Q_final, I_vsource = build_system_jit(V_final, vsource_vals, isource_vals, Q_prev,
                                                    integ_c0, device_arrays_arg, gmin, gshunt,
                                                    integ_c1, integ_d1, _dQdt_prev, integ_c2, _Q_prev2)

        # Compute dQdt for next timestep
        dQdt_final = integ_c0 * Q_final + integ_c1 * Q_prev + integ_d1 * _dQdt_prev + integ_c2 * _Q_prev2

        return V_final, iterations, converged, max_f, Q_final, dQdt_final, I_vsource

    logger.info(f"Creating Spineax NR solver: V({n_nodes})")
    return jax.jit(nr_solve)


def make_umfpack_solver(
    build_system_jit: Callable,
    n_nodes: int,
    nse: int,
    bcsr_indptr: Array,
    bcsr_indices: Array,
    noi_indices: Optional[Array] = None,
    max_iterations: int = MAX_NR_ITERATIONS,
    abstol: float = DEFAULT_ABSTOL,
    max_step: float = 1.0,
    coo_sort_perm: Optional[Array] = None,
    csr_segment_ids: Optional[Array] = None,
) -> Callable:
    """Create a JIT-compiled sparse NR solver using UMFPACK.

    Uses UMFPACK with cached symbolic factorization for fast CPU solving.

    Args:
        build_system_jit: JIT-wrapped function returning (J_bcoo, f, Q)
        n_nodes: Total node count including ground
        nse: Number of stored elements after summing duplicates
        bcsr_indptr: Pre-computed BCSR row pointers
        bcsr_indices: Pre-computed BCSR column indices
        noi_indices: Optional array of NOI node indices
        max_iterations: Maximum NR iterations
        abstol: Absolute tolerance for convergence
        max_step: Maximum voltage step per iteration
        coo_sort_perm: Pre-computed COO→CSR permutation
        csr_segment_ids: Pre-computed segment IDs

    Returns:
        JIT-compiled UMFPACK solver function
    """
    from jax.experimental.sparse import BCSR

    from jax_spice.analysis.umfpack_solver import UMFPACKSolver

    n_unknowns = n_nodes - 1
    use_precomputed = coo_sort_perm is not None and csr_segment_ids is not None

    masks = _compute_noi_masks(noi_indices, n_nodes, bcsr_indptr, bcsr_indices)
    residual_mask = masks['residual_mask']
    noi_row_mask = masks['noi_row_mask']
    noi_col_mask = masks['noi_col_mask']
    noi_diag_indices = masks['noi_diag_indices']
    noi_res_indices_arr = masks['noi_res_indices_arr']

    # Create UMFPACK solver with cached symbolic factorization
    umfpack_solver = UMFPACKSolver(bcsr_indptr, bcsr_indices)
    logger.info("Created UMFPACK solver with cached symbolic factorization")

    def nr_solve(V_init: Array, vsource_vals: Array, isource_vals: Array,
                 Q_prev: Array, integ_c0: float | Array,
                 device_arrays_arg: Dict[str, Array],
                 gmin: float | Array = 1e-12, gshunt: float | Array = 0.0,
                 integ_c1: float | Array = 0.0, integ_d1: float | Array = 0.0,
                 dQdt_prev: Array | None = None,
                 integ_c2: float | Array = 0.0, Q_prev2: Array | None = None):

        # Ensure dQdt_prev is a proper array for JIT tracing
        _dQdt_prev = dQdt_prev if dQdt_prev is not None else jnp.zeros(n_unknowns, dtype=jnp.float64)
        # Ensure Q_prev2 is a proper array for JIT tracing (Gear2 method)
        _Q_prev2 = Q_prev2 if Q_prev2 is not None else jnp.zeros(n_unknowns, dtype=jnp.float64)

        init_Q = jnp.zeros(n_unknowns, dtype=jnp.float64)
        init_state = (
            V_init,
            jnp.array(0, dtype=jnp.int32),
            jnp.array(False),
            jnp.array(jnp.inf),
            jnp.array(jnp.inf),
            init_Q,
        )

        def cond_fn(state):
            V, iteration, converged, max_f, max_delta, Q = state
            return jnp.logical_and(~converged, iteration < max_iterations)

        def body_fn(state):
            V, iteration, _, _, _, _ = state

            J_bcoo, f, Q, _ = build_system_jit(V, vsource_vals, isource_vals, Q_prev,
                                              integ_c0, device_arrays_arg, gmin, gshunt,
                                              integ_c1, integ_d1, _dQdt_prev, integ_c2, _Q_prev2)

            if residual_mask is not None:
                f_masked = jnp.where(residual_mask, f, 0.0)
                max_f = jnp.max(jnp.abs(f_masked))
            else:
                max_f = jnp.max(jnp.abs(f))
            residual_converged = max_f < abstol

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

            max_delta = jnp.max(jnp.abs(delta))
            scale = jnp.where(max_delta > max_step, max_step / max_delta, 1.0)
            delta = delta * scale

            # Update V with damping
            V_candidate = V[1:] + delta
            V_damped = apply_voltage_damping(V_candidate, V[1:], DAMPING_VT, DAMPING_VCRIT, nr_damping=_NR_DAMPING)
            V_new = V.at[1:].set(V_damped)

            if noi_indices is not None and len(noi_indices) > 0:
                V_new = V_new.at[noi_indices].set(0.0)

            delta_converged = max_delta < 1e-12
            converged = jnp.logical_or(residual_converged, delta_converged)

            return (V_new, iteration + 1, converged, max_f, max_delta, Q)

        V_final, iterations, converged, max_f, max_delta, _ = lax.while_loop(
            cond_fn, body_fn, init_state
        )

        _, _, Q_final, I_vsource = build_system_jit(V_final, vsource_vals, isource_vals, Q_prev,
                                                    integ_c0, device_arrays_arg, gmin, gshunt,
                                                    integ_c1, integ_d1, _dQdt_prev, integ_c2, _Q_prev2)

        # Compute dQdt for next timestep
        dQdt_final = integ_c0 * Q_final + integ_c1 * Q_prev + integ_d1 * _dQdt_prev + integ_c2 * _Q_prev2

        return V_final, iterations, converged, max_f, Q_final, dQdt_final, I_vsource

    logger.info(f"Creating UMFPACK NR solver: V({n_nodes})")
    return jax.jit(nr_solve)


# =============================================================================
# UMFPACK FFI Solver (eliminates pure_callback overhead)
# =============================================================================

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
    max_iterations: int = MAX_NR_ITERATIONS,
    abstol: float = DEFAULT_ABSTOL,
    max_step: float = 1.0,
    coo_sort_perm: Optional[Array] = None,
    csr_segment_ids: Optional[Array] = None,
) -> Callable:
    """Create a JIT-compiled UMFPACK FFI NR solver for full MNA formulation.

    This is the FFI-based version that eliminates the ~100ms pure_callback overhead.
    Uses UMFPACK directly via XLA FFI for fast CPU solving.

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

    Returns:
        JIT-compiled solver function
    """
    from jax.experimental.sparse import BCSR
    from jax_spice.sparse import umfpack_jax

    if not umfpack_jax.is_available():
        raise RuntimeError(
            "UMFPACK FFI extension not available. "
            "Install with: cd jax_spice/sparse && pip install ."
        )

    n_unknowns = n_nodes - 1
    n_augmented = n_unknowns + n_vsources
    n_total = n_nodes

    use_precomputed = coo_sort_perm is not None and csr_segment_ids is not None

    masks = _compute_noi_masks(noi_indices, n_nodes, bcsr_indptr, bcsr_indices)
    noi_res_idx = masks['noi_res_idx']
    noi_row_mask = masks['noi_row_mask']
    noi_col_mask = masks['noi_col_mask']
    noi_diag_indices = masks['noi_diag_indices']
    noi_res_indices_arr = masks['noi_res_indices_arr']

    if masks['residual_mask'] is not None:
        residual_mask = jnp.concatenate([
            masks['residual_mask'],
            jnp.ones(n_vsources, dtype=jnp.bool_)
        ])
    else:
        residual_mask = None

    logger.info("Creating UMFPACK FFI full MNA solver with zero callback overhead")

    def nr_solve(X_init: Array, vsource_vals: Array, isource_vals: Array,
                 Q_prev: Array, integ_c0: float | Array,
                 device_arrays_arg: Dict[str, Array],
                 gmin: float | Array = 1e-12, gshunt: float | Array = 0.0,
                 integ_c1: float | Array = 0.0, integ_d1: float | Array = 0.0,
                 dQdt_prev: Array | None = None,
                 integ_c2: float | Array = 0.0, Q_prev2: Array | None = None):

        _dQdt_prev = dQdt_prev if dQdt_prev is not None else jnp.zeros(n_unknowns, dtype=jnp.float64)
        _Q_prev2 = Q_prev2 if Q_prev2 is not None else jnp.zeros(n_unknowns, dtype=jnp.float64)

        init_Q = jnp.zeros(n_unknowns, dtype=jnp.float64)
        init_state = (
            X_init,
            jnp.array(0, dtype=jnp.int32),
            jnp.array(False),
            jnp.array(jnp.inf),
            jnp.array(jnp.inf),
            init_Q,
        )

        def cond_fn(state):
            X, iteration, converged, max_f, max_delta, Q = state
            return jnp.logical_and(~converged, iteration < max_iterations)

        def body_fn(state):
            X, iteration, _, _, _, _ = state

            J_bcoo, f, Q, _ = build_system_jit(X, vsource_vals, isource_vals, Q_prev,
                                                integ_c0, device_arrays_arg, gmin, gshunt,
                                                integ_c1, integ_d1, _dQdt_prev, integ_c2, _Q_prev2)

            if residual_mask is not None:
                f_masked = jnp.where(residual_mask, f, 0.0)
                max_f = jnp.max(jnp.abs(f_masked))
            else:
                max_f = jnp.max(jnp.abs(f))
            residual_converged = max_f < abstol

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

            max_delta = jnp.max(jnp.abs(delta))
            scale = jnp.where(max_delta > max_step, max_step / max_delta, 1.0)
            delta = delta * scale

            # Update X with damping for node voltages
            V_candidate = X[1:n_total] + delta[:n_unknowns]
            V_damped = apply_voltage_damping(V_candidate, X[1:n_total], DAMPING_VT, DAMPING_VCRIT, nr_damping=_NR_DAMPING)
            X_new = X.at[1:n_total].set(V_damped)
            X_new = X_new.at[n_total:].add(delta[n_unknowns:])

            if noi_indices is not None and len(noi_indices) > 0:
                X_new = X_new.at[noi_indices].set(0.0)

            delta_converged = max_delta < 1e-12
            converged = jnp.logical_or(residual_converged, delta_converged)

            return (X_new, iteration + 1, converged, max_f, max_delta, Q)

        X_final, iterations, converged, max_f, max_delta, _ = lax.while_loop(
            cond_fn, body_fn, init_state
        )

        _, _, Q_final, I_vsource = build_system_jit(X_final, vsource_vals, isource_vals, Q_prev,
                                                    integ_c0, device_arrays_arg, gmin, gshunt,
                                                    integ_c1, integ_d1, _dQdt_prev, integ_c2, _Q_prev2)

        dQdt_final = integ_c0 * Q_final + integ_c1 * Q_prev + integ_d1 * _dQdt_prev + integ_c2 * _Q_prev2

        return X_final, iterations, converged, max_f, Q_final, dQdt_final, I_vsource

    logger.info(f"Creating UMFPACK FFI full MNA solver: V({n_nodes}) + I({n_vsources})")
    return jax.jit(nr_solve)


def make_umfpack_ffi_solver(
    build_system_jit: Callable,
    n_nodes: int,
    nse: int,
    bcsr_indptr: Array,
    bcsr_indices: Array,
    noi_indices: Optional[Array] = None,
    max_iterations: int = MAX_NR_ITERATIONS,
    abstol: float = DEFAULT_ABSTOL,
    max_step: float = 1.0,
    coo_sort_perm: Optional[Array] = None,
    csr_segment_ids: Optional[Array] = None,
) -> Callable:
    """Create a JIT-compiled sparse NR solver using UMFPACK FFI.

    This is the FFI-based version that eliminates the ~100ms pure_callback overhead.

    Args:
        build_system_jit: JIT-wrapped function returning (J_bcoo, f, Q)
        n_nodes: Total node count including ground
        nse: Number of stored elements after summing duplicates
        bcsr_indptr: Pre-computed BCSR row pointers
        bcsr_indices: Pre-computed BCSR column indices
        noi_indices: Optional array of NOI node indices
        max_iterations: Maximum NR iterations
        abstol: Absolute tolerance for convergence
        max_step: Maximum voltage step per iteration
        coo_sort_perm: Pre-computed COO→CSR permutation
        csr_segment_ids: Pre-computed segment IDs

    Returns:
        JIT-compiled solver function
    """
    from jax.experimental.sparse import BCSR
    from jax_spice.sparse import umfpack_jax

    if not umfpack_jax.is_available():
        raise RuntimeError(
            "UMFPACK FFI extension not available. "
            "Install with: cd jax_spice/sparse && pip install ."
        )

    n_unknowns = n_nodes - 1
    use_precomputed = coo_sort_perm is not None and csr_segment_ids is not None

    masks = _compute_noi_masks(noi_indices, n_nodes, bcsr_indptr, bcsr_indices)
    noi_res_idx = masks['noi_res_idx']
    residual_mask = masks['residual_mask']
    noi_row_mask = masks['noi_row_mask']
    noi_col_mask = masks['noi_col_mask']
    noi_diag_indices = masks['noi_diag_indices']
    noi_res_indices_arr = masks['noi_res_indices_arr']

    logger.info("Creating UMFPACK FFI NR solver with zero callback overhead")

    def nr_solve(V_init: Array, vsource_vals: Array, isource_vals: Array,
                 Q_prev: Array, integ_c0: float | Array,
                 device_arrays_arg: Dict[str, Array],
                 gmin: float | Array = 1e-12, gshunt: float | Array = 0.0,
                 integ_c1: float | Array = 0.0, integ_d1: float | Array = 0.0,
                 dQdt_prev: Array | None = None,
                 integ_c2: float | Array = 0.0, Q_prev2: Array | None = None):

        _dQdt_prev = dQdt_prev if dQdt_prev is not None else jnp.zeros(n_unknowns, dtype=jnp.float64)
        _Q_prev2 = Q_prev2 if Q_prev2 is not None else jnp.zeros(n_unknowns, dtype=jnp.float64)

        init_Q = jnp.zeros(n_unknowns, dtype=jnp.float64)
        init_state = (
            V_init,
            jnp.array(0, dtype=jnp.int32),
            jnp.array(False),
            jnp.array(jnp.inf),
            jnp.array(jnp.inf),
            init_Q,
        )

        def cond_fn(state):
            V, iteration, converged, max_f, max_delta, Q = state
            return jnp.logical_and(~converged, iteration < max_iterations)

        def body_fn(state):
            V, iteration, _, _, _, _ = state

            J_bcoo, f, Q, _ = build_system_jit(V, vsource_vals, isource_vals, Q_prev, integ_c0,
                                                device_arrays_arg, gmin, gshunt,
                                                integ_c1, integ_d1, _dQdt_prev, integ_c2, _Q_prev2)

            if residual_mask is not None:
                f_masked = jnp.where(residual_mask, f, 0.0)
                max_f = jnp.max(jnp.abs(f_masked))
            else:
                max_f = jnp.max(jnp.abs(f))
            residual_converged = max_f < abstol

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

            max_delta = jnp.max(jnp.abs(delta))
            scale = jnp.where(max_delta > max_step, max_step / max_delta, 1.0)
            delta = delta * scale

            # Update V with damping
            V_candidate = V[1:] + delta
            V_damped = apply_voltage_damping(V_candidate, V[1:], DAMPING_VT, DAMPING_VCRIT, nr_damping=_NR_DAMPING)
            V_new = V.at[1:].set(V_damped)

            if noi_indices is not None and len(noi_indices) > 0:
                V_new = V_new.at[noi_indices].set(0.0)

            delta_converged = max_delta < 1e-12
            converged = jnp.logical_or(residual_converged, delta_converged)

            return (V_new, iteration + 1, converged, max_f, max_delta, Q)

        V_final, iterations, converged, max_f, max_delta, _ = lax.while_loop(
            cond_fn, body_fn, init_state
        )

        _, _, Q_final, I_vsource = build_system_jit(V_final, vsource_vals, isource_vals, Q_prev,
                                                    integ_c0, device_arrays_arg, gmin, gshunt,
                                                    integ_c1, integ_d1, _dQdt_prev, integ_c2, _Q_prev2)

        dQdt_final = integ_c0 * Q_final + integ_c1 * Q_prev + integ_d1 * _dQdt_prev + integ_c2 * _Q_prev2

        return V_final, iterations, converged, max_f, Q_final, dQdt_final, I_vsource

    logger.info(f"Creating UMFPACK FFI NR solver: V({n_nodes})")
    return jax.jit(nr_solve)
