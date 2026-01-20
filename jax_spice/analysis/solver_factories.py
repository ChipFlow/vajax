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
    -> (V_final, iterations, converged, max_residual, Q_final, dQdt_final)

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

# Newton-Raphson solver constants
MAX_NR_ITERATIONS = 100
DEFAULT_ABSTOL = 1e4  # Corresponds to ~10nV voltage accuracy with G=1e12


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
             integ_c1, integ_d1, dQdt_prev, integ_c2, Q_prev2) -> (J, f, Q)
        n_nodes: Total node count including ground
        noi_indices: Optional array of NOI node indices to constrain to 0V
        max_iterations: Maximum NR iterations
        abstol: Absolute tolerance for convergence
        max_step: Maximum voltage step per iteration

    Returns:
        JIT-compiled solver function with signature:
            (V, vsource_vals, isource_vals, Q_prev, integ_c0, device_arrays, gmin, gshunt,
             integ_c1, integ_d1, dQdt_prev, integ_c2, Q_prev2) -> (V, iterations, converged, max_f, Q, dQdt)
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

            J, f, Q = build_system_jit(V, vsource_vals, isource_vals, Q_prev, integ_c0,
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

            # Solve linear system
            delta = jax.scipy.linalg.solve(J, -f)

            # Step limiting
            max_delta = jnp.max(jnp.abs(delta))
            scale = jnp.where(max_delta > max_step, max_step / max_delta, 1.0)
            delta = delta * scale

            # Update V (ground stays fixed)
            V_new = V.at[1:].add(delta)

            # Clamp NOI nodes to 0V
            if noi_indices is not None and len(noi_indices) > 0:
                V_new = V_new.at[noi_indices].set(0.0)

            delta_converged = max_delta < 1e-12
            converged = jnp.logical_or(residual_converged, delta_converged)

            return (V_new, iteration + 1, converged, max_f, max_delta, Q)

        V_final, iterations, converged, max_f, max_delta, _ = lax.while_loop(
            cond_fn, body_fn, init_state
        )

        # Recompute Q from converged voltage
        _, _, Q_final = build_system_jit(V_final, vsource_vals, isource_vals, Q_prev,
                                         integ_c0, device_arrays_arg, gmin, gshunt,
                                         integ_c1, integ_d1, _dQdt_prev, integ_c2, _Q_prev2)

        # Compute dQdt for next timestep (needed for trapezoidal and Gear2 methods)
        # dQdt = c0 * Q + c1 * Q_prev + d1 * dQdt_prev + c2 * Q_prev2
        dQdt_final = integ_c0 * Q_final + integ_c1 * Q_prev + integ_d1 * _dQdt_prev + integ_c2 * _Q_prev2

        return V_final, iterations, converged, max_f, Q_final, dQdt_final

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

            J_bcoo, f, Q = build_system_jit(V, vsource_vals, isource_vals, Q_prev,
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

            V_new = V.at[1:].add(delta)

            if noi_indices is not None and len(noi_indices) > 0:
                V_new = V_new.at[noi_indices].set(0.0)

            delta_converged = max_delta < 1e-12
            converged = jnp.logical_or(residual_converged, delta_converged)

            return (V_new, iteration + 1, converged, max_f, max_delta, Q)

        V_final, iterations, converged, max_f, max_delta, _ = lax.while_loop(
            cond_fn, body_fn, init_state
        )

        _, _, Q_final = build_system_jit(V_final, vsource_vals, isource_vals, Q_prev,
                                         integ_c0, device_arrays_arg, gmin, gshunt,
                                         integ_c1, integ_d1, _dQdt_prev, integ_c2, _Q_prev2)

        # Compute dQdt for next timestep
        dQdt_final = integ_c0 * Q_final + integ_c1 * Q_prev + integ_d1 * _dQdt_prev + integ_c2 * _Q_prev2

        return V_final, iterations, converged, max_f, Q_final, dQdt_final

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

            J_bcoo, f, Q = build_system_jit(V, vsource_vals, isource_vals, Q_prev,
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

            V_new = V.at[1:].add(delta)

            if noi_indices is not None and len(noi_indices) > 0:
                V_new = V_new.at[noi_indices].set(0.0)

            delta_converged = max_delta < 1e-12
            converged = jnp.logical_or(residual_converged, delta_converged)

            return (V_new, iteration + 1, converged, max_f, max_delta, Q)

        V_final, iterations, converged, max_f, max_delta, _ = lax.while_loop(
            cond_fn, body_fn, init_state
        )

        _, _, Q_final = build_system_jit(V_final, vsource_vals, isource_vals, Q_prev,
                                         integ_c0, device_arrays_arg, gmin, gshunt,
                                         integ_c1, integ_d1, _dQdt_prev, integ_c2, _Q_prev2)

        # Compute dQdt for next timestep
        dQdt_final = integ_c0 * Q_final + integ_c1 * Q_prev + integ_d1 * _dQdt_prev + integ_c2 * _Q_prev2

        return V_final, iterations, converged, max_f, Q_final, dQdt_final

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

            J_bcoo, f, Q = build_system_jit(V, vsource_vals, isource_vals, Q_prev,
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

            V_new = V.at[1:].add(delta)

            if noi_indices is not None and len(noi_indices) > 0:
                V_new = V_new.at[noi_indices].set(0.0)

            delta_converged = max_delta < 1e-12
            converged = jnp.logical_or(residual_converged, delta_converged)

            return (V_new, iteration + 1, converged, max_f, max_delta, Q)

        V_final, iterations, converged, max_f, max_delta, _ = lax.while_loop(
            cond_fn, body_fn, init_state
        )

        _, _, Q_final = build_system_jit(V_final, vsource_vals, isource_vals, Q_prev,
                                         integ_c0, device_arrays_arg, gmin, gshunt,
                                         integ_c1, integ_d1, _dQdt_prev, integ_c2, _Q_prev2)

        # Compute dQdt for next timestep
        dQdt_final = integ_c0 * Q_final + integ_c1 * Q_prev + integ_d1 * _dQdt_prev + integ_c2 * _Q_prev2

        return V_final, iterations, converged, max_f, Q_final, dQdt_final

    logger.info(f"Creating UMFPACK NR solver: V({n_nodes})")
    return jax.jit(nr_solve)
