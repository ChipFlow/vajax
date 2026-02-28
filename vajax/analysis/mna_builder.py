"""MNA system builder for VAJAX.

This module provides the factory function that creates the GPU-resident
build_system function for Modified Nodal Analysis (MNA) formulation.

The build_system function is the core of the Newton-Raphson solver - it
evaluates all devices, assembles residuals and Jacobians, and returns
the system for solving.

Two assembly modes are supported:
- COO mode: Collects COO (row, col, val) triples, then assembles dense/sparse Jacobian.
  Used for dense solvers and sparse solvers without CSR direct stamping.
- CSR direct mode: Stamps device contributions directly into a pre-allocated CSR data
  array using pre-computed position mappings. Eliminates COO intermediates (~1 GB savings
  for large circuits like mul64 with 266k devices).
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import jax.numpy as jnp
import numpy as np
from jax import Array, lax

from vajax import get_float_dtype
from vajax.analysis.mna import (
    assemble_coo_max_abs,
    assemble_coo_vector,
    assemble_dense_jacobian,
    assemble_jacobian_coo,
    assemble_sparse_jacobian,
    build_vsource_equations,
    build_vsource_incidence_coo,
    build_vsource_kcl_contribution,
    combine_transient_residual,
    compute_vsource_current_from_kcl,
    mask_coo_matrix,
    mask_coo_vector,
)

logger = logging.getLogger(__name__)


def build_stamp_index_mapping(
    model_type: str,
    device_contexts: List[Dict],
    ground: int,
    compiled_models: Dict[str, Any],
) -> Dict[str, Array]:
    """Pre-compute index mappings for COO-based stamping.

    Called once per model type during setup. Returns arrays that map
    (device_idx, entry_idx) to global matrix indices.

    Args:
        model_type: OpenVAF model type (e.g., 'psp103')
        device_contexts: List of device context dicts with node_map
        ground: Ground node index
        compiled_models: Dict of compiled model info

    Returns:
        Dict with:
            res_indices: (n_devices, n_residuals) row indices for f vector, -1 for ground
            jac_row_indices: (n_devices, n_jac_entries) row indices for J
            jac_col_indices: (n_devices, n_jac_entries) col indices for J
    """
    compiled = compiled_models.get(model_type)
    if not compiled:
        return {}

    metadata = compiled["dae_metadata"]
    node_names = metadata["node_names"]  # Residual node names
    jacobian_keys = metadata["jacobian_keys"]  # (row_name, col_name) pairs

    n_devices = len(device_contexts)
    n_residuals = len(node_names)
    n_jac_entries = len(jacobian_keys)

    # Build residual index array
    res_indices = np.full((n_devices, n_residuals), -1, dtype=np.int32)

    for dev_idx, ctx in enumerate(device_contexts):
        node_map = ctx["node_map"]
        for res_idx, node_name in enumerate(node_names):
            # Map node name to global index
            # V2 API provides clean names like 'D', 'G', 'S', 'B', 'NOI', etc.
            node_idx = node_map.get(node_name, None)

            if node_idx is not None and node_idx != ground and node_idx > 0:
                res_indices[dev_idx, res_idx] = node_idx - 1  # 0-indexed residual

    # Build Jacobian index arrays
    jac_row_indices = np.full((n_devices, n_jac_entries), -1, dtype=np.int32)
    jac_col_indices = np.full((n_devices, n_jac_entries), -1, dtype=np.int32)

    for dev_idx, ctx in enumerate(device_contexts):
        node_map = ctx["node_map"]
        for jac_idx, (row_name, col_name) in enumerate(jacobian_keys):
            # Map row/col nodes - V2 API provides clean names
            row_idx = node_map.get(row_name, None)
            col_idx = node_map.get(col_name, None)

            if (
                row_idx is not None
                and col_idx is not None
                and row_idx != ground
                and col_idx != ground
                and row_idx > 0
                and col_idx > 0
            ):
                jac_row_indices[dev_idx, jac_idx] = row_idx - 1
                jac_col_indices[dev_idx, jac_idx] = col_idx - 1

    return {
        "res_indices": jnp.array(res_indices),
        "jac_row_indices": jnp.array(jac_row_indices),
        "jac_col_indices": jnp.array(jac_col_indices),
    }


# =============================================================================
# CSR Direct Stamping Position Computation
# =============================================================================


def compute_csr_stamp_positions(
    jac_row_indices: np.ndarray,
    jac_col_indices: np.ndarray,
    unique_linear: np.ndarray,
    n_augmented: int,
) -> np.ndarray:
    """Compute CSR data positions for device Jacobian entries.

    Maps each (row, col) pair from device stamp indices to its position in the
    CSR data array. Uses vectorized searchsorted for O(N log M) performance
    where N = total device entries and M = CSR non-zeros.

    Args:
        jac_row_indices: (n_devices, n_jac_entries) int32, -1 for ground/invalid
        jac_col_indices: (n_devices, n_jac_entries) int32, -1 for ground/invalid
        unique_linear: Sorted unique linear indices (row * n_augmented + col) of
            CSR entries. Same order as CSR data array.
        n_augmented: Total matrix dimension (n_unknowns + n_vsources)

    Returns:
        (n_devices, n_jac_entries) int32: CSR data position, or -1 for invalid
    """
    valid = (jac_row_indices >= 0) & (jac_col_indices >= 0)
    linear = jac_row_indices.astype(np.int64) * n_augmented + jac_col_indices.astype(np.int64)

    flat_linear = linear.ravel()
    positions = np.searchsorted(unique_linear, flat_linear)
    positions = np.clip(positions, 0, len(unique_linear) - 1)
    matched = (unique_linear[positions] == flat_linear) & valid.ravel()

    result = np.where(matched, positions, -1).astype(np.int32)
    return result.reshape(jac_row_indices.shape)


def compute_csr_diagonal_positions(
    n_unknowns: int,
    unique_linear: np.ndarray,
    n_augmented: int,
) -> np.ndarray:
    """Compute CSR data positions for diagonal regularization entries.

    Args:
        n_unknowns: Number of node voltage unknowns (diagonal entries 0..n-1)
        unique_linear: Sorted unique linear indices of CSR entries
        n_augmented: Total matrix dimension

    Returns:
        (n_unknowns,) int32: CSR data position for each diagonal entry
    """
    diag_rows = np.arange(n_unknowns, dtype=np.int64)
    diag_linear = diag_rows * n_augmented + diag_rows
    diag_pos = np.searchsorted(unique_linear, diag_linear)
    diag_pos = np.clip(diag_pos, 0, len(unique_linear) - 1)
    matched = unique_linear[diag_pos] == diag_linear
    assert np.all(matched), (
        f"All {n_unknowns} diagonal entries must exist in CSR structure, "
        f"but {np.sum(~matched)} are missing"
    )
    return diag_pos.astype(np.int32)


def compute_csr_vsource_positions(
    vsource_node_p: np.ndarray,
    vsource_node_n: np.ndarray,
    n_unknowns: int,
    n_vsources: int,
    unique_linear: np.ndarray,
    n_augmented: int,
) -> Optional[Dict[str, np.ndarray]]:
    """Compute CSR data positions for vsource B/B^T incidence entries.

    The B block maps branch currents to node KCL equations.
    The B^T block provides voltage constraint equations.

    Args:
        vsource_node_p: Positive terminal circuit node indices, shape (m,)
        vsource_node_n: Negative terminal circuit node indices, shape (m,)
        n_unknowns: Number of node voltage unknowns
        n_vsources: Number of voltage sources (m)
        unique_linear: Sorted unique linear indices of CSR entries
        n_augmented: Total matrix dimension

    Returns:
        Dict with CSR positions for B and B^T entries, or None if no vsources.
        Each array is (n_vsources,) int32, -1 for invalid (ground node).
    """
    if n_vsources == 0:
        return None

    branch_indices = np.arange(n_vsources, dtype=np.int64)
    n_ul = len(unique_linear)

    def _find_positions(rows, cols, valid_mask):
        linear = rows.astype(np.int64) * n_augmented + cols.astype(np.int64)
        pos = np.searchsorted(unique_linear, linear)
        pos = np.clip(pos, 0, n_ul - 1)
        matched = (unique_linear[pos] == linear) & valid_mask
        return np.where(matched, pos, -1).astype(np.int32)

    valid_p = vsource_node_p > 0
    valid_n = vsource_node_n > 0

    return {
        # B block: row=node_p-1, col=n_unknowns+i (positive terminal)
        "b_p": _find_positions(
            np.where(valid_p, vsource_node_p - 1, 0),
            np.where(valid_p, n_unknowns + branch_indices, 0),
            valid_p,
        ),
        # B block: row=node_n-1, col=n_unknowns+i (negative terminal)
        "b_n": _find_positions(
            np.where(valid_n, vsource_node_n - 1, 0),
            np.where(valid_n, n_unknowns + branch_indices, 0),
            valid_n,
        ),
        # B^T block: row=n_unknowns+i, col=node_p-1 (positive terminal)
        "bt_p": _find_positions(
            np.where(valid_p, n_unknowns + branch_indices, 0),
            np.where(valid_p, vsource_node_p - 1, 0),
            valid_p,
        ),
        # B^T block: row=n_unknowns+i, col=node_n-1 (negative terminal)
        "bt_n": _find_positions(
            np.where(valid_n, n_unknowns + branch_indices, 0),
            np.where(valid_n, vsource_node_n - 1, 0),
            valid_n,
        ),
        "valid_p": valid_p.astype(np.bool_),
        "valid_n": valid_n.astype(np.bool_),
    }


def make_mna_build_system_fn(
    source_device_data: Dict[str, Any],
    static_inputs_cache: Dict[str, Tuple],
    compiled_models: Dict[str, Dict[str, Any]],
    gmin: float,
    n_unknowns: int,
    use_dense: bool = True,
    csr_stamp_info: Optional[Dict[str, Any]] = None,
    batch_size: Optional[int] = None,
) -> Tuple[Callable, Dict[str, Array], int]:
    """Create GPU-resident build_system function for MNA formulation.

    Uses true Modified Nodal Analysis with branch currents as explicit unknowns,
    providing more accurate current extraction than the high-G (G=1e12) approximation.

    MNA augments the system from n×n to (n+m)×(n+m) where m = number
    of voltage sources. The branch currents become primary unknowns:

        ┌───────────────┐   ┌───┐   ┌───────┐
        │  G + c0*C   B │   │ V │   │ f_node│
        │               │ × │   │ = │       │
        │    B^T      0 │   │ J │   │ E - V │
        └───────────────┘   └───┘   └───────┘

    Where:
    - G = device conductance matrix (n×n)
    - C = device capacitance matrix (n×n)
    - B = incidence matrix mapping currents to nodes (n×m)
    - V = node voltages (n×1)
    - J = branch currents (m×1) - primary unknowns for vsources
    - f_node = device current contributions
    - E = voltage source values (m×1)

    Args:
        source_device_data: Pre-computed source device stamp templates.
            Expected structure: {
                "vsource": {"names": [...], "node_p": [...], "node_n": [...]},
                "isource": {"f_indices": array, ...}
            }
        static_inputs_cache: Dict of static inputs per model type.
            Each entry: (voltage_indices, stamp_indices, voltage_node1, voltage_node2, cache, _)
        compiled_models: Dict of compiled model info per model type.
            Each entry should have: vmapped_split_eval, shared_params, device_params,
            voltage_positions_in_varying, shared_cache, device_cache, uses_analysis,
            uses_simparam_gmin, use_device_limiting, num_limit_states, default_simparams
        gmin: Minimum conductance for diagonal regularization (from options.gmin)
        n_unknowns: Number of node voltage unknowns (n_total - 1)
        use_dense: Whether to use dense or sparse matrix assembly
        csr_stamp_info: Pre-computed CSR stamp positions for direct CSR assembly.
            When provided (and use_dense=False), device Jacobian entries are stamped
            directly into a CSR data array, eliminating COO intermediates. Structure:
            {"nse": int, "model_csr_pos": {model_type: jnp.array}, ...}
        batch_size: If set, process devices in fixed-size batches via lax.scan.
            Only used when csr_stamp_info is also provided. Reduces peak VRAM
            by processing batch_size devices at a time instead of all at once.

    Returns:
        Tuple of:
        - build_system function with signature:
            build_system(X, vsource_vals, isource_vals, Q_prev, integ_c0, device_arrays,
                         gmin, gshunt, ...) -> (J, f, Q, I_vsource, limit_state_out)
          where X = [V; I_branch] is the augmented solution vector
        - device_arrays: Dict[model_type, cache] to pass to build_system
        - total_limit_states: Total size of limit state array
    """
    # Get number of voltage sources for augmentation
    n_vsources = len(source_device_data.get("vsource", {}).get("names", []))
    n_augmented = n_unknowns + n_vsources

    # Capture model types as static list (unrolled at trace time)
    model_types = list(static_inputs_cache.keys())

    # Split cache into metadata (captured) and arrays (passed as argument)
    static_metadata: Dict[str, Tuple] = {}
    device_arrays: Dict[str, Array] = {}
    split_eval_info: Dict[str, Dict[str, Any]] = {}

    for model_type in model_types:
        voltage_indices, stamp_indices, voltage_node1, voltage_node2, cache, _ = (
            static_inputs_cache[model_type]
        )
        static_metadata[model_type] = (
            voltage_indices,
            stamp_indices,
            voltage_node1,
            voltage_node2,
        )

        compiled = compiled_models.get(model_type, {})
        n_devices = compiled["device_params"].shape[0]
        num_limit_states = compiled.get("num_limit_states", 0)
        split_eval_info[model_type] = {
            "vmapped_split_eval": compiled["vmapped_split_eval"],
            "shared_params": compiled["shared_params"],
            "device_params": compiled["device_params"],
            "voltage_positions": compiled["voltage_positions_in_varying"],
            "shared_cache": compiled["shared_cache"],
            "default_simparams": compiled.get("default_simparams", jnp.array([0.0, 1.0, 1e-12])),
            # Simparam metadata for correct index lookup
            "simparam_indices": compiled.get("simparam_indices", {}),
            # Device-level limiting info
            "use_device_limiting": compiled.get("use_device_limiting", False),
            "num_limit_states": num_limit_states,
            "n_devices": n_devices,
            # Analysis flags (captured for device evaluation)
            "uses_analysis": compiled.get("uses_analysis", False),
            "uses_simparam_gmin": compiled.get("uses_simparam_gmin", False),
        }
        device_arrays[model_type] = compiled["device_cache"]

    # Pre-compute vsource node indices as JAX arrays (captured in closure)
    if n_vsources > 0:
        vsource_node_p = jnp.array(source_device_data["vsource"]["node_p"], dtype=jnp.int32)
        vsource_node_n = jnp.array(source_device_data["vsource"]["node_n"], dtype=jnp.int32)
    else:
        vsource_node_p = jnp.zeros(0, dtype=jnp.int32)
        vsource_node_n = jnp.zeros(0, dtype=jnp.int32)

    # Capture gmin for diagonal regularization.
    # min_diag_reg must be consistent with the residual: the Jacobian has
    # min_diag_reg on the diagonal, and the residual must include min_diag_reg*V.
    # Using gmin (typically 1e-12) keeps the regularization negligible relative
    # to physical circuit conductances while preventing singular matrices.
    # IMPORTANT: A large min_diag_reg (e.g. 1e-6) without a matching residual
    # term masks the true sensitivity at weakly-grounded nodes, causing voltage
    # drift over transient timesteps (see graetz common-mode drift bug).
    min_diag_reg = gmin

    # Compute total limit state size and offsets per model type
    # limit_state is a flat array: [model0_device0_states, ..., model1_device0_states, ...]
    total_limit_states = 0
    limit_state_offsets: Dict[str, Tuple[int, int, int]] = {}
    for model_type in model_types:
        if model_type in split_eval_info:
            info = split_eval_info[model_type]
            if info.get("use_device_limiting", False) and info.get("num_limit_states", 0) > 0:
                n_devices = info["n_devices"]
                num_limit_states = info["num_limit_states"]
                limit_state_offsets[model_type] = (
                    total_limit_states,
                    n_devices,
                    num_limit_states,
                )
                total_limit_states += n_devices * num_limit_states

    # Determine assembly mode:
    # - COO mode: collect COO parts, assemble dense/sparse Jacobian
    # - CSR direct: stamp directly into CSR data array (eliminates COO intermediates)
    use_csr_direct = csr_stamp_info is not None and not use_dense

    # Pre-compute CSR stamp arrays captured in closure (factory time, not trace time)
    if use_csr_direct:
        _csr_nse = csr_stamp_info["nse"]
        _csr_model_pos = {
            mt: jnp.array(csr_stamp_info["model_csr_pos"][mt])
            for mt in model_types
            if mt in csr_stamp_info["model_csr_pos"]
        }
        _csr_diag_pos = jnp.array(csr_stamp_info["diag_positions"])
        _csr_vs_pos = None
        if csr_stamp_info.get("vsource_positions") is not None:
            vsp = csr_stamp_info["vsource_positions"]
            _csr_vs_pos = {
                k: jnp.array(v) for k, v in vsp.items()
            }

        # Pre-compute padded arrays for batched evaluation
        _batch_info: Dict[str, Dict[str, Any]] = {}
        if batch_size is not None:
            for model_type in model_types:
                info = split_eval_info[model_type]
                n_dev = info["n_devices"]
                if n_dev <= batch_size:
                    continue  # Small model, no batching needed

                n_batches = (n_dev + batch_size - 1) // batch_size
                padded_size = n_batches * batch_size

                # Pad device_params
                dp = info["device_params"]
                dp_padded = jnp.zeros((padded_size, dp.shape[1]), dtype=dp.dtype)
                dp_padded = dp_padded.at[:n_dev].set(dp)

                # Pad voltage_node1/node2 (use 0=ground for padding → voltage_update=0)
                _, stamp_indices, vn1, vn2 = static_metadata[model_type]
                vn1_padded = jnp.zeros((padded_size, vn1.shape[1]), dtype=vn1.dtype)
                vn1_padded = vn1_padded.at[:n_dev].set(vn1)
                vn2_padded = jnp.zeros((padded_size, vn2.shape[1]), dtype=vn2.dtype)
                vn2_padded = vn2_padded.at[:n_dev].set(vn2)

                # Pad CSR stamp positions (-1 for invalid → masked out)
                csr_pos = _csr_model_pos[model_type]
                csr_pos_padded = jnp.full(
                    (padded_size, csr_pos.shape[1]), -1, dtype=jnp.int32
                )
                csr_pos_padded = csr_pos_padded.at[:n_dev].set(csr_pos)

                # Pad res_indices (-1 for invalid → masked out)
                res_idx = stamp_indices["res_indices"]
                res_idx_padded = jnp.full(
                    (padded_size, res_idx.shape[1]), -1, dtype=jnp.int32
                )
                res_idx_padded = res_idx_padded.at[:n_dev].set(res_idx)

                _batch_info[model_type] = {
                    "n_batches": n_batches,
                    "padded_size": padded_size,
                    "device_params_padded": dp_padded,
                    "voltage_node1_padded": vn1_padded,
                    "voltage_node2_padded": vn2_padded,
                    "csr_pos_padded": csr_pos_padded,
                    "res_idx_padded": res_idx_padded,
                }

            if _batch_info:
                logger.info(
                    "CSR direct + batching: "
                    + ", ".join(
                        f"{mt}={_batch_info[mt]['n_batches']}×{batch_size}"
                        for mt in _batch_info
                    )
                )

    def _prepare_simparams(split_info, integ_c0, gmin_arg, nr_iteration):
        """Build simparams array for a model type. Shared by COO and CSR paths."""
        default_simparams = split_info["default_simparams"]
        simparam_indices = split_info.get("simparam_indices", {})

        analysis_type_val = jnp.where(integ_c0 > 0, 2.0, 0.0)
        iniLim_val = jnp.where(nr_iteration == 0, 1.0, 0.0)

        simparams = default_simparams
        if "$analysis_type" in simparam_indices:
            simparams = simparams.at[simparam_indices["$analysis_type"]].set(analysis_type_val)
        if "gmin" in simparam_indices:
            simparams = simparams.at[simparam_indices["gmin"]].set(gmin_arg)
        if "iniLim" in simparam_indices:
            simparams = simparams.at[simparam_indices["iniLim"]].set(iniLim_val)
        if "iteration" in simparam_indices:
            simparams = simparams.at[simparam_indices["iteration"]].set(
                jnp.asarray(nr_iteration, dtype=default_simparams.dtype)
            )
        return simparams

    def _get_model_limit_state_in(model_type, split_info, limit_state_in):
        """Extract model's limit_state slice. Shared by COO and CSR paths."""
        use_device_limiting = split_info.get("use_device_limiting", False)
        num_limit_states = split_info.get("num_limit_states", 0)
        n_dev = split_info["n_devices"]
        n_lim = max(1, num_limit_states)
        if (
            use_device_limiting
            and num_limit_states > 0
            and model_type in limit_state_offsets
            and limit_state_in is not None
        ):
            offset, _, n_lim = limit_state_offsets[model_type]
            return limit_state_in[offset : offset + n_dev * n_lim].reshape(n_dev, n_lim), n_lim
        return jnp.zeros((n_dev, n_lim), dtype=get_float_dtype()), n_lim

    def _stamp_vector(vec, indices, values):
        """Stamp values into dense vector at given indices, masking invalid (-1)."""
        valid = indices >= 0
        pos = jnp.where(valid, indices, 0)
        vals = jnp.where(valid, values, 0.0)
        vals = jnp.where(jnp.isnan(vals), 0.0, vals)
        return vec.at[pos].add(vals)

    def _stamp_max_abs(vec, indices, values):
        """Update max absolute value per index, masking invalid (-1)."""
        valid = indices >= 0
        pos = jnp.where(valid, indices, 0)
        vals = jnp.where(valid, jnp.abs(values), 0.0)
        vals = jnp.where(jnp.isnan(vals), 0.0, vals)
        return vec.at[pos].max(vals)

    def build_system_mna(
        X: Array,
        vsource_vals: Array,
        isource_vals: Array,
        Q_prev: Array,
        integ_c0: float | Array,
        device_arrays_arg: Dict[str, Array],
        gmin_arg: float | Array = 1e-12,
        gshunt: float | Array = 0.0,
        integ_c1: float | Array = 0.0,
        integ_d1: float | Array = 0.0,
        dQdt_prev: Array | None = None,
        integ_c2: float | Array = 0.0,
        Q_prev2: Array | None = None,
        limit_state_in: Array | None = None,
        nr_iteration: int | Array = 1,
    ) -> Tuple[Any, Array, Array, Array, Array, Array]:
        """Build augmented Jacobian J and residual f for MNA.

        Returns (J_or_csr_data, f, Q, I_vsource, limit_state_out, max_res_contrib).
        When use_csr_direct=True, J_or_csr_data is a 1D CSR data array.
        Otherwise, J_or_csr_data is a dense matrix or BCOO sparse matrix.
        """
        n_total = n_unknowns + 1
        V = X[:n_total]
        I_branch = X[n_total:] if n_vsources > 0 else jnp.zeros(0, dtype=get_float_dtype())

        limit_state_out = jnp.zeros(total_limit_states, dtype=get_float_dtype())

        if use_csr_direct:
            return _build_system_csr_direct(
                V, I_branch, vsource_vals, isource_vals, Q_prev, integ_c0,
                device_arrays_arg, gmin_arg, gshunt, integ_c1, integ_d1,
                dQdt_prev, integ_c2, Q_prev2, limit_state_in, nr_iteration,
                limit_state_out,
            )
        else:
            return _build_system_coo(
                V, I_branch, vsource_vals, isource_vals, Q_prev, integ_c0,
                device_arrays_arg, gmin_arg, gshunt, integ_c1, integ_d1,
                dQdt_prev, integ_c2, Q_prev2, limit_state_in, nr_iteration,
                limit_state_out,
            )

    def _build_system_coo(
        V, I_branch, vsource_vals, isource_vals, Q_prev, integ_c0,
        device_arrays_arg, gmin_arg, gshunt, integ_c1, integ_d1,
        dQdt_prev, integ_c2, Q_prev2, limit_state_in, nr_iteration,
        limit_state_out,
    ):
        """COO assembly path (original implementation)."""
        f_resist_parts: List[Any] = []
        f_react_parts: List[Any] = []
        j_resist_parts: List[Any] = []
        j_react_parts: List[Any] = []
        lim_rhs_resist_parts: List[Any] = []
        lim_rhs_react_parts: List[Any] = []

        # Current sources (residual only, no Jacobian)
        if "isource" in source_device_data and isource_vals.size > 0:
            d = source_device_data["isource"]
            f_vals = isource_vals[:, None] * jnp.array([1.0, -1.0])[None, :]
            f_idx = d["f_indices"].ravel()
            f_val = f_vals.ravel()
            f_resist_parts.append(mask_coo_vector(f_idx, f_val))

        # OpenVAF devices
        for model_type in model_types:
            voltage_indices, stamp_indices, voltage_node1, voltage_node2 = static_metadata[
                model_type
            ]
            cache = device_arrays_arg[model_type]
            voltage_updates = V[voltage_node1] - V[voltage_node2]

            split_info = split_eval_info[model_type]
            device_params = split_info["device_params"]
            voltage_positions = split_info["voltage_positions"]
            device_params_updated = device_params.at[:, voltage_positions].set(voltage_updates)

            if split_info["uses_analysis"]:
                analysis_type_val = jnp.where(integ_c0 > 0, 2.0, 0.0)
                device_params_updated = device_params_updated.at[:, -2].set(analysis_type_val)
                device_params_updated = device_params_updated.at[:, -1].set(gmin_arg)
            elif split_info["uses_simparam_gmin"]:
                device_params_updated = device_params_updated.at[:, -1].set(gmin_arg)

            simparams = _prepare_simparams(split_info, integ_c0, gmin_arg, nr_iteration)
            model_limit_state_in, n_lim = _get_model_limit_state_in(
                model_type, split_info, limit_state_in
            )
            n_dev = split_info["n_devices"]

            (
                batch_res_resist, batch_res_react, batch_jac_resist, batch_jac_react,
                batch_lim_rhs_resist, batch_lim_rhs_react, _, _, batch_limit_state_out,
            ) = split_info["vmapped_split_eval"](
                split_info["shared_params"], device_params_updated,
                split_info["shared_cache"], cache, simparams, model_limit_state_in,
            )

            use_device_limiting = split_info.get("use_device_limiting", False)
            if use_device_limiting and model_type in limit_state_offsets:
                offset, _, n_lim = limit_state_offsets[model_type]
                limit_state_out = limit_state_out.at[offset : offset + n_dev * n_lim].set(
                    batch_limit_state_out.ravel()
                )

            res_idx = stamp_indices["res_indices"].ravel()
            jac_row_idx = stamp_indices["jac_row_indices"].ravel()
            jac_col_idx = stamp_indices["jac_col_indices"].ravel()

            f_resist_parts.append(mask_coo_vector(res_idx, batch_res_resist.ravel()))
            f_react_parts.append(mask_coo_vector(res_idx, batch_res_react.ravel()))
            j_resist_parts.append(
                mask_coo_matrix(jac_row_idx, jac_col_idx, batch_jac_resist.ravel())
            )
            j_react_parts.append(
                mask_coo_matrix(jac_row_idx, jac_col_idx, batch_jac_react.ravel())
            )
            lim_rhs_resist_parts.append(mask_coo_vector(res_idx, batch_lim_rhs_resist.ravel()))
            lim_rhs_react_parts.append(mask_coo_vector(res_idx, batch_lim_rhs_react.ravel()))

        # Assemble vectors
        f_resist = assemble_coo_vector(f_resist_parts, n_unknowns)
        max_res_contrib = assemble_coo_max_abs(f_resist_parts, n_unknowns)
        Q = assemble_coo_vector(f_react_parts, n_unknowns)
        lim_rhs_resist = assemble_coo_vector(lim_rhs_resist_parts, n_unknowns)
        lim_rhs_react = assemble_coo_vector(lim_rhs_react_parts, n_unknowns)

        # Build residual and Jacobian
        f_resist, Q, I_vsource_kcl, f_augmented = _build_residual(
            V, I_branch, f_resist, Q, lim_rhs_resist, lim_rhs_react,
            vsource_vals, Q_prev, integ_c0, gshunt, integ_c1, integ_d1,
            dQdt_prev, integ_c2, Q_prev2,
        )

        all_j_rows, all_j_cols, all_j_vals = assemble_jacobian_coo(
            j_resist_parts, j_react_parts, integ_c0
        )
        if n_vsources > 0:
            b_rows, b_cols, b_vals = build_vsource_incidence_coo(
                vsource_node_p, vsource_node_n, n_unknowns, n_vsources
            )
            all_j_rows = jnp.concatenate([all_j_rows, b_rows])
            all_j_cols = jnp.concatenate([all_j_cols, b_cols])
            all_j_vals = jnp.concatenate([all_j_vals, b_vals])

        if use_dense:
            J = assemble_dense_jacobian(
                all_j_rows, all_j_cols, all_j_vals,
                n_augmented, n_unknowns, n_vsources, min_diag_reg, gshunt,
            )
        else:
            J = assemble_sparse_jacobian(
                all_j_rows, all_j_cols, all_j_vals,
                n_augmented, n_unknowns, min_diag_reg, gshunt,
            )

        return J, f_augmented, Q, I_vsource_kcl, limit_state_out, max_res_contrib

    def _build_system_csr_direct(
        V, I_branch, vsource_vals, isource_vals, Q_prev, integ_c0,
        device_arrays_arg, gmin_arg, gshunt, integ_c1, integ_d1,
        dQdt_prev, integ_c2, Q_prev2, limit_state_in, nr_iteration,
        limit_state_out,
    ):
        """CSR direct stamping path. Stamps device Jacobian entries directly
        into a pre-allocated CSR data array, eliminating COO intermediates."""
        dtype = get_float_dtype()
        csr_data = jnp.zeros(_csr_nse, dtype=dtype)
        f_resist = jnp.zeros(n_unknowns, dtype=dtype)
        Q = jnp.zeros(n_unknowns, dtype=dtype)
        max_res_contrib = jnp.zeros(n_unknowns, dtype=dtype)
        lim_rhs_resist = jnp.zeros(n_unknowns, dtype=dtype)
        lim_rhs_react = jnp.zeros(n_unknowns, dtype=dtype)

        # Current sources (residual only, no Jacobian)
        if "isource" in source_device_data and isource_vals.size > 0:
            d = source_device_data["isource"]
            f_vals = isource_vals[:, None] * jnp.array([1.0, -1.0])[None, :]
            f_idx = d["f_indices"].ravel()
            f_val = f_vals.ravel()
            f_resist = _stamp_vector(f_resist, f_idx, f_val)
            max_res_contrib = _stamp_max_abs(max_res_contrib, f_idx, f_val)

        # OpenVAF devices - stamp directly into CSR and dense vectors
        for model_type in model_types:
            voltage_indices, stamp_indices, voltage_node1, voltage_node2 = static_metadata[
                model_type
            ]
            cache = device_arrays_arg[model_type]
            split_info = split_eval_info[model_type]
            n_dev = split_info["n_devices"]
            use_device_limiting = split_info.get("use_device_limiting", False)

            simparams = _prepare_simparams(split_info, integ_c0, gmin_arg, nr_iteration)
            model_limit_state_in, n_lim = _get_model_limit_state_in(
                model_type, split_info, limit_state_in
            )

            model_csr_pos = _csr_model_pos[model_type]
            res_idx = stamp_indices["res_indices"]

            # Check if this model should use batched evaluation
            use_batched = (
                batch_size is not None
                and model_type in _batch_info
            )

            if use_batched:
                # Batched eval via lax.scan
                bi = _batch_info[model_type]
                n_batches = bi["n_batches"]
                dp_padded = bi["device_params_padded"]
                vn1_padded = bi["voltage_node1_padded"]
                vn2_padded = bi["voltage_node2_padded"]
                csr_pos_padded = bi["csr_pos_padded"]
                res_idx_padded = bi["res_idx_padded"]

                # Pad cache array
                dc = cache  # (n_dev, cache_dim)
                cache_padded = jnp.zeros(
                    (bi["padded_size"], dc.shape[1]), dtype=dc.dtype
                )
                cache_padded = cache_padded.at[:n_dev].set(dc)

                # Pad limit_state_in
                limit_padded = jnp.zeros(
                    (bi["padded_size"], n_lim), dtype=dtype
                )
                limit_padded = limit_padded.at[:n_dev].set(model_limit_state_in)

                shared_params = split_info["shared_params"]
                shared_cache = split_info["shared_cache"]
                voltage_positions = split_info["voltage_positions"]
                vmapped_fn = split_info["vmapped_split_eval"]
                uses_analysis = split_info["uses_analysis"]
                uses_simparam_gmin = split_info["uses_simparam_gmin"]

                def _scan_body(carry, batch_idx):
                    (c_csr, c_fr, c_Q, c_mrc, c_lrr, c_lrq) = carry
                    start = batch_idx * batch_size

                    # Dynamic slice this batch
                    b_dp = lax.dynamic_slice_in_dim(dp_padded, start, batch_size, 0)
                    b_vn1 = lax.dynamic_slice_in_dim(vn1_padded, start, batch_size, 0)
                    b_vn2 = lax.dynamic_slice_in_dim(vn2_padded, start, batch_size, 0)
                    b_cache = lax.dynamic_slice_in_dim(cache_padded, start, batch_size, 0)
                    b_limit = lax.dynamic_slice_in_dim(limit_padded, start, batch_size, 0)
                    b_csr_pos = lax.dynamic_slice_in_dim(csr_pos_padded, start, batch_size, 0)
                    b_res_idx = lax.dynamic_slice_in_dim(res_idx_padded, start, batch_size, 0)

                    # Update voltages for this batch
                    b_vu = V[b_vn1] - V[b_vn2]
                    b_dp = b_dp.at[:, voltage_positions].set(b_vu)

                    if uses_analysis:
                        atv = jnp.where(integ_c0 > 0, 2.0, 0.0)
                        b_dp = b_dp.at[:, -2].set(atv)
                        b_dp = b_dp.at[:, -1].set(gmin_arg)
                    elif uses_simparam_gmin:
                        b_dp = b_dp.at[:, -1].set(gmin_arg)

                    # Evaluate batch
                    (
                        b_rr, b_rc, b_jr, b_jc,
                        b_lrr, b_lrc, _, _, b_lim_out,
                    ) = vmapped_fn(shared_params, b_dp, shared_cache, b_cache, simparams, b_limit)

                    # Stamp Jacobian into CSR
                    combined_jac = b_jr + integ_c0 * b_jc
                    valid_j = b_csr_pos >= 0
                    j_pos = jnp.where(valid_j, b_csr_pos, 0).ravel()
                    j_vals = jnp.where(valid_j, combined_jac, 0.0).ravel()
                    j_vals = jnp.where(jnp.isnan(j_vals), 0.0, j_vals)
                    c_csr = c_csr.at[j_pos].add(j_vals)

                    # Stamp residuals and charges
                    b_res_flat = b_res_idx.ravel()
                    c_fr = _stamp_vector(c_fr, b_res_flat, b_rr.ravel())
                    c_mrc = _stamp_max_abs(c_mrc, b_res_flat, b_rr.ravel())
                    c_Q = _stamp_vector(c_Q, b_res_flat, b_rc.ravel())
                    c_lrr = _stamp_vector(c_lrr, b_res_flat, b_lrr.ravel())
                    c_lrq = _stamp_vector(c_lrq, b_res_flat, b_lrc.ravel())

                    return (c_csr, c_fr, c_Q, c_mrc, c_lrr, c_lrq), b_lim_out

                init_carry = (csr_data, f_resist, Q, max_res_contrib, lim_rhs_resist, lim_rhs_react)
                (csr_data, f_resist, Q, max_res_contrib, lim_rhs_resist, lim_rhs_react), all_lim = (
                    lax.scan(_scan_body, init_carry, jnp.arange(n_batches))
                )

                # Store limit_state_out: all_lim is (n_batches, batch_size, n_lim)
                if use_device_limiting and model_type in limit_state_offsets:
                    offset, _, n_lim_off = limit_state_offsets[model_type]
                    # Reshape and trim to n_dev
                    all_lim_flat = all_lim.reshape(-1, n_lim)[:n_dev]
                    limit_state_out = limit_state_out.at[
                        offset : offset + n_dev * n_lim_off
                    ].set(all_lim_flat.ravel())

            else:
                # Non-batched: evaluate all devices at once, stamp directly
                voltage_updates = V[voltage_node1] - V[voltage_node2]
                device_params = split_info["device_params"]
                voltage_positions = split_info["voltage_positions"]
                device_params_updated = device_params.at[:, voltage_positions].set(voltage_updates)

                if split_info["uses_analysis"]:
                    analysis_type_val = jnp.where(integ_c0 > 0, 2.0, 0.0)
                    device_params_updated = device_params_updated.at[:, -2].set(analysis_type_val)
                    device_params_updated = device_params_updated.at[:, -1].set(gmin_arg)
                elif split_info["uses_simparam_gmin"]:
                    device_params_updated = device_params_updated.at[:, -1].set(gmin_arg)

                (
                    res_r, res_c, jac_r, jac_c,
                    lim_r, lim_c, _, _, batch_limit_state_out,
                ) = split_info["vmapped_split_eval"](
                    split_info["shared_params"], device_params_updated,
                    split_info["shared_cache"], cache, simparams, model_limit_state_in,
                )

                if use_device_limiting and model_type in limit_state_offsets:
                    offset, _, n_lim_off = limit_state_offsets[model_type]
                    limit_state_out = limit_state_out.at[
                        offset : offset + n_dev * n_lim_off
                    ].set(batch_limit_state_out.ravel())

                # Stamp Jacobian into CSR
                combined_jac = jac_r + integ_c0 * jac_c
                valid_j = model_csr_pos >= 0
                j_pos = jnp.where(valid_j, model_csr_pos, 0).ravel()
                j_vals = jnp.where(valid_j, combined_jac, 0.0).ravel()
                j_vals = jnp.where(jnp.isnan(j_vals), 0.0, j_vals)
                csr_data = csr_data.at[j_pos].add(j_vals)

                # Stamp residuals and charges
                res_flat = res_idx.ravel()
                f_resist = _stamp_vector(f_resist, res_flat, res_r.ravel())
                max_res_contrib = _stamp_max_abs(max_res_contrib, res_flat, res_r.ravel())
                Q = _stamp_vector(Q, res_flat, res_c.ravel())
                lim_rhs_resist = _stamp_vector(lim_rhs_resist, res_flat, lim_r.ravel())
                lim_rhs_react = _stamp_vector(lim_rhs_react, res_flat, lim_c.ravel())

        # Add diagonal regularization to CSR (gmin + gshunt on node equations)
        csr_data = csr_data.at[_csr_diag_pos].add(min_diag_reg + gshunt)

        # Add vsource B/B^T incidence entries to CSR
        if n_vsources > 0 and _csr_vs_pos is not None:
            vp = _csr_vs_pos
            valid_p = vp["valid_p"]
            valid_n = vp["valid_n"]
            # B block: +1 at positive, -1 at negative
            csr_data = csr_data.at[jnp.where(valid_p, vp["b_p"], 0)].add(
                jnp.where(valid_p, 1.0, 0.0)
            )
            csr_data = csr_data.at[jnp.where(valid_n, vp["b_n"], 0)].add(
                jnp.where(valid_n, -1.0, 0.0)
            )
            # B^T block: +1 at positive, -1 at negative
            csr_data = csr_data.at[jnp.where(valid_p, vp["bt_p"], 0)].add(
                jnp.where(valid_p, 1.0, 0.0)
            )
            csr_data = csr_data.at[jnp.where(valid_n, vp["bt_n"], 0)].add(
                jnp.where(valid_n, -1.0, 0.0)
            )

        # Build residual (same as COO path)
        f_resist, Q, I_vsource_kcl, f_augmented = _build_residual(
            V, I_branch, f_resist, Q, lim_rhs_resist, lim_rhs_react,
            vsource_vals, Q_prev, integ_c0, gshunt, integ_c1, integ_d1,
            dQdt_prev, integ_c2, Q_prev2,
        )

        return csr_data, f_augmented, Q, I_vsource_kcl, limit_state_out, max_res_contrib

    def _build_residual(
        V, I_branch, f_resist, Q, lim_rhs_resist, lim_rhs_react,
        vsource_vals, Q_prev, integ_c0, gshunt, integ_c1, integ_d1,
        dQdt_prev, integ_c2, Q_prev2,
    ):
        """Build residual vector from assembled contributions.

        Shared by both COO and CSR direct paths.

        Returns:
            f_resist: Updated resistive residual (after lim_rhs subtraction + branch contrib)
            Q: Charge vector (unchanged)
            I_vsource_kcl: Vsource currents from KCL
            f_augmented: Full augmented residual [f_node; f_branch]
        """
        f_resist = f_resist - lim_rhs_resist

        I_vsource_kcl = compute_vsource_current_from_kcl(f_resist, vsource_node_p)

        _dQdt_prev = (
            dQdt_prev if dQdt_prev is not None else jnp.zeros(n_unknowns, dtype=get_float_dtype())
        )
        _Q_prev2 = (
            Q_prev2 if Q_prev2 is not None else jnp.zeros(n_unknowns, dtype=get_float_dtype())
        )

        if n_vsources > 0:
            f_branch_contrib = build_vsource_kcl_contribution(
                I_branch, vsource_node_p, vsource_node_n, n_unknowns
            )
            f_resist = f_resist + f_branch_contrib

        effective_shunt = min_diag_reg + gshunt
        f_node = combine_transient_residual(
            f_resist, Q, jnp.zeros_like(f_resist), lim_rhs_react,
            Q_prev, integ_c0, integ_c1, integ_d1, _dQdt_prev, integ_c2, _Q_prev2,
            effective_shunt, V[1:],
        )

        f_branch = build_vsource_equations(V, vsource_vals, vsource_node_p, vsource_node_n)
        f_augmented = jnp.concatenate([f_node, f_branch])

        return f_resist, Q, I_vsource_kcl, f_augmented

    return build_system_mna, device_arrays, total_limit_states
