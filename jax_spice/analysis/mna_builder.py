"""MNA system builder for JAX-SPICE.

This module provides the factory function that creates the GPU-resident
build_system function for Modified Nodal Analysis (MNA) formulation.

The build_system function is the core of the Newton-Raphson solver - it
evaluates all devices, assembles residuals and Jacobians, and returns
the system for solving.
"""

from typing import Any, Callable, Dict, List, Tuple

import jax.numpy as jnp
import numpy as np
from jax import Array

from jax_spice import get_float_dtype
from jax_spice.analysis.mna import (
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


def make_mna_build_system_fn(
    source_device_data: Dict[str, Any],
    static_inputs_cache: Dict[str, Tuple],
    compiled_models: Dict[str, Dict[str, Any]],
    gmin: float,
    n_unknowns: int,
    use_dense: bool = True,
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

    def build_system_mna(
        X: Array,  # Augmented solution: [V; I_branch] of size n_total + n_vsources
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

        The solution vector X has structure: [V1, V2, ..., Vn, I_vs1, I_vs2, ..., I_vsm]
        where V are node voltages (ground excluded) and I_vs are branch currents.

        Args:
            X: Augmented solution vector of size n_total + n_vsources
               X[:n_total] = node voltages (including ground at index 0)
               X[n_total:] = branch currents for voltage sources
            vsource_vals: Voltage source target values
            isource_vals: Current source values
            Q_prev: Charges from previous timestep
            integ_c0: Integration coefficient for current charges
            device_arrays_arg: Device cache arrays
            gmin_arg, gshunt: Regularization parameters
            integ_c1, integ_d1, dQdt_prev, integ_c2, Q_prev2: Integration history
            limit_state_in: Flat array of all limit states from previous iteration
            nr_iteration: Current NR iteration number (1-based). Used for iniLim/iteration simparams.

        Returns:
            J: Augmented Jacobian matrix (n_unknowns+n_vsources)²
            f: Augmented residual vector (n_unknowns+n_vsources)
            Q: Current charges (n_unknowns)
            I_vsource: Branch currents computed from KCL
            limit_state_out: Updated limit states
            max_res_contrib: Max absolute device current per node (n_unknowns)
        """
        # Extract voltage and current parts from augmented solution
        # X has structure: [V_ground=0, V_1, ..., V_n, I_vs1, ..., I_vsm]
        n_total = n_unknowns + 1  # Total nodes including ground
        V = X[:n_total]
        I_branch = X[n_total:] if n_vsources > 0 else jnp.zeros(0, dtype=get_float_dtype())

        # =====================================================================
        # Device contributions
        # =====================================================================
        f_resist_parts: List[Any] = []
        f_react_parts: List[Any] = []
        j_resist_parts: List[Any] = []
        j_react_parts: List[Any] = []
        lim_rhs_resist_parts: List[Any] = []
        lim_rhs_react_parts: List[Any] = []

        # Pre-allocate limit_state_out with static size
        limit_state_out = jnp.zeros(total_limit_states, dtype=get_float_dtype())

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
            uses_analysis = split_info["uses_analysis"]
            uses_simparam_gmin = split_info["uses_simparam_gmin"]
            shared_params = split_info["shared_params"]
            device_params = split_info["device_params"]
            voltage_positions = split_info["voltage_positions"]
            shared_cache = split_info["shared_cache"]

            device_params_updated = device_params.at[:, voltage_positions].set(voltage_updates)

            if uses_analysis:
                analysis_type_val = jnp.where(integ_c0 > 0, 2.0, 0.0)
                device_params_updated = device_params_updated.at[:, -2].set(analysis_type_val)
                device_params_updated = device_params_updated.at[:, -1].set(gmin_arg)
            elif uses_simparam_gmin:
                device_params_updated = device_params_updated.at[:, -1].set(gmin_arg)

            vmapped_split_eval = split_info["vmapped_split_eval"]
            default_simparams = split_info["default_simparams"]
            simparam_indices = split_info.get("simparam_indices", {})
            use_device_limiting = split_info.get("use_device_limiting", False)
            num_limit_states = split_info.get("num_limit_states", 0)

            # Build simparams using correct indices from the model's metadata
            analysis_type_val = jnp.where(integ_c0 > 0, 2.0, 0.0)
            # VACASK: iniLim=1 on first NR iteration, 0 otherwise (coreopnr.cpp:716)
            # nr_iteration is 0-indexed from solver (starts at 0, incremented after eval)
            iniLim_val = jnp.where(nr_iteration == 0, 1.0, 0.0)

            simparams = default_simparams
            # Set analysis_type if present
            if "$analysis_type" in simparam_indices:
                simparams = simparams.at[simparam_indices["$analysis_type"]].set(analysis_type_val)
            # Set gmin if present
            if "gmin" in simparam_indices:
                simparams = simparams.at[simparam_indices["gmin"]].set(gmin_arg)
            # Set iniLim if present (VACASK initialize_limiting support)
            if "iniLim" in simparam_indices:
                simparams = simparams.at[simparam_indices["iniLim"]].set(iniLim_val)
            # Set iteration if present
            if "iteration" in simparam_indices:
                simparams = simparams.at[simparam_indices["iteration"]].set(
                    jnp.asarray(nr_iteration, dtype=default_simparams.dtype)
                )

            # Get limit_state slice for this model type
            n_dev = split_info["n_devices"]
            n_lim = max(1, num_limit_states)
            if (
                use_device_limiting
                and num_limit_states > 0
                and model_type in limit_state_offsets
                and limit_state_in is not None
            ):
                offset, _, n_lim = limit_state_offsets[model_type]
                model_limit_state_in = limit_state_in[offset : offset + n_dev * n_lim].reshape(
                    n_dev, n_lim
                )
            else:
                model_limit_state_in = jnp.zeros((n_dev, n_lim), dtype=get_float_dtype())

            # Evaluate devices
            (
                batch_res_resist,
                batch_res_react,
                batch_jac_resist,
                batch_jac_react,
                batch_lim_rhs_resist,
                batch_lim_rhs_react,
                _,
                _,
                batch_limit_state_out,
            ) = vmapped_split_eval(
                shared_params,
                device_params_updated,
                shared_cache,
                cache,
                simparams,
                model_limit_state_in,
            )

            # Store limit_state_out at pre-computed offset
            if use_device_limiting and model_type in limit_state_offsets:
                offset, _, n_lim = limit_state_offsets[model_type]
                limit_state_out = limit_state_out.at[offset : offset + n_dev * n_lim].set(
                    batch_limit_state_out.ravel()
                )

            # Extract stamp indices (flattened for COO format)
            res_idx = stamp_indices["res_indices"].ravel()
            jac_row_idx = stamp_indices["jac_row_indices"].ravel()
            jac_col_idx = stamp_indices["jac_col_indices"].ravel()

            # Mask and collect residual contributions
            f_resist_parts.append(mask_coo_vector(res_idx, batch_res_resist.ravel()))
            f_react_parts.append(mask_coo_vector(res_idx, batch_res_react.ravel()))

            # Mask and collect Jacobian contributions
            j_resist_parts.append(
                mask_coo_matrix(jac_row_idx, jac_col_idx, batch_jac_resist.ravel())
            )
            j_react_parts.append(mask_coo_matrix(jac_row_idx, jac_col_idx, batch_jac_react.ravel()))

            # Mask and collect limiting RHS contributions
            lim_rhs_resist_parts.append(mask_coo_vector(res_idx, batch_lim_rhs_resist.ravel()))
            lim_rhs_react_parts.append(mask_coo_vector(res_idx, batch_lim_rhs_react.ravel()))

        # =====================================================================
        # Assemble vectors and matrices
        # =====================================================================
        f_resist = assemble_coo_vector(f_resist_parts, n_unknowns)
        max_res_contrib = assemble_coo_max_abs(f_resist_parts, n_unknowns)
        Q = assemble_coo_vector(f_react_parts, n_unknowns)
        lim_rhs_resist = assemble_coo_vector(lim_rhs_resist_parts, n_unknowns)
        lim_rhs_react = assemble_coo_vector(lim_rhs_react_parts, n_unknowns)

        f_resist = f_resist - lim_rhs_resist

        # Compute I_vsource from device residuals via KCL (before adding I_branch)
        I_vsource_kcl = compute_vsource_current_from_kcl(f_resist, vsource_node_p)

        _dQdt_prev = (
            dQdt_prev if dQdt_prev is not None else jnp.zeros(n_unknowns, dtype=get_float_dtype())
        )
        _Q_prev2 = (
            Q_prev2 if Q_prev2 is not None else jnp.zeros(n_unknowns, dtype=get_float_dtype())
        )

        # Add branch current contribution to KCL at vsource nodes
        if n_vsources > 0:
            f_branch_contrib = build_vsource_kcl_contribution(
                I_branch, vsource_node_p, vsource_node_n, n_unknowns
            )
            f_resist = f_resist + f_branch_contrib

        # Combine for transient.
        # Use min_diag_reg + gshunt as the effective shunt so that the residual
        # is consistent with the Jacobian diagonal (which has min_diag_reg + gshunt).
        effective_shunt = min_diag_reg + gshunt
        f_node = combine_transient_residual(
            f_resist,
            Q,
            jnp.zeros_like(f_resist),
            lim_rhs_react,
            Q_prev,
            integ_c0,
            integ_c1,
            integ_d1,
            _dQdt_prev,
            integ_c2,
            _Q_prev2,
            effective_shunt,
            V[1:],
        )

        # Voltage source equations: V_p - V_n - E = 0
        f_branch = build_vsource_equations(V, vsource_vals, vsource_node_p, vsource_node_n)

        # Combine node and branch residuals
        f_augmented = jnp.concatenate([f_node, f_branch])

        # =====================================================================
        # Build Jacobian
        # =====================================================================
        all_j_rows, all_j_cols, all_j_vals = assemble_jacobian_coo(
            j_resist_parts, j_react_parts, integ_c0
        )

        # Add B and B^T blocks for voltage sources
        if n_vsources > 0:
            b_rows, b_cols, b_vals = build_vsource_incidence_coo(
                vsource_node_p, vsource_node_n, n_unknowns, n_vsources
            )
            all_j_rows = jnp.concatenate([all_j_rows, b_rows])
            all_j_cols = jnp.concatenate([all_j_cols, b_cols])
            all_j_vals = jnp.concatenate([all_j_vals, b_vals])

        # Assemble final Jacobian matrix
        if use_dense:
            J = assemble_dense_jacobian(
                all_j_rows,
                all_j_cols,
                all_j_vals,
                n_augmented,
                n_unknowns,
                n_vsources,
                min_diag_reg,
                gshunt,
            )
        else:
            J = assemble_sparse_jacobian(
                all_j_rows,
                all_j_cols,
                all_j_vals,
                n_augmented,
                n_unknowns,
                min_diag_reg,
                gshunt,
            )

        return J, f_augmented, Q, I_vsource_kcl, limit_state_out, max_res_contrib

    return build_system_mna, device_arrays, total_limit_states
