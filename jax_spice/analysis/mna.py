"""Modified Nodal Analysis (MNA) types for JAX-SPICE

This module provides core types used by the VACASK benchmark runner.
The actual MNA matrix assembly is done in the runner using OpenVAF-compiled models.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

# =============================================================================
# Parameter Evaluation
# =============================================================================


def eval_param_simple(value, vdd: float = 1.2, defaults: dict = None):
    """Simple parameter evaluation for common cases.

    Handles:
    - Numbers (int, float)
    - String 'vdd' -> vdd value
    - String '0' -> 0.0
    - String 'w', 'l' -> default MOSFET dimensions
    - SPICE number suffixes (1u, 100n, etc.)
    """
    if defaults is None:
        defaults = {
            "w": 1e-6,  # Default MOSFET width = 1u
            "l": 0.2e-6,  # Default MOSFET length = 0.2u
            "ld": 0.5e-6,  # Default drain extension
            "ls": 0.5e-6,  # Default source extension
        }

    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, str):
        value_lower = value.lower().strip()

        # Common parameter references
        if value_lower == "vdd":
            return vdd
        if value_lower in ("0", "0.0", "vss", "gnd"):
            return 0.0
        if value_lower in defaults:
            return defaults[value_lower]

        # SPICE number suffixes
        suffixes = {
            "t": 1e12,
            "g": 1e9,
            "meg": 1e6,
            "k": 1e3,
            "m": 1e-3,
            "u": 1e-6,
            "n": 1e-9,
            "p": 1e-12,
            "f": 1e-15,
        }
        for suffix, mult in sorted(suffixes.items(), key=lambda x: -len(x[0])):
            if value_lower.endswith(suffix):
                try:
                    return float(value_lower[: -len(suffix)]) * mult
                except ValueError:
                    pass

        # Try direct conversion
        try:
            return float(value)
        except ValueError:
            pass

    return 0.0


# =============================================================================
# Device Type Enumeration
# =============================================================================


class DeviceType(Enum):
    """Enumeration of supported device types.

    All non-source devices use OpenVAF (VERILOG_A).
    VSOURCE and ISOURCE are handled with simple large-conductance models.
    """

    VSOURCE = "vsource"
    ISOURCE = "isource"
    VERILOG_A = (
        "verilog_a"  # All OpenVAF-compiled devices (resistor, capacitor, diode, psp103, etc.)
    )


# =============================================================================
# Device Info
# =============================================================================


@dataclass
class DeviceInfo:
    """Information about a device instance for simulation."""

    name: str
    model_name: str
    terminals: List[str]  # Terminal names
    node_indices: List[int]  # Corresponding node indices
    params: Dict[str, Any]  # Instance parameters
    eval_fn: Optional[Callable] = None  # Device evaluation function
    is_openvaf: bool = False  # True if device uses OpenVAF-compiled Verilog-A model


# =============================================================================
# Full MNA Branch Current Structures
# =============================================================================


@dataclass
class BranchInfo:
    """Information about a branch current unknown in full MNA formulation.

    In full MNA, voltage sources have their currents as explicit unknowns.
    The augmented system becomes:

        ┌───────────────┐   ┌───┐   ┌───┐
        │     G     B   │   │ V │   │ I │
        │               │ × │   │ = │   │
        │    B^T    0   │   │ J │   │ E │
        └───────────────┘   └───┘   └───┘

    Where:
    - G = device conductance matrix (n×n)
    - B = incidence matrix mapping branch currents to nodes (n×m)
    - V = node voltages (n×1)
    - J = branch currents (m×1) - these are the primary unknowns for vsources
    - I = device current sources (n×1)
    - E = voltage source values (m×1)

    Attributes:
        name: Source device name (e.g., 'vdd', 'v1')
        node_p: Positive terminal node index
        node_n: Negative terminal node index
        branch_idx: Index into the branch current solution vector
        dc_value: DC voltage value (for initialization)
    """

    name: str
    node_p: int
    node_n: int
    branch_idx: int
    dc_value: float = 0.0


@dataclass
class MNABranchData:
    """Data structure for full MNA branch currents.

    Contains all information needed to:
    1. Build the augmented MNA matrix with B and B^T blocks
    2. Extract branch currents from the solution vector
    3. Map source names to their branch current indices

    Attributes:
        n_branches: Number of branch current unknowns (= number of vsources)
        branches: List of BranchInfo for each voltage source
        name_to_idx: Mapping from source name to branch index
        node_p: Array of positive terminal indices for all vsources
        node_n: Array of negative terminal indices for all vsources
        dc_values: Array of DC voltage values
    """

    n_branches: int
    branches: List[BranchInfo]
    name_to_idx: Dict[str, int]
    node_p: List[int]
    node_n: List[int]
    dc_values: List[float]

    @classmethod
    def from_devices(cls, devices: List[Dict], node_names: Dict[str, int]) -> "MNABranchData":
        """Create MNABranchData from device list.

        Args:
            devices: List of device dicts with 'model', 'name', 'nodes', 'params'
            node_names: Mapping from node name to node index

        Returns:
            MNABranchData with branch info for all voltage sources
        """
        branches = []
        name_to_idx = {}
        node_p_list = []
        node_n_list = []
        dc_values = []

        branch_idx = 0
        for dev in devices:
            if dev.get("model") == "vsource":
                name = dev["name"]
                nodes = dev["nodes"]
                params = dev.get("params", {})

                # Get node indices (nodes are already indices in most cases)
                if len(nodes) >= 2:
                    p_node = (
                        nodes[0] if isinstance(nodes[0], int) else node_names.get(str(nodes[0]), 0)
                    )
                    n_node = (
                        nodes[1] if isinstance(nodes[1], int) else node_names.get(str(nodes[1]), 0)
                    )
                else:
                    p_node = (
                        nodes[0] if isinstance(nodes[0], int) else node_names.get(str(nodes[0]), 0)
                    )
                    n_node = 0  # Ground

                dc_val = float(params.get("dc", 0.0))

                branch = BranchInfo(
                    name=name, node_p=p_node, node_n=n_node, branch_idx=branch_idx, dc_value=dc_val
                )
                branches.append(branch)
                name_to_idx[name] = branch_idx
                node_p_list.append(p_node)
                node_n_list.append(n_node)
                dc_values.append(dc_val)
                branch_idx += 1

        return cls(
            n_branches=len(branches),
            branches=branches,
            name_to_idx=name_to_idx,
            node_p=node_p_list,
            node_n=node_n_list,
            dc_values=dc_values,
        )


# =============================================================================
# COO Assembly Helpers for MNA System Building
# =============================================================================
#
# These helpers reduce code duplication in the MNA build_system function by
# providing common patterns for:
# - Masking invalid COO entries (index < 0 or NaN values)
# - Accumulating COO contributions via segment_sum
# - Building incidence matrix entries for voltage sources
#
# Array Conventions:
#   circuit_node_idx → MNA_idx: mna_idx = circuit_idx - 1
#   MNA_idx → circuit_node_idx: circuit_idx = mna_idx + 1
#   Ground (circuit index 0) is excluded from MNA equations

from typing import NamedTuple, Tuple

import jax
import jax.numpy as jnp
from jax import Array

from jax_spice import get_float_dtype


class COOVector(NamedTuple):
    """COO-format contribution to a vector (residual or charge).

    Attributes:
        indices: Target positions in the vector, shape (n_entries,).
                 -1 indicates invalid entry (will be masked out).
        values: Values to accumulate at each position, shape (n_entries,).
    """

    indices: Array
    values: Array


class COOMatrix(NamedTuple):
    """COO-format contribution to a matrix (Jacobian).

    Attributes:
        rows: Row indices, shape (n_entries,). -1 indicates invalid.
        cols: Column indices, shape (n_entries,). -1 indicates invalid.
        values: Values to accumulate, shape (n_entries,).
    """

    rows: Array
    cols: Array
    values: Array


def mask_coo_vector(indices: Array, values: Array) -> COOVector:
    """Apply validity mask and NaN handling to COO vector data.

    Entries with index < 0 are masked out (set to index 0, value 0).
    NaN values are replaced with 0.

    Args:
        indices: Raw indices from device evaluation, shape (n,)
        values: Raw values from device evaluation, shape (n,)

    Returns:
        COOVector with masked indices and values
    """
    valid = indices >= 0
    masked_idx = jnp.where(valid, indices, 0)
    masked_val = jnp.where(valid, values, 0.0)
    masked_val = jnp.where(jnp.isnan(masked_val), 0.0, masked_val)
    return COOVector(masked_idx, masked_val)


def mask_coo_matrix(rows: Array, cols: Array, values: Array) -> COOMatrix:
    """Apply validity mask and NaN handling to COO matrix data.

    Entries with row < 0 or col < 0 are masked out.
    NaN values are replaced with 0.

    Args:
        rows: Raw row indices, shape (n,)
        cols: Raw column indices, shape (n,)
        values: Raw values, shape (n,)

    Returns:
        COOMatrix with masked rows, cols, and values
    """
    valid = (rows >= 0) & (cols >= 0)
    masked_rows = jnp.where(valid, rows, 0)
    masked_cols = jnp.where(valid, cols, 0)
    masked_vals = jnp.where(valid, values, 0.0)
    masked_vals = jnp.where(jnp.isnan(masked_vals), 0.0, masked_vals)
    return COOMatrix(masked_rows, masked_cols, masked_vals)


def assemble_coo_vector(parts: List[COOVector], size: int) -> Array:
    """Assemble a vector from COO contributions using segment_sum.

    Accumulates all contributions into a dense vector. Multiple contributions
    to the same index are summed.

    Args:
        parts: List of COOVector contributions
        size: Size of the output vector

    Returns:
        Dense vector of shape (size,)
    """
    if not parts:
        return jnp.zeros(size, dtype=get_float_dtype())

    all_idx = jnp.concatenate([p.indices for p in parts])
    all_val = jnp.concatenate([p.values for p in parts])
    return jax.ops.segment_sum(all_val, all_idx, num_segments=size)


def assemble_coo_max_abs(parts: List[COOVector], size: int) -> Array:
    """Max absolute contribution per index (VACASK maxResidualContribution).

    For each vector index, computes the maximum absolute value across all
    COO contributions. This is used for VACASK-style relative residual
    tolerance: tol[i] = max(maxResContrib[i] * reltol, abstol).

    Nodes with no device contributions get 0.0, so their tolerance falls
    back to abstol (since max(0 * reltol, abstol) = abstol).

    Args:
        parts: List of COOVector contributions
        size: Size of the output vector

    Returns:
        Dense vector of shape (size,) with max |value| per index
    """
    if not parts:
        return jnp.zeros(size, dtype=get_float_dtype())

    all_idx = jnp.concatenate([p.indices for p in parts])
    all_val = jnp.concatenate([jnp.abs(p.values) for p in parts])
    # segment_max returns -inf for empty segments; clamp to 0
    raw_max = jax.ops.segment_max(all_val, all_idx, num_segments=size)
    return jnp.maximum(raw_max, 0.0)


def assemble_jacobian_coo(
    resist_parts: List[COOMatrix],
    react_parts: List[COOMatrix],
    integ_c0: float | Array,
) -> Tuple[Array, Array, Array]:
    """Combine resistive and reactive Jacobian contributions.

    Produces combined COO arrays: J = G + c0*C where G is from resist_parts
    and C is from react_parts.

    Args:
        resist_parts: Resistive (conductance) Jacobian contributions
        react_parts: Reactive (capacitance) Jacobian contributions
        integ_c0: Integration coefficient (0 for DC, 1/dt for transient)

    Returns:
        Tuple of (rows, cols, values) arrays for combined Jacobian
    """
    # Collect resistive contributions
    if resist_parts:
        all_resist_rows = jnp.concatenate([p.rows for p in resist_parts])
        all_resist_cols = jnp.concatenate([p.cols for p in resist_parts])
        all_resist_vals = jnp.concatenate([p.values for p in resist_parts])
    else:
        all_resist_rows = jnp.zeros(0, dtype=jnp.int32)
        all_resist_cols = jnp.zeros(0, dtype=jnp.int32)
        all_resist_vals = jnp.zeros(0, dtype=get_float_dtype())

    # Collect reactive contributions (scaled by integ_c0)
    if react_parts:
        all_react_rows = jnp.concatenate([p.rows for p in react_parts])
        all_react_cols = jnp.concatenate([p.cols for p in react_parts])
        all_react_vals = jnp.concatenate([p.values for p in react_parts])
    else:
        all_react_rows = jnp.zeros(0, dtype=jnp.int32)
        all_react_cols = jnp.zeros(0, dtype=jnp.int32)
        all_react_vals = jnp.zeros(0, dtype=get_float_dtype())

    # Combine: J = G + c0*C
    all_rows = jnp.concatenate([all_resist_rows, all_react_rows])
    all_cols = jnp.concatenate([all_resist_cols, all_react_cols])
    all_vals = jnp.concatenate([all_resist_vals, integ_c0 * all_react_vals])

    return all_rows, all_cols, all_vals


def build_vsource_incidence_coo(
    vsource_node_p: Array,
    vsource_node_n: Array,
    n_unknowns: int,
    n_vsources: int,
) -> Tuple[Array, Array, Array]:
    """Build COO entries for B and B^T blocks of the MNA matrix.

    The incidence matrix B maps branch currents to node KCL equations:
    - At positive terminal: +I flows into the node
    - At negative terminal: -I flows out of the node

    B^T provides the voltage constraint equations: V_p - V_n = E

    Args:
        vsource_node_p: Positive terminal circuit node indices, shape (m,)
        vsource_node_n: Negative terminal circuit node indices, shape (m,)
        n_unknowns: Number of node voltage unknowns
        n_vsources: Number of voltage sources (m)

    Returns:
        Tuple of (rows, cols, values) for combined B and B^T entries
    """
    if n_vsources == 0:
        empty = jnp.zeros(0, dtype=jnp.int32)
        empty_f = jnp.zeros(0, dtype=get_float_dtype())
        return empty, empty, empty_f

    # Validity masks (ground node = 0 doesn't contribute to MNA equations)
    valid_p = vsource_node_p > 0
    valid_n = vsource_node_n > 0

    branch_indices = jnp.arange(n_vsources, dtype=jnp.int32)

    # B block entries: df_node/dI_branch
    # At node p: df/dI = +1
    b_rows_p = jnp.where(valid_p, vsource_node_p - 1, 0)  # MNA index
    b_cols_p = jnp.where(valid_p, n_unknowns + branch_indices, 0)
    b_vals_p = jnp.where(valid_p, 1.0, 0.0)

    # At node n: df/dI = -1
    b_rows_n = jnp.where(valid_n, vsource_node_n - 1, 0)
    b_cols_n = jnp.where(valid_n, n_unknowns + branch_indices, 0)
    b_vals_n = jnp.where(valid_n, -1.0, 0.0)

    # B^T block entries: df_branch/dV
    # For vsource i: df_i/dV_p = +1, df_i/dV_n = -1
    bt_rows_p = jnp.where(valid_p, n_unknowns + branch_indices, 0)
    bt_cols_p = jnp.where(valid_p, vsource_node_p - 1, 0)
    bt_vals_p = jnp.where(valid_p, 1.0, 0.0)

    bt_rows_n = jnp.where(valid_n, n_unknowns + branch_indices, 0)
    bt_cols_n = jnp.where(valid_n, vsource_node_n - 1, 0)
    bt_vals_n = jnp.where(valid_n, -1.0, 0.0)

    # Combine all entries
    all_rows = jnp.concatenate([b_rows_p, b_rows_n, bt_rows_p, bt_rows_n])
    all_cols = jnp.concatenate([b_cols_p, b_cols_n, bt_cols_p, bt_cols_n])
    all_vals = jnp.concatenate([b_vals_p, b_vals_n, bt_vals_p, bt_vals_n])

    return all_rows, all_cols, all_vals


def build_vsource_kcl_contribution(
    I_branch: Array,
    vsource_node_p: Array,
    vsource_node_n: Array,
    n_unknowns: int,
) -> Array:
    """Compute branch current contribution to node KCL equations.

    For each voltage source connecting nodes p and n:
    - At node p: add +I_branch (current flows in)
    - At node n: add -I_branch (current flows out)

    Args:
        I_branch: Branch currents, shape (m,)
        vsource_node_p: Positive terminal circuit indices, shape (m,)
        vsource_node_n: Negative terminal circuit indices, shape (m,)
        n_unknowns: Number of node voltage unknowns

    Returns:
        KCL contribution vector of shape (n_unknowns,)
    """
    n_vsources = I_branch.shape[0]
    if n_vsources == 0:
        return jnp.zeros(n_unknowns, dtype=get_float_dtype())

    # Convert to MNA indices (0-indexed)
    p_mna = vsource_node_p - 1
    n_mna = vsource_node_n - 1

    # Validity masks
    valid_p = vsource_node_p > 0
    valid_n = vsource_node_n > 0

    # Build COO entries
    all_idx = jnp.concatenate(
        [
            jnp.where(valid_p, p_mna, 0),
            jnp.where(valid_n, n_mna, 0),
        ]
    )
    all_val = jnp.concatenate(
        [
            jnp.where(valid_p, I_branch, 0.0),
            jnp.where(valid_n, -I_branch, 0.0),
        ]
    )

    return jax.ops.segment_sum(all_val, all_idx, num_segments=n_unknowns)


def build_vsource_equations(
    V: Array,
    vsource_vals: Array,
    vsource_node_p: Array,
    vsource_node_n: Array,
) -> Array:
    """Build voltage source constraint equations: V_p - V_n - E = 0.

    Args:
        V: Full voltage array including ground, shape (n_nodes,)
        vsource_vals: Target voltage values, shape (m,)
        vsource_node_p: Positive terminal circuit indices, shape (m,)
        vsource_node_n: Negative terminal circuit indices, shape (m,)

    Returns:
        Residual for voltage source equations, shape (m,)
    """
    if vsource_vals.size == 0:
        return jnp.zeros(0, dtype=get_float_dtype())

    Vp = V[vsource_node_p]
    Vn = V[vsource_node_n]
    return Vp - Vn - vsource_vals


def compute_vsource_current_from_kcl(
    f_device: Array,
    vsource_node_p: Array,
) -> Array:
    """Compute voltage source currents from device KCL residuals.

    By KCL, the sum of currents at any node is zero. The voltage source
    current equals the negative of all other device currents at its
    positive terminal.

    This is more accurate than using I_branch from the solution vector,
    especially for initial conditions (UIC) where I_branch may be zero.

    Args:
        f_device: Device contribution to residuals (before adding I_branch),
                  shape (n_unknowns,)
        vsource_node_p: Positive terminal circuit indices, shape (m,)

    Returns:
        Voltage source currents, shape (m,)
    """
    if vsource_node_p.size == 0:
        return jnp.zeros(0, dtype=get_float_dtype())

    # Convert to MNA index (0-indexed)
    vsource_node_p_mna = vsource_node_p - 1

    # Handle ground (index 0) - contribution is 0
    valid_nodes = vsource_node_p > 0
    f_at_p = jnp.where(valid_nodes, f_device[vsource_node_p_mna], 0.0)

    # I_vsource = -f_device (by KCL: sum of currents = 0)
    return -f_at_p


def assemble_dense_jacobian(
    j_rows: Array,
    j_cols: Array,
    j_vals: Array,
    n_augmented: int,
    n_unknowns: int,
    n_vsources: int,
    min_diag_reg: float,
    gshunt: float | Array,
) -> Array:
    """Assemble dense Jacobian matrix from COO data.

    Args:
        j_rows, j_cols, j_vals: COO Jacobian data
        n_augmented: Total matrix size
        n_unknowns: Number of node unknowns (for regularization)
        n_vsources: Number of branch unknowns
        min_diag_reg: Minimum diagonal regularization
        gshunt: Additional shunt conductance for homotopy

    Returns:
        Dense Jacobian matrix of shape (n_augmented, n_augmented)
    """
    # COO -> dense via segment_sum
    flat_indices = j_rows * n_augmented + j_cols
    J_flat = jax.ops.segment_sum(j_vals, flat_indices, num_segments=n_augmented * n_augmented)
    J = J_flat.reshape((n_augmented, n_augmented))

    # Add diagonal regularization to node equations only
    # Branch equations (voltage constraints) should NOT have regularization
    diag_reg = jnp.concatenate(
        [
            jnp.full(n_unknowns, min_diag_reg + gshunt),
            jnp.zeros(n_vsources),
        ]
    )
    J = J + jnp.diag(diag_reg)

    return J


def assemble_sparse_jacobian(
    j_rows: Array,
    j_cols: Array,
    j_vals: Array,
    n_augmented: int,
    n_unknowns: int,
    min_diag_reg: float,
    gshunt: float | Array,
):
    """Assemble sparse BCOO Jacobian matrix from COO data.

    Args:
        j_rows, j_cols, j_vals: COO Jacobian data
        n_augmented: Total matrix size
        n_unknowns: Number of node unknowns (for regularization)
        min_diag_reg: Minimum diagonal regularization
        gshunt: Additional shunt conductance for homotopy

    Returns:
        BCOO sparse Jacobian matrix
    """
    from jax.experimental.sparse import BCOO

    # Add diagonal regularization for node equations only
    diag_idx = jnp.arange(n_unknowns, dtype=jnp.int32)
    j_rows = jnp.concatenate([j_rows, diag_idx])
    j_cols = jnp.concatenate([j_cols, diag_idx])
    j_vals = jnp.concatenate([j_vals, jnp.full(n_unknowns, min_diag_reg + gshunt)])

    indices = jnp.stack([j_rows, j_cols], axis=1)
    return BCOO((j_vals, indices), shape=(n_augmented, n_augmented))


def combine_transient_residual(
    f_resist: Array,
    Q: Array,
    lim_rhs_resist: Array,
    lim_rhs_react: Array,
    Q_prev: Array,
    integ_c0: float | Array,
    integ_c1: float | Array,
    integ_d1: float | Array,
    dQdt_prev: Array,
    integ_c2: float | Array,
    Q_prev2: Array,
    gshunt: float | Array,
    V_nodes: Array,
) -> Array:
    """Combine device contributions into transient residual.

    The transient residual equation is:
        f = f_resist - lim_rhs_resist
            + c0*(Q - lim_rhs_react) + c1*Q_prev + d1*dQdt_prev + c2*Q_prev2
            + gshunt*V

    Args:
        f_resist: Resistive (DC) residual
        Q: Current charges
        lim_rhs_resist: Limiting RHS resistive contribution
        lim_rhs_react: Limiting RHS reactive contribution
        Q_prev: Charges from previous timestep
        integ_c0, integ_c1, integ_d1, integ_c2: Integration coefficients
        dQdt_prev: dQ/dt from previous timestep
        Q_prev2: Charges from two timesteps ago (for Gear-2)
        gshunt: Shunt conductance for homotopy
        V_nodes: Node voltages (excluding ground)

    Returns:
        Combined residual vector
    """
    f_node = (
        f_resist
        - lim_rhs_resist
        + integ_c0 * (Q - lim_rhs_react)
        + integ_c1 * Q_prev
        + integ_d1 * dQdt_prev
        + integ_c2 * Q_prev2
    )
    f_node = f_node + gshunt * V_nodes
    return f_node
