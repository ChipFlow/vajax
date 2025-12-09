"""GPU-native MNA stamping functions using JAX scatter operations

This module provides GPU-friendly stamping functions that replace Python loops
with JAX's at[].add() scatter operations for efficient GPU execution.

Key difference from standard MNA stamping:
- Standard: Python loop over devices, stamping one at a time
- GPU: Single scatter operation per device type

Example performance difference:
- 10,000 resistors with Python loop: 10,000 Python indexing operations
- 10,000 resistors with scatter: 1 GPU kernel call
"""

from typing import Tuple, Dict
import jax.numpy as jnp
from jax import Array


def stamp_2terminal_residual_gpu(
    residual: Array,
    node_p: Array,
    node_n: Array,
    I_batch: Array,
    ground_node: int,
) -> Array:
    """Stamp 2-terminal device currents into residual using GPU scatter.

    For a 2-terminal device (resistor, voltage source, etc.):
    - Current flows from p to n terminal
    - f[p] += I (current out of p)
    - f[n] -= I (current into n)

    Args:
        residual: Residual vector (num_nodes-1,) excluding ground
        node_p: Positive node indices (n_devices,)
        node_n: Negative node indices (n_devices,)
        I_batch: Current values (n_devices,)
        ground_node: Index of ground node (excluded from residual)

    Returns:
        Updated residual vector
    """
    n_devices = node_p.shape[0]

    # Create masks for non-ground nodes
    # Node indices are 1-indexed (ground=0), residual is 0-indexed
    mask_p = node_p != ground_node
    mask_n = node_n != ground_node

    # Convert to 0-indexed residual indices (node_idx - 1)
    # Use 0 as placeholder for ground nodes (will be masked out)
    idx_p = jnp.where(mask_p, node_p - 1, 0)
    idx_n = jnp.where(mask_n, node_n - 1, 0)

    # Masked currents (0 for ground nodes)
    I_p = jnp.where(mask_p, I_batch, 0.0)
    I_n = jnp.where(mask_n, -I_batch, 0.0)

    # Scatter-add: single GPU operation per terminal
    residual = residual.at[idx_p].add(I_p)
    residual = residual.at[idx_n].add(I_n)

    return residual


def stamp_4terminal_residual_gpu(
    residual: Array,
    node_d: Array,
    node_g: Array,
    node_s: Array,
    node_b: Array,
    Ids: Array,
    ground_node: int,
) -> Array:
    """Stamp 4-terminal MOSFET currents into residual using GPU scatter.

    For a MOSFET:
    - Drain current Ids flows from drain to source
    - f[d] += Ids (current out of drain)
    - f[s] -= Ids (current into source)
    - Gate and bulk have no DC current

    Args:
        residual: Residual vector (num_nodes-1,) excluding ground
        node_d: Drain node indices (n_devices,)
        node_g: Gate node indices (n_devices,) - unused for DC
        node_s: Source node indices (n_devices,)
        node_b: Bulk node indices (n_devices,) - unused for DC
        Ids: Drain current values (n_devices,)
        ground_node: Index of ground node

    Returns:
        Updated residual vector
    """
    # Create masks for non-ground nodes
    mask_d = node_d != ground_node
    mask_s = node_s != ground_node

    # Convert to 0-indexed residual indices
    idx_d = jnp.where(mask_d, node_d - 1, 0)
    idx_s = jnp.where(mask_s, node_s - 1, 0)

    # Masked currents
    I_d = jnp.where(mask_d, Ids, 0.0)
    I_s = jnp.where(mask_s, -Ids, 0.0)

    # Scatter-add
    residual = residual.at[idx_d].add(I_d)
    residual = residual.at[idx_s].add(I_s)

    return residual


def stamp_vsource_residual_gpu(
    residual: Array,
    node_p: Array,
    node_n: Array,
    V_batch: Array,
    I_branch: Array,
    branch_indices: Array,
    V_nodes: Array,
    vdd_scale: float,
    ground_node: int,
) -> Array:
    """Stamp voltage source contributions to residual.

    Voltage sources use MNA formulation with branch currents:
    - KCL at p: f[p] += I_branch
    - KCL at n: f[n] -= I_branch
    - Branch equation: f[branch] = V[p] - V[n] - V_source

    Args:
        residual: Residual vector
        node_p: Positive node indices (n_devices,)
        node_n: Negative node indices (n_devices,)
        V_batch: Source voltages (n_devices,)
        I_branch: Branch currents from solution vector
        branch_indices: Indices into solution vector for branch currents
        V_nodes: Full node voltage vector
        vdd_scale: Scaling factor for voltage sources
        ground_node: Index of ground node

    Returns:
        Updated residual vector
    """
    n_devices = node_p.shape[0]

    # KCL contributions from branch currents
    mask_p = node_p != ground_node
    mask_n = node_n != ground_node

    idx_p = jnp.where(mask_p, node_p - 1, 0)
    idx_n = jnp.where(mask_n, node_n - 1, 0)

    I_p = jnp.where(mask_p, I_branch, 0.0)
    I_n = jnp.where(mask_n, -I_branch, 0.0)

    residual = residual.at[idx_p].add(I_p)
    residual = residual.at[idx_n].add(I_n)

    # Branch equations: V[p] - V[n] - V_source * scale = 0
    # Get node voltages (ground = 0)
    V_p = jnp.where(node_p == ground_node, 0.0, V_nodes[node_p])
    V_n = jnp.where(node_n == ground_node, 0.0, V_nodes[node_n])

    branch_residual = V_p - V_n - V_batch * vdd_scale

    # Stamp into branch equation rows
    residual = residual.at[branch_indices].set(branch_residual)

    return residual


def stamp_gmin_residual_gpu(
    residual: Array,
    V: Array,
    gmin: float,
    ground_node: int,
) -> Array:
    """Add GMIN contribution to all nodes (except ground).

    GMIN is a small conductance to ground that improves numerical stability.
    f[i] += gmin * V[i] for all non-ground nodes.

    Args:
        residual: Residual vector (num_nodes-1,) excluding ground
        V: Node voltage vector (num_nodes,)
        gmin: GMIN conductance value
        ground_node: Index of ground node

    Returns:
        Updated residual vector
    """
    # V is indexed by node, residual is indexed by node-1 (excluding ground)
    # For nodes 1 to num_nodes-1, add gmin * V[node]
    num_nodes = V.shape[0]

    # Create voltage vector excluding ground
    V_nonground = jnp.concatenate([
        V[1:ground_node] if ground_node > 1 else jnp.array([]),
        V[ground_node+1:] if ground_node < num_nodes - 1 else jnp.array([])
    ])

    # If ground is at 0, V_nonground = V[1:]
    if ground_node == 0:
        V_nonground = V[1:]

    residual = residual + gmin * V_nonground

    return residual


def build_mosfet_params_from_group(
    group,
    temperature: float = 300.0,
) -> Dict[str, Array]:
    """Build parameter dictionary for mosfet_batch from VectorizedDeviceGroup.

    Args:
        group: VectorizedDeviceGroup containing MOSFET devices
        temperature: Temperature in Kelvin

    Returns:
        Dict of parameter arrays for mosfet_batch
    """
    n = group.n_devices
    params = group.params

    # Default MOSFET parameters
    defaults = {
        'W': 1e-6,
        'L': 0.25e-6,
        'Vth0': 0.4,
        'gamma': 0.5,
        'phiB': 0.9,
        'u0': 400e-4,
        'theta': 0.2,
        'vsat': 1e5,
        'a0': 1.0,
        'lambda_': 0.05,
        'tox': 5e-9,
        'epsilon_ox': 3.9 * 8.854e-12,
        'n_sub': 1.5,
        'Ioff': 1e-12,
        'temp': temperature,
        'pmos': 0.0,
    }

    # Map from lowercase param names to uppercase for mosfet_batch
    param_aliases = {
        'W': ['W', 'w'],
        'L': ['L', 'l'],
    }

    result = {}
    for key, default in defaults.items():
        found = False
        # Check aliases for case-insensitive matching
        if key in param_aliases:
            for alias in param_aliases[key]:
                if alias in params:
                    result[key] = params[alias]
                    found = True
                    break
        elif key in params:
            result[key] = params[key]
            found = True

        if not found:
            result[key] = jnp.full(n, default)

    return result


def build_resistor_params_from_group(group) -> Tuple[Array, Array, Array]:
    """Build parameter arrays for resistor_batch from VectorizedDeviceGroup.

    Args:
        group: VectorizedDeviceGroup containing resistor devices

    Returns:
        Tuple of (node_p, node_n, R) arrays
    """
    node_p = group.node_indices[:, 0]
    node_n = group.node_indices[:, 1]
    R = group.params.get('R', jnp.ones(group.n_devices) * 1000.0)
    return node_p, node_n, R
