"""DC operating point computation for JAX-SPICE.

This module contains the DC operating point solver, which finds the steady-state
solution for a circuit with all time-varying sources at their DC values.
"""

import logging
from typing import Callable, Dict, List, Optional, Tuple

import jax.numpy as jnp
from jax import Array

from jax_spice import get_float_dtype
from jax_spice.analysis.homotopy import HomotopyConfig, run_homotopy_chain
from jax_spice.analysis.options import SimulationOptions

logger = logging.getLogger(__name__)


def initialize_dc_voltages(
    n_nodes: int,
    vdd_value: float,
    node_names: Dict[str, int],
    devices: List[Dict],
    device_internal_nodes: Optional[Dict[str, Dict[str, int]]] = None,
) -> Tuple[Array, Optional[Array]]:
    """Initialize voltage vector with good starting values for DC convergence.

    This sets up initial voltages that help Newton-Raphson converge:
    - Ground to 0V
    - VDD/VCC nodes to supply voltage
    - Voltage source nodes to their target DC values
    - Internal NOI nodes to 0V (high-conductance noise nodes)
    - Body internal nodes to bulk voltage

    Args:
        n_nodes: Total number of nodes
        vdd_value: Supply voltage value (V)
        node_names: Dict mapping node name to index
        devices: List of device dictionaries
        device_internal_nodes: Map of device name -> {node_name: circuit_node_idx}

    Returns:
        Tuple of (V, noi_indices) where:
            - V: Initial voltage array shape (n_nodes,)
            - noi_indices: Array of NOI node indices or None
    """
    # Start with mid-rail voltage for all nodes
    mid_rail = vdd_value / 2.0
    V = jnp.full(n_nodes, mid_rail, dtype=get_float_dtype())
    V = V.at[0].set(0.0)  # Ground is always 0

    # Set VDD nodes to full supply voltage (name-based heuristic)
    for name, idx in node_names.items():
        name_lower = name.lower()
        if "vdd" in name_lower or "vcc" in name_lower:
            V = V.at[idx].set(vdd_value)
            logger.debug(f"  Initialized VDD node '{name}' (idx {idx}) to {vdd_value}V")
        elif name_lower in ("gnd", "vss", "0"):
            V = V.at[idx].set(0.0)
            logger.debug(f"  Initialized ground node '{name}' (idx {idx}) to 0V")

    # Initialize ALL voltage source nodes to their target DC values
    # This is critical for convergence - vsource stamps have G=1e12, so
    # any deviation from target creates residuals of order 1e12 * delta_V
    vsources_initialized = 0
    for dev in devices:
        if dev["model"] == "vsource":
            nodes = dev.get("nodes", [])
            if len(nodes) >= 2:
                p_node, n_node = nodes[0], nodes[1]
                dc_val = float(dev["params"].get("dc", 0.0))
                if n_node == 0:
                    # Negative node is ground, set positive node to DC value
                    if p_node > 0:
                        V = V.at[p_node].set(dc_val)
                        vsources_initialized += 1
                else:
                    # Both nodes non-ground - set relative voltage
                    # Set p_node = n_node_voltage + dc_val
                    if p_node > 0:
                        V = V.at[p_node].set(float(V[n_node]) + dc_val)
                        vsources_initialized += 1
    if vsources_initialized > 0:
        logger.debug(
            f"  Initialized {vsources_initialized} voltage source nodes to target DC values"
        )

    # Initialize PSP103 internal nodes
    noi_indices: List[int] = []
    if device_internal_nodes:
        noi_nodes_initialized = 0
        body_nodes_initialized = 0
        device_external_nodes = {dev["name"]: dev.get("nodes", []) for dev in devices}

        for dev_name, internal_nodes in device_internal_nodes.items():
            if "node4" in internal_nodes:
                noi_idx = internal_nodes["node4"]
                V = V.at[noi_idx].set(0.0)
                noi_indices.append(noi_idx)
                noi_nodes_initialized += 1

            ext_nodes = device_external_nodes.get(dev_name, [])
            if len(ext_nodes) >= 4:
                b_circuit_node = ext_nodes[3]
                b_voltage = float(V[b_circuit_node])
                for body_node_name in ["node8", "node9", "node10", "node11"]:
                    if body_node_name in internal_nodes:
                        body_idx = internal_nodes[body_node_name]
                        if body_idx > 0 and abs(V[body_idx] - mid_rail) < 0.01:
                            V = V.at[body_idx].set(b_voltage)
                            body_nodes_initialized += 1

        if noi_nodes_initialized > 0:
            logger.debug(f"  Initialized {noi_nodes_initialized} NOI nodes to 0V")
        if body_nodes_initialized > 0:
            logger.debug(f"  Initialized {body_nodes_initialized} body internal nodes")

    noi_indices_arr = jnp.array(noi_indices, dtype=jnp.int32) if noi_indices else None
    logger.debug(f"  Initial V: ground=0V, VDD={vdd_value}V, others={mid_rail}V")

    return V, noi_indices_arr


def compute_dc_operating_point(
    n_nodes: int,
    node_names: Dict[str, int],
    devices: List[Dict],
    nr_solve: Callable,
    device_arrays: Dict[str, Array],
    vsource_dc_vals: Array,
    isource_dc_vals: Array,
    options: SimulationOptions,
    vdd_value: float,
    device_internal_nodes: Optional[Dict[str, Dict[str, int]]] = None,
) -> Array:
    """Compute DC operating point using VACASK-style homotopy chain.

    Uses the homotopy chain (gdev -> gshunt -> src) to find the DC operating
    point even for difficult circuits like ring oscillators where simple
    Newton-Raphson fails due to near-singular Jacobians.

    Args:
        n_nodes: Number of nodes in the system
        node_names: Dict mapping node names to indices
        devices: List of device dictionaries
        nr_solve: The Newton-Raphson solver function
        device_arrays: Dict[model_type, cache] - passed to nr_solve
        vsource_dc_vals: DC values for voltage sources
        isource_dc_vals: DC values for current sources
        options: Simulation options containing NR and homotopy settings
        vdd_value: Supply voltage value (for initialization)
        device_internal_nodes: Map of device name -> {node_name: circuit_node_idx}

    Returns:
        DC operating point voltages (shape: [n_nodes])
    """
    logger.info("Computing DC operating point...")

    # Initialize voltages with good starting values
    V, noi_indices = initialize_dc_voltages(
        n_nodes, vdd_value, node_names, devices, device_internal_nodes
    )

    n_unknowns = n_nodes - 1
    Q_prev = jnp.zeros(n_unknowns, dtype=get_float_dtype())

    # First try direct NR without homotopy (works for well-initialized circuits)
    logger.info("  Trying direct NR solver first...")
    V_new, nr_iters, is_converged, max_f, _, _, _, _, _ = nr_solve(
        V,
        vsource_dc_vals,
        isource_dc_vals,
        Q_prev,
        0.0,  # integ_c0=0 for DC
        device_arrays,
    )

    if is_converged:
        V = V_new
        logger.info(
            f"  DC operating point converged via direct NR ({nr_iters} iters, residual={max_f:.2e})"
        )
    else:
        # Fall back to homotopy chain using the NR solver
        logger.info("  Direct NR failed, trying homotopy chain...")

        # Configure homotopy from SimulationOptions
        homotopy_config = HomotopyConfig(
            gmin=options.gmin,
            gdev_start=options.homotopy_startgmin,
            gdev_target=options.homotopy_mingmin,
            gmin_factor=options.homotopy_gminfactor,
            gmin_factor_min=options.homotopy_mingminfactor,
            gmin_factor_max=options.homotopy_maxgminfactor,
            gmin_max=options.homotopy_maxgmin,
            gmin_max_steps=options.homotopy_gminsteps,
            source_step=options.homotopy_srcstep,
            source_step_min=options.homotopy_minsrcstep,
            source_scale=options.homotopy_srcscale,
            source_max_steps=options.homotopy_srcsteps,
            chain=options.op_homotopy,
            max_iterations=options.op_itlcont,
            abstol=options.abstol,
            debug=0,
        )

        result = run_homotopy_chain(
            nr_solve,
            V,
            vsource_dc_vals,
            isource_dc_vals,
            Q_prev,
            device_arrays,
            homotopy_config,
        )

        if result.converged:
            V = result.V
            logger.info(
                f"  DC operating point converged via {result.method} "
                f"({result.iterations} total iters, {result.homotopy_steps} homotopy steps)"
            )
        else:
            logger.warning(f"  Homotopy chain did not converge (method={result.method})")
            # For oscillator circuits, accept the best solution we have
            # The DC operating point might be metastable anyway
            V = result.V
            logger.info("  Using best available solution for metastable circuit")

    # Clamp NOI nodes after DC solution
    if noi_indices is not None:
        V = V.at[noi_indices].set(0.0)

    # Log key node voltages
    n_external = len(node_names)
    logger.info(f"  DC solution: {min(n_external, 5)} node voltages:")
    for i in range(min(n_external, 5)):
        name = next((n for n, idx in node_names.items() if idx == i), str(i))
        logger.info(f"    Node {name} (idx {i}): {float(V[i]):.6f}V")

    return V
