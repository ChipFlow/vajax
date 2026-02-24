"""Node setup and collapse handling for VA-JAX.

This module handles:
- Union-find algorithm for node collapse computation
- Internal node allocation for OpenVAF devices
- Node collapse decisions based on device parameters
"""

import logging
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


def compute_collapse_roots(
    collapsible_pairs: List[Tuple[int, int]], n_nodes: int
) -> Dict[int, int]:
    """Compute the collapse root for each node using union-find.

    Collapsible pairs (a, b) mean nodes a and b should be the same electrical node.
    We use union-find to compute equivalence classes, preferring external nodes
    (indices 0-3) as roots.

    Args:
        collapsible_pairs: List of (node1, node2) pairs that should collapse
        n_nodes: Total number of model nodes

    Returns:
        Dict mapping each node index to its root (representative) node index
    """
    # Initialize parent array (each node is its own parent)
    parent = list(range(n_nodes))

    def find(x: int) -> int:
        if parent[x] != x:
            parent[x] = find(parent[x])  # Path compression
        return parent[x]

    def union(x: int, y: int) -> None:
        px, py = find(x), find(y)
        if px != py:
            # Prefer external nodes (0-3) as root
            if py < 4:
                parent[px] = py
            elif px < 4:
                parent[py] = px
            else:
                parent[py] = px

    # Apply collapse pairs
    for a, b in collapsible_pairs:
        if b != 4294967295:  # u32::MAX = collapse to ground (handled separately)
            if a < n_nodes and b < n_nodes:
                union(a, b)

    # Build root mapping for all nodes
    return {i: find(i) for i in range(n_nodes)}


def setup_internal_nodes(
    devices: List[Dict],
    num_nodes: int,
    compiled_models: Dict[str, Any],
    device_collapse_decisions: Dict[str, List[Tuple[int, int]]],
) -> Tuple[int, Dict[str, Dict[str, int]]]:
    """Set up internal nodes for OpenVAF devices with node collapse support.

    Node collapse eliminates unnecessary internal nodes when model parameters
    indicate they should be merged (e.g., when resistance parameters are 0).

    Args:
        devices: List of device dicts with 'is_openvaf', 'model', 'name', 'nodes' keys
        num_nodes: Number of external circuit nodes
        compiled_models: Dict of compiled model info with 'nodes' key
        device_collapse_decisions: Dict mapping device name to collapse pairs

    Returns:
        (total_nodes, device_internal_nodes) where device_internal_nodes maps
        device name to dict of internal node name -> global index
    """
    n_external = num_nodes
    next_internal = n_external
    device_internal_nodes: Dict[str, Dict[str, int]] = {}

    # Cache collapse roots for devices with identical collapse patterns
    collapse_roots_cache: Dict[Tuple[Tuple[int, int], ...], Dict[int, int]] = {}

    for dev in devices:
        if not dev.get("is_openvaf"):
            continue

        model_type = dev["model"]
        compiled = compiled_models.get(model_type)
        if not compiled:
            continue

        model_nodes = compiled["nodes"]
        n_model_nodes = len(model_nodes)

        # Get precomputed collapse pairs
        device_name = dev["name"]
        collapse_pairs = device_collapse_decisions.get(device_name, [])

        # Cache collapse roots by pattern (most devices of same type will share)
        pairs_key = tuple(sorted(collapse_pairs))
        if pairs_key not in collapse_roots_cache:
            collapse_roots_cache[pairs_key] = compute_collapse_roots(collapse_pairs, n_model_nodes)
        collapse_roots = collapse_roots_cache[pairs_key]

        # Include all internal nodes including branch currents
        # Branch currents (names like 'br[Branch(BranchId(N))]') are system unknowns
        # VACASK counts branch currents as unknowns, and we must match for node counts
        n_internal_end = n_model_nodes

        # Map external nodes to device's external circuit nodes
        # Number of external terminals is determined by the device instance
        ext_nodes = dev["nodes"]
        n_ext_terminals = len(ext_nodes)
        ext_node_map = {}
        for i in range(n_ext_terminals):
            ext_node_map[i] = ext_nodes[i]

        # Build node mapping using collapse roots
        # Track which internal root nodes need circuit node allocation
        internal_root_to_circuit: Dict[int, int] = {}
        node_mapping: Dict[int, int] = {}

        # Internal nodes start after external terminals
        for i in range(n_ext_terminals, n_internal_end):
            root = collapse_roots.get(i, i)

            if root < n_ext_terminals:
                # Root is an external node - use its circuit node
                node_mapping[i] = ext_node_map[root]
            else:
                # Root is internal - need to allocate/reuse a circuit node
                if root not in internal_root_to_circuit:
                    internal_root_to_circuit[root] = next_internal
                    next_internal += 1
                node_mapping[i] = internal_root_to_circuit[root]

        # Build internal_map: model node name -> circuit node index
        internal_map = {}
        for i in range(n_ext_terminals, n_internal_end):
            node_name = model_nodes[i]
            internal_map[node_name] = node_mapping[i]

        device_internal_nodes[dev["name"]] = internal_map

    if device_internal_nodes:
        n_internal = next_internal - n_external
        logger.info(
            f"Allocated {n_internal} internal nodes for {len(device_internal_nodes)} OpenVAF devices"
        )

    return next_internal, device_internal_nodes
