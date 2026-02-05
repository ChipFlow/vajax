"""Netlist parsing utilities for JAX-SPICE.

This module provides functions for parsing and flattening circuit netlists,
building device lists, and extracting node mappings.
"""

import logging
import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Parameters that should be kept as strings (not evaluated as expressions)
STRING_PARAMS = {"type", "wave"}


def parse_elaborate_directive(text: str) -> Optional[str]:
    """Parse 'elaborate circuit("subckt_name")' directive from text.

    Args:
        text: Full netlist text content

    Returns:
        Subcircuit name to elaborate, or None if not found
    """
    match = re.search(r'elaborate\s+circuit\s*\(\s*"([^"]+)"', text)
    if match:
        return match.group(1)
    return None


def flatten_instances(
    circuit: Any,
    elaborate_subckt: Optional[str],
    parse_number: Callable[[Any], float],
) -> List[Tuple[str, List[str], str, Dict[str, str]]]:
    """Flatten subcircuit instances to leaf devices.

    Args:
        circuit: Parsed Circuit object with subckts, top_instances, globals, etc.
        elaborate_subckt: Subcircuit name to elaborate (from parse_elaborate_directive)
        parse_number: Function to parse SPICE numbers (e.g., '1u' -> 1e-6)

    Returns:
        List of (name, terminals, model, params) tuples for leaf devices
    """
    from jax_spice.netlist.parser import Instance
    from jax_spice.utils.safe_eval import safe_eval_expr

    flat_instances: List[Tuple[str, List[str], str, Dict[str, str]]] = []
    ground = circuit.ground or "0"

    # Handle elaborate directive
    if elaborate_subckt:
        subckt = circuit.subckts.get(elaborate_subckt)
        if subckt:
            logger.debug(f"Elaborating subcircuit: {elaborate_subckt}")
            synthetic_inst = Instance(
                name="top",
                terminals=subckt.terminals,
                model=elaborate_subckt,
                params={},
            )
            circuit.top_instances.append(synthetic_inst)

    def eval_param_expr(key: str, expr: str, param_env: Dict[str, float]):
        """Evaluate a parameter expression like 'w*pfact' or '2*(w+ld)'."""
        if not isinstance(expr, str):
            return float(expr)

        stripped = expr.strip()
        # Check for quoted strings
        if (stripped.startswith('"') and stripped.endswith('"')) or (
            stripped.startswith("'") and stripped.endswith("'")
        ):
            return stripped[1:-1]

        # Check if key should be kept as string
        if key.lower() in STRING_PARAMS:
            return stripped

        return safe_eval_expr(expr, param_env, default=0.0)

    def flatten_instance(
        inst: Instance,
        prefix: str,
        port_map: Dict[str, str],
        param_env: Dict[str, float],
    ):
        """Recursively flatten an instance."""
        model_name = inst.model
        subckt = circuit.subckts.get(model_name)

        if subckt is None:
            # Leaf device
            mapped_terminals = []
            for t in inst.terminals:
                if t in port_map:
                    mapped_terminals.append(port_map[t])
                elif t in circuit.globals or t == ground:
                    mapped_terminals.append(t)
                elif prefix:
                    mapped_terminals.append(f"{prefix}.{t}")
                else:
                    mapped_terminals.append(t)

            inst_params = {}
            for k, v in inst.params.items():
                inst_params[k] = str(eval_param_expr(k, v, param_env))

            flat_name = f"{prefix}.{inst.name}" if prefix else inst.name
            flat_instances.append((flat_name, mapped_terminals, model_name, inst_params))
        else:
            # Subcircuit - recurse
            new_port_map = {}
            for i, term in enumerate(subckt.terminals):
                if i < len(inst.terminals):
                    inst_term = inst.terminals[i]
                    if inst_term in port_map:
                        new_port_map[term] = port_map[inst_term]
                    elif inst_term in circuit.globals or inst_term == ground:
                        new_port_map[term] = inst_term
                    elif prefix:
                        new_port_map[term] = f"{prefix}.{inst_term}"
                    else:
                        new_port_map[term] = inst_term

            new_param_env = dict(param_env)
            for k, v in subckt.params.items():
                new_param_env[k] = eval_param_expr(k, v, new_param_env)
            for k, v in inst.params.items():
                new_param_env[k] = eval_param_expr(k, v, param_env)

            new_prefix = f"{prefix}.{inst.name}" if prefix else inst.name
            for sub_inst in subckt.instances:
                flatten_instance(sub_inst, new_prefix, new_port_map, new_param_env)

    # Flatten all top-level instances
    for inst in circuit.top_instances:
        param_env = {k: parse_number(v) for k, v in circuit.params.items()}
        flatten_instance(inst, "", {}, param_env)

    return flat_instances


def build_node_mapping(
    flat_instances: List[Tuple[str, List[str], str, Dict[str, str]]],
    ground_name: str,
) -> Tuple[Dict[str, int], int]:
    """Build node name to index mapping from flattened instances.

    Args:
        flat_instances: List of (name, terminals, model, params) tuples
        ground_name: Name of ground node (usually "0")

    Returns:
        Tuple of (node_names dict mapping name->index, num_nodes)
    """
    node_set: Set[str] = {ground_name}
    for name, terminals, model, params in flat_instances:
        for t in terminals:
            node_set.add(t)

    node_names = {ground_name: 0}
    for i, name in enumerate(sorted(n for n in node_set if n != ground_name), start=1):
        node_names[name] = i

    return node_names, len(node_names)


def build_devices(
    flat_instances: List[Tuple[str, List[str], str, Dict[str, str]]],
    node_names: Dict[str, int],
    get_device_type: Callable[[str], str],
    get_model_params: Callable[[str], Dict[str, float]],
    parse_number_cached: Callable[[Any], float],
    openvaf_models: Set[str],
) -> Tuple[List[Dict], bool]:
    """Build device list from flattened instances.

    Args:
        flat_instances: List of (name, terminals, model, params) tuples
        node_names: Node name to index mapping
        get_device_type: Function to map model name to device type
        get_model_params: Function to get model parameters
        parse_number_cached: Cached SPICE number parser
        openvaf_models: Set of model types that use OpenVAF

    Returns:
        Tuple of (devices list, has_openvaf_devices flag)
    """
    devices = []
    has_openvaf = False

    for inst_name, inst_terminals, inst_model, inst_params in flat_instances:
        model_name = inst_model.lower()
        device_type = get_device_type(model_name)
        nodes = [node_names[t] for t in inst_terminals]

        # Get model parameters and instance parameters
        model_params = get_model_params(model_name)

        # Parse instance params, but keep string params as strings
        parsed_params = {}
        for k, v in inst_params.items():
            if k in STRING_PARAMS:
                parsed_params[k] = str(v).strip('"').strip("'")
            else:
                parsed_params[k] = parse_number_cached(v)

        # Merge model params with instance params (instance overrides model)
        params = {**model_params, **parsed_params}

        is_openvaf = device_type in openvaf_models

        devices.append(
            {
                "name": inst_name,
                "model": device_type,
                "nodes": nodes,
                "params": params,
                "original_params": parsed_params,
                "is_openvaf": is_openvaf,
            }
        )

        if is_openvaf:
            has_openvaf = True

    return devices, has_openvaf
