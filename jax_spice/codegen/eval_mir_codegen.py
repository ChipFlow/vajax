"""Generate eval() function from MIR with cache support."""

from .mir_parser import MIRFunction
from .control_flow_codegen import generate_python_with_control_flow
from typing import Dict, List, Tuple


def generate_eval_from_mir(
    mir_func: MIRFunction,
    param_map: Dict[str, str],
    cache_param_indices: List[int],
    model_name: str = "device"
) -> str:
    """Generate eval() function from eval MIR with cache support.

    The eval() function computes residuals and Jacobian from voltages and cached values.
    It's called many times per timestep during simulation.

    Args:
        mir_func: Parsed eval MIR function
        param_map: Maps MIR SSA names to semantic parameter names
        cache_param_indices: List of parameter indices that are cache slots
                            e.g. [5, 6] means Param[5] and Param[6] are cache
        model_name: Name of the device model (for function naming)

    Returns:
        Python source code for eval() function

    Example for capacitor:
        - mir_func.params = [v16(c), v17(V_A_B), v19(Q), v21(q), v25(mfactor), v37(cache), v40(cache)]
        - cache_param_indices = [5, 6]
        - Generated signature: def eval_capacitor(c, V_A_B, mfactor, cache):
        - v37 becomes cache[0], v40 becomes cache[1]
    """
    # Separate parameters into regular and cache
    regular_params = []
    cache_mappings = {}  # MIR name -> cache index

    cache_base_idx = min(cache_param_indices) if cache_param_indices else 999

    for i, param in enumerate(mir_func.params):
        if i in cache_param_indices:
            # This is a cache slot
            cache_idx = i - cache_base_idx
            cache_mappings[param.name] = cache_idx
        else:
            # Regular parameter - check if it's actually used
            param_semantic_name = param_map.get(param.name, param.name)

            # Skip hidden_state params - they're not actually used (inlined by optimizer)
            # We can detect them by checking if the name contains "hidden"
            if param_semantic_name and 'hidden' not in param_semantic_name.lower():
                regular_params.append(param_semantic_name)

    # Update param_map to map cache parameters to cache array access
    extended_param_map = param_map.copy()
    for mir_name, cache_idx in cache_mappings.items():
        extended_param_map[mir_name] = f'cache[{cache_idx}]'

    # Generate core function body using control flow codegen
    # This will handle branches, PHI nodes, etc.
    core_code = generate_python_with_control_flow(mir_func, extended_param_map)

    # Build final function with proper signature
    lines = []

    # Function signature
    sig_params = regular_params + ['cache']
    lines.append(f'def eval_{model_name}({", ".join(sig_params)}):')
    lines.append(f'    """Evaluate {model_name} model - compute residuals and Jacobian."""')
    lines.append('')

    # Note about cache
    if cache_mappings:
        lines.append('    # Cache slots:')
        for mir_name, cache_idx in sorted(cache_mappings.items(), key=lambda x: x[1]):
            semantic_name = param_map.get(mir_name, mir_name)
            lines.append(f'    #   cache[{cache_idx}] = {semantic_name}')
        lines.append('')

    # Extract the function body from core_code (skip the def line and final return)
    core_lines = core_code.split('\n')
    in_function_body = False
    for line in core_lines:
        # Skip def line
        if line.strip().startswith('def '):
            in_function_body = True
            continue

        if in_function_body and line.strip():
            # Keep the line
            lines.append(line)

    return '\n'.join(lines)


def identify_output_variables(mir_func: MIRFunction, param_map: Dict[str, str]) -> Tuple[List[str], List[str]]:
    """Identify which computed values are residuals vs Jacobian entries.

    This requires understanding the MIR's output mapping (from OSDI).
    For now, return all computed values.

    Returns:
        (residual_vars, jacobian_vars) tuple
    """
    # TODO: Use OSDI output metadata to identify residuals and Jacobian
    # For now, return empty lists and let caller filter
    return ([], [])
