"""Generate setup_instance() function from MIR init function."""

from .mir_parser import MIRFunction
from .control_flow_codegen import generate_python_with_control_flow
from typing import Dict, List, Tuple


def generate_setup_instance_from_mir(
    mir_func: MIRFunction,
    param_map: Dict[str, str],
    cache_mapping: List[Tuple[str, int]],
    model_name: str = "device"
) -> str:
    """Generate setup_instance() function from init MIR.

    The setup_instance() function computes cached values from model/instance parameters.
    It's called once per device instance.

    Args:
        mir_func: Parsed init MIR function
        param_map: Maps MIR SSA names to semantic parameter names
        cache_mapping: List of (init_value, eval_param_idx) tuples
                      e.g. [('v27', 5), ('v28', 6)]
        model_name: Name of the device model (for function naming)

    Returns:
        Python source code for setup_instance() function
    """
    # Use the control flow codegen to generate the core logic
    core_code = generate_python_with_control_flow(mir_func, param_map)

    # Wrap it in a setup_instance() function with proper interface
    lines = []
    lines.append(f'def setup_instance_{model_name}(**params):')
    lines.append(f'    """Compute cached values for {model_name} instance."""')
    lines.append('')
    lines.append('    # Extract parameters and given flags')

    # Build parameter extraction code
    # The MIR function signature tells us which parameters are expected
    for param in mir_func.params:
        param_name = param_map.get(param.name) or param.name

        # Distinguish between value and given flag
        if param_name.endswith('_given'):
            base_name = param_name[:-6]  # Remove '_given' suffix
            lines.append(f'    {param_name} = params.get("{base_name}_given", False)')
        elif param_name == 'mfactor':
            # System function with default value
            lines.append(f'    {param_name} = params.get("mfactor", 1.0)')
        else:
            lines.append(f'    {param_name} = params.get("{param_name}", 0.0)')

    lines.append('')
    lines.append('    # Compute cache values from MIR')

    # Extract the function body from core_code (skip the def line and final return)
    core_lines = core_code.split('\n')
    in_final_return = False
    for line in core_lines:
        # Skip def line
        if line.strip().startswith('def '):
            continue

        # Detect start of final return statement
        if '# Collect results' in line or (line.strip().startswith('return {') and not in_final_return):
            in_final_return = True
            continue

        # Skip lines in final return
        if in_final_return:
            if line.strip() == '}':
                in_final_return = False
            continue

        # Add the line
        if line.strip():
            lines.append(line)

    lines.append('')
    lines.append('    # Return cache values (computed by MIR)')
    lines.append('    cache = [')

    # Build cache array in order of eval param indices
    # cache_mapping is [(init_value, eval_param_idx), ...]
    # Sort by eval_param_idx to ensure correct ordering
    sorted_cache = sorted(cache_mapping, key=lambda x: x[1])

    for init_value, eval_idx in sorted_cache:
        # Map MIR SSA name to semantic name
        cache_var = param_map.get(init_value) or init_value
        lines.append(f'        {cache_var},  # cache[{eval_idx - sorted_cache[0][1]}] -> eval param[{eval_idx}]')

    lines.append('    ]')
    lines.append('    return cache')

    return '\n'.join(lines)
