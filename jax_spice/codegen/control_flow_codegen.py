"""Python code generation from MIR with control flow support."""

from .mir_parser import MIRFunction, MIRBlock, MIRInstruction
from typing import Dict, Set


def generate_python_with_control_flow(mir_func: MIRFunction, param_map: Dict[str, str]) -> str:
    """Generate Python code from MIR with branches, jumps, and PHI nodes.

    Strategy: Generate a state machine with block labels.
    Each block becomes a code section, branches become if/else, jumps become state transitions.

    Args:
        mir_func: Parsed MIR function with control flow
        param_map: Maps MIR names to semantic names

    Returns:
        Python function source code
    """
    # Build function signature
    sig_params = [param_map.get(p.name) or p.name for p in mir_func.params]
    func_lines = []

    func_lines.append(f"def {mir_func.name}({', '.join(sig_params)}):")
    func_lines.append('    """Generated from MIR with control flow."""')
    func_lines.append('')

    # Initialize constants first
    func_lines.append('    # Initialize constants')
    for const_name, const_val in sorted(mir_func.constants.items()):
        mapped_name = param_map.get(const_name, const_name)
        # Handle special float values
        if const_val == '+Inf' or (isinstance(const_val, float) and const_val > 1e308):
            func_lines.append(f'    {mapped_name} = math.inf')
        elif const_val == '-Inf' or (isinstance(const_val, float) and const_val < -1e308):
            func_lines.append(f'    {mapped_name} = -math.inf')
        elif isinstance(const_val, str) and const_val.startswith('0x'):
            # Hex float literal
            func_lines.append(f'    {mapped_name} = float.fromhex("{const_val}")')
        else:
            func_lines.append(f'    {mapped_name} = {const_val}')
    func_lines.append('')

    # Declare other MIR variables that might be referenced across blocks
    # (For PHI nodes, we need variables to exist before they're assigned)
    all_values = collect_all_values(mir_func)
    func_lines.append('    # Declare variables for SSA values')
    for val in sorted(all_values):
        mapped_name = param_map.get(val, val)
        if mapped_name not in sig_params and val not in mir_func.constants:  # Don't redeclare parameters or constants
            func_lines.append(f'    {mapped_name} = None')
    func_lines.append('')

    # Entry point
    first_block = mir_func.blocks[0].name if mir_func.blocks else None
    func_lines.append(f'    # Start execution at {first_block}')
    func_lines.append(f'    current_block = "{first_block}"')
    func_lines.append(f'    prev_block = None  # Track predecessor for PHI nodes')
    func_lines.append('')

    # Generate block dispatch loop
    func_lines.append('    while current_block is not None:')

    # Generate each block as a case in the state machine
    for i, block in enumerate(mir_func.blocks):
        if i == 0:
            func_lines.append(f'        if current_block == "{block.name}":')
        else:
            func_lines.append(f'        elif current_block == "{block.name}":')

        block_code = generate_block_code(block, param_map)
        if not block_code:
            # Empty block - add pass
            func_lines.append('            pass')
        else:
            for line in block_code:
                func_lines.append(f'            {line}')
        func_lines.append('')

    # Default case - exit
    func_lines.append('        else:')
    func_lines.append('            break  # Unknown block, exit')
    func_lines.append('')

    # Return results
    func_lines.append('    # Collect results')
    func_lines.append('    return {')
    for val in sorted(all_values):
        mapped_name = param_map.get(val, val)
        if mapped_name not in sig_params:
            func_lines.append(f'        "{mapped_name}": {mapped_name},')
    func_lines.append('    }')

    return '\n'.join(func_lines)


def collect_all_values(mir_func: MIRFunction) -> Set[str]:
    """Collect all SSA value names from MIR function."""
    values = set()

    for block in mir_func.blocks:
        for inst in block.instructions:
            if inst.result:
                values.add(inst.result.name)
            for arg in inst.args:
                values.add(arg.name)

    # Add constants
    values.update(mir_func.constants.keys())

    return values


def generate_block_code(block: MIRBlock, param_map: Dict[str, str]) -> list:
    """Generate Python code for a single MIR block.

    Returns:
        List of Python statements (without indentation)
    """
    lines = []
    has_terminator = False  # Track if block has br/jmp

    for inst in block.instructions:
        if inst.opcode == 'br':
            # Conditional branch: if cond then block_true else block_false
            cond = param_map.get(inst.args[0].name, inst.args[0].name)
            true_block, false_block = inst.target_blocks
            lines.append(f'if {cond}:')
            lines.append(f'    prev_block, current_block = current_block, "{true_block}"')
            lines.append(f'else:')
            lines.append(f'    prev_block, current_block = current_block, "{false_block}"')
            has_terminator = True

        elif inst.opcode == 'jmp':
            # Unconditional jump
            target = inst.target_blocks[0]
            lines.append(f'prev_block, current_block = current_block, "{target}"')
            has_terminator = True

        elif inst.opcode == 'call':
            # Callback - for now, just comment
            lines.append('# callback (validation)')
            lines.append('pass  # TODO: implement callback')

        elif inst.opcode == 'phi':
            # PHI node - select value based on predecessor block
            result = param_map.get(inst.result.name, inst.result.name)
            if not inst.phi_args:
                # Fallback if no PHI args parsed
                lines.append(f'# PHI: {result} = ... (no args)')
                lines.append('pass')
            else:
                # Generate if/elif chain for each predecessor
                for i, (value_name, block_name) in enumerate(inst.phi_args):
                    mapped_value = param_map.get(value_name, value_name)
                    if i == 0:
                        lines.append(f'if prev_block == "{block_name}":')
                        lines.append(f'    {result} = {mapped_value}')
                    else:
                        lines.append(f'elif prev_block == "{block_name}":')
                        lines.append(f'    {result} = {mapped_value}')
                # Add else case for unexpected predecessor
                lines.append(f'else:')
                lines.append(f'    {result} = None  # Unexpected predecessor')

        elif inst.opcode == 'optbarrier':
            # Pass-through
            if inst.result:
                result = param_map.get(inst.result.name, inst.result.name)
                arg = param_map.get(inst.args[0].name, inst.args[0].name)
                lines.append(f'{result} = {arg}  # optbarrier')

        elif inst.result:
            # Regular instruction with result
            python_stmt = translate_instruction_simple(inst, param_map)
            if python_stmt:
                lines.append(python_stmt)

    # If block doesn't have a terminator (br/jmp), exit the loop
    if not has_terminator:
        lines.append('prev_block, current_block = current_block, None  # Exit loop')

    return lines


def translate_instruction_simple(inst: MIRInstruction, param_map: Dict[str, str]) -> str:
    """Translate a single MIR instruction to Python (arithmetic/comparison only)."""
    result = param_map.get(inst.result.name, inst.result.name)
    opcode = inst.opcode
    args = [param_map.get(a.name, a.name) for a in inst.args]

    # Arithmetic operations
    if opcode == 'fdiv':
        return f"{result} = {args[0]} / {args[1]}"
    elif opcode == 'fmul':
        return f"{result} = {args[0]} * {args[1]}"
    elif opcode == 'fadd':
        return f"{result} = {args[0]} + {args[1]}"
    elif opcode == 'fsub':
        return f"{result} = {args[0]} - {args[1]}"
    elif opcode == 'fneg':
        return f"{result} = -{args[0]}"

    # Comparison operations
    elif opcode == 'fle':  # Float less-or-equal
        return f"{result} = {args[0]} <= {args[1]}"
    elif opcode == 'flt':  # Float less-than
        return f"{result} = {args[0]} < {args[1]}"
    elif opcode == 'fge':  # Float greater-or-equal
        return f"{result} = {args[0]} >= {args[1]}"
    elif opcode == 'fgt':  # Float greater-than
        return f"{result} = {args[0]} > {args[1]}"
    elif opcode == 'feq':  # Float equal
        return f"{result} = {args[0]} == {args[1]}"

    # Math functions
    elif opcode == 'sqrt':
        return f"{result} = math.sqrt({args[0]})"
    elif opcode == 'exp':
        return f"{result} = math.exp({args[0]})"

    # Type conversions
    elif opcode == 'ifcast':  # Integer to float cast
        return f"{result} = float({args[0]})"
    elif opcode == 'ficast':  # Float to integer cast
        return f"{result} = int({args[0]})"

    # Optimization barriers (just pass through the value)
    elif opcode == 'optbarrier':
        return f"{result} = {args[0]}  # optbarrier"

    else:
        return f"# Unsupported: {opcode}"
