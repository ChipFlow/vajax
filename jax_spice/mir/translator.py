"""MIR to JAX translator with control flow support"""

from typing import Dict, List, Callable, Any, Tuple, Set, Optional
import jax.numpy as jnp
from jax import lax

from jax_spice.mir.parser import (
    MIRFunction, DaeSystem, OpCode, Instruction, Block, BranchInst,
    PhiOperand, FunctionDecl, parse_mir, parse_system
)


class MIRToJAX:
    """Translates OpenVAF MIR to JAX functions with control flow support"""

    def __init__(self, mir: MIRFunction, system: DaeSystem):
        self.mir = mir
        self.system = system
        # Track which block each variable is defined in
        self.var_blocks: Dict[str, str] = {}
        # Track the execution path for PHI resolution
        self.execution_path: List[str] = []

    @classmethod
    def from_snapshots(cls, mir_text: str, system_text: str) -> "MIRToJAX":
        """Create translator from snapshot file contents"""
        mir = parse_mir(mir_text)
        system = parse_system(system_text)
        return cls(mir, system)

    def translate(self) -> Callable:
        """Generate a JAX function from the MIR

        Returns a function with signature:
            f(inputs: List[float]) -> (residuals: Dict, jacobian: Dict)
        """
        # Build the computation using an interpreter approach
        # This handles control flow by evaluating both branches and using jnp.where

        # Generate Python/JAX code
        code_lines = self._generate_code()
        code = '\n'.join(code_lines)

        # Compile and return
        local_ns = {'jnp': jnp, 'lax': lax}
        exec(code, local_ns)
        return local_ns['device_eval']

    def _generate_code(self) -> List[str]:
        """Generate the JAX function code"""
        lines = []
        lines.append("def device_eval(inputs):")
        lines.append("    import jax.numpy as jnp")
        lines.append("    from jax import lax")
        lines.append("")

        # Initialize constants
        lines.append("    # Constants")
        for name, value in self.mir.constants.items():
            lines.append(f"    {name} = {repr(value)}")

        # v3 is typically zero (used for unused components)
        if 'v3' not in self.mir.constants:
            lines.append("    v3 = 0.0")

        lines.append("")

        # Map function parameters to inputs
        lines.append("    # Input parameters")
        for i, param in enumerate(self.mir.params):
            lines.append(f"    {param} = inputs[{i}]")

        lines.append("")

        # Process blocks in topological order
        # For the diode, we need to handle the control flow properly
        block_order = self._topological_sort()

        # Track which variables have been defined
        defined_vars: Set[str] = set(self.mir.constants.keys())
        defined_vars.update(self.mir.params)
        defined_vars.add('v3')

        # Track condition variables for PHI resolution
        condition_vars: Dict[str, str] = {}  # block -> condition that led there

        lines.append("    # Computation")

        for block_name in block_order:
            if block_name not in self.mir.blocks:
                continue

            block = self.mir.blocks[block_name]
            lines.append(f"")
            lines.append(f"    # {block_name}")

            # Get the condition that leads to this block (for PHI nodes)
            for pred_name in block.predecessors:
                if pred_name in self.mir.blocks:
                    pred = self.mir.blocks[pred_name]
                    if pred.terminator and pred.terminator.condition:
                        if pred.terminator.true_block == block_name:
                            condition_vars[block_name] = f"{pred.terminator.condition}"
                        elif pred.terminator.false_block == block_name:
                            condition_vars[block_name] = f"jnp.logical_not({pred.terminator.condition})"

            for inst in block.instructions:
                expr = self._translate_instruction(inst, defined_vars, condition_vars)
                if expr:
                    lines.append(f"    {inst.result} = {expr}")
                    defined_vars.add(inst.result)

        lines.append("")

        # Build output expressions
        lines.append("    # Build outputs")
        lines.append("    residuals = {")
        for node, res in self.system.residuals.items():
            resist_val = res.resist if res.resist in defined_vars else '0.0'
            react_val = res.react if res.react in defined_vars else '0.0'
            lines.append(f"        '{node}': {{'resist': {resist_val}, 'react': {react_val}}},")
        lines.append("    }")

        lines.append("    jacobian = {")
        for entry in self.system.jacobian:
            key = f"('{entry.row}', '{entry.col}')"
            resist_val = entry.resist if entry.resist in defined_vars else '0.0'
            react_val = entry.react if entry.react in defined_vars else '0.0'
            lines.append(f"        {key}: {{'resist': {resist_val}, 'react': {react_val}}},")
        lines.append("    }")

        lines.append("    return residuals, jacobian")

        return lines

    def _topological_sort(self) -> List[str]:
        """Sort blocks in topological order"""
        if not self.mir.blocks:
            return []

        # Find entry block (usually block with smallest number or no predecessors)
        entry_blocks = [name for name, block in self.mir.blocks.items()
                       if not block.predecessors]
        if not entry_blocks:
            # Fall back to sorted order
            entry_blocks = sorted(self.mir.blocks.keys(),
                                 key=lambda x: int(x.replace('block', '')))

        visited = set()
        result = []

        def visit(name: str):
            if name in visited or name not in self.mir.blocks:
                return
            visited.add(name)
            block = self.mir.blocks[name]
            for succ in block.successors:
                visit(succ)
            result.append(name)

        for entry in entry_blocks:
            visit(entry)

        return list(reversed(result))

    def _translate_instruction(
        self,
        inst: Instruction,
        defined_vars: Set[str],
        condition_vars: Dict[str, str]
    ) -> Optional[str]:
        """Translate a single instruction to JAX expression"""

        def get_operand(op: str) -> str:
            if op in defined_vars:
                return op
            if op in self.mir.constants:
                return repr(self.mir.constants[op])
            # Check if it's a hex float constant
            if op.startswith('0x') or op.startswith('-0x'):
                try:
                    return repr(float.fromhex(op.replace('-0x', '-0x')))
                except ValueError:
                    pass
            return op

        if inst.opcode == OpCode.FCONST:
            # Already handled in constants
            return None

        elif inst.opcode == OpCode.SCONST:
            # String constant - return as string literal
            if inst.result in self.mir.string_constants:
                return repr(self.mir.string_constants[inst.result])
            return None

        elif inst.opcode == OpCode.FADD:
            ops = [get_operand(op) for op in inst.operands]
            return f"({ops[0]} + {ops[1]})"

        elif inst.opcode == OpCode.FSUB:
            ops = [get_operand(op) for op in inst.operands]
            return f"({ops[0]} - {ops[1]})"

        elif inst.opcode == OpCode.FMUL:
            ops = [get_operand(op) for op in inst.operands]
            return f"({ops[0]} * {ops[1]})"

        elif inst.opcode == OpCode.FDIV:
            ops = [get_operand(op) for op in inst.operands]
            return f"({ops[0]} / {ops[1]})"

        elif inst.opcode == OpCode.FNEG:
            ops = [get_operand(op) for op in inst.operands]
            return f"(-{ops[0]})"

        elif inst.opcode == OpCode.EXP:
            ops = [get_operand(op) for op in inst.operands]
            return f"jnp.exp({ops[0]})"

        elif inst.opcode == OpCode.LN:
            ops = [get_operand(op) for op in inst.operands]
            return f"jnp.log({ops[0]})"

        elif inst.opcode == OpCode.SQRT:
            ops = [get_operand(op) for op in inst.operands]
            return f"jnp.sqrt({ops[0]})"

        elif inst.opcode == OpCode.POW:
            ops = [get_operand(op) for op in inst.operands]
            return f"jnp.power({ops[0]}, {ops[1]})"

        elif inst.opcode == OpCode.FEQ:
            ops = [get_operand(op) for op in inst.operands]
            return f"({ops[0]} == {ops[1]})"

        elif inst.opcode == OpCode.FLT:
            ops = [get_operand(op) for op in inst.operands]
            return f"({ops[0]} < {ops[1]})"

        elif inst.opcode == OpCode.FGT:
            ops = [get_operand(op) for op in inst.operands]
            return f"({ops[0]} > {ops[1]})"

        elif inst.opcode == OpCode.FLE:
            ops = [get_operand(op) for op in inst.operands]
            return f"({ops[0]} <= {ops[1]})"

        elif inst.opcode == OpCode.FGE:
            ops = [get_operand(op) for op in inst.operands]
            return f"({ops[0]} >= {ops[1]})"

        elif inst.opcode == OpCode.OPTBARRIER:
            # Optimization barrier - just pass through
            ops = [get_operand(op) for op in inst.operands]
            return ops[0]

        elif inst.opcode == OpCode.PHI:
            # PHI node - select value based on control flow
            if inst.phi_operands and len(inst.phi_operands) >= 2:
                # For a 2-way PHI, use jnp.where
                phi_ops = inst.phi_operands

                # Find the condition that determines which value to use
                # Look at the first operand's block to find the branch condition
                first_block = phi_ops[0].block
                second_block = phi_ops[1].block

                # Get condition from the common predecessor
                cond = None
                for pred_name in self.mir.blocks.get(inst.block, Block(inst.block)).predecessors:
                    if pred_name in self.mir.blocks:
                        pred = self.mir.blocks[pred_name]
                        if pred.terminator and pred.terminator.condition:
                            if pred.terminator.true_block == first_block:
                                cond = pred.terminator.condition
                                break
                            elif pred.terminator.true_block == second_block:
                                cond = pred.terminator.condition
                                # Swap operands since condition is for second block
                                phi_ops = [phi_ops[1], phi_ops[0]]
                                break

                if cond and cond in defined_vars:
                    val0 = get_operand(phi_ops[0].value)
                    val1 = get_operand(phi_ops[1].value)
                    return f"jnp.where({cond}, {val0}, {val1})"
                else:
                    # Fallback: just use first value
                    return get_operand(phi_ops[0].value)
            elif inst.operands:
                return get_operand(inst.operands[0])
            return '0.0'

        elif inst.opcode == OpCode.CALL:
            # Function call - handle known functions
            if inst.call_target and inst.call_target in self.mir.function_decls:
                fn_decl = self.mir.function_decls[inst.call_target]
                fn_name = fn_decl.name

                if fn_name == 'simparam_opt':
                    # $simparam("name", default) - return the default value
                    # The first arg is the parameter name (string), second is default
                    if len(inst.operands) >= 2:
                        default_val = get_operand(inst.operands[1])
                        return default_val
                    return '1e-12'  # gmin default

                elif fn_name == 'ddt':
                    # Time derivative - for DC analysis, return 0
                    # For transient, this would need state handling
                    return '0.0'

                elif fn_name.startswith('ddx_'):
                    # Derivative with respect to a variable
                    # For now, return 0 (would need AD in full implementation)
                    return '0.0'

                elif fn_name.startswith('flickr_noise') or fn_name.startswith('white_noise'):
                    # Noise functions - return 0 for DC analysis
                    return '0.0'

                elif fn_name.startswith('collapse_node'):
                    # Node collapsing - side effect only
                    return None

            # Unknown function - return 0
            return '0.0'

        return None

    def get_parameter_info(self) -> List[str]:
        """Get the list of input parameters expected by the function"""
        return self.mir.params

    def get_node_mapping(self) -> Dict[str, str]:
        """Get the mapping from sim_node to actual node names"""
        return self.system.unknowns

    def get_function_summary(self) -> str:
        """Get a summary of the parsed MIR function"""
        lines = []
        lines.append(f"Parameters: {len(self.mir.params)}")
        lines.append(f"Constants: {len(self.mir.constants)}")
        lines.append(f"Blocks: {len(self.mir.blocks)}")
        lines.append(f"Instructions: {len(self.mir.instructions)}")
        lines.append(f"Function declarations: {len(self.mir.function_decls)}")

        if self.mir.function_decls:
            lines.append("Functions used:")
            for name, decl in self.mir.function_decls.items():
                lines.append(f"  {name}: {decl.name}({decl.num_args}) -> {decl.num_returns}")

        lines.append(f"\nSystem nodes: {list(self.system.unknowns.values())}")
        lines.append(f"Residuals: {len(self.system.residuals)}")
        lines.append(f"Jacobian entries: {len(self.system.jacobian)}")

        return '\n'.join(lines)
