"""MIR data structures for OpenVAF to JAX translation.

This module provides dataclasses representing MIR (Mid-level IR) structures
as returned by openvaf_py.VaModule methods.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, NewType, Optional

# Type-safe identifiers for MIR elements
# These are strings at runtime but distinct types for type checking
BlockId = NewType('BlockId', str)
"""Identifier for a basic block (e.g., 'block0')."""

ValueId = NewType('ValueId', str)
"""Identifier for an SSA value (e.g., 'v123')."""


# Pre-allocated MIR constants (from OpenVAF mir/src/dfg/values.rs:57-76)
# These are always present at fixed positions but not shown in MIR dumps
V_GRAVESTONE = ValueId('v0')  # Dead value placeholder
V_FALSE = ValueId('v1')       # bool: false
V_TRUE = ValueId('v2')        # bool: true
V_F_ZERO = ValueId('v3')      # f64: 0.0
V_ZERO = ValueId('v4')        # i32: 0
V_ONE = ValueId('v5')         # i32: 1
V_F_ONE = ValueId('v6')       # f64: 1.0
V_F_N_ONE = ValueId('v7')     # f64: -1.0


@dataclass
class PhiOperand:
    """An operand of a PHI node.

    PHI nodes merge values from different control flow paths.
    Each operand specifies:
    - block: The predecessor block this value comes from
    - value: The value ID from that block
    """
    block: BlockId
    value: ValueId


@dataclass
class MIRInstruction:
    """A single MIR instruction.

    MIR instructions are the basic unit of computation. Each instruction:
    - Has an opcode (e.g., 'fadd', 'phi', 'br')
    - May produce a result value
    - Has zero or more operands

    Special instructions:
    - PHI: Merges values from multiple predecessors
    - BR: Conditional branch with true/false targets
    - JMP: Unconditional jump
    - EXIT: Function return
    - CALL: Function call
    """
    opcode: str
    block: str
    result: Optional[ValueId] = None
    operands: List[ValueId] = field(default_factory=list)

    # PHI-specific fields
    phi_operands: Optional[List[PhiOperand]] = None

    # Branch-specific fields
    condition: Optional[ValueId] = None
    true_block: Optional[str] = None
    false_block: Optional[str] = None
    loop_entry: bool = False  # True if this branch is a loop entry point
    target_block: Optional[str] = None  # For JMP

    # Call-specific fields
    func_name: Optional[str] = None

    @property
    def is_phi(self) -> bool:
        return self.opcode.lower() == 'phi'

    @property
    def is_terminator(self) -> bool:
        """Returns True if this instruction ends a basic block."""
        return self.opcode.lower() in ('br', 'jmp', 'exit')

    @property
    def is_branch(self) -> bool:
        return self.opcode.lower() == 'br'

    @property
    def is_jump(self) -> bool:
        return self.opcode.lower() == 'jmp'

    @property
    def is_exit(self) -> bool:
        return self.opcode.lower() == 'exit'


@dataclass
class Block:
    """A basic block in the control flow graph.

    A basic block is a sequence of instructions with:
    - Single entry point (first instruction)
    - Single exit point (last instruction, a terminator)
    - No branches except at the end
    """
    name: str
    instructions: List[MIRInstruction] = field(default_factory=list)
    predecessors: List[str] = field(default_factory=list)
    successors: List[str] = field(default_factory=list)

    @property
    def phi_nodes(self) -> List[MIRInstruction]:
        """Get all PHI nodes at the start of this block."""
        return [inst for inst in self.instructions if inst.is_phi]

    @property
    def terminator(self) -> Optional[MIRInstruction]:
        """Get the terminating instruction of this block."""
        for inst in reversed(self.instructions):
            if inst.is_terminator:
                return inst
        return None

    @property
    def body_instructions(self) -> List[MIRInstruction]:
        """Get non-PHI, non-terminator instructions."""
        return [inst for inst in self.instructions
                if not inst.is_phi and not inst.is_terminator]


@dataclass
class MIRFunction:
    """A complete MIR function (init or eval).

    Contains all blocks, constants, and parameters for a function.
    """
    name: str
    blocks: Dict[str, Block] = field(default_factory=dict)

    # Constants
    constants: Dict[str, float] = field(default_factory=dict)
    bool_constants: Dict[str, bool] = field(default_factory=dict)
    int_constants: Dict[str, int] = field(default_factory=dict)
    str_constants: Dict[str, str] = field(default_factory=dict)

    # Parameters (value IDs in order)
    params: List[str] = field(default_factory=list)

    # Entry block name
    entry_block: str = 'block0'

    @property
    def all_instructions(self) -> List[MIRInstruction]:
        """Get all instructions across all blocks."""
        result = []
        for block in self.blocks.values():
            result.extend(block.instructions)
        return result

    def get_instruction_by_result(self, result: ValueId) -> Optional[MIRInstruction]:
        """Find instruction that produces a given result value."""
        for block in self.blocks.values():
            for inst in block.instructions:
                if inst.result == result:
                    return inst
        return None


def parse_mir_function(
    name: str,
    mir_data: Dict[str, Any],
    str_constants: Optional[Dict[str, str]] = None
) -> MIRFunction:
    """Parse MIR data from openvaf_py into MIRFunction.

    Args:
        name: Function name ('init' or 'eval')
        mir_data: Dict from get_mir_instructions() or get_init_mir_instructions()
        str_constants: Optional dict of string constants from module.get_str_constants()

    Returns:
        Parsed MIRFunction
    """
    # Extract function declarations for resolving func_ref in call instructions
    function_decls = dict(mir_data.get('function_decls', {}))

    func = MIRFunction(
        name=name,
        constants=dict(mir_data.get('constants', {})),
        bool_constants=dict(mir_data.get('bool_constants', {})),
        int_constants=dict(mir_data.get('int_constants', {})),
        str_constants=dict(str_constants) if str_constants else {},
        params=list(mir_data.get('params', [])),
    )

    # Parse blocks
    blocks_data = mir_data.get('blocks', {})
    for block_name, block_data in blocks_data.items():
        block = Block(
            name=block_name,
            predecessors=list(block_data.get('predecessors', [])),
            successors=list(block_data.get('successors', [])),
        )
        func.blocks[block_name] = block

    # Parse instructions and add to blocks
    instructions = mir_data.get('instructions', [])
    for inst_data in instructions:
        inst = _parse_instruction(inst_data, function_decls)
        block_name = inst.block
        if block_name in func.blocks:
            func.blocks[block_name].instructions.append(inst)

    # Set entry block
    # The entry block is the one with no predecessors
    if func.blocks:
        # First try 'block0' which is the conventional entry
        if 'block0' in func.blocks:
            func.entry_block = 'block0'
        else:
            # Find block with no predecessors
            for block_name, block in func.blocks.items():
                if not block.predecessors:
                    func.entry_block = block_name
                    break
            else:
                # Fallback: use block with lowest numeric suffix
                def block_num(name: str) -> int:
                    try:
                        return int(name.replace('block', ''))
                    except ValueError:
                        return 2**31 - 1  # Large int as fallback
                func.entry_block = min(func.blocks.keys(), key=block_num)

    return func


def _parse_instruction(
    inst_data: Dict[str, Any],
    function_decls: Optional[Dict[str, Dict[str, Any]]] = None
) -> MIRInstruction:
    """Parse a single instruction from MIR data.

    Args:
        inst_data: Raw instruction dict from MIR
        function_decls: Function declarations dict for resolving func_ref
    """
    opcode = inst_data.get('opcode', '').lower()

    # Wrap result with ValueId if present
    result_str = inst_data.get('result')
    result = ValueId(result_str) if result_str else None

    # Wrap operands with ValueId
    operands = [ValueId(op) for op in inst_data.get('operands', [])]

    inst = MIRInstruction(
        opcode=opcode,
        block=inst_data.get('block', ''),
        result=result,
        operands=operands,
    )

    # Parse PHI operands
    if opcode == 'phi':
        phi_ops = inst_data.get('phi_operands', [])
        inst.phi_operands = [
            PhiOperand(block=BlockId(op['block']), value=ValueId(op['value']))
            for op in phi_ops
        ]

    # Parse branch-specific fields
    if opcode == 'br':
        cond_str = inst_data.get('condition')
        inst.condition = ValueId(cond_str) if cond_str else None
        inst.true_block = inst_data.get('true_block')
        inst.false_block = inst_data.get('false_block')
        inst.loop_entry = inst_data.get('loop_entry', False)
    elif opcode == 'jmp':
        # JMP uses 'destination' in raw MIR data
        inst.target_block = inst_data.get('destination') or inst_data.get('target_block')

    # Parse call-specific fields
    if opcode == 'call':
        # Try direct func_name first
        func_name = inst_data.get('func_name')
        if not func_name:
            # Look up via func_ref in function_decls
            func_ref = inst_data.get('func_ref')
            if func_ref and function_decls and func_ref in function_decls:
                func_name = function_decls[func_ref].get('name')
        inst.func_name = func_name

    return inst
