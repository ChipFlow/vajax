"""Parser for OpenVAF MIR snapshot files"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any
from enum import Enum, auto


class OpCode(Enum):
    """MIR operation codes"""
    # Constants
    FCONST = auto()
    SCONST = auto()
    # Arithmetic
    FADD = auto()
    FSUB = auto()
    FMUL = auto()
    FDIV = auto()
    FNEG = auto()
    # Math functions
    EXP = auto()
    LN = auto()
    SQRT = auto()
    POW = auto()
    # Comparisons
    FEQ = auto()
    FLT = auto()
    FGT = auto()
    FLE = auto()
    FGE = auto()
    # Control flow
    PHI = auto()
    BR = auto()
    JMP = auto()
    # Function calls
    CALL = auto()
    # Other
    OPTBARRIER = auto()
    CONST = auto()  # For const fn declarations


@dataclass
class PhiOperand:
    """A single PHI operand with value and source block"""
    value: str
    block: str


@dataclass
class Instruction:
    """A single MIR instruction"""
    result: str  # Result variable (e.g., "v21")
    opcode: OpCode
    operands: List[str]  # Operand variables or constants
    block: Optional[str] = None  # Block this instruction is in
    phi_operands: Optional[List[PhiOperand]] = None  # For PHI nodes
    call_target: Optional[str] = None  # For CALL instructions


@dataclass
class BranchInst:
    """A branch instruction (block terminator)"""
    condition: Optional[str]  # None for unconditional jump
    true_block: str
    false_block: Optional[str] = None  # None for unconditional jump


@dataclass
class Block:
    """A basic block in the MIR"""
    name: str
    instructions: List[Instruction] = field(default_factory=list)
    terminator: Optional[BranchInst] = None
    predecessors: List[str] = field(default_factory=list)
    successors: List[str] = field(default_factory=list)


@dataclass
class FunctionDecl:
    """A function declaration (inst0 = const fn %name(...) -> ...)"""
    name: str
    num_args: int
    num_returns: int


@dataclass
class MIRFunction:
    """Parsed MIR function"""
    params: List[str]  # Input parameters
    constants: Dict[str, float]  # Named constants (fconst)
    string_constants: Dict[str, str]  # String constants (sconst)
    blocks: Dict[str, Block]  # Basic blocks
    instructions: List[Instruction]  # All instructions in order
    function_decls: Dict[str, FunctionDecl]  # Function declarations


@dataclass
class Residual:
    """Residual equation for a node"""
    resist: str  # Resistive component variable
    react: str   # Reactive component variable


@dataclass
class JacobianEntry:
    """Jacobian matrix entry"""
    row: str     # Row node
    col: str     # Column node
    resist: str  # Resistive component variable
    react: str   # Reactive component variable


@dataclass
class DaeSystem:
    """Parsed DAE system from OpenVAF"""
    unknowns: Dict[str, str]  # sim_nodeN -> nodeN
    residuals: Dict[str, Residual]
    jacobian: List[JacobianEntry]
    model_inputs: List[Tuple[int, int]]  # Input port pairs
    num_resistive: int
    num_reactive: int


def parse_float_hex(hex_str: str) -> float:
    """Parse hexadecimal float representation (e.g., 0x1.0000000000000p0)"""
    hex_str = hex_str.strip()
    if hex_str.startswith("-"):
        return -float.fromhex(hex_str[1:])
    return float.fromhex(hex_str)


def parse_mir(mir_text: str) -> MIRFunction:
    """Parse MIR text into a MIRFunction structure"""
    lines = mir_text.strip().split('\n')

    # Parse function signature
    func_match = re.match(r'function %\(([^)]*)\)', lines[0])
    if not func_match:
        raise ValueError(f"Invalid MIR function signature: {lines[0]}")

    params = [p.strip() for p in func_match.group(1).split(',') if p.strip()]

    constants: Dict[str, float] = {}
    string_constants: Dict[str, str] = {}
    blocks: Dict[str, Block] = {}
    instructions: List[Instruction] = []
    function_decls: Dict[str, FunctionDecl] = {}
    current_block: Optional[Block] = None

    # Opcode mapping
    opcode_map = {
        'fconst': OpCode.FCONST,
        'sconst': OpCode.SCONST,
        'fadd': OpCode.FADD,
        'fsub': OpCode.FSUB,
        'fmul': OpCode.FMUL,
        'fdiv': OpCode.FDIV,
        'fneg': OpCode.FNEG,
        'exp': OpCode.EXP,
        'ln': OpCode.LN,
        'sqrt': OpCode.SQRT,
        'pow': OpCode.POW,
        'feq': OpCode.FEQ,
        'flt': OpCode.FLT,
        'fgt': OpCode.FGT,
        'fle': OpCode.FLE,
        'fge': OpCode.FGE,
        'phi': OpCode.PHI,
        'optbarrier': OpCode.OPTBARRIER,
        'call': OpCode.CALL,
        'const': OpCode.CONST,
    }

    for line in lines[1:]:
        line = line.strip()
        if not line or line == '}':
            continue

        # Block definition
        block_match = re.match(r'block(\d+):', line)
        if block_match:
            block_name = f"block{block_match.group(1)}"
            current_block = Block(name=block_name)
            blocks[block_name] = current_block
            continue

        # Branch instruction: br v18, block2, block4
        br_match = re.match(r'(?:@[0-9a-fA-F]+\s+)?br\s+(\w+),\s*(\w+),\s*(\w+)', line)
        if br_match:
            if current_block:
                current_block.terminator = BranchInst(
                    condition=br_match.group(1),
                    true_block=br_match.group(2),
                    false_block=br_match.group(3)
                )
                current_block.successors = [br_match.group(2), br_match.group(3)]
            continue

        # Jump instruction: jmp block4
        jmp_match = re.match(r'(?:@[0-9a-fA-F]+\s+)?jmp\s+(\w+)', line)
        if jmp_match:
            if current_block:
                current_block.terminator = BranchInst(
                    condition=None,
                    true_block=jmp_match.group(1)
                )
                current_block.successors = [jmp_match.group(1)]
            continue

        # Function declaration: inst0 = const fn %ddt(1) -> 1
        fn_decl_match = re.match(r'(\w+)\s*=\s*(?:const\s+)?fn\s+%(\w+)(?:\([^)]*\))?\((\d+)\)\s*->\s*(\d+)', line)
        if fn_decl_match:
            inst_name = fn_decl_match.group(1)
            fn_name = fn_decl_match.group(2)
            num_args = int(fn_decl_match.group(3))
            num_returns = int(fn_decl_match.group(4))
            function_decls[inst_name] = FunctionDecl(fn_name, num_args, num_returns)
            continue

        # Also match: inst4 = fn %collapse_node3_Some(node1)(0) -> 0
        fn_decl_match2 = re.match(r'(\w+)\s*=\s*(?:const\s+)?fn\s+%([^\(]+)\([^)]*\)\((\d+)\)\s*->\s*(\d+)', line)
        if fn_decl_match2:
            inst_name = fn_decl_match2.group(1)
            fn_name = fn_decl_match2.group(2)
            num_args = int(fn_decl_match2.group(3))
            num_returns = int(fn_decl_match2.group(4))
            function_decls[inst_name] = FunctionDecl(fn_name, num_args, num_returns)
            continue

        # Call with no result: call inst4()
        call_no_result = re.match(r'(?:@[0-9a-fA-F]+\s+)?call\s+(\w+)\(\)', line)
        if call_no_result:
            # Side-effect only call, skip for now
            continue

        # Instruction with optional address prefix
        # Format: @ADDR  v21 = opcode operands
        # or:     v21 = opcode operands
        inst_match = re.match(r'(?:@[0-9a-fA-F]+\s+)?(\w+)\s*=\s*(\w+)\s*(.*)', line)
        if inst_match:
            result = inst_match.group(1)
            opcode_str = inst_match.group(2).lower()
            operands_str = inst_match.group(3).strip()

            opcode = opcode_map.get(opcode_str)
            if opcode is None:
                # Skip unknown opcodes
                continue

            # Parse operands based on opcode type
            phi_operands = None
            call_target = None
            operands = []

            if opcode == OpCode.FCONST:
                # Floating point constant
                try:
                    val = parse_float_hex(operands_str)
                    constants[result] = val
                except ValueError:
                    # Handle special cases like 0.0
                    if operands_str == '0.0':
                        constants[result] = 0.0
                    else:
                        constants[result] = float(operands_str)
                operands = [operands_str]

            elif opcode == OpCode.SCONST:
                # String constant: "<DUMMY>"
                str_match = re.match(r'"([^"]*)"', operands_str)
                if str_match:
                    string_constants[result] = str_match.group(1)
                operands = [operands_str]

            elif opcode == OpCode.PHI:
                # phi [v21, block2], [v19, block20]
                phi_matches = re.findall(r'\[(\w+),\s*(\w+)\]', operands_str)
                phi_operands = [PhiOperand(value=m[0], block=m[1]) for m in phi_matches]
                operands = [m[0] for m in phi_matches]

            elif opcode == OpCode.CALL:
                # call inst1(v125, v126)
                call_match = re.match(r'(\w+)\(([^)]*)\)', operands_str)
                if call_match:
                    call_target = call_match.group(1)
                    args_str = call_match.group(2)
                    operands = re.findall(r'\w+', args_str) if args_str else []

            elif opcode == OpCode.CONST:
                # const fn declaration - already handled above
                continue

            else:
                # Regular operands: v19, v20 or hex constants
                operands = re.findall(r'v\d+', operands_str)

            inst = Instruction(
                result=result,
                opcode=opcode,
                operands=operands,
                block=current_block.name if current_block else None,
                phi_operands=phi_operands,
                call_target=call_target
            )
            instructions.append(inst)
            if current_block:
                current_block.instructions.append(inst)

    # Build predecessor lists
    for block_name, block in blocks.items():
        for succ_name in block.successors:
            if succ_name in blocks:
                if block_name not in blocks[succ_name].predecessors:
                    blocks[succ_name].predecessors.append(block_name)

    return MIRFunction(
        params=params,
        constants=constants,
        string_constants=string_constants,
        blocks=blocks,
        instructions=instructions,
        function_decls=function_decls
    )


def parse_system(system_text: str) -> DaeSystem:
    """Parse DaeSystem text into a DaeSystem structure"""
    unknowns: Dict[str, str] = {}
    residuals: Dict[str, Residual] = {}
    jacobian: List[JacobianEntry] = []
    model_inputs: List[Tuple[int, int]] = []
    num_resistive = 0
    num_reactive = 0

    # Parse unknowns
    unknowns_match = re.search(r'unknowns:\s*\{([^}]+)\}', system_text, re.DOTALL)
    if unknowns_match:
        for line in unknowns_match.group(1).strip().split('\n'):
            match = re.match(r'\s*(sim_node\d+):\s*(\w+)', line)
            if match:
                unknowns[match.group(1)] = match.group(2)

    # Parse residuals
    residual_pattern = re.compile(
        r'(sim_node\d+):\s*Residual\s*\{[^}]*resist:\s*(\w+)[^}]*react:\s*(\w+)',
        re.DOTALL
    )
    for match in residual_pattern.finditer(system_text):
        node = match.group(1)
        residuals[node] = Residual(
            resist=match.group(2),
            react=match.group(3)
        )

    # Parse Jacobian entries
    jacobian_pattern = re.compile(
        r'j\d+:\s*MatrixEntry\s*\{[^}]*row:\s*(sim_node\d+)[^}]*col:\s*(sim_node\d+)[^}]*resist:\s*(\w+)[^}]*react:\s*(\w+)',
        re.DOTALL
    )
    for match in jacobian_pattern.finditer(system_text):
        jacobian.append(JacobianEntry(
            row=match.group(1),
            col=match.group(2),
            resist=match.group(3),
            react=match.group(4)
        ))

    # Parse model_inputs
    inputs_match = re.search(r'model_inputs:\s*\[(.*?)\]', system_text, re.DOTALL)
    if inputs_match:
        pair_pattern = re.compile(r'\(\s*(\d+)\s*,\s*(\d+)\s*,?\s*\)')
        for match in pair_pattern.finditer(inputs_match.group(1)):
            model_inputs.append((int(match.group(1)), int(match.group(2))))

    # Parse counts
    num_resist_match = re.search(r'num_resistive:\s*(\d+)', system_text)
    if num_resist_match:
        num_resistive = int(num_resist_match.group(1))

    num_react_match = re.search(r'num_reactive:\s*(\d+)', system_text)
    if num_react_match:
        num_reactive = int(num_react_match.group(1))

    return DaeSystem(
        unknowns=unknowns,
        residuals=residuals,
        jacobian=jacobian,
        model_inputs=model_inputs,
        num_resistive=num_resistive,
        num_reactive=num_reactive
    )
