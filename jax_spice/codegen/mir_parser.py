"""Simple MIR parser for OpenVAF's text format."""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple


@dataclass
class MIRValue:
    """A value in MIR (SSA form - assigned once)."""
    name: str  # e.g., "v18"

    def __str__(self):
        return self.name


@dataclass
class MIRInstruction:
    """A single MIR instruction."""
    result: Optional[MIRValue]  # None for void instructions
    opcode: str  # e.g., "fdiv", "fmul", "fneg", "br", "jmp"
    args: List[MIRValue]  # Operands
    target_blocks: List[str] = field(default_factory=list)  # For br/jmp - target block names
    phi_args: List[Tuple[str, str]] = field(default_factory=list)  # For PHI: [(value, block), ...]

    def __str__(self):
        if self.result:
            return f"{self.result} = {self.opcode}({', '.join(str(a) for a in self.args)})"
        elif self.target_blocks:
            return f"{self.opcode} {', '.join(self.target_blocks)}"
        else:
            return f"{self.opcode}({', '.join(str(a) for a in self.args)})"


@dataclass
class MIRBlock:
    """A basic block in MIR."""
    name: str  # e.g., "block5"
    instructions: List[MIRInstruction]

    def __str__(self):
        insts = '\n  '.join(str(i) for i in self.instructions)
        return f"{self.name}:\n  {insts}"


@dataclass
class MIRFunction:
    """A complete MIR function."""
    name: str
    params: List[MIRValue]  # Function parameters
    blocks: List[MIRBlock]
    constants: Dict[str, any]  # Constant definitions

    def __str__(self):
        blocks_str = '\n\n'.join(str(b) for b in self.blocks)
        return f"function {self.name}({', '.join(str(p) for p in self.params)}):\n{blocks_str}"


def parse_mir_function(mir_text: str) -> MIRFunction:
    """Parse MIR function from OpenVAF's text dump format.

    Example input:
        Optimized evaluation MIR of resistor
        function %(v16, v17, v20, v21) {
            block5:
                v18 = fdiv v16, v17
                v55 = fneg v18
        }

    Note: OpenVAF pre-allocates common constants at fixed positions:
        v0: GRAVESTONE (dead PHI placeholder)
        v1: FALSE (bool = false)
        v2: TRUE (bool = true)
        v3: F_ZERO (f64 = 0.0)
        v4: ZERO (i32 = 0)
        v5: ONE (i32 = 1)
        v6: F_ONE (f64 = 1.0)
        v7: F_N_ONE (f64 = -1.0)
    """
    lines = mir_text.strip().split('\n')

    # Find function signature
    func_pattern = r'function %\((.*?)\)'
    params = []

    # Pre-allocated OpenVAF constants (from openvaf/mir/src/dfg/values.rs)
    constants = {
        'v0': None,     # GRAVESTONE - placeholder for dead values
        'v1': False,    # FALSE
        'v2': True,     # TRUE
        'v3': 0.0,      # F_ZERO
        'v4': 0,        # ZERO (i32)
        'v5': 1,        # ONE (i32)
        'v6': 1.0,      # F_ONE
        'v7': -1.0,     # F_N_ONE
    }
    blocks = []

    current_block = None

    for line in lines:
        line = line.strip()

        # Parse function signature
        if line.startswith('function'):
            match = re.search(func_pattern, line)
            if match:
                param_str = match.group(1)
                if param_str:
                    params = [MIRValue(p.strip()) for p in param_str.split(',')]

        # Parse constant definitions
        elif '= fconst' in line or '= iconst' in line or '= bconst' in line:
            parts = line.split('=')
            if len(parts) == 2:
                name = parts[0].strip()
                value_str = parts[1].strip()
                # Extract the constant value
                if 'fconst' in value_str:
                    const_val = value_str.replace('fconst', '').strip()
                    constants[name] = const_val
                elif 'iconst' in value_str:
                    const_val = int(value_str.replace('iconst', '').strip())
                    constants[name] = const_val
                elif 'bconst' in value_str:
                    const_val = value_str.replace('bconst', '').strip() == 'true'
                    constants[name] = const_val

        # Parse block start
        elif line.startswith('block') and line.endswith(':'):
            if current_block:
                blocks.append(current_block)
            block_name = line.rstrip(':')
            current_block = MIRBlock(block_name, [])

        # Parse instruction
        elif current_block and not line.startswith('//'):
            # Remove annotation prefix like @0006
            if line.startswith('@'):
                line = line.split(None, 1)[1] if ' ' in line else ''
                if not line:
                    continue

            # Handle control flow instructions (br, jmp)
            if line.startswith('br '):
                # Format: "br v20, block2, block11"
                parts = line.split()
                if len(parts) >= 4:
                    cond = MIRValue(parts[1].rstrip(','))
                    true_block = parts[2].rstrip(',')
                    false_block = parts[3].rstrip(',')
                    inst = MIRInstruction(
                        result=None,
                        opcode='br',
                        args=[cond],
                        target_blocks=[true_block, false_block]
                    )
                    current_block.instructions.append(inst)

            elif line.startswith('jmp '):
                # Format: "jmp block4"
                parts = line.split()
                if len(parts) >= 2:
                    target = parts[1].rstrip(',')
                    inst = MIRInstruction(
                        result=None,
                        opcode='jmp',
                        args=[],
                        target_blocks=[target]
                    )
                    current_block.instructions.append(inst)

            elif line.startswith('call '):
                # Format: "call inst0()"
                # For now, treat as void instruction
                inst = MIRInstruction(
                    result=None,
                    opcode='call',
                    args=[],
                    target_blocks=[]
                )
                current_block.instructions.append(inst)

            elif '=' in line:
                # Regular instruction with result
                parts = line.split('=', 1)
                if len(parts) == 2:
                    result_name = parts[0].strip()
                    inst_str = parts[1].strip()

                    # Handle PHI nodes specially
                    if inst_str.startswith('phi '):
                        # Format: "phi [v1, block2], [v24, block7]"
                        # Parse PHI arguments: [(value, block), ...]
                        phi_pairs = []
                        # Find all [value, block] pairs
                        pair_pattern = r'\[(\w+),\s*(\w+)\]'
                        for match in re.finditer(pair_pattern, inst_str):
                            value_name = match.group(1)
                            block_name = match.group(2)
                            phi_pairs.append((value_name, block_name))

                        inst = MIRInstruction(
                            result=MIRValue(result_name),
                            opcode='phi',
                            args=[],
                            target_blocks=[],
                            phi_args=phi_pairs
                        )
                        current_block.instructions.append(inst)
                    else:
                        # Parse opcode and args
                        # Format: "fdiv v16, v17" or "fle v3, v19"
                        tokens = inst_str.split()
                        if tokens:
                            opcode = tokens[0]
                            args = [MIRValue(t.rstrip(',')) for t in tokens[1:] if t and not t.startswith('(')]

                            inst = MIRInstruction(
                                result=MIRValue(result_name),
                                opcode=opcode,
                                args=args,
                                target_blocks=[]
                            )
                            current_block.instructions.append(inst)

    # Don't forget the last block
    if current_block:
        blocks.append(current_block)

    return MIRFunction(
        name="eval",
        params=params,
        blocks=blocks,
        constants=constants
    )


def parse_mir_dict(mir_dict: Dict) -> MIRFunction:
    """Parse MIR function from dict format (from openvaf_py).

    The dict has keys: 'params', 'instructions', 'blocks', 'constants', etc.
    Instructions are dictionaries with 'block', 'opcode', 'operands', etc.

    Example input:
        {
            'params': ['v18', 'v20', 'v32'],
            'constants': {'v3': 0.0, 'v6': 1.0, ...},
            'bool_constants': {'v1': False, 'v2': True},
            'int_constants': {'v4': 0, 'v5': 1, ...},
            'instructions': [
                {'block': 'block2', 'result': 'v18', 'opcode': 'fmul', 'operands': ['v16', 'v17']},
                {'block': 'block2', 'opcode': 'br', 'condition': 'v20', 'true_block': 'block5', 'false_block': 'block6'},
                ...
            ],
            'blocks': {
                'block2': {'predecessors': [...], 'successors': [...]},
                ...
            }
        }
    """
    # Parse parameters
    params = [MIRValue(name) for name in mir_dict['params']]

    # Merge all constants into one dict
    constants = {}
    if 'constants' in mir_dict:
        constants.update(mir_dict['constants'])
    if 'bool_constants' in mir_dict:
        constants.update(mir_dict['bool_constants'])
    if 'int_constants' in mir_dict:
        constants.update(mir_dict['int_constants'])

    # Group instructions by block
    blocks_dict: Dict[str, List[Dict]] = {}
    for inst in mir_dict['instructions']:
        block_name = inst['block']
        if block_name not in blocks_dict:
            blocks_dict[block_name] = []
        blocks_dict[block_name].append(inst)

    # Convert instructions to MIRInstruction objects
    blocks = []
    for block_name, insts_list in blocks_dict.items():
        mir_insts = []
        for inst in insts_list:
            opcode = inst['opcode']
            result = MIRValue(inst['result']) if 'result' in inst else None

            if opcode == 'br':
                # Branch instruction
                cond = MIRValue(inst['condition'])
                true_block = inst['true_block']
                false_block = inst['false_block']
                mir_inst = MIRInstruction(
                    result=None,
                    opcode='br',
                    args=[cond],
                    target_blocks=[true_block, false_block]
                )
            elif opcode == 'jmp':
                # Jump instruction
                target = inst['destination']
                mir_inst = MIRInstruction(
                    result=None,
                    opcode='jmp',
                    args=[],
                    target_blocks=[target]
                )
            elif opcode == 'phi':
                # PHI node
                phi_operands = inst['phi_operands']
                phi_args = [(op['value'], op['block']) for op in phi_operands]
                mir_inst = MIRInstruction(
                    result=result,
                    opcode='phi',
                    args=[],
                    target_blocks=[],
                    phi_args=phi_args
                )
            elif opcode == 'call':
                # Function call
                mir_inst = MIRInstruction(
                    result=result,
                    opcode='call',
                    args=[],
                    target_blocks=[]
                )
            else:
                # Regular instruction
                operands = inst.get('operands', [])
                args = [MIRValue(op) for op in operands]
                mir_inst = MIRInstruction(
                    result=result,
                    opcode=opcode,
                    args=args,
                    target_blocks=[]
                )

            mir_insts.append(mir_inst)

        blocks.append(MIRBlock(block_name, mir_insts))

    return MIRFunction(
        name="init",
        params=params,
        blocks=blocks,
        constants=constants
    )
