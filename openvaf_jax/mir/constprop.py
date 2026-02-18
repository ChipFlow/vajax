"""Sparse Conditional Constant Propagation (SCCP) for MIR.

This module implements the Wegman-Zadeck SCCP algorithm to:
1. Propagate constants through the MIR
2. Identify unreachable code based on constant branch conditions
3. Simplify PHI nodes by eliminating dead predecessors

For PSP102 and similar models with NMOS/PMOS branching, this allows
static evaluation of TYPE==1 conditions to eliminate the unused path.

Reference: Wegman & Zadeck, "Constant Propagation with Conditional Branches"
           ACM TOPLAS 1991
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple

from .types import (
    V_F_N_ONE,
    V_F_ONE,
    V_F_ZERO,
    V_FALSE,
    V_ONE,
    V_TRUE,
    V_ZERO,
    MIRFunction,
    MIRInstruction,
    ValueId,
)


class LatticeState(Enum):
    """State in the constant propagation lattice."""
    TOP = auto()      # Unknown/unexecuted - could be anything
    CONSTANT = auto()  # Known constant value
    BOTTOM = auto()    # Not constant - varies at runtime


class ConstType(Enum):
    """Type of a constant value."""
    BOOL = "bool"
    INT = "int"
    FLOAT = "float"
    STRING = "string"


@dataclass(frozen=True)
class LatticeValue:
    """A value in the SCCP lattice.

    The lattice is:
        TOP (unknown)
       /|\\
      c1 c2 c3 ... (constants)
       \\|/
       BOTTOM (not constant)

    TOP means we haven't computed this value yet.
    CONSTANT means we know the exact value.
    BOTTOM means the value varies (not constant).
    """
    state: LatticeState
    const_type: Optional[ConstType] = None
    value: Optional[Any] = None  # The constant value if state == CONSTANT

    @classmethod
    def top(cls) -> 'LatticeValue':
        """Create TOP (unknown) lattice value."""
        return cls(LatticeState.TOP)

    @classmethod
    def bottom(cls) -> 'LatticeValue':
        """Create BOTTOM (not constant) lattice value."""
        return cls(LatticeState.BOTTOM)

    @classmethod
    def constant(cls, value: Any, const_type: ConstType) -> 'LatticeValue':
        """Create a constant lattice value."""
        return cls(LatticeState.CONSTANT, const_type, value)

    @classmethod
    def from_python(cls, value: Any) -> 'LatticeValue':
        """Create lattice value from Python value, inferring type."""
        if isinstance(value, bool):
            return cls.constant(value, ConstType.BOOL)
        elif isinstance(value, int):
            return cls.constant(value, ConstType.INT)
        elif isinstance(value, float):
            return cls.constant(value, ConstType.FLOAT)
        elif isinstance(value, str):
            return cls.constant(value, ConstType.STRING)
        else:
            return cls.bottom()

    def is_top(self) -> bool:
        return self.state == LatticeState.TOP

    def is_bottom(self) -> bool:
        return self.state == LatticeState.BOTTOM

    def is_constant(self) -> bool:
        return self.state == LatticeState.CONSTANT

    def __repr__(self) -> str:
        if self.state == LatticeState.TOP:
            return "⊤"
        elif self.state == LatticeState.BOTTOM:
            return "⊥"
        else:
            return f"Const({self.value})"


def meet(a: LatticeValue, b: LatticeValue) -> LatticeValue:
    """Compute the meet (greatest lower bound) of two lattice values.

    Meet rules:
    - TOP ⊓ x = x (TOP is the identity)
    - BOTTOM ⊓ x = BOTTOM (BOTTOM absorbs everything)
    - Const(c1) ⊓ Const(c2) = Const(c1) if c1 == c2, else BOTTOM
    """
    if a.is_top():
        return b
    if b.is_top():
        return a
    if a.is_bottom() or b.is_bottom():
        return LatticeValue.bottom()
    # Both are constants
    if a.value == b.value and a.const_type == b.const_type:
        return a
    return LatticeValue.bottom()


def meet_all(values: List[LatticeValue]) -> LatticeValue:
    """Compute the meet of a list of lattice values."""
    if not values:
        return LatticeValue.top()
    result = values[0]
    for v in values[1:]:
        result = meet(result, v)
        if result.is_bottom():
            break  # Early exit - can't get lower
    return result


# Edge type for tracking executable edges
Edge = Tuple[str, str]  # (from_block, to_block)


class SCCP:
    """Sparse Conditional Constant Propagation.

    Implements the Wegman-Zadeck algorithm for combined constant propagation
    and unreachable code detection.
    """

    # Built-in MIR constants (always defined)
    BUILTIN_CONSTANTS: Dict[str, LatticeValue] = {
        str(V_FALSE): LatticeValue.constant(False, ConstType.BOOL),
        str(V_TRUE): LatticeValue.constant(True, ConstType.BOOL),
        str(V_F_ZERO): LatticeValue.constant(0.0, ConstType.FLOAT),
        str(V_ZERO): LatticeValue.constant(0, ConstType.INT),
        str(V_ONE): LatticeValue.constant(1, ConstType.INT),
        str(V_F_ONE): LatticeValue.constant(1.0, ConstType.FLOAT),
        str(V_F_N_ONE): LatticeValue.constant(-1.0, ConstType.FLOAT),
    }

    def __init__(
        self,
        mir_func: MIRFunction,
        known_values: Optional[Dict[str, Any]] = None,
    ):
        """Initialize SCCP analysis.

        Args:
            mir_func: The MIR function to analyze
            known_values: Dict of value_id -> value for known constants
                         (e.g., {'v50609': 1} for TYPE parameter)
        """
        self.mir_func = mir_func
        self.known_values = known_values or {}

        # Lattice value for each SSA value
        self.lattice: Dict[str, LatticeValue] = {}

        # Executable edges (from_block, to_block)
        self.executable_edges: Set[Edge] = set()

        # Track which blocks have been visited (at least one incoming edge executable)
        self.visited_blocks: Set[str] = set()

        # Worklists
        self.cfg_worklist: List[str] = []  # Blocks to process
        self.ssa_worklist: List[str] = []  # Values that changed

        # Build def-use chains for SSA propagation
        self.uses: Dict[str, List[Tuple[str, MIRInstruction]]] = {}  # value -> [(block, inst)]
        self._build_uses()

        # Run analysis
        self._run()

    def _build_uses(self):
        """Build def-use chains: for each value, track all instructions that use it."""
        for block_name, block in self.mir_func.blocks.items():
            for inst in block.instructions:
                # Regular operands
                for op in inst.operands:
                    op_str = str(op)
                    if op_str not in self.uses:
                        self.uses[op_str] = []
                    self.uses[op_str].append((block_name, inst))

                # PHI operands
                if inst.phi_operands:
                    for phi_op in inst.phi_operands:
                        val_str = str(phi_op.value)
                        if val_str not in self.uses:
                            self.uses[val_str] = []
                        self.uses[val_str].append((block_name, inst))

                # Branch condition
                if inst.condition:
                    cond_str = str(inst.condition)
                    if cond_str not in self.uses:
                        self.uses[cond_str] = []
                    self.uses[cond_str].append((block_name, inst))

    def _run(self):
        """Run SCCP analysis."""
        # Initialize lattice
        self._initialize()

        # Process worklists until empty
        while self.cfg_worklist or self.ssa_worklist:
            # Process CFG worklist first (mark blocks executable)
            while self.cfg_worklist:
                block_name = self.cfg_worklist.pop()
                self._process_block(block_name)

            # Process SSA worklist (propagate constants)
            while self.ssa_worklist:
                value_id = self.ssa_worklist.pop()
                self._propagate_value(value_id)

    def _initialize(self):
        """Initialize lattice values and worklists."""
        # Initialize all values to TOP
        for block in self.mir_func.blocks.values():
            for inst in block.instructions:
                if inst.result:
                    self.lattice[str(inst.result)] = LatticeValue.top()

        # Set built-in constants
        for value_id, lattice_val in self.BUILTIN_CONSTANTS.items():
            self.lattice[value_id] = lattice_val

        # Set MIR constants from function data
        for value_id, const_val in self.mir_func.constants.items():
            self.lattice[value_id] = LatticeValue.constant(const_val, ConstType.FLOAT)
        for value_id, const_val in self.mir_func.int_constants.items():
            self.lattice[value_id] = LatticeValue.constant(const_val, ConstType.INT)
        for value_id, const_val in self.mir_func.bool_constants.items():
            self.lattice[value_id] = LatticeValue.constant(const_val, ConstType.BOOL)
        for value_id, const_val in self.mir_func.str_constants.items():
            self.lattice[value_id] = LatticeValue.constant(const_val, ConstType.STRING)

        # Initialize function params to BOTTOM (they're runtime values)
        # unless they're in known_values
        for param_id in self.mir_func.params:
            if param_id not in self.known_values:
                self.lattice[param_id] = LatticeValue.bottom()

        # Set known parameter values (overrides BOTTOM for specific params)
        for value_id, value in self.known_values.items():
            self.lattice[value_id] = LatticeValue.from_python(value)

        # Start from entry block
        entry = self.mir_func.entry_block
        if entry:
            self.cfg_worklist.append(entry)
            self.visited_blocks.add(entry)

    def _process_block(self, block_name: str):
        """Process a block: evaluate all instructions."""
        block = self.mir_func.blocks.get(block_name)
        if not block:
            return

        # Evaluate all instructions in the block
        for inst in block.instructions:
            self._evaluate_instruction(inst, block_name)

        # Handle terminator to add successor edges
        term = block.terminator
        if term:
            self._process_terminator(term, block_name)

    def _evaluate_instruction(self, inst: MIRInstruction, block_name: str):
        """Evaluate an instruction and update lattice if result changes."""
        if not inst.result:
            return

        result_str = str(inst.result)
        old_val = self.lattice.get(result_str, LatticeValue.top())

        # Already at BOTTOM - can't change
        if old_val.is_bottom():
            return

        # Compute new value
        if inst.is_phi:
            new_val = self._evaluate_phi(inst, block_name)
        else:
            new_val = self._evaluate_regular(inst)

        # Update if changed (can only go down in lattice)
        if new_val != old_val:
            self.lattice[result_str] = new_val
            self.ssa_worklist.append(result_str)

    def _evaluate_phi(self, phi: MIRInstruction, block_name: str) -> LatticeValue:
        """Evaluate a PHI node considering only executable incoming edges."""
        if not phi.phi_operands:
            return LatticeValue.bottom()

        values = []
        for phi_op in phi.phi_operands:
            # Only consider values from executable edges
            edge = (phi_op.block, block_name)
            if edge in self.executable_edges:
                val_str = str(phi_op.value)
                val = self.lattice.get(val_str, LatticeValue.top())
                values.append(val)

        if not values:
            # No executable incoming edges yet
            return LatticeValue.top()

        return meet_all(values)

    def _evaluate_regular(self, inst: MIRInstruction) -> LatticeValue:
        """Evaluate a regular (non-PHI) instruction."""
        # Get operand values
        operand_vals = []
        for op in inst.operands:
            op_str = str(op)
            val = self.lattice.get(op_str, LatticeValue.top())
            if val.is_top():
                # Can't evaluate yet - operand unknown
                return LatticeValue.top()
            if val.is_bottom():
                # Operand not constant - result not constant
                return LatticeValue.bottom()
            operand_vals.append(val)

        # Try to evaluate the opcode
        return self._eval_opcode(inst.opcode, operand_vals)

    def _eval_opcode(self, opcode: str, operands: List[LatticeValue]) -> LatticeValue:
        """Evaluate an opcode with constant operands."""
        op = opcode.lower()

        if len(operands) == 2:
            a_val, b_val = operands[0].value, operands[1].value

            # Integer comparisons
            if op == 'ieq':
                return LatticeValue.constant(a_val == b_val, ConstType.BOOL)
            if op == 'ine':
                return LatticeValue.constant(a_val != b_val, ConstType.BOOL)
            if op == 'ilt':
                return LatticeValue.constant(a_val < b_val, ConstType.BOOL)
            if op == 'ile':
                return LatticeValue.constant(a_val <= b_val, ConstType.BOOL)
            if op == 'igt':
                return LatticeValue.constant(a_val > b_val, ConstType.BOOL)
            if op == 'ige':
                return LatticeValue.constant(a_val >= b_val, ConstType.BOOL)

            # Float comparisons
            if op == 'feq':
                return LatticeValue.constant(a_val == b_val, ConstType.BOOL)
            if op == 'fne':
                return LatticeValue.constant(a_val != b_val, ConstType.BOOL)
            if op == 'flt':
                return LatticeValue.constant(a_val < b_val, ConstType.BOOL)
            if op == 'fle':
                return LatticeValue.constant(a_val <= b_val, ConstType.BOOL)
            if op == 'fgt':
                return LatticeValue.constant(a_val > b_val, ConstType.BOOL)
            if op == 'fge':
                return LatticeValue.constant(a_val >= b_val, ConstType.BOOL)

            # Boolean operations
            if op == 'and':
                return LatticeValue.constant(a_val and b_val, ConstType.BOOL)
            if op == 'or':
                return LatticeValue.constant(a_val or b_val, ConstType.BOOL)

            # Integer arithmetic
            if op == 'iadd':
                return LatticeValue.constant(a_val + b_val, ConstType.INT)
            if op == 'isub':
                return LatticeValue.constant(a_val - b_val, ConstType.INT)
            if op == 'imul':
                return LatticeValue.constant(a_val * b_val, ConstType.INT)

            # Float arithmetic
            if op == 'fadd':
                return LatticeValue.constant(a_val + b_val, ConstType.FLOAT)
            if op == 'fsub':
                return LatticeValue.constant(a_val - b_val, ConstType.FLOAT)
            if op == 'fmul':
                return LatticeValue.constant(a_val * b_val, ConstType.FLOAT)
            if op == 'fdiv' and b_val != 0:
                return LatticeValue.constant(a_val / b_val, ConstType.FLOAT)

        if len(operands) == 1:
            a_val = operands[0].value
            a_type = operands[0].const_type

            if op == 'not':
                return LatticeValue.constant(not a_val, ConstType.BOOL)
            if op == 'fneg':
                return LatticeValue.constant(-a_val, ConstType.FLOAT)
            if op == 'ineg':
                return LatticeValue.constant(-a_val, ConstType.INT)

            # Type conversions
            if op == 'itof':
                return LatticeValue.constant(float(a_val), ConstType.FLOAT)
            if op == 'ftoi':
                return LatticeValue.constant(int(a_val), ConstType.INT)
            if op == 'btoi':
                return LatticeValue.constant(int(a_val), ConstType.INT)
            if op == 'itob':
                return LatticeValue.constant(bool(a_val), ConstType.BOOL)

        # Can't evaluate - mark as not constant
        return LatticeValue.bottom()

    def _process_terminator(self, term: MIRInstruction, block_name: str):
        """Process a terminator instruction to add executable edges."""
        if term.is_branch:
            # Conditional branch
            cond_str = str(term.condition) if term.condition else None
            cond_val = self.lattice.get(cond_str, LatticeValue.top()) if cond_str else LatticeValue.bottom()

            if cond_val.is_top():
                # Condition unknown - don't add any edges yet
                pass
            elif cond_val.is_bottom():
                # Condition not constant - both branches possible
                if term.true_block:
                    self._add_edge(block_name, term.true_block)
                if term.false_block:
                    self._add_edge(block_name, term.false_block)
            else:
                # Condition is constant - only one branch taken
                if cond_val.value:
                    if term.true_block:
                        self._add_edge(block_name, term.true_block)
                else:
                    if term.false_block:
                        self._add_edge(block_name, term.false_block)

        elif term.is_jump:
            # Unconditional jump
            if term.target_block:
                self._add_edge(block_name, term.target_block)

        # EXIT has no successors

    def _add_edge(self, from_block: str, to_block: str):
        """Add an executable edge and schedule the target block if new."""
        edge = (from_block, to_block)
        if edge not in self.executable_edges:
            self.executable_edges.add(edge)

            if to_block not in self.visited_blocks:
                # First time visiting this block
                self.visited_blocks.add(to_block)
                self.cfg_worklist.append(to_block)
            else:
                # Block already visited - re-evaluate PHIs with new edge
                block = self.mir_func.blocks.get(to_block)
                if block:
                    for phi in block.phi_nodes:
                        if phi.result:
                            result_str = str(phi.result)
                            old_val = self.lattice.get(result_str, LatticeValue.top())
                            new_val = self._evaluate_phi(phi, to_block)
                            if new_val != old_val:
                                self.lattice[result_str] = new_val
                                self.ssa_worklist.append(result_str)

    def _propagate_value(self, value_id: str):
        """Propagate a changed value to all its uses."""
        if value_id not in self.uses:
            return

        for block_name, inst in self.uses[value_id]:
            # Only process if the block is executable
            if block_name in self.visited_blocks:
                self._evaluate_instruction(inst, block_name)

                # If this is a terminator with a condition that changed,
                # we may need to add new edges
                if inst.is_terminator and inst.condition and str(inst.condition) == value_id:
                    self._process_terminator(inst, block_name)

    # =========================================================================
    # Public API
    # =========================================================================

    def is_edge_executable(self, from_block: str, to_block: str) -> bool:
        """Check if an edge is executable."""
        return (from_block, to_block) in self.executable_edges

    def is_block_reachable(self, block_name: str) -> bool:
        """Check if a block is reachable (has at least one executable incoming edge)."""
        return block_name in self.visited_blocks

    def is_block_dead(self, block_name: str) -> bool:
        """Check if a block is dead (unreachable)."""
        return block_name not in self.visited_blocks

    def get_constant(self, value_id: str) -> Optional[LatticeValue]:
        """Get the lattice value for a value ID.

        Returns the LatticeValue, or None if not in lattice.
        """
        return self.lattice.get(value_id)

    def get_constant_value(self, value_id: str) -> Optional[Any]:
        """Get the constant value for a value ID if it's a known constant.

        Returns the Python value (int, float, bool) or None if not constant.
        """
        val = self.lattice.get(value_id)
        if val and val.is_constant():
            return val.value
        return None

    def get_live_phi_value(self, phi: MIRInstruction, block_name: str) -> Optional[ValueId]:
        """Get the single live value for a PHI if only one predecessor is executable.

        This is useful for simplifying PHIs when constant propagation has
        eliminated branches.

        Returns:
            The single live ValueId if exactly one executable predecessor,
            or None if multiple predecessors are executable.
        """
        if not phi.is_phi or not phi.phi_operands:
            return None

        live_values: List[ValueId] = []
        for phi_op in phi.phi_operands:
            edge = (phi_op.block, block_name)
            if edge in self.executable_edges:
                live_values.append(phi_op.value)

        # If exactly one live value, return it
        if len(live_values) == 1:
            return live_values[0]

        # If multiple live values but all the same, return it
        if live_values and all(v == live_values[0] for v in live_values):
            return live_values[0]

        return None

    def get_dead_blocks(self) -> Set[str]:
        """Get all dead (unreachable) blocks."""
        all_blocks = set(self.mir_func.blocks.keys())
        return all_blocks - self.visited_blocks

    def get_static_branch_direction(self, block_name: str) -> Optional[bool]:
        """Get the static branch direction for a block if the condition is constant.

        Returns:
            True if the true branch is taken,
            False if the false branch is taken,
            None if the condition is not constant.
        """
        block = self.mir_func.blocks.get(block_name)
        if not block:
            return None

        term = block.terminator
        if not term or not term.is_branch or not term.condition:
            return None

        cond_val = self.lattice.get(str(term.condition))
        if cond_val and cond_val.is_constant():
            return bool(cond_val.value)

        return None

    def print_summary(self):
        """Print a summary of the analysis results."""
        dead = self.get_dead_blocks()
        constants = [(k, v) for k, v in self.lattice.items() if v.is_constant()]

        print(f"SCCP Analysis Summary for {self.mir_func.name}")
        print(f"  Total blocks: {len(self.mir_func.blocks)}")
        print(f"  Reachable blocks: {len(self.visited_blocks)}")
        print(f"  Dead blocks: {len(dead)}")
        print(f"  Executable edges: {len(self.executable_edges)}")
        print(f"  Constants found: {len(constants)}")

        # Show static branches
        static_branches = []
        for block_name in self.mir_func.blocks:
            direction = self.get_static_branch_direction(block_name)
            if direction is not None:
                static_branches.append((block_name, direction))

        if static_branches:
            print(f"  Static branches: {len(static_branches)}")
            for block_name, direction in static_branches[:10]:
                branch = "T" if direction else "F"
                print(f"    {block_name} -> {branch}")
            if len(static_branches) > 10:
                print(f"    ... and {len(static_branches) - 10} more")


# Backward compatibility alias
ConstantPropagator = SCCP
