"""SSA analysis for PHI node resolution.

This module provides SSA-specific analysis including:
- Branch condition mapping
- PHI node resolution using successor-pair lookup (no dominator computation needed)
- Multi-way PHI handling for complex control flow
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, FrozenSet
from enum import Enum

from .types import MIRFunction, MIRInstruction, Block
from .cfg import CFGAnalyzer, LoopInfo


@dataclass
class BranchInfo:
    """Information about a conditional branch.

    Maps a block to its branch condition and targets.
    """
    block: str
    condition: str  # The value ID used as condition
    true_block: str
    false_block: str


class PHIResolutionType(Enum):
    """Type of PHI resolution strategy."""
    TWO_WAY = "two_way"  # Simple jnp.where(cond, true, false)
    MULTI_WAY = "multi_way"  # Nested where for 3+ predecessors
    LOOP_INIT = "loop_init"  # Initial value from outside loop
    LOOP_UPDATE = "loop_update"  # Updated value from loop body
    FALLBACK = "fallback"  # Single value, no condition


@dataclass
class PHIResolution:
    """Resolution strategy for a PHI node.

    Contains all information needed to generate code for a PHI node.
    """
    type: PHIResolutionType

    # For TWO_WAY
    condition: Optional[str] = None
    true_value: Optional[str] = None
    false_value: Optional[str] = None

    # For MULTI_WAY: list of (condition, value) pairs + default
    cases: Optional[List[Tuple[str, str]]] = None
    default: Optional[str] = None

    # For LOOP_INIT/LOOP_UPDATE
    init_value: Optional[str] = None  # Value from outside loop
    update_value: Optional[str] = None  # Value from loop iteration

    # For FALLBACK
    single_value: Optional[str] = None


class SSAAnalyzer:
    """SSA-specific analysis for PHI resolution.

    Uses a simple successor-pair lookup approach (like the OSDI pipeline)
    instead of expensive dominator computation.
    """

    def __init__(self, mir_func: MIRFunction, cfg: CFGAnalyzer):
        """Initialize SSA analyzer.

        Args:
            mir_func: The MIR function to analyze
            cfg: Pre-computed CFG analysis
        """
        self.mir_func = mir_func
        self.cfg = cfg

        # Cached analysis - built lazily
        self._branch_conditions: Optional[Dict[str, Dict[str, Tuple[str, bool]]]] = None
        self._succ_pair_map: Optional[Dict[FrozenSet[str], List[str]]] = None

    @property
    def branch_conditions(self) -> Dict[str, Dict[str, Tuple[str, bool]]]:
        """Get branch condition info for all branching blocks.

        Returns:
            Dict mapping block -> {successor: (condition_var, is_true_branch)}
        """
        if self._branch_conditions is None:
            self._branch_conditions = self._build_branch_conditions()
        return self._branch_conditions

    @property
    def succ_pair_map(self) -> Dict[FrozenSet[str], List[str]]:
        """Get map from successor pairs to blocks that branch to them.

        Returns:
            Dict mapping frozenset(successors) -> [block_names]
        """
        if self._succ_pair_map is None:
            self._succ_pair_map = self._build_succ_pair_map()
        return self._succ_pair_map

    def resolve_phi(self, phi: MIRInstruction,
                    loop: Optional[LoopInfo] = None) -> PHIResolution:
        """Resolve a PHI node to a code generation strategy.

        Args:
            phi: The PHI instruction to resolve
            loop: If the PHI is in a loop header, the loop info

        Returns:
            PHIResolution with the strategy and values
        """
        assert phi.is_phi and phi.phi_operands is not None

        # Handle loop PHI specially
        if loop is not None and phi.block == loop.header:
            return self._resolve_loop_phi(phi, loop)

        num_ops = len(phi.phi_operands)

        if num_ops == 0:
            return PHIResolution(type=PHIResolutionType.FALLBACK, single_value='0')

        if num_ops == 1:
            return PHIResolution(
                type=PHIResolutionType.FALLBACK,
                single_value=phi.phi_operands[0].value
            )

        if num_ops == 2:
            return self._resolve_two_way_phi(phi)

        return self._resolve_multi_way_phi(phi)

    def _build_branch_conditions(self) -> Dict[str, Dict[str, Tuple[str, bool]]]:
        """Build map of block -> {successor: (condition, is_true_branch)}.

        For each block with a branch instruction, maps to its condition and targets.
        """
        conditions: Dict[str, Dict[str, Tuple[str, bool]]] = {}

        for block_name, block in self.mir_func.blocks.items():
            terminator = block.terminator
            if terminator and terminator.is_branch:
                cond = terminator.condition
                true_block = terminator.true_block
                false_block = terminator.false_block
                if cond and true_block and false_block:
                    conditions[block_name] = {
                        true_block: (cond, True),
                        false_block: (cond, False),
                    }

        return conditions

    def _build_succ_pair_map(self) -> Dict[FrozenSet[str], List[str]]:
        """Build map from successor pairs to block names.

        This is an optimization for O(1) PHI resolution lookup.
        """
        succ_to_blocks: Dict[FrozenSet[str], List[str]] = {}

        for block_name, block in self.mir_func.blocks.items():
            succs = block.successors
            if len(succs) == 2:  # Only care about binary branches
                key = frozenset(succs)
                if key not in succ_to_blocks:
                    succ_to_blocks[key] = []
                succ_to_blocks[key].append(block_name)

        return succ_to_blocks

    def _resolve_loop_phi(self, phi: MIRInstruction,
                          loop: LoopInfo) -> PHIResolution:
        """Resolve PHI in a loop header.

        Loop PHIs have two kinds of operands:
        - Init value: from predecessor outside the loop
        - Update value: from predecessor inside the loop (back-edge)
        """
        init_value = None
        update_value = None

        for op in phi.phi_operands:
            if op.block in loop.body and op.block != loop.header:
                # This predecessor is in the loop body (back-edge)
                update_value = op.value
            else:
                # This predecessor is outside the loop
                init_value = op.value

        # If we couldn't determine, fall back to first operand
        if init_value is None:
            init_value = phi.phi_operands[0].value
        if update_value is None and len(phi.phi_operands) > 1:
            update_value = phi.phi_operands[1].value
        elif update_value is None:
            update_value = init_value

        return PHIResolution(
            type=PHIResolutionType.LOOP_INIT,
            init_value=init_value,
            update_value=update_value,
        )

    def _resolve_two_way_phi(self, phi: MIRInstruction) -> PHIResolution:
        """Resolve a two-way PHI node using successor-pair lookup.

        Uses the same approach as OpenVAF's OSDI pipeline:
        1. Check if either predecessor is a branching block that targets phi's block
        2. Look up in succ_pair_map to find a block whose successors match the predecessors
        """
        assert phi.phi_operands and len(phi.phi_operands) == 2

        pred0 = phi.phi_operands[0].block
        pred1 = phi.phi_operands[1].block
        val0 = phi.phi_operands[0].value
        val1 = phi.phi_operands[1].value

        branch_conds = self.branch_conditions

        # Strategy 1: Check if either predecessor is the branching block
        if pred0 in branch_conds:
            cond_info = branch_conds[pred0].get(phi.block)
            if cond_info:
                cond_var, is_true = cond_info
                if is_true:
                    return PHIResolution(
                        type=PHIResolutionType.TWO_WAY,
                        condition=cond_var,
                        true_value=val0,
                        false_value=val1,
                    )
                else:
                    return PHIResolution(
                        type=PHIResolutionType.TWO_WAY,
                        condition=cond_var,
                        true_value=val1,
                        false_value=val0,
                    )

        if pred1 in branch_conds:
            cond_info = branch_conds[pred1].get(phi.block)
            if cond_info:
                cond_var, is_true = cond_info
                if is_true:
                    return PHIResolution(
                        type=PHIResolutionType.TWO_WAY,
                        condition=cond_var,
                        true_value=val1,
                        false_value=val0,
                    )
                else:
                    return PHIResolution(
                        type=PHIResolutionType.TWO_WAY,
                        condition=cond_var,
                        true_value=val0,
                        false_value=val1,
                    )

        # Strategy 2: Look up in succ_pair_map
        pred_key = frozenset([pred0, pred1])
        candidate_blocks = self.succ_pair_map.get(pred_key, [])

        for block_name in candidate_blocks:
            if block_name in branch_conds:
                cond_info = branch_conds[block_name]
                if pred0 in cond_info and pred1 in cond_info:
                    cond_var, is_true0 = cond_info[pred0]
                    if is_true0:
                        return PHIResolution(
                            type=PHIResolutionType.TWO_WAY,
                            condition=cond_var,
                            true_value=val0,
                            false_value=val1,
                        )
                    else:
                        return PHIResolution(
                            type=PHIResolutionType.TWO_WAY,
                            condition=cond_var,
                            true_value=val1,
                            false_value=val0,
                        )

        # Fallback: couldn't find condition
        return PHIResolution(
            type=PHIResolutionType.FALLBACK,
            single_value=val0
        )

    def _resolve_multi_way_phi(self, phi: MIRInstruction) -> PHIResolution:
        """Resolve a multi-way PHI node (3+ predecessors).

        Groups predecessors by value and builds nested where expressions.
        """
        assert phi.phi_operands and len(phi.phi_operands) >= 3

        pred_blocks = [op.block for op in phi.phi_operands]
        val_by_pred = {op.block: op.value for op in phi.phi_operands}

        # Group predecessors by value
        val_to_preds: Dict[str, List[str]] = {}
        for pred in pred_blocks:
            val = val_by_pred.get(pred, '_ZERO')
            if val not in val_to_preds:
                val_to_preds[val] = []
            val_to_preds[val].append(pred)

        unique_vals = list(val_to_preds.keys())

        # If only 2 unique values, simplify to TWO_WAY
        if len(unique_vals) == 2:
            val_a, val_b = unique_vals
            preds_a = val_to_preds[val_a]
            preds_b = val_to_preds[val_b]

            # Find condition that separates these groups
            condition = self._find_condition_for_groups(preds_a, preds_b)
            if condition:
                return PHIResolution(
                    type=PHIResolutionType.TWO_WAY,
                    condition=condition,
                    true_value=val_a,
                    false_value=val_b,
                )

        # Build nested cases
        cases: List[Tuple[str, str]] = []
        remaining = list(pred_blocks)

        # Try to peel off predecessors one at a time
        while len(remaining) > 1:
            peeled = self._peel_one_predecessor(remaining, val_by_pred)
            if peeled is None:
                break
            cond, value, pred = peeled
            cases.append((cond, value))
            remaining.remove(pred)

        # Default is the remaining predecessor's value
        default = val_by_pred.get(remaining[0], phi.phi_operands[0].value) if remaining else phi.phi_operands[0].value

        if not cases:
            return PHIResolution(
                type=PHIResolutionType.FALLBACK,
                single_value=default
            )

        return PHIResolution(
            type=PHIResolutionType.MULTI_WAY,
            cases=cases,
            default=default,
        )

    def _find_condition_for_groups(self, preds_a: List[str],
                                    preds_b: List[str]) -> Optional[str]:
        """Find a condition that separates two groups of predecessors."""
        branch_conds = self.branch_conditions

        # Look for a branch where true leads to preds_a and false leads to preds_b
        for block_name, cond_info in branch_conds.items():
            targets = list(cond_info.keys())
            if len(targets) != 2:
                continue

            t0, t1 = targets
            cond_var, is_t0_true = cond_info[t0]

            # Check if t0 leads to preds_a and t1 leads to preds_b
            t0_reaches_a = t0 in preds_a or self._any_reachable(t0, preds_a)
            t1_reaches_b = t1 in preds_b or self._any_reachable(t1, preds_b)

            if t0_reaches_a and t1_reaches_b:
                return cond_var if is_t0_true else f"!{cond_var}"

            # Or vice versa
            t0_reaches_b = t0 in preds_b or self._any_reachable(t0, preds_b)
            t1_reaches_a = t1 in preds_a or self._any_reachable(t1, preds_a)

            if t0_reaches_b and t1_reaches_a:
                return cond_var if not is_t0_true else f"!{cond_var}"

        return None

    def _any_reachable(self, start: str, targets: List[str]) -> bool:
        """Check if any target is reachable from start."""
        if start in targets:
            return True

        visited = set()
        stack = [start]

        while stack:
            block = stack.pop()
            if block in visited:
                continue
            visited.add(block)

            if block in targets:
                return True

            if block in self.mir_func.blocks:
                for succ in self.mir_func.blocks[block].successors:
                    if succ not in visited:
                        stack.append(succ)

        return False

    def _peel_one_predecessor(self, remaining: List[str],
                               val_by_pred: Dict[str, str]
                               ) -> Optional[Tuple[str, str, str]]:
        """Try to peel off one predecessor from the remaining set.

        Returns (condition, value, predecessor) if successful.
        """
        branch_conds = self.branch_conditions

        for block_name, cond_info in branch_conds.items():
            for target, (cond_var, is_true) in cond_info.items():
                # Check if this target is one of our remaining predecessors
                if target in remaining:
                    # Check if only this predecessor is reachable from this target
                    reachable_preds = [p for p in remaining
                                       if p == target or self._any_reachable(target, [p])]

                    if len(reachable_preds) == 1:
                        pred = reachable_preds[0]
                        value = val_by_pred.get(pred)
                        if value is None:
                            continue

                        condition = cond_var if is_true else f"!{cond_var}"
                        return (condition, value, pred)

        return None
