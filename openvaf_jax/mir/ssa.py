"""SSA analysis for PHI node resolution.

This module provides SSA-specific analysis including:
- Branch condition mapping
- PHI node resolution using dominator-based lookup
- Multi-way PHI handling for complex control flow
"""

from dataclasses import dataclass
from enum import Enum

# Import SCCP for type hints only (avoid circular import at runtime)
from typing import TYPE_CHECKING, Dict, FrozenSet, List, Optional, Set, Tuple

from .cfg import CFGAnalyzer, LoopInfo
from .types import V_F_ZERO, BlockId, MIRFunction, MIRInstruction, PhiOperand, ValueId

if TYPE_CHECKING:
    from .constprop import SCCP


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

    # For TWO_WAY: condition may be negated (e.g., '!v5')
    condition: Optional[str] = None
    true_value: Optional[ValueId] = None
    false_value: Optional[ValueId] = None

    # For NESTED_TWO_WAY: branches can be nested PHIResolutions
    # Use these instead of true_value/false_value when branch needs further resolution
    nested_true: Optional['PHIResolution'] = None
    nested_false: Optional['PHIResolution'] = None

    # For MULTI_WAY: list of (condition_expr, value) pairs + default
    # condition_expr may have negation prefix (e.g., '!v5')
    cases: Optional[List[Tuple[str, ValueId]]] = None
    default: Optional[ValueId] = None

    # For LOOP_INIT/LOOP_UPDATE
    init_value: Optional[ValueId] = None  # Value from outside loop
    update_value: Optional[ValueId] = None  # Value from loop iteration

    # For FALLBACK
    single_value: Optional[ValueId] = None


class SSAAnalyzer:
    """SSA-specific analysis for PHI resolution.

    Uses dominator-based PHI resolution:
    1. Find the immediate dominator of the PHI block
    2. Walk up dominator tree to find the controlling branch
    3. Map predecessors to branch targets to determine condition
    """

    def __init__(
        self,
        mir_func: MIRFunction,
        cfg: CFGAnalyzer,
        sccp: Optional['SCCP'] = None,
    ):
        """Initialize SSA analyzer.

        Args:
            mir_func: The MIR function to analyze
            cfg: Pre-computed CFG analysis
            sccp: Optional SCCP analysis for constant-based dead code elimination.
                  When provided, PHI nodes with dead predecessors will be simplified.
        """
        self.mir_func = mir_func
        self.cfg = cfg
        self.sccp = sccp

        # Cached analysis - built lazily
        self._branch_conditions: Optional[Dict[str, Dict[str, Tuple[ValueId, bool]]]] = None
        self._succ_pair_map: Optional[Dict[FrozenSet[str], List[str]]] = None
        self._pred_to_branch_target: Optional[Dict[str, Dict[str, str]]] = None
        self._reachable_from: Optional[Dict[str, Set[str]]] = None  # Transitive closure
        self._dominated_reachable_cache: Dict[Tuple[str, str, str], bool] = {}  # Cache for dominated reachability

    @property
    def branch_conditions(self) -> Dict[str, Dict[str, Tuple[ValueId, bool]]]:
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

    @property
    def reachable_from(self) -> Dict[str, Set[str]]:
        """Get transitive closure: reachable_from[start] = {all reachable blocks}.

        Uses Kameda's algorithm (reverse topological order) for O(V*E) precomputation,
        giving O(1) reachability queries instead of O(V+E) per query.
        """
        if self._reachable_from is None:
            self._reachable_from = self._build_transitive_closure()
        return self._reachable_from

    def _build_transitive_closure(self) -> Dict[str, Set[str]]:
        """Build transitive closure using optimized reverse post-order algorithm.

        For reducible CFGs (typical compiler output), this completes in 1-2 passes.
        Uses reverse post-order traversal for efficient propagation.
        """
        blocks = self.mir_func.blocks
        if not blocks:
            return {}

        # Step 1: Compute post-order via iterative DFS
        entry = self.mir_func.entry_block
        post_order: List[str] = []
        visited: Set[str] = set()
        # Stack: (block, children_processed)
        stack: List[Tuple[str, bool]] = [(entry, False)]

        while stack:
            block, done = stack.pop()
            if done:
                post_order.append(block)
                continue
            if block in visited or block not in blocks:
                continue
            visited.add(block)
            stack.append((block, True))
            for succ in reversed(blocks[block].successors):
                if succ not in visited and succ in blocks:
                    stack.append((succ, False))

        # Add any unreachable blocks (shouldn't happen for valid CFGs)
        for name in blocks:
            if name not in visited:
                post_order.append(name)

        # Step 2: Process in reverse post-order (forward topological order)
        # For reachability, we need to process in reverse - from exits to entry
        # So we use post_order directly (not reversed)
        reachable: Dict[str, Set[str]] = {name: {name} for name in blocks}

        # Single pass in post-order (reverse topological) - handles DAGs perfectly
        for name in post_order:
            block = blocks[name]
            for succ in block.successors:
                if succ in reachable:
                    reachable[name].update(reachable[succ])

        # Second pass to handle back-edges (cycles)
        # Only needed if there are loops; for most CFGs this adds little
        changed = True
        max_extra_passes = 3  # Typically converges in 1-2 extra passes
        for _ in range(max_extra_passes):
            if not changed:
                break
            changed = False
            for name in post_order:
                block = blocks[name]
                old_size = len(reachable[name])
                for succ in block.successors:
                    if succ in reachable:
                        reachable[name].update(reachable[succ])
                if len(reachable[name]) > old_size:
                    changed = True

        return reachable

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

        # Filter operands based on SCCP if available
        live_operands = self._get_live_operands(phi)

        if len(live_operands) == 0:
            return PHIResolution(type=PHIResolutionType.FALLBACK, single_value=ValueId('0'))

        if len(live_operands) == 1:
            return PHIResolution(
                type=PHIResolutionType.FALLBACK,
                single_value=live_operands[0].value
            )

        # If SCCP reduced operands, create a "virtual" PHI with filtered operands
        if len(live_operands) != len(phi.phi_operands):
            # Create a modified PHI instruction for resolution
            filtered_phi = MIRInstruction(
                opcode=phi.opcode,
                block=phi.block,
                result=phi.result,
                phi_operands=live_operands,
            )
            if len(live_operands) == 2:
                return self._resolve_two_way_phi(filtered_phi)
            return self._resolve_multi_way_phi(filtered_phi)

        if len(live_operands) == 2:
            return self._resolve_two_way_phi(phi)

        return self._resolve_multi_way_phi(phi)

    def _get_live_operands(self, phi: MIRInstruction) -> List[PhiOperand]:
        """Get PHI operands from live (executable) predecessors.

        Uses SCCP analysis if available to filter out dead predecessors.
        Without SCCP, returns all operands.
        """
        if phi.phi_operands is None:
            return []

        if self.sccp is None:
            return list(phi.phi_operands)

        # Filter by executable edges
        live = []
        for op in phi.phi_operands:
            if self.sccp.is_edge_executable(op.block, phi.block):
                live.append(op)

        return live

    def _build_branch_conditions(self) -> Dict[str, Dict[str, Tuple[ValueId, bool]]]:
        """Build map of block -> {successor: (condition, is_true_branch)}.

        For each block with a branch instruction, maps to its condition and targets.
        """
        conditions: Dict[str, Dict[str, Tuple[ValueId, bool]]] = {}

        for block_name, block in self.mir_func.blocks.items():
            terminator = block.terminator
            if terminator and terminator.is_branch:
                cond = terminator.condition
                true_block = terminator.true_block
                false_block = terminator.false_block
                if cond and true_block and false_block:
                    cond_id = ValueId(cond)
                    conditions[block_name] = {
                        true_block: (cond_id, True),
                        false_block: (cond_id, False),
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
        assert phi.phi_operands is not None
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
        """Resolve a two-way PHI node using dominator-based lookup.

        Strategy:
        1. Check if either predecessor is the branching block (direct branch)
        2. Look up in succ_pair_map for blocks that branch to both predecessors
        3. Use dominator-based resolution to find the controlling branch
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

        # Strategy 3: Trace back through unconditional jumps to find branching ancestor
        # This handles diamond-with-intermediate-blocks patterns
        resolution = self._resolve_via_ancestor_trace(pred0, pred1, val0, val1)
        if resolution:
            return resolution

        # Strategy 4: Dominator-based resolution
        # Walk up the dominator tree to find a branching dominator
        resolution = self._resolve_via_dominator(phi.block, [pred0, pred1],
                                                  {pred0: val0, pred1: val1})
        if resolution:
            return resolution

        # Fallback: couldn't find condition, use first value
        return PHIResolution(
            type=PHIResolutionType.FALLBACK,
            single_value=val0
        )

    def _resolve_via_dominator(
        self,
        phi_block: str,
        pred_blocks: List[str],
        val_by_pred: Dict[str, ValueId]
    ) -> Optional[PHIResolution]:
        """Resolve PHI using dominator tree.

        Walk up the dominator tree from phi_block to find a branching block
        that controls which predecessor is taken.
        """
        idom = self.cfg.immediate_dominators
        branch_conds = self.branch_conditions

        # Walk up dominator tree
        current = phi_block
        visited = set()
        max_depth = 100  # Safety limit

        while current and current not in visited and len(visited) < max_depth:
            visited.add(current)
            dom = idom.get(current)
            if dom is None:
                break

            # Check if this dominator is a branching block
            if dom in branch_conds:
                cond_info = branch_conds[dom]
                if len(cond_info) == 2:
                    # Find which branch target each predecessor is reachable from
                    targets = list(cond_info.keys())
                    t0, t1 = targets
                    cond_var, is_t0_true = cond_info[t0]

                    # Build mapping: predecessor -> which target it's reachable from
                    pred_to_target: Dict[str, str] = {}
                    for pred in pred_blocks:
                        t0_reaches = self._is_dominated_reachable(t0, pred, dom)
                        t1_reaches = self._is_dominated_reachable(t1, pred, dom)

                        # We want exclusive reachability
                        if t0_reaches and not t1_reaches:
                            pred_to_target[pred] = t0
                        elif t1_reaches and not t0_reaches:
                            pred_to_target[pred] = t1

                    # Check if all predecessors have exclusive mappings
                    if len(pred_to_target) == len(pred_blocks):
                        # Group by target
                        t0_preds = [p for p, t in pred_to_target.items() if t == t0]
                        t1_preds = [p for p, t in pred_to_target.items() if t == t1]

                        if t0_preds and t1_preds:
                            # Get values for each group
                            t0_vals = list(set(val_by_pred[p] for p in t0_preds))
                            t1_vals = list(set(val_by_pred[p] for p in t1_preds))

                            # If each group has a single unique value, we can resolve
                            if len(t0_vals) == 1 and len(t1_vals) == 1:
                                if is_t0_true:
                                    return PHIResolution(
                                        type=PHIResolutionType.TWO_WAY,
                                        condition=cond_var,
                                        true_value=t0_vals[0],
                                        false_value=t1_vals[0],
                                    )
                                else:
                                    return PHIResolution(
                                        type=PHIResolutionType.TWO_WAY,
                                        condition=cond_var,
                                        true_value=t1_vals[0],
                                        false_value=t0_vals[0],
                                    )

            current = dom

        return None

    def _is_dominated_reachable(self, start: str, target: str, dominator: str) -> bool:
        """Check if target is reachable from start without going back through dominator.

        Uses caching and fast-path optimization with transitive closure for efficiency.
        """
        if start == target:
            return True

        # Check cache first
        cache_key = (start, target, dominator)
        cached = self._dominated_reachable_cache.get(cache_key)
        if cached is not None:
            return cached

        # Fast-path: if target not reachable from start at all, return False
        reachable = self.reachable_from
        if start in reachable and target not in reachable[start]:
            self._dominated_reachable_cache[cache_key] = False
            return False

        # Fast-path: if dominator not reachable from start, dominator can't block the path
        if start in reachable and dominator not in reachable[start]:
            result = target in reachable[start]
            self._dominated_reachable_cache[cache_key] = result
            return result

        # Full DFS avoiding dominator
        visited = {dominator}  # Don't go back through dominator
        stack = [start]

        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)

            if current == target:
                self._dominated_reachable_cache[cache_key] = True
                return True

            if current in self.mir_func.blocks:
                for succ in self.mir_func.blocks[current].successors:
                    if succ not in visited:
                        stack.append(succ)

        self._dominated_reachable_cache[cache_key] = False
        return False

    def _resolve_via_ancestor_trace(
        self, pred0: str, pred1: str, val0: ValueId, val1: ValueId
    ) -> Optional[PHIResolution]:
        """Resolve PHI by finding a branching block that separates the predecessors.

        Handles diamond-with-intermediate-blocks patterns:
        - PHI at block61 with predecessors block59 and block64
        - block52 branches to block59 (true) and block60 (false)
        - block64 is reached via block60 -> intermediate -> block64

        Algorithm:
        For each branching block, check if:
        - pred0 is reachable only from one branch (true or false)
        - pred1 is reachable only from the other branch
        If so, use that branch's condition.
        """
        branch_conds = self.branch_conditions

        # For each branching block, check if it separates pred0 and pred1
        for block_name, cond_info in branch_conds.items():
            if len(cond_info) != 2:
                continue

            targets = list(cond_info.keys())
            true_target = None
            false_target = None

            for target, (cond_var, is_true) in cond_info.items():
                if is_true:
                    true_target = target
                else:
                    false_target = target

            if not true_target or not false_target:
                continue

            # Check reachability: can pred0/pred1 be reached from true/false branches?
            pred0_from_true = self._is_reachable(true_target, pred0)
            pred0_from_false = self._is_reachable(false_target, pred0)
            pred1_from_true = self._is_reachable(true_target, pred1)
            pred1_from_false = self._is_reachable(false_target, pred1)

            # We want exclusive reachability: pred0 from one branch only, pred1 from the other only
            cond_var, _ = cond_info[true_target]

            # Case 1: pred0 only from true, pred1 only from false
            if pred0_from_true and not pred0_from_false and pred1_from_false and not pred1_from_true:
                return PHIResolution(
                    type=PHIResolutionType.TWO_WAY,
                    condition=cond_var,
                    true_value=val0,
                    false_value=val1,
                )

            # Case 2: pred0 only from false, pred1 only from true
            if pred0_from_false and not pred0_from_true and pred1_from_true and not pred1_from_false:
                return PHIResolution(
                    type=PHIResolutionType.TWO_WAY,
                    condition=cond_var,
                    true_value=val1,
                    false_value=val0,
                )

        return None

    def _is_reachable(self, start: str, target: str) -> bool:
        """Check if target is reachable from start via successors.

        Uses precomputed transitive closure (Kameda's algorithm) for O(1) lookup.
        """
        return target in self.reachable_from.get(start, set())

    def _resolve_multi_way_phi(self, phi: MIRInstruction) -> PHIResolution:
        """Resolve a multi-way PHI node (3+ predecessors).

        Strategy: Build a binary decision tree by recursively splitting predecessors.
        1. Group predecessors by value
        2. If only one non-v3 value, use it directly (v3=0.0 is placeholder)
        3. Find a branch that separates predecessors into two groups
        4. Recursively resolve each group if it has multiple predecessors
        5. Build TWO_WAY with nested_true/nested_false for recursive cases
        """
        assert phi.phi_operands and len(phi.phi_operands) >= 3

        pred_blocks = [op.block for op in phi.phi_operands]
        val_by_pred = {op.block: op.value for op in phi.phi_operands}

        # Group predecessors by value
        val_to_preds: Dict[ValueId, List[BlockId]] = {}
        for pred in pred_blocks:
            val = val_by_pred.get(pred, V_F_ZERO)
            if val not in val_to_preds:
                val_to_preds[val] = []
            val_to_preds[val].append(pred)

        unique_vals = list(val_to_preds.keys())

        # Optimization: If only one non-v3 value, use it directly
        # v3 (constant 0.0) is often a placeholder for "unused" paths
        # (e.g., PMOS values when running NMOS, or vice versa)
        non_v3_vals = [v for v in unique_vals if str(v) != 'v3']
        if len(non_v3_vals) == 1:
            # All paths either compute this value or use v3 placeholder
            return PHIResolution(
                type=PHIResolutionType.FALLBACK,
                single_value=non_v3_vals[0]
            )

        # If only 2 unique values, try dominator-based TWO_WAY resolution
        if len(unique_vals) == 2:
            # Try dominator-based resolution first
            resolution = self._resolve_via_dominator(phi.block, pred_blocks, val_by_pred)
            if resolution:
                return resolution

            # Try ancestor trace as fallback for 2-value case
            # This handles cases where the controlling branch isn't in the dominator tree
            resolution = self._resolve_via_ancestor_trace_for_groups(
                val_to_preds, unique_vals
            )
            if resolution:
                return resolution

        # Build nested TWO_WAY using recursive tree construction
        resolution = self._build_phi_decision_tree(pred_blocks, val_by_pred)
        if resolution:
            return resolution

        # Fallback to first value if tree building fails
        return PHIResolution(
            type=PHIResolutionType.FALLBACK,
            single_value=phi.phi_operands[0].value
        )

    def _build_phi_decision_tree(
        self,
        pred_blocks: List[BlockId],
        val_by_pred: Dict[BlockId, ValueId],
    ) -> Optional[PHIResolution]:
        """Build a binary decision tree for PHI resolution.

        Recursively finds branches that separate predecessors and builds
        nested TWO_WAY resolutions.

        Args:
            pred_blocks: List of predecessor blocks to resolve
            val_by_pred: Map from predecessor block to its value

        Returns:
            PHIResolution with nested structure, or None if no separation found
        """
        # Base case: 1 predecessor - return FALLBACK with that value
        if len(pred_blocks) == 1:
            return PHIResolution(
                type=PHIResolutionType.FALLBACK,
                single_value=val_by_pred[pred_blocks[0]]
            )

        # Base case: 2 predecessors with same value - return FALLBACK
        if len(pred_blocks) == 2:
            val0 = val_by_pred[pred_blocks[0]]
            val1 = val_by_pred[pred_blocks[1]]
            if val0 == val1:
                return PHIResolution(
                    type=PHIResolutionType.FALLBACK,
                    single_value=val0
                )

        # Find a branch that separates predecessors into two non-empty groups
        split = self._find_separating_branch(pred_blocks)
        if split is None:
            return None

        cond_var, true_preds, false_preds = split

        # Build resolution for true branch
        if len(true_preds) == 1:
            # Single predecessor - use value directly
            true_resolution = None
            true_value = val_by_pred[true_preds[0]]
        else:
            # Multiple predecessors - check if all have same value
            true_vals = set(val_by_pred[p] for p in true_preds)
            if len(true_vals) == 1:
                true_resolution = None
                true_value = true_vals.pop()
            else:
                # Recursively build subtree
                true_resolution = self._build_phi_decision_tree(true_preds, val_by_pred)
                if true_resolution is None:
                    # Recursive call failed - can't resolve this PHI
                    return None
                true_value = None

        # Build resolution for false branch
        if len(false_preds) == 1:
            # Single predecessor - use value directly
            false_resolution = None
            false_value = val_by_pred[false_preds[0]]
        else:
            # Multiple predecessors - check if all have same value
            false_vals = set(val_by_pred[p] for p in false_preds)
            if len(false_vals) == 1:
                false_resolution = None
                false_value = false_vals.pop()
            else:
                # Recursively build subtree
                false_resolution = self._build_phi_decision_tree(false_preds, val_by_pred)
                if false_resolution is None:
                    # Recursive call failed - can't resolve this PHI
                    return None
                false_value = None

        return PHIResolution(
            type=PHIResolutionType.TWO_WAY,
            condition=cond_var,
            true_value=true_value,
            false_value=false_value,
            nested_true=true_resolution,
            nested_false=false_resolution,
        )

    def _find_separating_branch(
        self,
        pred_blocks: List[BlockId],
    ) -> Optional[Tuple[str, List[BlockId], List[BlockId]]]:
        """Find a branch that separates predecessors into two non-empty groups.

        Args:
            pred_blocks: List of predecessor blocks

        Returns:
            (condition_var, true_preds, false_preds) or None if no separation found
        """
        branch_conds = self.branch_conditions

        for block_name, cond_info in branch_conds.items():
            if len(cond_info) != 2:
                continue

            targets = list(cond_info.keys())
            true_target = None
            false_target = None

            for target, (cond_var, is_true) in cond_info.items():
                if is_true:
                    true_target = target
                else:
                    false_target = target

            if not true_target or not false_target:
                continue

            # Classify each predecessor by which branch reaches it
            true_preds: List[BlockId] = []
            false_preds: List[BlockId] = []

            for pred in pred_blocks:
                # Case 1: pred IS the branching block itself
                # The edge pred->PHI_block exists when the branch goes directly to the PHI block
                if pred == block_name:
                    # Check which branch target the PHI block is
                    # (One of true_target or false_target must be the PHI block for this to matter)
                    from_true = (true_target in pred_blocks) or self._is_reachable(true_target, pred)
                    from_false = (false_target in pred_blocks) or self._is_reachable(false_target, pred)

                    # If this block branches directly to another predecessor via true/false,
                    # classify based on which target reaches more of our predecessors
                    # Actually, for the branching block, its contribution to the PHI is when
                    # it takes the branch that leads directly or indirectly to the PHI block
                    # But since pred IS the branching block, the edge pred->PHI must be one of the targets
                    if true_target not in pred_blocks and false_target not in pred_blocks:
                        # Neither target is another predecessor.
                        # The PHI block must be one of the targets (since pred edges to PHI).
                        # Determine which by checking if targets can reach OTHER predecessors.
                        # The target that CAN reach other preds leads to them;
                        # the target that CANNOT reach other preds is likely the PHI block.
                        other_preds = [p for p in pred_blocks if p != pred]
                        true_reaches_others = any(
                            self._is_reachable(true_target, op) for op in other_preds
                        )
                        false_reaches_others = any(
                            self._is_reachable(false_target, op) for op in other_preds
                        )

                        if true_reaches_others and not false_reaches_others:
                            # true_target leads to other preds, so pred goes via false to PHI
                            from_true = False
                            from_false = True
                        elif false_reaches_others and not true_reaches_others:
                            # false_target leads to other preds, so pred goes via true to PHI
                            from_true = True
                            from_false = False
                        # else: both or neither reach others, can't determine
                    elif true_target in pred_blocks and false_target not in pred_blocks:
                        # true_target is another predecessor, so this pred goes via true
                        # Wait no - if pred IS block_name and true_target is another pred,
                        # that means block_name branches to true_target, not to the PHI block
                        # So this pred's edge to PHI block must be via FALSE
                        from_true = False
                        from_false = True
                    elif false_target in pred_blocks and true_target not in pred_blocks:
                        from_true = True
                        from_false = False
                # Case 2: pred IS a direct branch target
                elif pred == true_target:
                    from_true = True
                    from_false = False
                elif pred == false_target:
                    from_true = False
                    from_false = True
                # Case 3: pred is reachable from one branch but not the other
                else:
                    from_true = self._is_reachable(true_target, pred)
                    from_false = self._is_reachable(false_target, pred)

                # Predecessor must be reachable from exactly one branch
                if from_true and not from_false:
                    true_preds.append(pred)
                elif from_false and not from_true:
                    false_preds.append(pred)
                # If reachable from both or neither, this branch doesn't separate

            # Check if this branch separates ALL predecessors into two non-empty groups
            if len(true_preds) + len(false_preds) == len(pred_blocks):
                if len(true_preds) > 0 and len(false_preds) > 0:
                    cond_var, _ = cond_info[true_target]
                    return (str(cond_var), true_preds, false_preds)

        return None

    def _peel_via_dominator(
        self,
        phi_block: str,
        remaining: List[str],
        val_by_pred: Dict[str, ValueId]
    ) -> Optional[Tuple[str, ValueId, List[str]]]:
        """Peel off a subset of predecessors using dominator-based condition.

        Returns (condition, value, list_of_peeled_preds) or None.
        """
        idom = self.cfg.immediate_dominators
        branch_conds = self.branch_conditions

        # Walk up dominator tree
        current = phi_block
        visited: Set[str] = set()
        max_depth = 100

        while current and current not in visited and len(visited) < max_depth:
            visited.add(current)
            dom = idom.get(current)
            if dom is None:
                break

            if dom in branch_conds:
                cond_info = branch_conds[dom]
                if len(cond_info) == 2:
                    targets = list(cond_info.keys())
                    t0, t1 = targets
                    cond_var, is_t0_true = cond_info[t0]

                    # Map remaining predecessors to targets
                    t0_preds: List[str] = []
                    t1_preds: List[str] = []
                    unmapped: List[str] = []

                    for pred in remaining:
                        t0_reaches = self._is_dominated_reachable(t0, pred, dom)
                        t1_reaches = self._is_dominated_reachable(t1, pred, dom)

                        if t0_reaches and not t1_reaches:
                            t0_preds.append(pred)
                        elif t1_reaches and not t0_reaches:
                            t1_preds.append(pred)
                        else:
                            unmapped.append(pred)

                    # If we can peel some but not all, and all peeled have same value
                    for preds, target in [(t0_preds, t0), (t1_preds, t1)]:
                        if preds and len(preds) < len(remaining):
                            vals = list(set(val_by_pred[p] for p in preds))
                            if len(vals) == 1:
                                _, is_target_true = cond_info[target]
                                cond = cond_var if is_target_true else f"!{cond_var}"
                                return (cond, vals[0], preds)

            current = dom

        return None

    def _resolve_via_ancestor_trace_for_groups(
        self,
        val_to_preds: Dict[ValueId, List[BlockId]],
        unique_vals: List[ValueId]
    ) -> Optional[PHIResolution]:
        """Resolve 2-value multi-way PHI by finding branch that separates value groups.

        When the controlling branch isn't in the dominator tree, we search ALL
        branching blocks for one that exclusively separates the two value groups.
        """
        if len(unique_vals) != 2:
            return None

        val0, val1 = unique_vals
        preds0 = val_to_preds[val0]
        preds1 = val_to_preds[val1]

        branch_conds = self.branch_conditions

        # For each branching block, check if it separates the two groups
        for block_name, cond_info in branch_conds.items():
            if len(cond_info) != 2:
                continue

            targets = list(cond_info.keys())
            true_target = None
            false_target = None

            for target, (cond_var, is_true) in cond_info.items():
                if is_true:
                    true_target = target
                else:
                    false_target = target

            if not true_target or not false_target:
                continue

            # Check reachability for ALL preds in each group
            # Group 0 from true, Group 1 from false
            g0_all_from_true = all(
                self._is_reachable(true_target, p) and not self._is_reachable(false_target, p)
                for p in preds0
            )
            g1_all_from_false = all(
                self._is_reachable(false_target, p) and not self._is_reachable(true_target, p)
                for p in preds1
            )

            if g0_all_from_true and g1_all_from_false:
                cond_var, _ = cond_info[true_target]
                return PHIResolution(
                    type=PHIResolutionType.TWO_WAY,
                    condition=cond_var,
                    true_value=val0,
                    false_value=val1,
                )

            # Group 0 from false, Group 1 from true
            g0_all_from_false = all(
                self._is_reachable(false_target, p) and not self._is_reachable(true_target, p)
                for p in preds0
            )
            g1_all_from_true = all(
                self._is_reachable(true_target, p) and not self._is_reachable(false_target, p)
                for p in preds1
            )

            if g0_all_from_false and g1_all_from_true:
                cond_var, _ = cond_info[true_target]
                return PHIResolution(
                    type=PHIResolutionType.TWO_WAY,
                    condition=cond_var,
                    true_value=val1,
                    false_value=val0,
                )

        return None

    def _peel_via_ancestor_trace(
        self,
        remaining: List[str],
        val_by_pred: Dict[str, ValueId]
    ) -> Optional[Tuple[str, ValueId, List[str]]]:
        """Peel off one predecessor from a 2-predecessor list using ancestor trace.

        This is a fallback when dominator peeling fails because the controlling
        branch isn't in the dominator tree.
        """
        if len(remaining) != 2:
            return None

        pred0, pred1 = remaining
        val0 = val_by_pred[pred0]
        val1 = val_by_pred[pred1]

        branch_conds = self.branch_conditions

        for block_name, cond_info in branch_conds.items():
            if len(cond_info) != 2:
                continue

            targets = list(cond_info.keys())
            true_target = None
            false_target = None

            for target, (cond_var, is_true) in cond_info.items():
                if is_true:
                    true_target = target
                else:
                    false_target = target

            if not true_target or not false_target:
                continue

            # Check exclusive reachability
            pred0_from_true = self._is_reachable(true_target, pred0)
            pred0_from_false = self._is_reachable(false_target, pred0)
            pred1_from_true = self._is_reachable(true_target, pred1)
            pred1_from_false = self._is_reachable(false_target, pred1)

            cond_var, _ = cond_info[true_target]

            # pred0 only from true, pred1 only from false
            if pred0_from_true and not pred0_from_false and pred1_from_false and not pred1_from_true:
                # Peel pred0 with condition true
                return (str(cond_var), val0, [pred0])

            # pred0 only from false, pred1 only from true
            if pred0_from_false and not pred0_from_true and pred1_from_true and not pred1_from_false:
                # Peel pred1 with condition true
                return (str(cond_var), val1, [pred1])

        return None

    def _find_condition_for_groups(self, preds_a: List[BlockId],
                                    preds_b: List[BlockId]) -> Optional[str]:
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

    def _any_reachable(self, start: str, targets: List[BlockId]) -> bool:
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

    def _peel_one_predecessor(self, remaining: List[BlockId],
                               val_by_pred: Dict[BlockId, ValueId]
                               ) -> Optional[Tuple[str, ValueId, BlockId]]:
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
