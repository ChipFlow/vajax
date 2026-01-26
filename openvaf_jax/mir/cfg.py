"""Control Flow Graph analysis for MIR.

This module provides CFG analysis including:
- Dominator computation using iterative algorithm
- Natural loop detection via back-edge identification
- Topological ordering with loops as single units
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Union

from .types import MIRFunction


@dataclass
class LoopInfo:
    """Information about a natural loop in the CFG.

    A natural loop is defined by a back-edge (edge to a dominator).
    The loop header dominates all nodes in the loop body.
    """
    header: str
    body: Set[str]  # All blocks in the loop (including header)
    exits: List[str]  # Blocks outside the loop reachable from loop
    back_edges: List[Tuple[str, str]]  # (source, header) pairs


class CFGAnalyzer:
    """Analyzes control flow graph structure.

    Provides dominator computation, loop detection, and topological ordering.
    All analysis is computed lazily on first access and cached.
    """

    def __init__(self, mir_func: MIRFunction):
        """Initialize analyzer with a MIR function.

        Args:
            mir_func: The MIR function to analyze
        """
        self.mir_func = mir_func
        self.blocks = mir_func.blocks
        self.entry = mir_func.entry_block

        # Cached analysis results
        self._dominators: Optional[Dict[str, Set[str]]] = None
        self._idom: Optional[Dict[str, Optional[str]]] = None
        self._loops: Optional[List[LoopInfo]] = None
        self._post_order: Optional[List[str]] = None
        self._reachable: Optional[Set[str]] = None

    @property
    def reachable_blocks(self) -> Set[str]:
        """Get set of blocks reachable from entry."""
        if self._reachable is None:
            self._reachable = self._compute_reachable()
        return self._reachable

    @property
    def dominators(self) -> Dict[str, Set[str]]:
        """Get dominator sets for each block.

        WARNING: This is O(n^2) for large CFGs and should be avoided.
        Use SSAAnalyzer's successor-pair lookup for PHI resolution instead.

        Returns:
            Dict mapping block name -> set of blocks that dominate it
        """
        if self._dominators is None:
            self._dominators = self._compute_dominators()
        return self._dominators

    @property
    def immediate_dominators(self) -> Dict[str, Optional[str]]:
        """Get immediate dominator for each block.

        Returns:
            Dict mapping block name -> immediate dominator (None for entry)
        """
        if self._idom is None:
            self._idom = self._compute_immediate_dominators()
        return self._idom

    @property
    def loops(self) -> List[LoopInfo]:
        """Get all natural loops in the CFG.

        Uses DFS-based back-edge detection (O(V+E)) instead of dominator
        computation (O(n^2)). This is much faster for large CFGs.

        Returns:
            List of LoopInfo for each detected loop
        """
        if self._loops is None:
            self._loops = self._find_loops_dfs()
        return self._loops

    def topological_order(self) -> List[Union[str, LoopInfo]]:
        """Get blocks in topological order with loops as units.

        Returns a list where:
        - Regular blocks are represented by their name (str)
        - Loops are represented by LoopInfo objects

        The ordering respects data dependencies:
        - A block appears after all its non-loop predecessors
        - A loop appears after all blocks that precede its header
        """
        return self._compute_topological_order()

    def dominates(self, a: str, b: str) -> bool:
        """Check if block a dominates block b.

        Block a dominates b if every path from entry to b goes through a.
        """
        return a in self.dominators.get(b, set())

    def _compute_reachable(self) -> Set[str]:
        """Compute blocks reachable from entry using DFS."""
        if self.entry not in self.blocks:
            return set()

        visited = set()
        stack = [self.entry]

        while stack:
            block = stack.pop()
            if block in visited:
                continue
            visited.add(block)
            if block in self.blocks:
                for succ in self.blocks[block].successors:
                    if succ not in visited:
                        stack.append(succ)

        return visited

    def _compute_dominators(self) -> Dict[str, Set[str]]:
        """Compute dominator sets from immediate dominators.

        Derives full dominator sets by walking up the idom tree.
        This is computed on-demand since most uses only need idom.
        """
        idom = self.immediate_dominators
        dom: Dict[str, Set[str]] = {}

        # Sorted iteration for deterministic dict key order
        for block in sorted(self.reachable_blocks):
            # Walk up idom tree to collect all dominators
            dominators = {block}
            current = idom.get(block)
            while current is not None:
                dominators.add(current)
                current = idom.get(current)
            dom[block] = dominators

        return dom

    def _compute_immediate_dominators(self) -> Dict[str, Optional[str]]:
        """Compute immediate dominators using Cooper et al. algorithm.

        This is the "simple and fast" algorithm from:
        Cooper, Harvey, Kennedy. "A Simple, Fast Dominance Algorithm" (2001)

        Time complexity: O(n) for reducible CFGs (typical), O(n²) worst case.
        Much faster than the iterative dataflow algorithm for large CFGs.
        """
        reachable = self.reachable_blocks
        if not reachable:
            return {}

        # Step 1: Compute reverse post-order numbering via iterative DFS
        post_order: List[str] = []
        visited: Set[str] = set()
        # Stack: (block, have_visited_children)
        stack: List[Tuple[str, bool]] = [(self.entry, False)]

        while stack:
            block, children_done = stack.pop()

            if children_done:
                # All children processed, add to post-order
                post_order.append(block)
                continue

            if block in visited or block not in reachable:
                continue

            visited.add(block)
            # Push self again to add to post-order after children
            stack.append((block, True))

            # Push children (in reverse to maintain order)
            if block in self.blocks:
                for succ in reversed(self.blocks[block].successors):
                    if succ not in visited:
                        stack.append((succ, False))

        # Reverse to get reverse post-order
        rpo = list(reversed(post_order))
        rpo_number: Dict[str, int] = {block: i for i, block in enumerate(rpo)}

        # Step 2: Initialize idom
        idom: Dict[str, Optional[str]] = {block: None for block in reachable}
        idom[self.entry] = self.entry  # Entry's idom is itself (sentinel)

        def intersect(b1: str, b2: str) -> str:
            """Find common dominator by walking up idom tree."""
            finger1, finger2 = b1, b2
            while finger1 != finger2:
                while rpo_number.get(finger1, 0) > rpo_number.get(finger2, 0):
                    finger1 = idom[finger1] or finger1
                while rpo_number.get(finger2, 0) > rpo_number.get(finger1, 0):
                    finger2 = idom[finger2] or finger2
            return finger1

        # Step 3: Iterate until convergence
        changed = True
        while changed:
            changed = False
            for block in rpo:
                if block == self.entry:
                    continue

                # Get processed predecessors (those with idom already set)
                preds = [p for p in self.blocks[block].predecessors
                         if p in reachable and idom.get(p) is not None]

                if not preds:
                    continue

                # Start with first processed predecessor
                new_idom = preds[0]

                # Intersect with other predecessors
                for pred in preds[1:]:
                    if idom.get(pred) is not None:
                        new_idom = intersect(pred, new_idom)

                if idom[block] != new_idom:
                    idom[block] = new_idom
                    changed = True

        # Fix entry's idom to be None (we used self as sentinel)
        idom[self.entry] = None

        return idom

    def _find_loops(self) -> List[LoopInfo]:
        """Find all natural loops via back-edge detection.

        A back-edge is an edge from n to h where h dominates n.
        The natural loop for this back-edge includes:
        - All blocks that can reach n without going through h
        - Plus h (the header)

        IMPORTANT: Iterations are sorted for deterministic output.
        """
        loops = []
        reachable = self.reachable_blocks
        dom = self.dominators

        # Find all back-edges (sorted iteration for determinism)
        back_edges: List[Tuple[str, str]] = []
        for block_name in sorted(reachable):
            block = self.blocks[block_name]
            for succ in block.successors:
                if succ in dom.get(block_name, set()):
                    # succ dominates block_name, so this is a back-edge
                    back_edges.append((block_name, succ))

        # Group back-edges by header
        header_to_edges: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        for source, header in back_edges:
            header_to_edges[header].append((source, header))

        # For each header, compute the loop body (sorted for determinism)
        for header in sorted(header_to_edges.keys()):
            edges = header_to_edges[header]
            body = self._compute_loop_body(header, edges)
            exits = self._compute_loop_exits(body)
            loops.append(LoopInfo(
                header=header,
                body=body,
                exits=exits,
                back_edges=edges,
            ))

        return loops

    def _compute_loop_body(self, header: str,
                           back_edges: List[Tuple[str, str]]) -> Set[str]:
        """Compute all blocks in a natural loop.

        The loop body includes all blocks that can reach the back-edge
        source without going through the header.
        """
        body = {header}

        # Work backwards from each back-edge source
        for source, _ in back_edges:
            if source == header:
                continue

            worklist = [source]
            while worklist:
                block = worklist.pop()
                if block in body:
                    continue
                body.add(block)

                # Add predecessors (except those outside the loop)
                if block in self.blocks:
                    for pred in self.blocks[block].predecessors:
                        if pred not in body and pred in self.reachable_blocks:
                            worklist.append(pred)

        return body

    def _compute_loop_exits(self, body: Set[str]) -> List[str]:
        """Find blocks outside the loop reachable from loop blocks."""
        exits = []
        # Sorted iteration for determinism
        for block_name in sorted(body):
            if block_name in self.blocks:
                for succ in self.blocks[block_name].successors:
                    if succ not in body and succ not in exits:
                        exits.append(succ)
        return exits

    def _find_loops_dfs(self) -> List[LoopInfo]:
        """Find all natural loops via DFS-based back-edge detection.

        Uses iterative DFS traversal to find back-edges (edges to ancestors
        in DFS tree). This is O(V+E) compared to O(n^2) for dominator-based
        detection.

        For reducible CFGs (typical of compiler-generated code), this gives
        the same results as dominator-based loop detection.
        """
        if self.entry not in self.blocks:
            return []

        reachable = self.reachable_blocks
        back_edges: List[Tuple[str, str]] = []

        # Iterative DFS with explicit stack to avoid recursion limit
        # Stack contains (block, iterator_index, is_entering)
        # - is_entering=True: first visit, add to path
        # - is_entering=False: finished with children, remove from path
        visited: Set[str] = set()
        on_path: Set[str] = set()  # Nodes currently on the DFS path
        stack: List[Tuple[str, int, bool]] = [(self.entry, 0, True)]

        while stack:
            block, succ_idx, is_entering = stack.pop()

            if is_entering:
                if block in visited:
                    continue
                visited.add(block)
                on_path.add(block)

                if block in self.blocks:
                    successors = self.blocks[block].successors

                    # Push exit marker
                    stack.append((block, 0, False))

                    # Process successors
                    for succ in successors:
                        if succ not in reachable:
                            continue
                        if succ in on_path:
                            # Back-edge: edge to ancestor on current path
                            back_edges.append((block, succ))
                        elif succ not in visited:
                            stack.append((succ, 0, True))
                else:
                    # No successors, just remove from path immediately
                    on_path.discard(block)
            else:
                # Exiting: remove from path
                on_path.discard(block)

        # Group back-edges by header (target of back-edge)
        header_to_edges: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        for source, header in back_edges:
            header_to_edges[header].append((source, header))

        # Build LoopInfo for each unique header (sorted for determinism)
        loops = []
        for header in sorted(header_to_edges.keys()):
            edges = header_to_edges[header]
            body = self._compute_loop_body(header, edges)
            exits = self._compute_loop_exits(body)
            loops.append(LoopInfo(
                header=header,
                body=body,
                exits=exits,
                back_edges=edges,
            ))

        return loops

    def _compute_topological_order(self) -> List[Union[str, LoopInfo]]:
        """Compute topological order with loops as single nodes.

        Uses Kahn's algorithm on a condensed graph where each loop
        is treated as a single node.

        IMPORTANT: All set/dict iterations are sorted to ensure deterministic
        output across Python process runs (hash randomization).
        """
        reachable = self.reachable_blocks
        if not reachable:
            return []

        loops = self.loops

        # Map blocks to their containing loop (if any)
        block_to_loop: Dict[str, int] = {}
        for i, loop in enumerate(loops):
            for block in loop.body:
                block_to_loop[block] = i

        # Build condensed graph
        # Nodes are either block names (str) or loop indices (int)
        def get_node(block: str) -> Union[str, int]:
            if block in block_to_loop:
                return block_to_loop[block]
            return block

        # Helper for sorting mixed str/int nodes deterministically
        def node_sort_key(n: Union[str, int]) -> Tuple[int, Union[str, int]]:
            # Ints (loop indices) sort before strings
            if isinstance(n, int):
                return (0, n)
            return (1, n)

        # Compute in-degree for each node
        in_degree: Dict[Union[str, int], int] = defaultdict(int)
        edges: Dict[Union[str, int], Set[Union[str, int]]] = defaultdict(set)

        # Add all nodes (sorted for determinism)
        for block in sorted(reachable):
            node = get_node(block)
            if node not in in_degree:
                in_degree[node] = 0

        # Add edges (skip intra-loop edges) - sorted iteration for determinism
        for block_name in sorted(reachable):
            if block_name not in self.blocks:
                continue
            src_node = get_node(block_name)
            for succ in self.blocks[block_name].successors:
                if succ not in reachable:
                    continue
                dst_node = get_node(succ)
                if src_node != dst_node and dst_node not in edges[src_node]:
                    edges[src_node].add(dst_node)
                    in_degree[dst_node] += 1

        # Kahn's algorithm
        # Start with entry block/loop
        entry_node = get_node(self.entry)
        queue = [entry_node]
        result: List[Union[str, LoopInfo]] = []
        visited: Set[Union[str, int]] = set()

        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)

            if isinstance(node, int):
                result.append(loops[node])
            else:
                result.append(node)

            # Sort successors for deterministic ordering
            for succ in sorted(edges[node], key=node_sort_key):
                in_degree[succ] -= 1
                if in_degree[succ] == 0 and succ not in visited:
                    queue.append(succ)

        # Add any remaining blocks not reached (sorted for determinism)
        for node in sorted(in_degree.keys(), key=node_sort_key):
            if node not in visited:
                if isinstance(node, int):
                    result.append(loops[node])
                else:
                    result.append(node)

        return result

    def get_block_predecessors_outside_loop(self, block: str,
                                             loop: LoopInfo) -> List[str]:
        """Get predecessors of a block that are outside the given loop."""
        if block not in self.blocks:
            return []
        return [p for p in self.blocks[block].predecessors
                if p not in loop.body]

    def get_block_successors_outside_loop(self, block: str,
                                           loop: LoopInfo) -> List[str]:
        """Get successors of a block that are outside the given loop."""
        if block not in self.blocks:
            return []
        return [s for s in self.blocks[block].successors
                if s not in loop.body]
