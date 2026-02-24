"""MIR Graph for debugging queries.

Provides a NetworkX-based graph representation of MIR for intuitive queries:
- Value definition/usage tracing
- Control flow path finding
- PHI node analysis
- Param-to-value mapping
- DAE residual/jacobian variable lookup

Example usage:
    from vajax.debug import MIRGraph

    # Build from VA file
    graph = MIRGraph.from_va_file('model.va', func='eval')

    # Query value definition
    graph.who_defines('v273116')  # Returns instruction info

    # Query value usage
    graph.who_uses('v142825')  # Returns list of instructions

    # Trace backwards from a value
    graph.trace_back('v273116', depth=5)  # Returns dependency chain

    # Find path between blocks
    graph.path_to_block('block1458')  # Returns path from entry

    # Get PHI resolution info
    graph.phi_info('block1458')  # Returns PHI nodes and their sources

    # Find what param corresponds to a value
    graph.value_to_param('v142825')  # Returns 'rth'

    # Find DAE residual variable
    graph.dae_residual('dt')  # Returns 'v273116'
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import networkx as nx

    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    nx = None


@dataclass
class NodeInfo:
    """Information about a graph node."""

    node_type: str  # 'value', 'block', 'param', 'constant'
    name: str
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EdgeInfo:
    """Information about a graph edge."""

    edge_type: str  # 'defines', 'uses', 'flows_to', 'phi_from', 'bound_to', 'dae_residual'
    data: Dict[str, Any] = field(default_factory=dict)


class MIRGraph:
    """Graph representation of MIR for debugging queries.

    Node types:
    - value:vXXXX - SSA values
    - block:blockXXX - Basic blocks
    - param:name - Model parameters
    - const:vXXX - Constants (float, int, bool, str)
    - dae:node_name - DAE system nodes

    Edge types:
    - defines: instruction -> result value
    - uses: operand value -> instruction result
    - flows_to: block -> successor block (with 'branch' label for T/F)
    - phi_from: (predecessor block, value) -> phi result
    - bound_to: param -> value
    - dae_residual: dae node -> resist/react MIR value
    - dae_jacobian: (dae row, dae col) -> resist/react MIR value
    """

    def __init__(self):
        if not HAS_NETWORKX:
            raise ImportError(
                "networkx is required for MIRGraph. Install with: pip install networkx"
            )

        self.graph = nx.DiGraph()
        self._entry_block: Optional[str] = None
        self._param_names: List[str] = []
        self._param_kinds: List[str] = []
        self._param_value_indices: List[int] = []
        self._dae_data: Optional[Dict] = None
        self._node_names: List[str] = []

    @classmethod
    def from_va_file(
        cls,
        va_path: Union[str, Path],
        func: str = "eval",
        include_dae: bool = True,
    ) -> "MIRGraph":
        """Build MIR graph from a Verilog-A file.

        Args:
            va_path: Path to .va file
            func: 'eval' or 'init'
            include_dae: Include DAE system mappings

        Returns:
            MIRGraph instance
        """
        import openvaf_py

        modules = openvaf_py.compile_va(str(va_path))
        if not modules:
            raise ValueError(f"No modules found in {va_path}")
        module = modules[0]

        return cls.from_module(module, func=func, include_dae=include_dae)

    @classmethod
    def from_module(
        cls,
        module,
        func: str = "eval",
        include_dae: bool = True,
    ) -> "MIRGraph":
        """Build MIR graph from a compiled VaModule.

        Args:
            module: VaModule from openvaf_py.compile_va()
            func: 'eval' or 'init'
            include_dae: Include DAE system mappings

        Returns:
            MIRGraph instance
        """
        from openvaf_jax.mir import parse_mir_function

        graph = cls()

        # Get MIR data
        if func == "eval":
            mir_data = module.get_mir_instructions()
        else:
            mir_data = module.get_init_mir_instructions()

        str_constants = dict(module.get_str_constants())
        mir_func = parse_mir_function(func, mir_data, str_constants)

        # Store metadata
        graph._entry_block = mir_func.entry_block
        graph._param_names = list(module.param_names)
        graph._param_kinds = list(module.param_kinds)
        graph._param_value_indices = list(module.param_value_indices)

        if include_dae and func == "eval":
            graph._dae_data = module.get_dae_system()
            desc = module.get_osdi_descriptor()
            graph._node_names = [n["name"] for n in desc["nodes"]]

        # Build graph
        graph._add_constants(mir_func)
        graph._add_params(module)
        graph._add_blocks(mir_func)
        graph._add_instructions(mir_func)

        if include_dae and func == "eval":
            graph._add_dae_mappings()

        return graph

    def _add_constants(self, mir_func):
        """Add constant nodes."""
        # Float constants
        for val_id, value in mir_func.constants.items():
            node_id = f"const:{val_id}"
            self.graph.add_node(
                node_id, node_type="constant", value_id=val_id, value=value, const_type="float"
            )
            # Also add as value node for lookups
            self.graph.add_node(
                f"value:{val_id}",
                node_type="value",
                value_id=val_id,
                is_constant=True,
                const_value=value,
                const_type="float",
            )

        # Bool constants
        for val_id, value in mir_func.bool_constants.items():
            node_id = f"const:{val_id}"
            self.graph.add_node(
                node_id, node_type="constant", value_id=val_id, value=value, const_type="bool"
            )
            self.graph.add_node(
                f"value:{val_id}",
                node_type="value",
                value_id=val_id,
                is_constant=True,
                const_value=value,
                const_type="bool",
            )

        # Int constants
        for val_id, value in mir_func.int_constants.items():
            node_id = f"const:{val_id}"
            self.graph.add_node(
                node_id, node_type="constant", value_id=val_id, value=value, const_type="int"
            )
            self.graph.add_node(
                f"value:{val_id}",
                node_type="value",
                value_id=val_id,
                is_constant=True,
                const_value=value,
                const_type="int",
            )

        # String constants
        for val_id, value in mir_func.str_constants.items():
            node_id = f"const:{val_id}"
            self.graph.add_node(
                node_id, node_type="constant", value_id=val_id, value=value, const_type="str"
            )
            self.graph.add_node(
                f"value:{val_id}",
                node_type="value",
                value_id=val_id,
                is_constant=True,
                const_value=value,
                const_type="str",
            )

    def _add_params(self, module):
        """Add parameter nodes and bindings."""
        for i, name in enumerate(self._param_names):
            kind = self._param_kinds[i]
            val_idx = self._param_value_indices[i] if i < len(self._param_value_indices) else None
            val_id = f"v{val_idx}" if val_idx is not None else None

            node_id = f"param:{name}"
            self.graph.add_node(
                node_id, node_type="param", name=name, kind=kind, param_index=i, value_id=val_id
            )

            # Add binding edge
            if val_id:
                value_node = f"value:{val_id}"
                if value_node not in self.graph:
                    self.graph.add_node(
                        value_node,
                        node_type="value",
                        value_id=val_id,
                        is_param=True,
                        param_name=name,
                        param_kind=kind,
                    )
                else:
                    # Update existing node
                    self.graph.nodes[value_node]["is_param"] = True
                    self.graph.nodes[value_node]["param_name"] = name
                    self.graph.nodes[value_node]["param_kind"] = kind

                self.graph.add_edge(node_id, value_node, edge_type="bound_to")

    def _add_blocks(self, mir_func):
        """Add block nodes and control flow edges."""
        for block_name, block in mir_func.blocks.items():
            node_id = f"block:{block_name}"

            # Get terminator info
            term = block.terminator
            term_info = {}
            if term:
                term_info["terminator"] = term.opcode
                if term.is_branch:
                    term_info["condition"] = term.condition
                    term_info["true_block"] = term.true_block
                    term_info["false_block"] = term.false_block
                elif term.is_jump:
                    term_info["target"] = term.target_block

            self.graph.add_node(
                node_id,
                node_type="block",
                name=block_name,
                predecessors=list(block.predecessors),
                successors=list(block.successors),
                num_phis=len(block.phi_nodes),
                num_instructions=len(block.instructions),
                **term_info,
            )

            # Add control flow edges
            if term:
                if term.is_branch:
                    if term.true_block:
                        self.graph.add_edge(
                            node_id,
                            f"block:{term.true_block}",
                            edge_type="flows_to",
                            branch="true",
                            condition=term.condition,
                        )
                    if term.false_block:
                        self.graph.add_edge(
                            node_id,
                            f"block:{term.false_block}",
                            edge_type="flows_to",
                            branch="false",
                            condition=term.condition,
                        )
                elif term.is_jump and term.target_block:
                    self.graph.add_edge(
                        node_id,
                        f"block:{term.target_block}",
                        edge_type="flows_to",
                        branch="unconditional",
                    )

    def _add_instructions(self, mir_func):
        """Add instruction edges (defines/uses)."""
        for block_name, block in mir_func.blocks.items():
            for inst in block.instructions:
                if inst.result:
                    result_node = f"value:{inst.result}"

                    # Ensure result node exists
                    if result_node not in self.graph:
                        self.graph.add_node(result_node, node_type="value", value_id=inst.result)

                    # Update with instruction info
                    self.graph.nodes[result_node]["opcode"] = inst.opcode
                    self.graph.nodes[result_node]["block"] = block_name
                    self.graph.nodes[result_node]["is_phi"] = inst.is_phi

                    # Add operand edges (uses -> defines)
                    for operand in inst.operands:
                        op_node = f"value:{operand}"
                        if op_node not in self.graph:
                            self.graph.add_node(op_node, node_type="value", value_id=operand)
                        self.graph.add_edge(
                            op_node,
                            result_node,
                            edge_type="uses",
                            opcode=inst.opcode,
                            block=block_name,
                        )

                    # Add PHI-specific edges
                    if inst.is_phi and inst.phi_operands:
                        self.graph.nodes[result_node]["phi_operands"] = [
                            (op.block, op.value) for op in inst.phi_operands
                        ]
                        for phi_op in inst.phi_operands:
                            op_node = f"value:{phi_op.value}"
                            if op_node not in self.graph:
                                self.graph.add_node(
                                    op_node, node_type="value", value_id=phi_op.value
                                )
                            self.graph.add_edge(
                                op_node, result_node, edge_type="phi_from", from_block=phi_op.block
                            )

    def _add_dae_mappings(self):
        """Add DAE system mappings."""
        if not self._dae_data:
            return

        # Add DAE nodes
        for i, name in enumerate(self._node_names):
            node_id = f"dae:{name}"
            self.graph.add_node(node_id, node_type="dae", name=name, index=i)

        # Add residual mappings
        for res in self._dae_data.get("residuals", []):
            node_name = res["node_name"]
            dae_node = f"dae:{node_name}"

            resist_var = res.get("resist_var", "")
            react_var = res.get("react_var", "")

            if resist_var and resist_var.startswith("mir_"):
                val_id = f"v{resist_var[4:]}"
                value_node = f"value:{val_id}"
                if value_node not in self.graph:
                    self.graph.add_node(value_node, node_type="value", value_id=val_id)
                self.graph.add_edge(
                    dae_node,
                    value_node,
                    edge_type="dae_residual",
                    resist_or_react="resist",
                    mir_var=resist_var,
                )
                self.graph.nodes[dae_node]["resist_var"] = val_id

            if react_var and react_var.startswith("mir_"):
                val_id = f"v{react_var[4:]}"
                value_node = f"value:{val_id}"
                if value_node not in self.graph:
                    self.graph.add_node(value_node, node_type="value", value_id=val_id)
                self.graph.add_edge(
                    dae_node,
                    value_node,
                    edge_type="dae_residual",
                    resist_or_react="react",
                    mir_var=react_var,
                )
                self.graph.nodes[dae_node]["react_var"] = val_id

    # =========================================================================
    # Query Methods
    # =========================================================================

    def who_defines(self, value: str) -> Optional[Dict[str, Any]]:
        """Find what instruction defines a value.

        Args:
            value: Value ID like 'v273116' or 'value:v273116'

        Returns:
            Dict with opcode, block, operands, etc. or None if not found
        """
        if not value.startswith("value:"):
            value = f"value:{value}"

        if value not in self.graph:
            return None

        node_data = dict(self.graph.nodes[value])

        # Get incoming edges (what feeds into this value)
        predecessors = []
        for pred in self.graph.predecessors(value):
            edge_data = self.graph.edges[pred, value]
            predecessors.append(
                {"source": pred, "edge_type": edge_data.get("edge_type"), **edge_data}
            )
        node_data["inputs"] = predecessors

        return node_data

    def who_uses(self, value: str) -> List[Dict[str, Any]]:
        """Find what instructions use a value.

        Args:
            value: Value ID like 'v142825' or 'value:v142825'

        Returns:
            List of dicts with info about each usage
        """
        if not value.startswith("value:"):
            value = f"value:{value}"

        if value not in self.graph:
            return []

        usages = []
        for succ in self.graph.successors(value):
            edge_data = self.graph.edges[value, succ]
            if edge_data.get("edge_type") in ("uses", "phi_from"):
                succ_data = dict(self.graph.nodes[succ])
                usages.append(
                    {
                        "target": succ,
                        "edge_type": edge_data.get("edge_type"),
                        "opcode": succ_data.get("opcode"),
                        "block": succ_data.get("block"),
                        **edge_data,
                    }
                )
        return usages

    def trace_back(self, value: str, depth: int = 10) -> List[Dict[str, Any]]:
        """Trace a value's dependencies backwards.

        Args:
            value: Value ID to start from
            depth: Maximum depth to trace

        Returns:
            List of dependency info, ordered from immediate to distant
        """
        if not value.startswith("value:"):
            value = f"value:{value}"

        if value not in self.graph:
            return []

        result = []
        visited = set()
        queue = [(value, 0)]

        while queue:
            current, current_depth = queue.pop(0)

            if current in visited or current_depth > depth:
                continue
            visited.add(current)

            if not current.startswith("value:"):
                continue

            node_data = dict(self.graph.nodes[current])
            node_data["_depth"] = current_depth
            node_data["_node_id"] = current
            result.append(node_data)

            # Queue predecessors
            for pred in self.graph.predecessors(current):
                if pred.startswith("value:"):
                    queue.append((pred, current_depth + 1))

        return result

    def trace_forward(self, value: str, depth: int = 10) -> List[Dict[str, Any]]:
        """Trace what depends on a value forwards.

        Args:
            value: Value ID to start from
            depth: Maximum depth to trace

        Returns:
            List of dependent value info
        """
        if not value.startswith("value:"):
            value = f"value:{value}"

        if value not in self.graph:
            return []

        result = []
        visited = set()
        queue = [(value, 0)]

        while queue:
            current, current_depth = queue.pop(0)

            if current in visited or current_depth > depth:
                continue
            visited.add(current)

            if not current.startswith("value:"):
                continue

            node_data = dict(self.graph.nodes[current])
            node_data["_depth"] = current_depth
            node_data["_node_id"] = current
            result.append(node_data)

            # Queue successors
            for succ in self.graph.successors(current):
                if succ.startswith("value:"):
                    queue.append((succ, current_depth + 1))

        return result

    def path_to_block(self, target: str, source: Optional[str] = None) -> Optional[List[str]]:
        """Find control flow path to a block.

        Args:
            target: Target block like 'block1458' or 'block:block1458'
            source: Source block (default: entry block)

        Returns:
            List of block names in path, or None if no path exists
        """
        if not target.startswith("block:"):
            target = f"block:{target}"

        if source is None:
            source = f"block:{self._entry_block}"
        elif not source.startswith("block:"):
            source = f"block:{source}"

        if source not in self.graph or target not in self.graph:
            return None

        try:
            # Only follow control flow edges
            path = nx.shortest_path(self.graph, source, target)
            # Filter to only block nodes
            return [n.replace("block:", "") for n in path if n.startswith("block:")]
        except nx.NetworkXNoPath:
            return None

    def phi_info(self, block: str) -> List[Dict[str, Any]]:
        """Get PHI node information for a block.

        Args:
            block: Block name like 'block1458'

        Returns:
            List of PHI info dicts with result, operands, etc.
        """
        if not block.startswith("block:"):
            pass
        else:
            block = block.replace("block:", "")

        phis = []
        for node_id, data in self.graph.nodes(data=True):
            if (
                data.get("node_type") == "value"
                and data.get("is_phi")
                and data.get("block") == block
            ):
                phis.append(
                    {
                        "result": data.get("value_id"),
                        "phi_operands": data.get("phi_operands", []),
                        "opcode": data.get("opcode"),
                    }
                )
        return phis

    def value_to_param(self, value: str) -> Optional[str]:
        """Find what parameter a value corresponds to.

        Args:
            value: Value ID like 'v142825'

        Returns:
            Parameter name or None
        """
        if not value.startswith("value:"):
            value = f"value:{value}"

        if value not in self.graph:
            return None

        data = self.graph.nodes[value]
        return data.get("param_name")

    def param_to_value(self, param: str) -> Optional[str]:
        """Find what value ID a parameter is bound to.

        Args:
            param: Parameter name like 'rth'

        Returns:
            Value ID or None
        """
        param_node = f"param:{param}"
        if param_node not in self.graph:
            return None

        data = self.graph.nodes[param_node]
        return data.get("value_id")

    def dae_residual(self, node_name: str) -> Dict[str, Optional[str]]:
        """Find DAE residual MIR variables for a node.

        Args:
            node_name: DAE node name like 'dt'

        Returns:
            Dict with 'resist' and 'react' value IDs
        """
        dae_node = f"dae:{node_name}"
        if dae_node not in self.graph:
            return {"resist": None, "react": None}

        data = self.graph.nodes[dae_node]
        return {
            "resist": data.get("resist_var"),
            "react": data.get("react_var"),
        }

    def is_constant(self, value: str) -> Tuple[bool, Any]:
        """Check if a value is a constant.

        Args:
            value: Value ID like 'v3'

        Returns:
            Tuple of (is_constant, value) where value is None if not constant
        """
        if not value.startswith("value:"):
            value = f"value:{value}"

        if value not in self.graph:
            return (False, None)

        data = self.graph.nodes[value]
        if data.get("is_constant"):
            return (True, data.get("const_value"))
        return (False, None)

    def blocks_with_condition(self, condition_value: str) -> List[str]:
        """Find blocks that branch on a specific condition.

        Args:
            condition_value: Value ID of the condition

        Returns:
            List of block names
        """
        blocks = []
        for node_id, data in self.graph.nodes(data=True):
            if data.get("node_type") == "block" and data.get("condition") == condition_value:
                blocks.append(data.get("name"))
        return blocks

    def branch_condition(self, block: str) -> Optional[Dict[str, Any]]:
        """Get branch condition info for a block.

        Args:
            block: Block name

        Returns:
            Dict with condition, true_block, false_block or None
        """
        if not block.startswith("block:"):
            block = f"block:{block}"

        if block not in self.graph:
            return None

        data = self.graph.nodes[block]
        if data.get("terminator") == "br":
            return {
                "condition": data.get("condition"),
                "true_block": data.get("true_block"),
                "false_block": data.get("false_block"),
            }
        return None

    # =========================================================================
    # Summary/Stats Methods
    # =========================================================================

    def summary(self) -> Dict[str, Any]:
        """Get graph summary statistics."""
        node_types = {}
        edge_types = {}

        for _, data in self.graph.nodes(data=True):
            nt = data.get("node_type", "unknown")
            node_types[nt] = node_types.get(nt, 0) + 1

        for _, _, data in self.graph.edges(data=True):
            et = data.get("edge_type", "unknown")
            edge_types[et] = edge_types.get(et, 0) + 1

        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "node_types": node_types,
            "edge_types": edge_types,
            "entry_block": self._entry_block,
            "num_params": len(self._param_names),
            "num_dae_nodes": len(self._node_names),
        }

    def print_summary(self):
        """Print graph summary."""
        s = self.summary()
        print("MIR Graph Summary")
        print(f"  Total nodes: {s['total_nodes']}")
        print(f"  Total edges: {s['total_edges']}")
        print(f"  Entry block: {s['entry_block']}")
        print(f"  Parameters: {s['num_params']}")
        print(f"  DAE nodes: {s['num_dae_nodes']}")
        print(f"  Node types: {s['node_types']}")
        print(f"  Edge types: {s['edge_types']}")
