"""Circuit data structures for JAX-SPICE

Represents parsed netlist as Python objects.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class Model:
    """Model definition mapping name to device module"""
    name: str
    module: str
    params: Dict[str, str] = field(default_factory=dict)


@dataclass
class Instance:
    """Device or subcircuit instance"""
    name: str
    terminals: List[str]
    model: str
    params: Dict[str, str] = field(default_factory=dict)


@dataclass
class Subcircuit:
    """Subcircuit definition"""
    name: str
    terminals: List[str]
    params: Dict[str, str] = field(default_factory=dict)
    instances: List[Instance] = field(default_factory=list)


@dataclass
class Circuit:
    """Top-level circuit containing all definitions"""
    title: Optional[str] = None
    loads: List[str] = field(default_factory=list)
    includes: List[str] = field(default_factory=list)
    globals: List[str] = field(default_factory=list)
    ground: Optional[str] = None
    models: Dict[str, Model] = field(default_factory=dict)
    subckts: Dict[str, Subcircuit] = field(default_factory=dict)
    params: Dict[str, str] = field(default_factory=dict)
    top_instances: List[Instance] = field(default_factory=list)

    def flatten(self, top_subckt: str) -> Tuple[List[Instance], Dict[str, int]]:
        """Flatten hierarchy starting from given subcircuit

        Returns:
            Tuple of (flat instance list, node name to index mapping)
        """
        flat_instances = []
        nodes = {self.ground or '0': 0}  # Ground is node 0

        # Add global nodes
        for g in self.globals:
            if g not in nodes:
                nodes[g] = len(nodes)

        def add_node(name: str) -> int:
            if name not in nodes:
                nodes[name] = len(nodes)
            return nodes[name]

        def flatten_subckt(subckt: Subcircuit, prefix: str, port_map: Dict[str, str]):
            """Recursively flatten a subcircuit"""
            for inst in subckt.instances:
                # Map terminal names through port_map and hierarchy
                mapped_terminals = []
                for t in inst.terminals:
                    if t in port_map:
                        mapped_terminals.append(port_map[t])
                    elif t in self.globals or t == (self.ground or '0'):
                        mapped_terminals.append(t)
                    else:
                        mapped_terminals.append(f"{prefix}.{t}")

                # Check if this is a subcircuit instance
                if inst.model in self.subckts:
                    # Recursive flatten
                    sub = self.subckts[inst.model]
                    new_prefix = f"{prefix}.{inst.name}"
                    new_port_map = dict(zip(sub.terminals, mapped_terminals))
                    flatten_subckt(sub, new_prefix, new_port_map)
                else:
                    # Leaf instance - add to flat list
                    for t in mapped_terminals:
                        add_node(t)

                    flat_inst = Instance(
                        name=f"{prefix}.{inst.name}",
                        terminals=mapped_terminals,
                        model=inst.model,
                        params={**inst.params}  # Copy params
                    )
                    flat_instances.append(flat_inst)

        # Start flattening from top subcircuit
        if top_subckt not in self.subckts:
            raise ValueError(f"Subcircuit '{top_subckt}' not found")

        top = self.subckts[top_subckt]
        # Top subcircuit has no external connections (empty port map for internal nodes)
        flatten_subckt(top, top_subckt, {})

        # Also add top-level instances
        for inst in self.top_instances:
            mapped_terminals = []
            for t in inst.terminals:
                if t in self.globals or t == (self.ground or '0'):
                    mapped_terminals.append(t)
                else:
                    mapped_terminals.append(t)
                add_node(mapped_terminals[-1])

            if inst.model in self.subckts:
                sub = self.subckts[inst.model]
                new_port_map = dict(zip(sub.terminals, mapped_terminals))
                flatten_subckt(sub, inst.name, new_port_map)
            else:
                flat_inst = Instance(
                    name=inst.name,
                    terminals=mapped_terminals,
                    model=inst.model,
                    params={**inst.params}
                )
                flat_instances.append(flat_inst)

        return flat_instances, nodes

    def stats(self) -> Dict[str, int]:
        """Return statistics about the circuit"""
        return {
            'num_subckts': len(self.subckts),
            'num_models': len(self.models),
            'num_top_instances': len(self.top_instances),
            'num_globals': len(self.globals),
            'num_loads': len(self.loads),
        }
