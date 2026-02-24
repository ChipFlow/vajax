"""Circuit data structures for VA-JAX

Represents parsed netlist as Python objects.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ============================================================================
# Source Location Tracking (mirrors VACASK FileStack/Loc from sourceloc.cpp)
# ============================================================================


@dataclass
class SourceFile:
    """Entry in file stack for tracking includes.

    Similar to VACASK FileStackEntry (filestack.h:24-47).
    """

    filename: str
    canonical_path: str
    parent_id: Optional[int] = None  # ID of including file
    inclusion_line: int = 0  # Line in parent where included


@dataclass
class FileStack:
    """Track files for source location reporting.

    Similar to VACASK FileStack (filestack.h:49-120).
    Enables error messages with inclusion chains like:
        12:8 in models.inc
          included on line 3 of runme.sim
    """

    files: List[SourceFile] = field(default_factory=list)

    def add_file(
        self,
        filename: str,
        parent_id: Optional[int] = None,
        inclusion_line: int = 0,
    ) -> int:
        """Add a file to the stack, return its ID."""
        try:
            canonical = str(Path(filename).resolve())
        except (OSError, ValueError):
            canonical = filename
        self.files.append(SourceFile(filename, canonical, parent_id, inclusion_line))
        return len(self.files) - 1

    def get_file(self, file_id: int) -> Optional[SourceFile]:
        """Get file entry by ID."""
        if 0 <= file_id < len(self.files):
            return self.files[file_id]
        return None

    def format_location(self, file_id: int, line: int, col: int) -> str:
        """Format location with inclusion chain (like VACASK Loc::toString)."""
        parts = []
        parts.append(f"{line}:{col}")

        current_id = file_id
        first = True
        while current_id is not None:
            f = self.get_file(current_id)
            if f is None:
                break
            if first:
                parts.append(f" in {f.filename}")
                first = False
            else:
                parts.append(f"\n  included on line {f.inclusion_line} of {f.filename}")
            current_id = f.parent_id

        return "".join(parts)


@dataclass
class SourceLocation:
    """Source location with file tracking.

    Similar to VACASK Loc (sourceloc.h).
    """

    file_id: int = 0
    line: int = 0
    col: int = 0
    offset: int = 0  # Byte offset in file (for showing source lines)

    def to_string(self, file_stack: FileStack) -> str:
        """Format location with inclusion chain."""
        return file_stack.format_location(self.file_id, self.line, self.col)


# ============================================================================
# Control Section AST (mirrors VACASK parseroutput.h)
# ============================================================================


@dataclass
class OptionsDirective:
    """Options command in control block.

    Similar to VACASK PTCommand with name="options" (parseroutput.h:691-725).
    Maps to: options tran_method="trap" reltol=1e-3 abstol=1e-9
    """

    params: Dict[str, Any] = field(default_factory=dict)
    loc: Optional[SourceLocation] = None


@dataclass
class AnalysisDirective:
    """Analysis command in control block.

    Similar to VACASK PTAnalysis (parseroutput.h:617-663).
    Maps to: analysis tran1 tran step=0.05n stop=1u maxstep=0.05n icmode="op"
    """

    name: str  # "tran1"
    analysis_type: str  # "tran", "op", "ac", "hb"
    params: Dict[str, Any] = field(default_factory=dict)  # step, stop, maxstep, icmode
    loc: Optional[SourceLocation] = None


@dataclass
class SaveDirective:
    """Save command in control block.

    Similar to VACASK PTSave (parseroutput.h:521-547).
    Maps to: save v(out) i(r1)
    """

    signals: List[str] = field(default_factory=list)
    loc: Optional[SourceLocation] = None


@dataclass
class VarDirective:
    """Var command in control block.

    Maps to: var TEMP=27 VDD=1.8
    """

    vars: Dict[str, Any] = field(default_factory=dict)
    loc: Optional[SourceLocation] = None


@dataclass
class PrintDirective:
    """Print command in control block.

    Maps to: print devices, print model "psp103n", print stats
    """

    subcommand: str  # "devices", "models", "stats", "instance", "model"
    args: List[str] = field(default_factory=list)
    loc: Optional[SourceLocation] = None


@dataclass
class ControlBlock:
    """Parsed control block.

    Similar to VACASK PTControl (parseroutput.h:727-728).
    Contains all directives from control...endc block.
    """

    options: Optional[OptionsDirective] = None
    analyses: List[AnalysisDirective] = field(default_factory=list)
    saves: List[SaveDirective] = field(default_factory=list)
    vars: List[VarDirective] = field(default_factory=list)
    prints: List[PrintDirective] = field(default_factory=list)


# ============================================================================
# Circuit Data Structures
# ============================================================================


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
    """Top-level circuit containing all definitions.

    Similar to VACASK ParserTables (parseroutput.h:732-803).
    """

    title: Optional[str] = None
    loads: List[str] = field(default_factory=list)
    includes: List[str] = field(default_factory=list)
    globals: List[str] = field(default_factory=list)
    ground: Optional[str] = None
    models: Dict[str, Model] = field(default_factory=dict)
    subckts: Dict[str, Subcircuit] = field(default_factory=dict)
    params: Dict[str, str] = field(default_factory=dict)
    top_instances: List[Instance] = field(default_factory=list)
    # New fields for control section and source tracking
    control: Optional[ControlBlock] = None
    file_stack: FileStack = field(default_factory=FileStack)

    def flatten(self, top_subckt: str) -> Tuple[List[Instance], Dict[str, int]]:
        """Flatten hierarchy starting from given subcircuit

        Returns:
            Tuple of (flat instance list, node name to index mapping)
        """
        flat_instances = []
        nodes = {self.ground or "0": 0}  # Ground is node 0

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
                    elif t in self.globals or t == (self.ground or "0"):
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
                        params={**inst.params},  # Copy params
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
                if t in self.globals or t == (self.ground or "0"):
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
                    params={**inst.params},
                )
                flat_instances.append(flat_inst)

        return flat_instances, nodes

    def stats(self) -> Dict[str, int]:
        """Return statistics about the circuit"""
        return {
            "num_subckts": len(self.subckts),
            "num_models": len(self.models),
            "num_top_instances": len(self.top_instances),
            "num_globals": len(self.globals),
            "num_loads": len(self.loads),
        }
