"""Parser for OpenVAF OSDI snapshot files.

Parses .snap files from openvaf/test_data/osdi/ to extract expected OSDI descriptor metadata
for comparison with openvaf-py output.

Snapshot format example:
    param "$mfactor"
    units = "", desc = "Multiplier (Verilog-A $mfactor)", flags = ParameterFlags(PARA_KIND_INST)
    param "R"
    units = "Ohm", desc = "Ohmic resistance", flags = ParameterFlags(0x0)

    2 terminals
    node "A" units = "V", runits = "A"
    node "B" units = "V", runits = "A"
    jacobian (A, A) JacobianFlags(JACOBIAN_ENTRY_RESIST | JACOBIAN_ENTRY_REACT_CONST)
    jacobian (A, B) JacobianFlags(JACOBIAN_ENTRY_RESIST | JACOBIAN_ENTRY_REACT_CONST)
    collapsible (CI, C)
    collapsible (dT, gnd)
    noise "unnamed0" (A, CI)
    0 states
    has bound_step false
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


# OSDI flag constants (matching osdi_0_4.rs)
PARA_KIND_MODEL = 0 << 30
PARA_KIND_INST = 1 << 30
PARA_TY_REAL = 0
PARA_TY_INT = 1
PARA_TY_STR = 2

JACOBIAN_ENTRY_RESIST_CONST = 1
JACOBIAN_ENTRY_REACT_CONST = 2
JACOBIAN_ENTRY_RESIST = 4
JACOBIAN_ENTRY_REACT = 8


@dataclass
class ParamInfo:
    """Parameter metadata from snapshot."""

    name: str
    units: str = ""
    description: str = ""
    flags: int = 0

    @property
    def is_instance(self) -> bool:
        """Check if this is an instance parameter (vs model parameter)."""
        return (self.flags & PARA_KIND_INST) != 0

    @property
    def is_model(self) -> bool:
        """Check if this is a model parameter."""
        return not self.is_instance


@dataclass
class NodeInfo:
    """Node metadata from snapshot."""

    name: str
    units: str = "V"
    residual_units: str = "A"
    is_flow: bool = False  # True for flow nodes like flow(br_in)


@dataclass
class JacobianEntry:
    """Jacobian entry metadata from snapshot."""

    row: str  # Node name
    col: str  # Node name
    flags: int = 0

    @property
    def has_resist(self) -> bool:
        return (self.flags & JACOBIAN_ENTRY_RESIST) != 0

    @property
    def has_react(self) -> bool:
        return (self.flags & JACOBIAN_ENTRY_REACT) != 0

    @property
    def resist_const(self) -> bool:
        return (self.flags & JACOBIAN_ENTRY_RESIST_CONST) != 0

    @property
    def react_const(self) -> bool:
        return (self.flags & JACOBIAN_ENTRY_REACT_CONST) != 0


@dataclass
class CollapsiblePair:
    """Collapsible node pair from snapshot."""

    node1: str
    node2: str  # "gnd" for ground


@dataclass
class NoiseSource:
    """Noise source from snapshot."""

    name: str
    node1: str
    node2: str  # "gnd" for ground


@dataclass
class SnapshotData:
    """Parsed OSDI snapshot data."""

    params: list[ParamInfo] = field(default_factory=list)
    nodes: list[NodeInfo] = field(default_factory=list)
    jacobian: list[JacobianEntry] = field(default_factory=list)
    collapsible: list[CollapsiblePair] = field(default_factory=list)
    noise_sources: list[NoiseSource] = field(default_factory=list)
    num_terminals: int = 0
    num_states: int = 0
    has_bound_step: bool = False


def parse_flags(flags_str: str) -> int:
    """Parse ParameterFlags or JacobianFlags string to integer.

    Examples:
        "ParameterFlags(PARA_KIND_INST)" -> PARA_KIND_INST
        "ParameterFlags(0x0)" -> 0
        "JacobianFlags(JACOBIAN_ENTRY_RESIST | JACOBIAN_ENTRY_REACT_CONST)" -> 5
    """
    # Extract the content inside parentheses
    match = re.search(r"\(([^)]*)\)", flags_str)
    if not match:
        return 0

    content = match.group(1).strip()
    if not content or content == "0x0":
        return 0

    # Parse hex value
    if content.startswith("0x"):
        return int(content, 16)

    # Parse flag names
    flags = 0
    for flag in content.split("|"):
        flag = flag.strip()
        if flag == "PARA_KIND_INST":
            flags |= PARA_KIND_INST
        elif flag == "PARA_KIND_MODEL":
            flags |= PARA_KIND_MODEL
        elif flag == "JACOBIAN_ENTRY_RESIST":
            flags |= JACOBIAN_ENTRY_RESIST
        elif flag == "JACOBIAN_ENTRY_REACT":
            flags |= JACOBIAN_ENTRY_REACT
        elif flag == "JACOBIAN_ENTRY_RESIST_CONST":
            flags |= JACOBIAN_ENTRY_RESIST_CONST
        elif flag == "JACOBIAN_ENTRY_REACT_CONST":
            flags |= JACOBIAN_ENTRY_REACT_CONST

    return flags


def parse_snap_file(path: str | Path) -> SnapshotData:
    """Parse an OpenVAF .snap file into structured data.

    Args:
        path: Path to the .snap file

    Returns:
        SnapshotData containing parsed metadata
    """
    path = Path(path)
    result = SnapshotData()

    with open(path) as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1

        if not line:
            continue

        # Parse param (spans two lines)
        if line.startswith('param "'):
            # Extract param name
            name_match = re.search(r'param "([^"]*)"', line)
            if name_match:
                name = name_match.group(1)
                # Read next line for metadata
                if i < len(lines):
                    meta_line = lines[i].strip()
                    i += 1

                    # Parse: units = "...", desc = "...", flags = ...
                    units = ""
                    desc = ""
                    flags = 0

                    units_match = re.search(r'units = "([^"]*)"', meta_line)
                    if units_match:
                        units = units_match.group(1)

                    desc_match = re.search(r'desc = "([^"]*)"', meta_line)
                    if desc_match:
                        desc = desc_match.group(1)

                    flags_match = re.search(r"flags = (ParameterFlags\([^)]*\))", meta_line)
                    if flags_match:
                        flags = parse_flags(flags_match.group(1))

                    result.params.append(ParamInfo(name=name, units=units, description=desc, flags=flags))

        # Parse terminal count
        elif re.match(r"^\d+ terminals?$", line):
            result.num_terminals = int(line.split()[0])

        # Parse node (regular or flow)
        elif line.startswith('node "') or line.startswith("node(flow)"):
            # Regular: node "A" units = "V", runits = "A"
            # Flow: node(flow) "flow(br_in)" units = "A", runits = "A"
            is_flow = line.startswith("node(flow)")

            # Extract the quoted name
            name_match = re.search(r'"([^"]*)"', line)
            if name_match:
                name = name_match.group(1)
                # Flow nodes default to "A" units (current), regular to "V" (voltage)
                units = "A" if is_flow else "V"
                runits = "A"

                units_match = re.search(r'units = "([^"]*)"', line)
                if units_match:
                    units = units_match.group(1)

                runits_match = re.search(r'runits = "([^"]*)"', line)
                if runits_match:
                    runits = runits_match.group(1)

                result.nodes.append(NodeInfo(name=name, units=units, residual_units=runits, is_flow=is_flow))

        # Parse jacobian entry
        elif line.startswith("jacobian ("):
            # jacobian (A, A) JacobianFlags(...)
            # jacobian (Inp, flow(br_in)) JacobianFlags(...) - node names can have parens
            # Extract content between "jacobian (" and ") Jacobian" or end of node section
            content_match = re.search(r"jacobian \((.+?)\) Jacobian", line)
            if content_match:
                content = content_match.group(1)
                # Split by ", " to get row and col (handles names with parens)
                parts = content.split(", ")
                if len(parts) == 2:
                    row = parts[0].strip()
                    col = parts[1].strip()
                    flags = 0

                    flags_match = re.search(r"(JacobianFlags\([^)]*\))", line)
                    if flags_match:
                        flags = parse_flags(flags_match.group(1))

                    result.jacobian.append(JacobianEntry(row=row, col=col, flags=flags))

        # Parse collapsible
        elif line.startswith("collapsible ("):
            # collapsible (CI, C) or collapsible (dT, gnd)
            match = re.search(r"collapsible \((\w+), (\w+)\)", line)
            if match:
                result.collapsible.append(CollapsiblePair(node1=match.group(1), node2=match.group(2)))

        # Parse noise
        elif line.startswith('noise "'):
            # noise "unnamed0" (A, CI)
            match = re.search(r'noise "([^"]*)" \((\w+), (\w+)\)', line)
            if match:
                result.noise_sources.append(
                    NoiseSource(name=match.group(1), node1=match.group(2), node2=match.group(3))
                )

        # Parse states
        elif re.match(r"^\d+ states?$", line):
            result.num_states = int(line.split()[0])

        # Parse bound_step
        elif line.startswith("has bound_step"):
            result.has_bound_step = line.endswith("true")

    return result


def load_snap_file(model_name: str, snap_dir: str | Path | None = None) -> SnapshotData:
    """Load a snapshot file by model name.

    Args:
        model_name: Name of the model (e.g., "resistor", "diode")
        snap_dir: Directory containing .snap files. Defaults to OpenVAF test_data/osdi.

    Returns:
        Parsed SnapshotData
    """
    if snap_dir is None:
        # Default to OpenVAF test data directory
        snap_dir = Path(__file__).parent.parent / "vendor" / "OpenVAF" / "openvaf" / "test_data" / "osdi"

    snap_path = Path(snap_dir) / f"{model_name}.snap"
    return parse_snap_file(snap_path)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python snap_parser.py <model_name|snap_file>")
        sys.exit(1)

    arg = sys.argv[1]
    if arg.endswith(".snap"):
        data = parse_snap_file(arg)
    else:
        data = load_snap_file(arg)

    print(f"Parameters: {len(data.params)}")
    for p in data.params:
        flag_str = "PARA_KIND_INST" if p.is_instance else "0x0"
        print(f'  param "{p.name}" units="{p.units}" desc="{p.description}" flags={flag_str}')

    print(f"\nNodes: {len(data.nodes)} ({data.num_terminals} terminals)")
    for n in data.nodes:
        print(f'  node "{n.name}" units="{n.units}" runits="{n.residual_units}"')

    print(f"\nJacobian entries: {len(data.jacobian)}")
    for j in data.jacobian:
        flags = []
        if j.has_resist:
            flags.append("RESIST")
        if j.has_react:
            flags.append("REACT")
        if j.resist_const:
            flags.append("RESIST_CONST")
        if j.react_const:
            flags.append("REACT_CONST")
        print(f"  ({j.row}, {j.col}): {' | '.join(flags)}")

    print(f"\nCollapsible: {len(data.collapsible)}")
    for c in data.collapsible:
        print(f"  ({c.node1}, {c.node2})")

    print(f"\nNoise sources: {len(data.noise_sources)}")
    for n in data.noise_sources:
        print(f'  "{n.name}" ({n.node1}, {n.node2})')

    print(f"\nStates: {data.num_states}")
    print(f"Has bound_step: {data.has_bound_step}")
