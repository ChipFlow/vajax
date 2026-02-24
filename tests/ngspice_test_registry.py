"""ngspice test discovery and registry.

Discovers test circuits from vendor/ngspice/tests/ and categorizes them
by device type and analysis compatibility with VA-JAX.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
NGSPICE_ROOT = PROJECT_ROOT / "vendor" / "ngspice"
NGSPICE_TESTS = NGSPICE_ROOT / "tests"


@dataclass
class NgspiceTestCase:
    """Specification for an ngspice regression test."""

    name: str
    netlist_path: Path
    analysis_type: str  # 'tran', 'dc', 'ac', 'op'
    device_types: Set[str] = field(default_factory=set)
    expected_nodes: List[str] = field(default_factory=list)
    reference_path: Optional[Path] = None  # .out or .standard reference file
    rtol: float = 0.05  # Relative tolerance (5%)
    atol: float = 1e-9  # Absolute tolerance
    xfail: bool = False  # Expected to fail
    xfail_reason: str = ""
    skip: bool = False  # Skip this test
    skip_reason: str = ""


def discover_ngspice_tests() -> List[NgspiceTestCase]:
    """Discover test cases from vendor/ngspice/tests/.

    Scans for .sp, .cir, .spice files and analyzes them.

    Returns:
        List of discovered test cases
    """
    tests = []

    if not NGSPICE_TESTS.exists():
        return tests

    # Scan for test files
    for pattern in ["**/*.sp", "**/*.cir", "**/*.spice"]:
        for netlist in NGSPICE_TESTS.glob(pattern):
            test_case = analyze_netlist(netlist)
            if test_case:
                tests.append(test_case)

    return tests


def analyze_netlist(netlist_path: Path) -> Optional[NgspiceTestCase]:
    """Analyze a netlist file to extract test parameters.

    Args:
        netlist_path: Path to ngspice netlist

    Returns:
        NgspiceTestCase or None if not a valid test
    """
    try:
        content = netlist_path.read_text()
    except Exception:
        return None

    content_lower = content.lower()

    # Skip files without analysis commands
    if not any(x in content_lower for x in [".tran", ".dc", ".ac", ".op"]):
        return None

    # Determine analysis type (priority: tran > dc > ac > op)
    analysis_type = "op"
    if ".tran" in content_lower:
        analysis_type = "tran"
    elif ".ac" in content_lower:
        analysis_type = "ac"
    elif ".dc" in content_lower:
        analysis_type = "dc"

    # Detect device types
    device_types = _detect_device_types(content)

    # Extract expected output nodes from .print or .plot commands
    expected_nodes = _extract_output_nodes(content)

    # Find reference file (.out in same dir, or .standard in reference/ subdir)
    reference_path = _find_reference_file(netlist_path)

    # Generate test name from path
    rel_path = netlist_path.relative_to(NGSPICE_TESTS)
    name = str(rel_path).replace("/", "_").replace("\\", "_").replace(".", "_")

    return NgspiceTestCase(
        name=name,
        netlist_path=netlist_path,
        analysis_type=analysis_type,
        device_types=device_types,
        expected_nodes=expected_nodes or ["1"],  # Default to node 1
        reference_path=reference_path,
    )


def _find_reference_file(netlist_path: Path) -> Optional[Path]:
    """Find reference output file for a netlist.

    Checks for:
    1. .out file in same directory (e.g., res_simple.out for res_simple.cir)
    2. .standard file in reference/ subdirectory

    Args:
        netlist_path: Path to netlist file

    Returns:
        Path to reference file or None if not found
    """
    # Check for .out file in same directory
    out_path = netlist_path.with_suffix(".out")
    if out_path.exists():
        return out_path

    # Check for .standard file in reference/ subdirectory
    test_dir = netlist_path.parent
    ref_dir = test_dir / "reference"
    if ref_dir.exists():
        # Look for matching .standard file
        stem = netlist_path.stem
        for std_file in ref_dir.glob(f"{stem}*.standard"):
            return std_file

    return None


def _detect_device_types(content: str) -> Set[str]:
    """Detect device types used in the netlist.

    Args:
        content: Netlist content

    Returns:
        Set of device type names
    """
    device_types: Set[str] = set()

    # Device patterns (first character of instance name)
    device_patterns = {
        "r": "resistor",
        "c": "capacitor",
        "l": "inductor",
        "d": "diode",
        "m": "mosfet",
        "q": "bjt",
        "j": "jfet",
        "v": "vsource",
        "i": "isource",
        "e": "vcvs",
        "f": "cccs",
        "g": "vccs",
        "h": "ccvs",
        "x": "subckt",
        "b": "bsource",
        "t": "tline",
    }

    for line in content.split("\n"):
        line = line.strip().lower()
        # Skip comments and directives
        if line.startswith("*") or line.startswith(".") or not line:
            continue

        first_char = line[0] if line else ""
        if first_char in device_patterns:
            device_types.add(device_patterns[first_char])

    return device_types


def _extract_output_signals(content: str) -> List[str]:
    """Extract output signal names from .print/.plot commands.

    Args:
        content: Netlist content

    Returns:
        List of signal names (e.g., 'v(1)', 'i(v1)', 'v1#branch')
    """
    signals = []

    # Match v(node) patterns
    for match in re.finditer(
        r"(?:\.print|\.plot)\s+(?:tran|ac|dc)?\s+.*?(v\(\w+\))", content, re.IGNORECASE
    ):
        sig = match.group(1).lower()
        if sig not in signals:
            signals.append(sig)

    # Match i(source) patterns - these map to source#branch in ngspice output
    for match in re.finditer(
        r"(?:\.print|\.plot)\s+(?:tran|ac|dc)?\s+.*?i\((\w+)\)", content, re.IGNORECASE
    ):
        source = match.group(1).lower()
        # ngspice outputs current as source#branch
        sig = f"{source}#branch"
        if sig not in signals:
            signals.append(sig)

    return signals


def _extract_output_nodes(content: str) -> List[str]:
    """Extract output node names from .print/.plot commands.

    Args:
        content: Netlist content

    Returns:
        List of signal names for comparison
    """
    return _extract_output_signals(content)


# Devices supported by VA-JAX (via OpenVAF or built-in)
SUPPORTED_DEVICES = {
    "resistor",
    "capacitor",
    "inductor",
    "diode",
    "vsource",
    "isource",
    "mosfet",  # Via PSP103 or other VA models
    "subckt",  # Subcircuit instantiation
}


def get_compatible_tests(
    analysis_types: Optional[List[str]] = None,
    device_types: Optional[Set[str]] = None,
    include_unsupported: bool = False,
) -> List[NgspiceTestCase]:
    """Get tests compatible with VA-JAX capabilities.

    Args:
        analysis_types: Filter by analysis type (default: ['tran'])
        device_types: Only include tests using these devices
            (default: SUPPORTED_DEVICES)
        include_unsupported: If True, include tests with unsupported devices

    Returns:
        List of compatible test cases
    """
    if analysis_types is None:
        analysis_types = ["tran"]  # Start with transient only

    if device_types is None:
        device_types = SUPPORTED_DEVICES

    tests = discover_ngspice_tests()
    compatible = []

    for test in tests:
        # Filter by analysis type
        if test.analysis_type not in analysis_types:
            continue

        # Filter by device types
        if not include_unsupported:
            if not test.device_types.issubset(device_types):
                continue

        compatible.append(test)

    return compatible


def get_tests_by_category() -> Dict[str, List[NgspiceTestCase]]:
    """Get all tests organized by their parent directory.

    Returns:
        Dict mapping category name to list of tests
    """
    tests = discover_ngspice_tests()
    by_category: Dict[str, List[NgspiceTestCase]] = {}

    for test in tests:
        # Use parent directory as category
        rel_path = test.netlist_path.relative_to(NGSPICE_TESTS)
        category = rel_path.parts[0] if rel_path.parts else "unknown"

        if category not in by_category:
            by_category[category] = []
        by_category[category].append(test)

    return by_category
