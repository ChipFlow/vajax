"""Xyce test discovery and registry.

Discovers test circuits from vendor/Xyce_Regression/Netlists/ and categorizes them
by device type and analysis compatibility with JAX-SPICE.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set


# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
XYCE_REGRESSION = PROJECT_ROOT / "vendor" / "Xyce_Regression"
XYCE_NETLISTS = XYCE_REGRESSION / "Netlists"
XYCE_OUTPUT = XYCE_REGRESSION / "OutputData"


@dataclass
class XyceTestCase:
    """Specification for a Xyce regression test."""

    name: str
    category: str
    netlist_path: Path
    output_path: Optional[Path]  # Expected .prn output
    analysis_type: str  # 'tran', 'dc', 'ac', 'op'
    device_types: Set[str] = field(default_factory=set)
    expected_nodes: List[str] = field(default_factory=list)
    rtol: float = 0.05  # Relative tolerance (5%)
    atol: float = 1e-9  # Absolute tolerance


def discover_xyce_tests() -> List[XyceTestCase]:
    """Discover test cases from vendor/Xyce_Regression/Netlists/.

    Returns:
        List of discovered test cases
    """
    tests = []

    if not XYCE_NETLISTS.exists():
        return tests

    # Scan each category directory
    for category_dir in sorted(XYCE_NETLISTS.iterdir()):
        if not category_dir.is_dir():
            continue

        category = category_dir.name

        # Find .cir files in this category
        for netlist in category_dir.glob("*.cir"):
            test_case = analyze_xyce_netlist(netlist, category)
            if test_case:
                tests.append(test_case)

    return tests


def analyze_xyce_netlist(netlist_path: Path, category: str) -> Optional[XyceTestCase]:
    """Analyze a Xyce netlist file to extract test parameters.

    Args:
        netlist_path: Path to Xyce netlist
        category: Category name (directory)

    Returns:
        XyceTestCase or None if not a valid test
    """
    try:
        content = netlist_path.read_text()
    except Exception:
        return None

    content_lower = content.lower()

    # Skip files without analysis commands
    if not any(x in content_lower for x in ['.tran', '.dc', '.ac', '.op']):
        return None

    # Determine analysis type (priority: tran > dc > ac > op)
    analysis_type = 'op'
    if '.tran' in content_lower:
        analysis_type = 'tran'
    elif '.ac' in content_lower:
        analysis_type = 'ac'
    elif '.dc' in content_lower:
        analysis_type = 'dc'

    # Detect device types
    device_types = _detect_device_types(content)

    # Extract expected output nodes from .print commands
    expected_nodes = _extract_output_nodes(content)

    # Check for expected output file
    output_path = XYCE_OUTPUT / category / f"{netlist_path.name}.prn"
    if not output_path.exists():
        output_path = None

    # Generate test name
    name = f"{category}_{netlist_path.stem}"

    return XyceTestCase(
        name=name,
        category=category,
        netlist_path=netlist_path,
        output_path=output_path,
        analysis_type=analysis_type,
        device_types=device_types,
        expected_nodes=expected_nodes or ['1'],
    )


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
        'r': 'resistor',
        'c': 'capacitor',
        'l': 'inductor',
        'd': 'diode',
        'm': 'mosfet',
        'q': 'bjt',
        'j': 'jfet',
        'v': 'vsource',
        'i': 'isource',
        'e': 'vcvs',
        'f': 'cccs',
        'g': 'vccs',
        'h': 'ccvs',
        'x': 'subckt',
        'b': 'bsource',
        't': 'tline',
        'y': 'pde',  # Xyce-specific PDE device
        'p': 'digital',  # Xyce digital device
        'u': 'mutual_inductor',
        'k': 'coupling',
    }

    for line in content.split('\n'):
        line = line.strip().lower()
        # Skip comments and directives
        if line.startswith('*') or line.startswith('.') or not line:
            continue

        first_char = line[0] if line else ''
        if first_char in device_patterns:
            device_types.add(device_patterns[first_char])

    return device_types


def _extract_output_nodes(content: str) -> List[str]:
    """Extract output node names from .print commands.

    Args:
        content: Netlist content

    Returns:
        List of node names
    """
    nodes = []

    # Match v(node) or i(source) patterns in .print
    for match in re.finditer(
        r'\.print\s+(?:tran|ac|dc)?\s+.*?v\((\w+)\)',
        content,
        re.IGNORECASE
    ):
        node = match.group(1)
        if node not in nodes:
            nodes.append(node)

    return nodes


# Devices supported by JAX-SPICE (via OpenVAF or built-in)
# Note: Missing devices can be created with VADistiller
SUPPORTED_DEVICES = {
    'resistor',
    'capacitor',
    'inductor',
    'diode',
    'vsource',
    'isource',
    'mosfet',  # Via PSP103 or other VA models
    'subckt',  # Subcircuit instantiation
}

# Devices that can be added via VADistiller
VADISTILLER_DEVICES = {
    'bjt',      # BJT transistor
    'jfet',     # JFET transistor
    'vcvs',     # Voltage-controlled voltage source
    'cccs',     # Current-controlled current source
    'vccs',     # Voltage-controlled current source
    'ccvs',     # Current-controlled voltage source
    'bsource',  # Behavioral source
}


def get_compatible_tests(
    analysis_types: Optional[List[str]] = None,
    device_types: Optional[Set[str]] = None,
    include_vadistiller: bool = False,
) -> List[XyceTestCase]:
    """Get tests compatible with JAX-SPICE capabilities.

    Args:
        analysis_types: Filter by analysis type (default: ['tran'])
        device_types: Only include tests using these devices
            (default: SUPPORTED_DEVICES)
        include_vadistiller: If True, also include tests needing VADistiller devices

    Returns:
        List of compatible test cases
    """
    if analysis_types is None:
        analysis_types = ['tran']

    if device_types is None:
        device_types = SUPPORTED_DEVICES.copy()
        if include_vadistiller:
            device_types.update(VADISTILLER_DEVICES)

    tests = discover_xyce_tests()
    compatible = []

    for test in tests:
        # Filter by analysis type
        if test.analysis_type not in analysis_types:
            continue

        # Filter by device types
        if not test.device_types.issubset(device_types):
            continue

        # Only include tests with expected output
        if test.output_path is None:
            continue

        compatible.append(test)

    return compatible


def get_tests_by_category() -> Dict[str, List[XyceTestCase]]:
    """Get all tests organized by category.

    Returns:
        Dict mapping category name to list of tests
    """
    tests = discover_xyce_tests()
    by_category: Dict[str, List[XyceTestCase]] = {}

    for test in tests:
        if test.category not in by_category:
            by_category[test.category] = []
        by_category[test.category].append(test)

    return by_category


def get_missing_devices() -> Dict[str, int]:
    """Get count of tests blocked by each missing device type.

    Returns:
        Dict mapping device type to count of blocked tests
    """
    tests = discover_xyce_tests()
    all_devices = SUPPORTED_DEVICES | VADISTILLER_DEVICES

    missing = {}
    for test in tests:
        unsupported = test.device_types - all_devices
        for dev in unsupported:
            missing[dev] = missing.get(dev, 0) + 1

    return dict(sorted(missing.items(), key=lambda x: -x[1]))
