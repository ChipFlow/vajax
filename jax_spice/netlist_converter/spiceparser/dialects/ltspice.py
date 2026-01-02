# SPDX-FileCopyrightText: 2025 ChipFlow
#
# SPDX-License-Identifier: AGPL-3.0-or-later

r"""LTSpice dialect implementation.

Analog Devices LTSpice has several differences from standard Ngspice:
- Extra parameters on passives: Rser, Lser, Rpar, Cpar
- Windows-style paths: C:\path\to\file
- LTC.lib shorthand for LTSpice component library
- .backanno directive for annotation
- Unicode characters in values (µ for micro)
- Values often without parameter names (R1 n1 n2 10k not R1 n1 n2 R=10k)
"""

import re
from typing import Any

from ..dialect import SpiceDialect, register_dialect

# SI prefix pattern - includes 'meg' for mega and µ for micro
_SI_PREFIX_PATTERN = re.compile(
    r"(\d+\.?\d*|\.\d+)\s*(t|g|meg|k|m|u|µ|n|p|f|mil)",
    re.IGNORECASE,
)

_SI_MULTIPLIERS = {
    "t": 1e12,
    "g": 1e9,
    "meg": 1e6,
    "k": 1e3,
    "m": 1e-3,
    "u": 1e-6,
    "µ": 1e-6,  # Unicode micro symbol
    "n": 1e-9,
    "p": 1e-12,
    "f": 1e-15,
    "mil": 25.4e-6,
}

# LTSpice-specific extra element parameters
_LTSPICE_EXTRA_PARAMS = {
    "c": ["Rser", "Lser", "Rpar", "Cpar"],  # Capacitor extras
    "l": ["Rser", "Rpar", "Cpar"],  # Inductor extras
    "v": ["Rser"],  # Voltage source extras
    "i": ["Rser"],  # Current source extras
}


@register_dialect("ltspice")
class LtspiceDialect(SpiceDialect):
    """LTSpice-specific parsing rules.

    LTSpice is Analog Devices' free SPICE simulator (formerly Linear Technology).
    It has several extensions for modeling real-world component behavior.

    Key differences from Ngspice:
    - Rser/Lser/Rpar/Cpar parameters on passive elements
    - Windows-style paths with backslashes
    - LTC.lib shorthand for component library
    - .backanno directive
    """

    @property
    def name(self) -> str:
        return "ltspice"

    # -------------------------------------------------------------------------
    # Include/Library handling
    # -------------------------------------------------------------------------

    def parse_include(self, line: str) -> tuple[str, str | None] | None:
        r"""Parse LTSpice .include directive.

        LTSpice syntax:
            .include file.lib
            .include C:\path\to\file.lib
        """
        lower = line.lower()

        if lower.startswith(".include "):
            rest = line[9:].strip()
        elif lower.startswith(".inc "):
            rest = line[5:].strip()
        else:
            return None

        # Remove quotes if present
        filepath = self._extract_path(rest)
        if filepath:
            return (self._normalize_path(filepath), None)
        return None

    def parse_library(self, line: str) -> tuple[str, str] | None:
        r"""Parse LTSpice .lib directive.

        LTSpice syntax:
            .lib file.lib
            .lib LTC.lib
            .lib C:\path\to\file.lib section
        """
        lower = line.lower()
        if not lower.startswith(".lib"):
            return None

        rest = line[4:].strip()

        # Skip empty directives
        if not rest:
            return None

        # First, try to extract quoted path (handles spaces in paths)
        filepath = None
        section = ""

        if rest.startswith('"') and '"' in rest[1:]:
            # Double-quoted path
            end = rest.find('"', 1)
            filepath = rest[1:end]
            # Check for section after the quoted path
            after = rest[end + 1 :].strip()
            if after:
                section = after.split()[0]
        elif rest.startswith("'") and "'" in rest[1:]:
            # Single-quoted path
            end = rest.find("'", 1)
            filepath = rest[1:end]
            # Check for section after the quoted path
            after = rest[end + 1 :].strip()
            if after:
                section = after.split()[0]
        else:
            # Unquoted - split on spaces
            tokens = rest.split()
            if not tokens:
                return None
            filepath = tokens[0]
            if len(tokens) >= 2:
                section = tokens[1]

        if not filepath:
            return None

        filepath = self._normalize_path(filepath)

        # If it looks like a path, treat as library file include
        if "." in filepath or "/" in filepath or "\\" in filepath:
            return (filepath, section)

        # Check if there's a section
        if section:
            return (filepath, section)

        # Otherwise it's a section start (single token without extension)
        return None

    def _extract_path(self, text: str) -> str | None:
        """Extract a file path from text, handling quotes."""
        text = text.strip()

        # Handle quoted paths
        if text.startswith('"') and '"' in text[1:]:
            end = text.find('"', 1)
            return text[1:end]
        if text.startswith("'") and "'" in text[1:]:
            end = text.find("'", 1)
            return text[1:end]

        # Unquoted - return first token
        parts = text.split()
        if parts:
            return parts[0]
        return None

    def _normalize_path(self, path: str) -> str:
        """Normalize a file path.

        LTSpice uses Windows-style paths. Convert to forward slashes
        for cross-platform compatibility.
        """
        # Replace backslashes with forward slashes
        return path.replace("\\", "/")

    # -------------------------------------------------------------------------
    # Model parsing
    # -------------------------------------------------------------------------

    def parse_model_type(self, type_str: str) -> str:
        """Normalize LTSpice model type."""
        return type_str.lower()

    def get_device_prefix_map(self) -> dict[str, str]:
        """Return LTSpice device prefix mapping.

        Same as Ngspice - LTSpice uses standard SPICE prefixes.
        """
        return {
            "R": "resistor",
            "C": "capacitor",
            "L": "inductor",
            "D": "diode",
            "Q": "bjt",
            "M": "mosfet",
            "J": "jfet",
            "Z": "mesfet",
            "B": "behavioral",
            "E": "vcvs",
            "F": "cccs",
            "G": "vccs",
            "H": "ccvs",
            "I": "isource",
            "V": "vsource",
            "X": "subcircuit",
            "S": "switch",
            "W": "cswitch",
            "T": "tline",
        }

    # -------------------------------------------------------------------------
    # Parameter parsing
    # -------------------------------------------------------------------------

    def parse_parameter_value(self, value_str: str) -> Any:
        """Parse LTSpice parameter value.

        Handles:
            - Numeric values with SI prefixes (including µ)
            - Plain numeric values
            - Expression strings
        """
        value_str = value_str.strip()

        # Handle braces around expressions (LTSpice uses {expr} sometimes)
        if value_str.startswith("{") and value_str.endswith("}"):
            return value_str[1:-1]

        # Try to parse as numeric with SI prefix
        match = _SI_PREFIX_PATTERN.fullmatch(value_str)
        if match:
            number = float(match.group(1))
            prefix = match.group(2).lower()
            # Handle µ separately since .lower() doesn't change it
            if prefix == "µ":
                prefix = "u"
            multiplier = _SI_MULTIPLIERS.get(prefix, 1.0)
            return number * multiplier

        # Try to parse as plain numeric
        try:
            if "." in value_str or "e" in value_str.lower():
                return float(value_str)
            return int(value_str)
        except ValueError:
            # Return as string expression
            return value_str

    # -------------------------------------------------------------------------
    # LTSpice-specific: Extra element parameters
    # -------------------------------------------------------------------------

    def get_extra_element_params(self, element_type: str) -> list[str]:
        """Return LTSpice-specific parameters for element types.

        LTSpice adds parasitic elements to passives:
        - Capacitors: Rser, Lser, Rpar, Cpar
        - Inductors: Rser, Rpar, Cpar
        - Voltage sources: Rser
        """
        return _LTSPICE_EXTRA_PARAMS.get(element_type.lower(), [])

    # -------------------------------------------------------------------------
    # LTSpice-specific directives
    # -------------------------------------------------------------------------

    def is_backanno_directive(self, line: str) -> bool:
        """Check if line is a .backanno directive."""
        return line.lower().strip() == ".backanno"

    def supports_conditional(self) -> bool:
        """LTSpice does not support .if/.else/.endif conditionals."""
        return False

    def parse_conditional(self, line: str) -> tuple[str, str | None] | None:
        """LTSpice doesn't support conditionals."""
        return None
