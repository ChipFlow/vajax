# SPDX-FileCopyrightText: 2025 ChipFlow
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""HSPICE dialect implementation.

Synopsys HSPICE has several extensions beyond standard SPICE3:
- Conditional blocks: .if/.else/.elseif/.endif
- Expression parameters in single quotes: param='1+x'
- Library sections: .lib 'file' section
- .ALTER for parameter sweeps
- .DATA for tabular data
- Spaces around = in parameter assignments
"""

import re
from typing import Any

from ..dialect import SpiceDialect, register_dialect

# HSPICE uses single quotes for expressions
_QUOTED_EXPR_PATTERN = re.compile(r"'([^']*)'")

# SI prefix pattern (same as Ngspice but HSPICE is case-insensitive)
_SI_PREFIX_PATTERN = re.compile(
    r"(\d+\.?\d*|\.\d+)\s*(t|g|meg|k|m|u|n|p|f|mil)",
    re.IGNORECASE,
)

_SI_MULTIPLIERS = {
    "t": 1e12,
    "g": 1e9,
    "meg": 1e6,
    "k": 1e3,
    "m": 1e-3,
    "u": 1e-6,
    "n": 1e-9,
    "p": 1e-12,
    "f": 1e-15,
    "mil": 25.4e-6,
}

# Conditional directive pattern
_CONDITIONAL_PATTERN = re.compile(
    r"\.(if|elseif|else|endif)\s*(?:\(([^)]*)\))?",
    re.IGNORECASE,
)


@register_dialect("hspice")
class HspiceDialect(SpiceDialect):
    """HSPICE-specific parsing rules.

    HSPICE is Synopsys' commercial SPICE simulator with many extensions.
    It's commonly used in industry PDKs (e.g., GlobalFoundries, TSMC).

    Key differences from Ngspice:
    - .if/.else/.endif conditional blocks
    - Single-quoted expressions: toxe='2.8e-9+dtoxe'
    - Spaces around = in parameters: level = 54
    - .lib 'file' section syntax
    """

    @property
    def name(self) -> str:
        return "hspice"

    # -------------------------------------------------------------------------
    # Include/Library handling
    # -------------------------------------------------------------------------

    def parse_include(self, line: str) -> tuple[str, str | None] | None:
        """Parse HSPICE .inc directive.

        HSPICE syntax:
            .inc 'filename'
            .inc "filename"
            .include 'filename'
        """
        lower = line.lower()

        # Handle both .inc and .include
        if lower.startswith(".inc "):
            rest = line[4:].strip()
        elif lower.startswith(".include"):
            rest = line[8:].strip()
        else:
            return None

        # Extract quoted path
        filepath = self._extract_quoted_string(rest)
        if filepath is None:
            # Try unquoted
            filepath = rest.split()[0] if rest else None

        if filepath:
            return (filepath, None)
        return None

    def parse_library(self, line: str) -> tuple[str, str] | None:
        """Parse HSPICE .lib directive with section.

        HSPICE syntax:
            .lib 'filename' section
            .lib "filename" section
        """
        lower = line.lower()
        if not lower.startswith(".lib"):
            return None

        rest = line[4:].strip()

        # Check if this is a section definition (no quotes)
        if not rest.startswith("'") and not rest.startswith('"'):
            # Could be .lib section_name (section start)
            parts = rest.split()
            if len(parts) == 1:
                return None  # Section start, not file include
            # Could be unquoted: .lib file section
            if len(parts) >= 2:
                return (parts[0], parts[1])
            return None

        # Extract quoted filepath
        filepath = self._extract_quoted_string(rest)
        if filepath is None:
            return None

        # Find section after the quoted path
        # Skip past the closing quote
        quote_char = rest[0]
        end_quote = rest.find(quote_char, 1)
        if end_quote == -1:
            return None

        section = rest[end_quote + 1 :].strip()
        if section:
            return (filepath, section)

        return None

    def _extract_quoted_string(self, text: str) -> str | None:
        """Extract a quoted string from text."""
        text = text.strip()
        if text.startswith("'"):
            end = text.find("'", 1)
            if end != -1:
                return text[1:end]
        elif text.startswith('"'):
            end = text.find('"', 1)
            if end != -1:
                return text[1:end]
        return None

    # -------------------------------------------------------------------------
    # Model parsing
    # -------------------------------------------------------------------------

    def parse_model_type(self, type_str: str) -> str:
        """Normalize HSPICE model type."""
        return type_str.lower()

    def get_device_prefix_map(self) -> dict[str, str]:
        """Return HSPICE device prefix mapping.

        Same as Ngspice - HSPICE uses standard SPICE prefixes.
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
        """Parse HSPICE parameter value.

        Handles:
            - Single-quoted expressions: '1+x*3'
            - Double-quoted strings
            - Numeric values with SI prefixes
            - Plain numeric values
        """
        value_str = value_str.strip()

        # Handle single-quoted expressions (HSPICE-specific)
        if value_str.startswith("'") and value_str.endswith("'"):
            return value_str[1:-1]  # Return expression without quotes

        # Handle double-quoted strings
        if value_str.startswith('"') and value_str.endswith('"'):
            return value_str[1:-1]

        # Try to parse as numeric with SI prefix
        match = _SI_PREFIX_PATTERN.fullmatch(value_str)
        if match:
            number = float(match.group(1))
            prefix = match.group(2).lower()
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
    # HSPICE-specific: Conditionals
    # -------------------------------------------------------------------------

    def supports_conditional(self) -> bool:
        """HSPICE supports .if/.else/.endif conditionals."""
        return True

    def parse_conditional(self, line: str) -> tuple[str, str | None] | None:
        """Parse HSPICE conditional directives.

        HSPICE syntax:
            .if (condition)
            .elseif (condition)
            .else
            .endif

        Returns:
            Tuple of (directive_type, condition) or None
        """
        match = _CONDITIONAL_PATTERN.match(line)
        if match:
            directive = match.group(1).lower()
            condition = match.group(2)  # May be None for .else/.endif
            return (directive, condition)
        return None

    # -------------------------------------------------------------------------
    # HSPICE-specific extensions
    # -------------------------------------------------------------------------

    def get_extra_element_params(self, element_type: str) -> list[str]:
        """HSPICE standard elements don't have extra params like LTSpice."""
        return []

    def parse_param_line(self, line: str) -> dict[str, str]:
        """Parse HSPICE .param line.

        HSPICE .param can have multiple assignments:
            .param name1='expr1' name2='expr2'
            .param name = value

        Returns:
            Dict of parameter name -> value
        """
        if not line.lower().startswith(".param"):
            return {}

        rest = line[6:].strip()
        params = {}

        # Handle quoted expressions
        # Pattern: name='expr' or name="expr" or name=value
        # HSPICE allows spaces around =

        # First, normalize spaces around =
        rest = re.sub(r"\s*=\s*", "=", rest)

        # Find all name=value pairs
        # This is tricky because values can contain spaces if quoted
        i = 0
        while i < len(rest):
            # Skip whitespace
            while i < len(rest) and rest[i] in " \t":
                i += 1
            if i >= len(rest):
                break

            # Find parameter name (up to =)
            start = i
            while i < len(rest) and rest[i] != "=":
                i += 1
            if i >= len(rest):
                break

            name = rest[start:i].strip()
            i += 1  # Skip =

            # Find value
            if i < len(rest) and rest[i] in "'\"":
                # Quoted value
                quote = rest[i]
                i += 1
                value_start = i
                while i < len(rest) and rest[i] != quote:
                    i += 1
                value = rest[value_start:i]
                if i < len(rest):
                    i += 1  # Skip closing quote
            else:
                # Unquoted value - goes until whitespace
                value_start = i
                while i < len(rest) and rest[i] not in " \t":
                    i += 1
                value = rest[value_start:i]

            if name:
                params[name.lower()] = value

        return params
