# SPDX-FileCopyrightText: 2025 ChipFlow
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Ngspice dialect implementation.

This is the reference implementation, based on the existing ng2vclib logic.
Ngspice is largely compatible with Berkeley SPICE3 with extensions.
"""

import re
from typing import Any

from ..dialect import SpiceDialect, register_dialect

# SI prefix patterns for Ngspice
# Ngspice uses: meg (M), g (G), t (T), mil (25.4e-6)
# Plus standard: f, p, n, u, m, k
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


@register_dialect("ngspice")
class NgspiceDialect(SpiceDialect):
    """Ngspice-specific parsing rules.

    Ngspice is the open-source continuation of Berkeley SPICE3. It has
    several extensions including:
    - OSDI device interface for Verilog-A models
    - Extended .lib syntax
    - Case-insensitive by default
    """

    @property
    def name(self) -> str:
        return "ngspice"

    # -------------------------------------------------------------------------
    # Include/Library handling
    # -------------------------------------------------------------------------

    def parse_include(self, line: str) -> tuple[str, str | None] | None:
        """Parse Ngspice .include directive.

        Ngspice syntax:
            .include "filename"
            .include 'filename'
            .include filename
        """
        lower = line.lower()
        if not lower.startswith(".include"):
            return None

        rest = line[8:].strip()

        # Remove quotes if present
        if rest.startswith('"') and rest.endswith('"'):
            filepath = rest[1:-1]
        elif rest.startswith("'") and rest.endswith("'"):
            filepath = rest[1:-1]
        else:
            filepath = rest

        return (filepath, None)

    def parse_library(self, line: str) -> tuple[str, str] | None:
        """Parse Ngspice .lib directive with section.

        Ngspice syntax:
            .lib "filename" section
            .lib 'filename' section
            .lib filename section
        """
        lower = line.lower()
        if not lower.startswith(".lib"):
            return None

        rest = line[4:].strip()

        # Check if this is a section definition (.lib name without file)
        parts = rest.split()
        if len(parts) == 1:
            # This is a section start, not a file include
            return None

        # Extract filepath and section
        if rest.startswith('"'):
            end_quote = rest.find('"', 1)
            if end_quote == -1:
                return None
            filepath = rest[1:end_quote]
            section = rest[end_quote + 1 :].strip()
        elif rest.startswith("'"):
            end_quote = rest.find("'", 1)
            if end_quote == -1:
                return None
            filepath = rest[1:end_quote]
            section = rest[end_quote + 1 :].strip()
        else:
            # Space-separated
            parts = rest.split(None, 1)
            if len(parts) != 2:
                return None
            filepath, section = parts

        return (filepath, section)

    # -------------------------------------------------------------------------
    # Model parsing
    # -------------------------------------------------------------------------

    def parse_model_type(self, type_str: str) -> str:
        """Normalize Ngspice model type.

        Ngspice model types (case-insensitive):
            r, res - Resistor
            c - Capacitor
            l - Inductor
            d - Diode
            npn, pnp - BJT
            nmos, pmos - MOSFET
            njf, pjf - JFET
            nmf, pmf - MESFET
            etc.
        """
        return type_str.lower()

    def get_device_prefix_map(self) -> dict[str, str]:
        """Return Ngspice device prefix mapping.

        Based on ng2vclib/dfl.py type_map.
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
            "U": "uniformrc",
            "O": "lossy_tline",
            "K": "coupling",
            "N": "osdi",  # OSDI device
        }

    # -------------------------------------------------------------------------
    # Parameter parsing
    # -------------------------------------------------------------------------

    def parse_parameter_value(self, value_str: str) -> Any:
        """Parse Ngspice parameter value.

        Handles:
            - Numeric values with SI prefixes (1k, 10meg, 5u)
            - Curly brace expressions {expr}
            - Quoted strings
            - Simple numeric values
        """
        value_str = value_str.strip()

        # Handle curly brace expressions
        if value_str.startswith("{") and value_str.endswith("}"):
            return value_str[1:-1]  # Return expression without braces

        # Handle quoted strings
        if (value_str.startswith('"') and value_str.endswith('"')) or (
            value_str.startswith("'") and value_str.endswith("'")
        ):
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
    # Ngspice-specific
    # -------------------------------------------------------------------------

    def supports_conditional(self) -> bool:
        """Ngspice does not support .if/.else/.endif."""
        return False

    def get_extra_element_params(self, element_type: str) -> list[str]:
        """Ngspice standard elements don't have extra params like LTSpice."""
        return []


# Convenience type map from dfl.py for reference
# This maps model type string to (extra_params, family, remove_level, remove_version)
NGSPICE_TYPE_MAP = {
    "r": ({}, "r", False, False),
    "res": ({}, "r", False, False),
    "c": ({}, "c", False, False),
    "l": ({}, "l", False, False),
    "d": ({}, "d", False, False),
    "npn": ({"type": "1"}, "bjt", True, False),
    "pnp": ({"type": "-1"}, "bjt", True, False),
    "njf": ({"type": "1"}, "jfet", True, False),
    "pjf": ({"type": "-1"}, "jfet", True, False),
    "nmf": ({"type": "1"}, "mes", True, False),
    "pmf": ({"type": "-1"}, "mes", True, False),
    "nhfet": ({"type": "1"}, "hemt", True, False),
    "phfet": ({"type": "-1"}, "hemt", True, False),
    "nmos": ({"type": "1"}, "mos", True, True),
    "pmos": ({"type": "-1"}, "mos", True, True),
    "nsoi": ({"type": "1"}, "soi", True, True),
    "psoi": ({"type": "-1"}, "soi", True, True),
}
