# SPDX-FileCopyrightText: 2025 ChipFlow
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Spectre dialect implementation.

Cadence Spectre is the de-facto industry standard for analog/mixed-signal
simulation. It uses a distinct syntax from traditional SPICE.

Key differences from SPICE dialects:
- No dot prefix on keywords: subckt, ends, model, include
- Nodes enclosed in parentheses: inst (n1 n2) model param=val
- 'parameters' keyword instead of 'params:'
- library/endlibrary and section/endsection blocks
- // for inline comments (in addition to * for line comments)
- 'simulator lang=spectre' for dialect switching in mixed files

This module provides two dialects:
- SpectreDialect: Pure Spectre syntax
- SpectreSpiceDialect: Spectre-SPICE mode with 'simulator lang=' switching
"""

import re
from typing import TYPE_CHECKING, Any

from ..dialect import SpiceDialect, register_dialect

if TYPE_CHECKING:
    from ..netlist import StatisticsBlock, VariationSpec

# SI prefix patterns - same as Ngspice but Spectre is case-sensitive for some
# In Spectre: M = 1e6 (mega), m = 1e-3 (milli) - case matters!
_SI_PREFIX_PATTERN = re.compile(
    r"(\d+\.?\d*|\.\d+)\s*(T|G|M|K|k|m|u|n|p|f|a)",
)

_SI_MULTIPLIERS = {
    "T": 1e12,
    "G": 1e9,
    "M": 1e6,  # Spectre: M = mega (unlike SPICE where M = milli)
    "K": 1e3,
    "k": 1e3,
    "m": 1e-3,
    "u": 1e-6,
    "n": 1e-9,
    "p": 1e-12,
    "f": 1e-15,
    "a": 1e-18,
}

# Pattern for simulator lang= directive
_SIMULATOR_LANG_PATTERN = re.compile(r"simulator\s+lang\s*=\s*(\w+)", re.IGNORECASE)


@register_dialect("spectre")
class SpectreDialect(SpiceDialect):
    """Spectre-specific parsing rules.

    Cadence Spectre uses a distinct syntax from traditional SPICE:
    - Keywords without dot prefix (subckt, ends, model, include)
    - Nodes in parentheses: M1 (d g s b) nch w=1u
    - library/endlibrary and section/endsection blocks
    - // for inline comments
    """

    @property
    def name(self) -> str:
        return "spectre"

    @property
    def comment_chars(self) -> tuple[str, ...]:
        """Spectre supports * for line comments and // for inline."""
        return ("*", "//")

    @property
    def continuation_char(self) -> str:
        """Spectre uses backslash for line continuation."""
        return "\\"

    # -------------------------------------------------------------------------
    # Include/Library handling
    # -------------------------------------------------------------------------

    def parse_include(self, line: str) -> tuple[str, str | None] | None:
        """Parse Spectre include directive.

        Spectre syntax (no dot prefix):
            include "filename"
            include "filename" section=name
        """
        # Check for include keyword (with or without dot for compatibility)
        lower = line.lower().strip()
        if lower.startswith("include "):
            rest = line[8:].strip()
        elif lower.startswith(".include "):
            rest = line[9:].strip()
        else:
            return None

        # Extract filepath (quoted or unquoted)
        filepath, section = self._parse_include_path(rest)
        if filepath:
            return (filepath, section)
        return None

    def _parse_include_path(self, rest: str) -> tuple[str | None, str | None]:
        """Extract filepath and optional section from include directive."""
        filepath = None
        section = None

        # Handle quoted paths
        if rest.startswith('"'):
            end_quote = rest.find('"', 1)
            if end_quote != -1:
                filepath = rest[1:end_quote]
                remaining = rest[end_quote + 1 :].strip()
                # Check for section=name
                if remaining.lower().startswith("section="):
                    section = remaining[8:].strip().strip("\"'")
        elif rest.startswith("'"):
            end_quote = rest.find("'", 1)
            if end_quote != -1:
                filepath = rest[1:end_quote]
                remaining = rest[end_quote + 1 :].strip()
                if remaining.lower().startswith("section="):
                    section = remaining[8:].strip().strip("\"'")
        else:
            # Unquoted - take first token
            parts = rest.split()
            if parts:
                filepath = parts[0]
                # Check for section=name in remaining parts
                for part in parts[1:]:
                    if part.lower().startswith("section="):
                        section = part[8:].strip("\"'")
                        break

        return filepath, section

    def parse_library(self, line: str) -> tuple[str, str] | None:
        """Parse Spectre library directive.

        Spectre uses structural library/endlibrary blocks, not file references.
        This method handles .lib file section syntax for compatibility.

        Spectre structural syntax (returns None - handled by parser):
            library name
            endlibrary name

        SPICE-style file reference:
            .lib "filename" section
        """
        lower = line.lower().strip()

        # Handle SPICE-style .lib file section
        if lower.startswith(".lib "):
            rest = line[5:].strip()
            parts = rest.split()
            if len(parts) >= 2:
                # .lib "file" section or .lib file section
                filepath = parts[0].strip("\"'")
                section = parts[1].strip("\"'")
                # Only treat as file reference if it looks like a path
                if "." in filepath or "/" in filepath:
                    return (filepath, section)

        return None

    def parse_library_block(self, line: str) -> tuple[str, str] | None:
        """Parse Spectre library/endlibrary block start/end.

        Returns:
            Tuple of (directive_type, name) where directive_type is
            "library" or "endlibrary", or None if not a library block.
        """
        lower = line.lower().strip()

        if lower.startswith("library "):
            name = line[8:].strip()
            return ("library", name)
        elif lower.startswith("endlibrary"):
            name = line[10:].strip() if len(line) > 10 else ""
            return ("endlibrary", name)

        return None

    def parse_section_block(self, line: str) -> tuple[str, str] | None:
        """Parse Spectre section/endsection block start/end.

        Returns:
            Tuple of (directive_type, name) where directive_type is
            "section" or "endsection", or None if not a section block.
        """
        lower = line.lower().strip()

        if lower.startswith("section "):
            name = line[8:].strip()
            return ("section", name)
        elif lower.startswith("endsection"):
            name = line[10:].strip() if len(line) > 10 else ""
            return ("endsection", name)

        return None

    # -------------------------------------------------------------------------
    # Model parsing
    # -------------------------------------------------------------------------

    def parse_model_type(self, type_str: str) -> str:
        """Normalize Spectre model type.

        Spectre model types are case-sensitive but we normalize to lowercase
        for consistency with other dialects.
        """
        return type_str.lower()

    def get_device_prefix_map(self) -> dict[str, str]:
        """Return Spectre device prefix mapping.

        Spectre uses the same prefix conventions as SPICE.
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
            "N": "osdi",
        }

    # -------------------------------------------------------------------------
    # Parameter parsing
    # -------------------------------------------------------------------------

    def parse_parameter_value(self, value_str: str) -> Any:
        """Parse Spectre parameter value.

        Handles:
            - Numeric values with SI prefixes (case-sensitive: M=mega, m=milli)
            - Quoted strings
            - Expressions (returned as strings)
        """
        value_str = value_str.strip()

        # Handle quoted strings
        if (value_str.startswith('"') and value_str.endswith('"')) or (
            value_str.startswith("'") and value_str.endswith("'")
        ):
            return value_str[1:-1]

        # Try to parse as numeric with SI prefix (case-sensitive)
        match = _SI_PREFIX_PATTERN.fullmatch(value_str)
        if match:
            number = float(match.group(1))
            prefix = match.group(2)  # Keep case for Spectre
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
    # Spectre-specific extensions
    # -------------------------------------------------------------------------

    def supports_conditional(self) -> bool:
        """Spectre does not use .if/.else/.endif like HSPICE."""
        return False

    def get_extra_element_params(self, element_type: str) -> list[str]:
        """Spectre standard elements don't have extra params."""
        return []

    def is_spectre_keyword(self, line: str) -> str | None:
        """Check if line starts with a Spectre keyword (no dot prefix).

        Returns the keyword if found, None otherwise.
        """
        lower = line.lower().strip()
        keywords = [
            "subckt",
            "ends",
            "model",
            "include",
            "library",
            "endlibrary",
            "section",
            "endsection",
            "parameters",
            "simulator",
            "inline",
            "statistics",
            "global",
            "ahdl_include",
            "real",
        ]
        for kw in keywords:
            if lower.startswith(kw + " ") or lower == kw:
                return kw
        return None

    def parse_simulator_lang(self, line: str) -> str | None:
        """Parse simulator lang= directive.

        Returns the language name (spectre, spice) or None.
        """
        match = _SIMULATOR_LANG_PATTERN.search(line)
        if match:
            return match.group(1).lower()
        return None

    # -------------------------------------------------------------------------
    # Statistics block parsing (for MC variation)
    # -------------------------------------------------------------------------

    def parse_statistics_block_start(self, line: str) -> bool:
        """Check if line starts a statistics block.

        Spectre statistics block syntax:
            statistics {
                process { vary param dist=gauss std=value }
                mismatch { vary param dist=gauss std=value }
            }
        """
        lower = line.lower().strip()
        return lower.startswith("statistics") and "{" in line

    def parse_statistics_content(self, lines: list[str]) -> "StatisticsBlock":
        """Parse a statistics block from its content lines.

        Args:
            lines: All lines from statistics block (including braces)

        Returns:
            StatisticsBlock with parsed variations
        """
        from ..netlist import StatisticsBlock

        block = StatisticsBlock(raw_content=lines)
        current_section: str | None = None  # "process" or "mismatch"

        for line in lines:
            stripped = line.strip()
            lower = stripped.lower()

            # Skip empty lines and opening/closing braces
            if not stripped or stripped in ("{", "}"):
                continue

            # Check for section start (process/mismatch)
            # Handle inline format: process { vary vth0 dist=gauss std=0.01 }
            if lower.startswith("process"):
                current_section = "process"
                # Check for inline vary directive
                if "vary " in lower:
                    vary_idx = lower.find("vary ")
                    vary_content = stripped[vary_idx + 5 :]
                    # Remove trailing brace if present
                    vary_content = vary_content.rstrip().rstrip("}")
                    variation = self._parse_vary_directive(vary_content)
                    if variation:
                        variation.variation_type = "process"
                        block.process_variations.append(variation)
                continue
            elif lower.startswith("mismatch"):
                current_section = "mismatch"
                # Check for inline vary directive
                if "vary " in lower:
                    vary_idx = lower.find("vary ")
                    vary_content = stripped[vary_idx + 5 :]
                    vary_content = vary_content.rstrip().rstrip("}")
                    variation = self._parse_vary_directive(vary_content)
                    if variation:
                        variation.variation_type = "mismatch"
                        block.mismatch_variations.append(variation)
                continue

            # Check for standalone vary directive
            if lower.startswith("vary "):
                variation = self._parse_vary_directive(stripped[5:])
                if variation:
                    variation.variation_type = current_section or "process"
                    if current_section == "mismatch":
                        block.mismatch_variations.append(variation)
                    else:
                        block.process_variations.append(variation)

        return block

    def _parse_vary_directive(self, content: str) -> "VariationSpec | None":
        """Parse a 'vary param dist=gauss std=value' directive.

        Args:
            content: Text after 'vary ' keyword

        Returns:
            VariationSpec or None if parsing fails
        """
        from ..netlist import VariationSpec

        parts = content.split()
        if not parts:
            return None

        param_name = parts[0]
        spec = VariationSpec(parameter=param_name)

        # Parse key=value pairs
        for part in parts[1:]:
            if "=" in part:
                key, value = part.split("=", 1)
                key_lower = key.lower()
                if key_lower == "dist":
                    spec.distribution = value
                elif key_lower == "std":
                    spec.std = value
                elif key_lower == "mean":
                    spec.mean = value

        return spec


@register_dialect("spectre-spice")
class SpectreSpiceDialect(SpectreDialect):
    """Spectre-SPICE mode for mixed-dialect files.

    This dialect handles files that switch between Spectre and SPICE syntax
    using 'simulator lang=' directives. It defaults to SPICE-style syntax
    (dot prefixes) but can switch to pure Spectre mode.

    This is commonly used in PDK files that need to work with both
    Spectre and SPICE simulators.
    """

    @property
    def name(self) -> str:
        return "spectre-spice"

    @property
    def comment_chars(self) -> tuple[str, ...]:
        """Support both SPICE (*) and Spectre (//) comments."""
        return ("*", "//")

    @property
    def continuation_char(self) -> str:
        """In SPICE mode, use + for continuation."""
        return "+"

    def parse_include(self, line: str) -> tuple[str, str | None] | None:
        """Parse include directive in either syntax.

        Handles both:
            .include "file"  (SPICE style)
            include "file"   (Spectre style)
        """
        lower = line.lower().strip()

        # SPICE style
        if lower.startswith(".include ") or lower.startswith(".inc "):
            prefix_len = 9 if lower.startswith(".include ") else 5
            rest = line[prefix_len:].strip()
        # Spectre style
        elif lower.startswith("include "):
            rest = line[8:].strip()
        else:
            return None

        filepath, section = self._parse_include_path(rest)
        if filepath:
            return (filepath, section)
        return None

    def parse_library(self, line: str) -> tuple[str, str] | None:
        """Parse library directive in either syntax.

        Handles both:
            .lib "file" section  (SPICE style)
            library name         (Spectre structural - returns None)
        """
        lower = line.lower().strip()

        # SPICE style .lib
        if lower.startswith(".lib "):
            rest = line[5:].strip()
            parts = rest.split()
            if len(parts) >= 2:
                filepath = parts[0].strip("\"'")
                section = parts[1].strip("\"'")
                if "." in filepath or "/" in filepath:
                    return (filepath, section)
            elif len(parts) == 1:
                # Could be section start: .lib section_name
                return None

        return None
