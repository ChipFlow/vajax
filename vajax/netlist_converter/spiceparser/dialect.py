# SPDX-FileCopyrightText: 2025 ChipFlow
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Base class for SPICE dialect implementations.

Each SPICE simulator has dialect-specific syntax and extensions. This module
provides an abstract base class that defines the interface for handling these
differences.
"""

from abc import ABC, abstractmethod
from typing import Any

# Registry for dialect implementations
_DIALECT_REGISTRY: dict[str, type["SpiceDialect"]] = {}


def register_dialect(name: str):
    """Decorator to register a dialect implementation."""

    def decorator(cls: type["SpiceDialect"]) -> type["SpiceDialect"]:
        _DIALECT_REGISTRY[name.lower()] = cls
        return cls

    return decorator


def get_dialect(name: str) -> "SpiceDialect":
    """Get a dialect instance by name.

    Args:
        name: Dialect name (ngspice, hspice, ltspice)

    Returns:
        Dialect instance

    Raises:
        ValueError: If dialect is not registered
    """
    name_lower = name.lower()
    if name_lower not in _DIALECT_REGISTRY:
        available = ", ".join(sorted(_DIALECT_REGISTRY.keys()))
        raise ValueError(f"Unknown dialect '{name}'. Available: {available}")
    return _DIALECT_REGISTRY[name_lower]()


def detect_dialect(content: str) -> str:
    """Detect SPICE dialect from file content.

    Analyzes the content for dialect-specific patterns and returns the
    most likely dialect name. If no specific patterns are found, defaults
    to "ngspice".

    Args:
        content: SPICE netlist content as a string

    Returns:
        Dialect name: "ltspice", "hspice", or "ngspice"
    """
    content_lower = content.lower()

    # LTSpice indicators
    ltspice_score = 0
    if ".backanno" in content_lower:
        ltspice_score += 3
    # Windows paths with backslashes
    if "\\" in content and (".lib" in content_lower or ".inc" in content_lower):
        ltspice_score += 2
    # Unicode micro symbol
    if "µ" in content:
        ltspice_score += 2
    # LTSpice-specific parameters on passives
    if "rser=" in content_lower or "lser=" in content_lower:
        ltspice_score += 2

    # HSPICE indicators
    hspice_score = 0
    if ".if " in content_lower or ".if(" in content_lower:
        hspice_score += 3
    if ".elseif" in content_lower:
        hspice_score += 2
    if ".endif" in content_lower:
        hspice_score += 2
    if ".alter" in content_lower:
        hspice_score += 3
    if ".data " in content_lower:
        hspice_score += 2
    if ".prot" in content_lower or ".unprot" in content_lower:
        hspice_score += 2
    # Single-quoted expressions (common in HSPICE)
    import re

    if re.search(r"=\s*'[^']+[+\-*/][^']+'", content):
        hspice_score += 2
    # Spaces around = in parameters (HSPICE style)
    if re.search(r"\s=\s", content):
        hspice_score += 1

    # Return highest scoring dialect, default to ngspice
    if ltspice_score > hspice_score and ltspice_score >= 2:
        return "ltspice"
    if hspice_score > ltspice_score and hspice_score >= 2:
        return "hspice"
    return "ngspice"


def detect_dialect_from_file(filepath: str) -> str:
    """Detect SPICE dialect from a file.

    Args:
        filepath: Path to SPICE netlist file

    Returns:
        Dialect name: "ltspice", "hspice", or "ngspice"
    """
    with open(filepath, encoding="utf-8", errors="replace") as f:
        content = f.read()
    return detect_dialect(content)


class SpiceDialect(ABC):
    """Base class for SPICE dialect implementations.

    Each dialect defines how to parse and interpret SPICE syntax specific
    to that simulator. Subclasses must implement the abstract methods to
    handle dialect-specific parsing.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the dialect name."""
        ...

    @property
    def comment_chars(self) -> tuple[str, ...]:
        """Characters that start a comment line. Default: ('*',)"""
        return ("*",)

    @property
    def continuation_char(self) -> str:
        """Character that indicates line continuation. Default: '+'"""
        return "+"

    # -------------------------------------------------------------------------
    # Include/Library handling
    # -------------------------------------------------------------------------

    @abstractmethod
    def parse_include(self, line: str) -> tuple[str, str | None] | None:
        """Parse an include directive.

        Args:
            line: The directive line (e.g., ".include 'file.spi'")

        Returns:
            Tuple of (filepath, section_name) or None if not an include.
            section_name is None for unconditional includes.
        """
        ...

    @abstractmethod
    def parse_library(self, line: str) -> tuple[str, str] | None:
        """Parse a library directive with section.

        Args:
            line: The directive line (e.g., ".lib 'file.lib' section")

        Returns:
            Tuple of (filepath, section_name) or None if not a library directive.
        """
        ...

    # -------------------------------------------------------------------------
    # Model parsing
    # -------------------------------------------------------------------------

    @abstractmethod
    def parse_model_type(self, type_str: str) -> str:
        """Normalize a model type string.

        Args:
            type_str: Raw model type (e.g., "NMOS", "nmos", "n")

        Returns:
            Normalized type string (e.g., "nmos")
        """
        ...

    @abstractmethod
    def get_device_prefix_map(self) -> dict[str, str]:
        """Return mapping from device prefix to device type.

        Returns:
            Dict mapping single-char prefix to device type.
            E.g., {"M": "mosfet", "Q": "bjt", "D": "diode", ...}
        """
        ...

    # -------------------------------------------------------------------------
    # Parameter parsing
    # -------------------------------------------------------------------------

    @abstractmethod
    def parse_parameter_value(self, value_str: str) -> Any:
        """Parse a parameter value, handling SI prefixes and expressions.

        Args:
            value_str: Raw parameter value (e.g., "1k", "1.5e-3", "'1+x'")

        Returns:
            Parsed value (float, str expression, etc.)
        """
        ...

    # -------------------------------------------------------------------------
    # Dialect-specific extensions
    # -------------------------------------------------------------------------

    def get_extra_element_params(self, element_type: str) -> list[str]:
        """Return dialect-specific parameters for an element type.

        For example, LTSpice supports Rser, Lser, Rpar, Cpar on passives.

        Args:
            element_type: Element type (e.g., "capacitor", "inductor")

        Returns:
            List of additional parameter names
        """
        return []

    def supports_conditional(self) -> bool:
        """Whether this dialect supports .if/.else/.endif conditionals.

        Returns:
            True if conditionals are supported (HSPICE), False otherwise.
        """
        return False

    def parse_conditional(self, line: str) -> tuple[str, str | None] | None:
        """Parse a conditional directive.

        Args:
            line: The directive line (e.g., ".if (cond)", ".else", ".endif")

        Returns:
            Tuple of (directive_type, condition) or None if not a conditional.
            directive_type is one of: "if", "elseif", "else", "endif"
        """
        return None
