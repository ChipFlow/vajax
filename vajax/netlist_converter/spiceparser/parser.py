# SPDX-FileCopyrightText: 2025 ChipFlow
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Unified SPICE netlist parser.

This module provides the main parsing logic that works with any registered
dialect. The dialect handles syntax-specific details while this module
manages the overall parsing flow.
"""

from pathlib import Path
from typing import Any, TextIO

from .dialect import SpiceDialect, get_dialect
from .netlist import (
    Instance,
    LibrarySection,
    ModelDef,
    Netlist,
    Subcircuit,
)


class ParseError(Exception):
    """Error during netlist parsing."""

    def __init__(self, message: str, file: Path | None = None, line: int | None = None):
        self.file = file
        self.line = line
        location = ""
        if file:
            location = f"{file}"
            if line:
                location += f":{line}"
            location += ": "
        super().__init__(f"{location}{message}")


def parse_netlist(
    source: str | Path | TextIO,
    dialect: SpiceDialect | str = "ngspice",
    search_paths: list[Path] | None = None,
) -> Netlist:
    """Parse a SPICE netlist file.

    Args:
        source: Path to file, file content string, or file object
        dialect: Dialect instance or name string
        search_paths: Additional paths to search for included files

    Returns:
        Parsed Netlist object

    Raises:
        ParseError: If parsing fails
        FileNotFoundError: If file not found
    """
    # Get dialect instance if string provided
    if isinstance(dialect, str):
        dialect = get_dialect(dialect)

    # Handle different source types
    if isinstance(source, Path):
        content = source.read_text(encoding="utf-8", errors="replace")
        file_path = source
    elif isinstance(source, str):
        # Check if it looks like content (contains newline) vs file path
        if "\n" in source:
            content = source
            file_path = None
        else:
            source_path = Path(source)
            if source_path.exists():
                content = source_path.read_text(encoding="utf-8", errors="replace")
                file_path = source_path
            else:
                # Assume it's content string if doesn't exist as file
                content = source
                file_path = None
    else:
        content = source.read()
        file_path = Path(source.name) if hasattr(source, "name") else None

    parser = NetlistParser(dialect, file_path, search_paths or [])
    return parser.parse(content)


class NetlistParser:
    """Internal parser class handling the actual parsing logic."""

    def __init__(
        self,
        dialect: SpiceDialect,
        source_file: Path | None,
        search_paths: list[Path],
    ):
        self.dialect = dialect
        self.source_file = source_file
        self.search_paths = search_paths
        if source_file:
            self.search_paths.insert(0, source_file.parent)

        # Parsing state
        self.lines: list[str] = []
        self.line_no = 0
        self.netlist: Netlist = Netlist(source_file=source_file)

        # Current parsing context
        self._current_subckt: Subcircuit | None = None
        self._current_lib_section: LibrarySection | None = None
        self._in_lib_section = False

    def parse(self, content: str) -> Netlist:
        """Parse netlist content and return Netlist object."""
        self.lines = self._join_continuation_lines(content.split("\n"))
        self.line_no = 0

        # First line is title
        if self.lines:
            self.netlist.title = self.lines[0].strip()

        # Parse remaining lines
        for line_no, line in enumerate(self.lines[1:], start=2):
            self.line_no = line_no
            line = line.strip()

            # Skip empty lines and comments
            if not line or any(line.startswith(c) for c in self.dialect.comment_chars):
                continue

            # Parse line
            self._parse_line(line)

        return self.netlist

    def _join_continuation_lines(self, lines: list[str]) -> list[str]:
        """Join lines that start with continuation character."""
        result = []
        current = ""
        cont_char = self.dialect.continuation_char

        for line in lines:
            stripped = line.strip()
            if stripped.startswith(cont_char):
                # Continuation line
                current += " " + stripped[1:].strip()
            else:
                if current:
                    result.append(current)
                current = stripped

        if current:
            result.append(current)

        return result

    def _parse_line(self, line: str) -> None:
        """Parse a single line and update state."""
        lower = line.lower()

        # Check for directives (start with '.')
        if line.startswith("."):
            self._parse_directive(line, lower)
        else:
            # Instance line
            self._parse_instance(line)

    def _parse_directive(self, line: str, lower: str) -> None:
        """Parse a directive line (starting with '.')."""
        # Include directive
        include = self.dialect.parse_include(line)
        if include:
            filepath, section = include
            self._handle_include(filepath, section)
            return

        # Library directive
        lib = self.dialect.parse_library(line)
        if lib:
            filepath, section = lib
            self._handle_library(filepath, section)
            return

        # Library section start (.lib name without file path)
        if lower.startswith(".lib ") and "'" not in line and '"' not in line:
            parts = line.split()
            if len(parts) == 2:
                section_name = parts[1]
                self._start_lib_section(section_name)
                return

        # Library section end
        if lower.startswith(".endl"):
            self._end_lib_section()
            return

        # Model definition
        if lower.startswith(".model "):
            model = self._parse_model(line)
            if model:
                self._add_model(model)
            return

        # Subcircuit start
        if lower.startswith(".subckt "):
            subckt = self._parse_subckt_header(line)
            if subckt:
                self._start_subckt(subckt)
            return

        # Subcircuit end
        if lower.startswith(".ends"):
            self._end_subckt()
            return

        # Conditional directives (HSPICE)
        if self.dialect.supports_conditional():
            cond = self.dialect.parse_conditional(line)
            if cond:
                # For now, just track but don't evaluate
                # TODO: Implement conditional evaluation
                return

        # Other directives - ignore for now
        # (.param, .option, .control, etc.)

    def _parse_model(self, line: str) -> ModelDef | None:
        """Parse a .model definition line."""
        # .model name type [(]params[)]
        # Remove the .model prefix
        rest = line[7:].strip()

        # Split into tokens, respecting parentheses and quotes
        tokens = self._tokenize(rest)
        if len(tokens) < 2:
            return None

        name = tokens[0]
        type_str = tokens[1]
        device_type = self.dialect.parse_model_type(type_str)

        model = ModelDef(
            name=name,
            device_type=device_type,
            source_file=self.source_file,
            line_number=self.line_no,
        )

        # Parse remaining tokens as parameters
        params = self._parse_params(tokens[2:])
        model.parameters = params

        # Extract level if present
        if "level" in params:
            try:
                model.level = int(params["level"])
            except (ValueError, TypeError):
                pass

        # Extract version if present
        if "version" in params:
            model.version = str(params["version"])

        return model

    def _parse_subckt_header(self, line: str) -> Subcircuit | None:
        """Parse a .subckt header line."""
        # .subckt name node1 node2 ... [param=value ...]
        rest = line[7:].strip()
        tokens = self._tokenize(rest)

        if not tokens:
            return None

        name = tokens[0]
        ports = []
        params = {}

        # Separate ports from parameters
        # Look for either "name=value" or "name = value" pattern
        i = 1
        while i < len(tokens):
            token = tokens[i]
            # Check for combined format (name=value)
            if "=" in token and token != "=":
                params = self._parse_params(tokens[i:])
                break
            # Check for spaced format (name = value)
            if i + 1 < len(tokens) and tokens[i + 1] == "=":
                params = self._parse_params(tokens[i:])
                break
            ports.append(token)
            i += 1

        return Subcircuit(
            name=name,
            ports=ports,
            parameters=params,
            source_file=self.source_file,
            line_number=self.line_no,
        )

    def _parse_instance(self, line: str) -> None:
        """Parse a device instance line."""
        tokens = self._tokenize(line)
        if not tokens:
            return

        name = tokens[0]
        prefix = name[0].upper()

        # Get device type from prefix
        prefix_map = self.dialect.get_device_prefix_map()
        device_type = prefix_map.get(prefix, "unknown")

        instance = Instance(
            name=name,
            device_type=device_type,
            source_file=self.source_file,
            line_number=self.line_no,
        )

        # Parse nodes and parameters based on device type
        # This is simplified - real implementation needs device-specific logic
        self._parse_instance_tokens(instance, tokens[1:])

        # Add to current context
        if self._current_subckt:
            self._current_subckt.instances.append(instance)
        else:
            self.netlist.instances.append(instance)

    def _parse_instance_tokens(self, instance: Instance, tokens: list[str]) -> None:
        """Parse instance nodes and parameters from tokens."""
        # Simplified parsing - nodes first, then parameters
        # Handle both "name=value" and "name = value" formats
        i = 0
        while i < len(tokens):
            token = tokens[i]

            # Check for combined parameter format (name=value)
            if "=" in token and token != "=":
                key, _, value = token.partition("=")
                instance.parameters[key.lower()] = self.dialect.parse_parameter_value(value)
                i += 1
            # Check for spaced parameter format (name = value)
            elif i + 2 < len(tokens) and tokens[i + 1] == "=":
                key = token
                value = tokens[i + 2]
                instance.parameters[key.lower()] = self.dialect.parse_parameter_value(value)
                i += 3
            elif instance.model_name is None and token and not token[0].isdigit():
                # Could be model name or node
                # Heuristic: if it looks like a node name, add as node
                # Otherwise treat as model name
                if token.lower() in ("vdd", "vss", "gnd", "0", "in", "out") or token.isdigit():
                    instance.nodes.append(token)
                else:
                    # Might be model name - check if we have enough nodes
                    # This is device-type dependent
                    instance.nodes.append(token)  # Simplified: just add as node
                i += 1
            else:
                instance.nodes.append(token)
                i += 1

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize a line respecting quotes and parentheses."""
        tokens = []
        current = ""
        in_quotes = False
        quote_char = None
        paren_depth = 0

        for char in text:
            if in_quotes:
                current += char
                if char == quote_char:
                    in_quotes = False
                    tokens.append(current)
                    current = ""
            elif char in ("'", '"'):
                in_quotes = True
                quote_char = char
                current += char
            elif char == "(":
                paren_depth += 1
                current += char
            elif char == ")":
                paren_depth -= 1
                current += char
            elif char in (" ", "\t") and paren_depth == 0:
                if current:
                    tokens.append(current)
                    current = ""
            else:
                current += char

        if current:
            tokens.append(current)

        return tokens

    def _parse_params(self, tokens: list[str]) -> dict[str, Any]:
        """Parse parameter tokens into a dictionary.

        Handles both formats:
        - name=value (Ngspice style, single token)
        - name = value (HSPICE style, three separate tokens)
        """
        params = {}
        i = 0
        while i < len(tokens):
            token = tokens[i]

            if "=" in token and token != "=":
                # Combined format: name=value
                key, _, value = token.partition("=")
                params[key.lower()] = self.dialect.parse_parameter_value(value)
                i += 1
            elif i + 2 < len(tokens) and tokens[i + 1] == "=" and "=" not in token:
                # Spaced format: name = value
                key = token
                value = tokens[i + 2]
                params[key.lower()] = self.dialect.parse_parameter_value(value)
                i += 3
            else:
                # Skip non-parameter tokens
                i += 1

        return params

    def _handle_include(self, filepath: str, section: str | None) -> None:
        """Handle an include directive."""
        # Resolve path
        resolved = self._resolve_path(filepath)
        if resolved:
            self.netlist.includes.append(resolved)
            # TODO: Actually parse the included file

    def _handle_library(self, filepath: str, section: str) -> None:
        """Handle a library directive with section."""
        resolved = self._resolve_path(filepath)
        if resolved:
            # TODO: Parse library file and extract section
            pass

    def _resolve_path(self, filepath: str) -> Path | None:
        """Resolve a file path using search paths."""
        # Remove quotes
        filepath = filepath.strip("'\"")
        path = Path(filepath)

        # Check if absolute
        if path.is_absolute() and path.exists():
            return path

        # Search in search paths
        for search_dir in self.search_paths:
            candidate = search_dir / filepath
            if candidate.exists():
                return candidate

        return None

    def _start_lib_section(self, name: str) -> None:
        """Start a new library section."""
        self._current_lib_section = LibrarySection(name=name)
        self._in_lib_section = True

    def _end_lib_section(self) -> None:
        """End current library section."""
        if self._current_lib_section:
            self.netlist.library_sections[self._current_lib_section.name] = (
                self._current_lib_section
            )
        self._current_lib_section = None
        self._in_lib_section = False

    def _start_subckt(self, subckt: Subcircuit) -> None:
        """Start parsing a subcircuit."""
        if self._current_subckt:
            # Nested subcircuit
            self._current_subckt.subcircuits.append(subckt)
        self._current_subckt = subckt

    def _end_subckt(self) -> None:
        """End current subcircuit."""
        if self._current_subckt:
            if self._in_lib_section and self._current_lib_section:
                self._current_lib_section.subcircuits.append(self._current_subckt)
            else:
                self.netlist.subcircuits.append(self._current_subckt)
        self._current_subckt = None

    def _add_model(self, model: ModelDef) -> None:
        """Add a model to current context."""
        if self._current_subckt:
            self._current_subckt.models.append(model)
        elif self._in_lib_section and self._current_lib_section:
            self._current_lib_section.models.append(model)
        else:
            self.netlist.models.append(model)
