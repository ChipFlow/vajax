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
    Comment,
    Instance,
    LibrarySection,
    ModelDef,
    Netlist,
    StatisticsBlock,
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

    # Device type to expected node count and value parameter name
    # Format: prefix -> (node_count, value_param_name or None)
    DEVICE_NODE_COUNTS: dict[str, tuple[int, str | None]] = {
        "R": (2, "r"),  # Resistor: 2 nodes, value as 'r' param
        "C": (2, "c"),  # Capacitor: 2 nodes, value as 'c' param
        "L": (2, "l"),  # Inductor: 2 nodes, value as 'l' param
        "D": (2, None),  # Diode: 2 nodes, no direct value
        "Q": (3, None),  # BJT: 3 nodes (C, B, E), optional 4th (substrate)
        "M": (4, None),  # MOSFET: 4 nodes (D, G, S, B)
        "J": (3, None),  # JFET: 3 nodes
        "V": (2, None),  # Voltage source: 2 nodes
        "I": (2, None),  # Current source: 2 nodes
        "X": (0, None),  # Subcircuit: variable nodes
    }

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

    def _looks_like_value(self, token: str) -> bool:
        """Check if a token looks like a numeric value (possibly with SI prefix).

        Returns True for tokens like: 1k, 10p, 1.5u, 100, 1e-9, etc.
        Returns False for model names, node names, expressions.
        """
        import re

        # SI prefix pattern: number followed by optional SI suffix
        # Handles: 1k, 10p, 1.5u, 100meg, 1e-9, etc.
        si_pattern = re.compile(
            r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?"
            r"(T|G|[Mm]eg|[Mm]|[Kk]|[Uu]|[Nn]|[Pp]|[Ff]|[Aa])?$",
            re.IGNORECASE,
        )
        return bool(si_pattern.match(token))

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

            # Skip empty lines
            if not line:
                continue

            # Check for comments
            is_comment = False
            for comment_char in self.dialect.comment_chars:
                if line.startswith(comment_char):
                    # Store comment (strip comment character prefix)
                    text = line[len(comment_char) :].strip()
                    comment = Comment(
                        text=text,
                        line_number=line_no,
                        inline=(comment_char == "//"),
                    )
                    self._add_comment(comment)
                    is_comment = True
                    break

            if is_comment:
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

    def _add_comment(self, comment: Comment) -> None:
        """Add a comment to the current context (subcircuit or netlist)."""
        if self._current_subckt:
            self._current_subckt.comments.append(comment)
        else:
            self.netlist.comments.append(comment)

    def _parse_line(self, line: str) -> None:
        """Parse a single line and update state."""
        lower = line.lower()

        # Check for directives (start with '.')
        if line.startswith("."):
            self._parse_directive(line, lower)
        # Check for Spectre-style keywords (no dot prefix)
        elif self._is_spectre_keyword(line):
            self._parse_spectre_keyword(line, lower)
        else:
            # Instance line
            self._parse_instance(line)

    def _is_spectre_keyword(self, line: str) -> bool:
        """Check if line starts with a Spectre keyword (no dot prefix)."""
        # Only check for Spectre dialects
        if hasattr(self.dialect, "is_spectre_keyword"):
            return self.dialect.is_spectre_keyword(line) is not None
        return False

    def _parse_spectre_keyword(self, line: str, lower: str) -> None:
        """Parse a Spectre-style keyword line (no dot prefix)."""
        # Include directive
        include = self.dialect.parse_include(line)
        if include:
            filepath, section = include
            self._handle_include(filepath, section)
            return

        # Library/section blocks (Spectre structural syntax)
        if hasattr(self.dialect, "parse_library_block"):
            lib_block = self.dialect.parse_library_block(line)
            if lib_block:
                directive_type, name = lib_block
                if directive_type == "library":
                    self._start_lib_section(name)
                elif directive_type == "endlibrary":
                    self._end_lib_section()
                return

        if hasattr(self.dialect, "parse_section_block"):
            section_block = self.dialect.parse_section_block(line)
            if section_block:
                directive_type, name = section_block
                if directive_type == "section":
                    self._start_lib_section(name)
                elif directive_type == "endsection":
                    self._end_lib_section()
                return

        # Model definition: model name type params
        if lower.startswith("model "):
            model = self._parse_spectre_model(line)
            if model:
                self._add_model(model)
            return

        # Subcircuit start: subckt name (ports) parameters ...
        if lower.startswith("subckt ") or lower.startswith("inline subckt "):
            subckt = self._parse_spectre_subckt_header(line)
            if subckt:
                self._start_subckt(subckt)
            return

        # Subcircuit end: ends [name]
        if lower.startswith("ends"):
            self._end_subckt()
            return

        # simulator lang= directive (for mixed-mode files)
        if hasattr(self.dialect, "parse_simulator_lang"):
            lang = self.dialect.parse_simulator_lang(line)
            if lang:
                # Track language switch but don't change parser behavior for now
                # In future, could switch dialect dynamically
                return

        # Statistics block (for MC variation)
        if hasattr(self.dialect, "parse_statistics_block_start"):
            if self.dialect.parse_statistics_block_start(line):
                stats_block = self._parse_statistics_block(line)
                if stats_block:
                    self.netlist.statistics_blocks.append(stats_block)
                return

        # Ignore other Spectre keywords for now (parameters, global, etc.)

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

        # Parameter definition (.param name=value or .param name = value)
        if lower.startswith(".param ") or lower.startswith(".params "):
            self._parse_param_directive(line)
            return

        # Other directives - ignore for now
        # (.option, .control, etc.)

    def _parse_param_directive(self, line: str) -> None:
        """Parse a .param or .params directive.

        Formats:
            .param name=value
            .param name = value
            .param name1=val1 name2=val2
            .params name=value
        """
        # Remove .param/.params prefix
        lower = line.lower()
        if lower.startswith(".params "):
            rest = line[8:].strip()
        else:
            rest = line[7:].strip()

        # Parse parameters
        tokens = self._tokenize(rest)
        i = 0
        while i < len(tokens):
            token = tokens[i]

            # Combined format: name=value
            if "=" in token and token != "=":
                key, _, value = token.partition("=")
                parsed_value = self.dialect.parse_parameter_value(value)
                self.netlist.parameters[key.lower()] = parsed_value
                i += 1
            # Spaced format: name = value
            elif i + 2 < len(tokens) and tokens[i + 1] == "=":
                key = token
                value = tokens[i + 2]
                parsed_value = self.dialect.parse_parameter_value(value)
                self.netlist.parameters[key.lower()] = parsed_value
                i += 3
            else:
                i += 1

    def _parse_spectre_model(self, line: str) -> ModelDef | None:
        """Parse a Spectre-style model definition (no dot prefix).

        Spectre syntax: model name type (params) or model name type param=val
        """
        # Remove 'model ' prefix
        rest = line[6:].strip()
        return self._parse_model_content(rest)

    def _parse_spectre_subckt_header(self, line: str) -> Subcircuit | None:
        """Parse a Spectre-style subckt header (no dot prefix).

        Spectre syntax: subckt name (port1 port2) parameters p1=v1 ...
        Or: inline subckt name (port1 port2) ...
        """
        # Handle 'inline subckt' prefix
        lower = line.lower()
        if lower.startswith("inline subckt "):
            rest = line[14:].strip()
        else:
            rest = line[7:].strip()  # Remove 'subckt ' prefix

        tokens = self._tokenize(rest)
        if not tokens:
            return None

        name = tokens[0]
        ports = []
        params = {}

        # In Spectre, ports are often in parentheses: subckt name (p1 p2 p3)
        i = 1
        if i < len(tokens) and tokens[i].startswith("("):
            # Extract ports from parenthesized group
            port_token = tokens[i]
            if port_token.startswith("(") and port_token.endswith(")"):
                # Ports are in a single token like "(a b c)"
                port_str = port_token[1:-1].strip()
                if port_str:
                    ports = port_str.split()
            i += 1

        # Look for 'parameters' keyword or param=value
        while i < len(tokens):
            token = tokens[i]
            token_lower = token.lower()

            # Check for 'parameters' keyword
            if token_lower == "parameters":
                i += 1
                params = self._parse_params(tokens[i:])
                break
            # Check for combined format (name=value)
            elif "=" in token and token != "=":
                params = self._parse_params(tokens[i:])
                break
            # Check for spaced format (name = value)
            elif i + 1 < len(tokens) and tokens[i + 1] == "=":
                params = self._parse_params(tokens[i:])
                break
            # Otherwise it's a port (for cases without parentheses)
            elif not ports:
                ports.append(token)
            i += 1

        return Subcircuit(
            name=name,
            ports=ports,
            parameters=params,
            source_file=self.source_file,
            line_number=self.line_no,
        )

    def _parse_model(self, line: str) -> ModelDef | None:
        """Parse a .model definition line."""
        # .model name type [(]params[)]
        # Remove the .model prefix
        rest = line[7:].strip()
        return self._parse_model_content(rest)

    def _parse_model_content(self, rest: str) -> ModelDef | None:
        """Parse model content (shared between SPICE and Spectre styles)."""

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
        """Parse instance nodes and parameters from tokens.

        Handles device-specific parsing:
        - R/C/L: 2 nodes, optional value, optional model
        - M: 4 nodes, model, parameters
        - Q: 3-4 nodes, model, parameters
        - X: variable nodes (until model/subckt name)
        """
        # Get device info from prefix
        prefix = instance.name[0].upper() if instance.name else ""
        node_count, value_param = self.DEVICE_NODE_COUNTS.get(prefix, (0, None))

        i = 0
        nodes_collected = 0
        value_collected = False

        while i < len(tokens):
            token = tokens[i]

            # Check for combined parameter format (name=value)
            if "=" in token and token != "=":
                key, _, value = token.partition("=")
                instance.parameters[key.lower()] = self.dialect.parse_parameter_value(value)
                i += 1
                continue

            # Check for spaced parameter format (name = value)
            if i + 2 < len(tokens) and tokens[i + 1] == "=":
                key = token
                value = tokens[i + 2]
                instance.parameters[key.lower()] = self.dialect.parse_parameter_value(value)
                i += 3
                continue

            # For passive elements (R/C/L), handle value after nodes
            if value_param and nodes_collected >= node_count and not value_collected:
                if self._looks_like_value(token):
                    # This is the element value, store as parameter
                    instance.parameters[value_param] = self.dialect.parse_parameter_value(token)
                    value_collected = True
                    i += 1
                    continue

            # For subcircuit calls (X prefix), nodes continue until we hit
            # what looks like a subcircuit name (non-numeric, not a common node)
            if prefix == "X":
                # Check if this could be the subcircuit name
                # It's the subcircuit name if:
                # - We have at least one node
                # - The token doesn't look like a common node name
                # - The next tokens (if any) are parameters
                if nodes_collected > 0 and instance.model_name is None:
                    # Check if remaining tokens are parameters
                    next_is_param = (
                        i + 1 >= len(tokens)
                        or "=" in tokens[i + 1]
                        or (i + 2 < len(tokens) and tokens[i + 2] == "=")
                    )
                    if next_is_param and not self._looks_like_value(token):
                        instance.model_name = token
                        i += 1
                        continue

            # Collect nodes based on expected count
            if node_count > 0 and nodes_collected < node_count:
                instance.nodes.append(token)
                nodes_collected += 1
                i += 1
                continue

            # For X prefix (subcircuits), keep collecting nodes until subckt name
            if prefix == "X" and instance.model_name is None:
                instance.nodes.append(token)
                nodes_collected += 1
                i += 1
                continue

            # After expected nodes, next non-value token could be model name
            if instance.model_name is None and not self._looks_like_value(token):
                instance.model_name = token
                i += 1
                continue

            # Anything else is added as a node (for devices with variable nodes)
            instance.nodes.append(token)
            nodes_collected += 1
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

    def _parse_statistics_block(self, first_line: str) -> StatisticsBlock | None:
        """Parse a complete statistics block, reading until closing brace.

        Spectre statistics block syntax:
            statistics {
                process { vary param dist=gauss std=value }
                mismatch { vary param dist=gauss std=value }
            }

        Args:
            first_line: The first line containing 'statistics {'

        Returns:
            StatisticsBlock with parsed variations, or None if parsing fails
        """
        lines = [first_line]
        brace_depth = first_line.count("{") - first_line.count("}")

        # Read lines until we close all braces
        # Note: self.line_no is 1-based from enumerate(start=2) over lines[1:]
        # So the actual array index is (self.line_no - 1) for the current line
        current_idx = self.line_no - 1  # Convert to 0-based array index
        while brace_depth > 0 and current_idx < len(self.lines) - 1:
            current_idx += 1
            line = self.lines[current_idx].strip()
            lines.append(line)
            brace_depth += line.count("{") - line.count("}")

        # Update line number to skip the lines we consumed
        # Convert back to the line_no format used by main loop
        self.line_no = current_idx + 1

        # Parse the block content using dialect method
        if hasattr(self.dialect, "parse_statistics_content"):
            return self.dialect.parse_statistics_content(lines)

        # Fallback: just store raw content
        return StatisticsBlock(raw_content=lines)
