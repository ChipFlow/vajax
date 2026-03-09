"""Core netlist builder classes."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, List, Optional


class SubcktBuilder:
    """Subcircuit definition builder.

    Used as a context manager via Netlist.subckt(). Collects instance
    and comment lines, then renders as VACASK subcircuit definition.
    """

    def __init__(self, name: str, terminals: List[str], params: Optional[Dict[str, str]] = None):
        self.name = name
        self.terminals = terminals
        self.params = params or {}
        self._lines: List[str] = []

    def inst(self, name: str, model: str, terminals: List[str], **params: str) -> None:
        """Add an instance to this subcircuit.

        Args:
            name: Instance name (e.g., "mp1")
            model: Model/subcircuit name (e.g., "pmos", "and")
            terminals: Terminal connections
            **params: Instance parameters (e.g., w="1u", l="0.2u")
        """
        terms = " ".join(terminals)
        param_str = " ".join(f"{k}={v}" for k, v in params.items())
        if param_str:
            self._lines.append(f"  {name} ({terms}) {model} {param_str}")
        else:
            self._lines.append(f"  {name} ({terms}) {model}")

    def comment(self, text: str) -> None:
        """Add a comment line inside the subcircuit."""
        self._lines.append(f"  // {text}")

    def blank(self) -> None:
        """Add a blank line inside the subcircuit."""
        self._lines.append("")

    def __enter__(self) -> SubcktBuilder:
        return self

    def __exit__(self, *args) -> None:
        pass

    def render(self) -> str:
        """Render this subcircuit as VACASK text."""
        terms = " ".join(self.terminals)
        lines = [f"subckt {self.name}({terms})"]
        if self.params:
            param_str = " ".join(f"{k}={v}" for k, v in self.params.items())
            lines.append(f"  parameters {param_str}")
        lines.extend(self._lines)
        lines.append("ends")
        return "\n".join(lines)


class Netlist:
    """Top-level VACASK netlist builder.

    Collects globals, ground, models, subcircuits, top-level instances,
    and comments, then renders to VACASK netlist format via str() or print().
    """

    def __init__(
        self,
        globals: Optional[List[str]] = None,
        ground: Optional[str] = None,
        title: Optional[str] = None,
    ):
        self._title = title
        self._globals = globals or []
        self._ground = ground
        self._sections: List[object] = []  # ordered list of renderable items

    @contextmanager
    def subckt(self, name: str, terminals: List[str], **default_params: str):
        """Define a subcircuit as a context manager.

        Args:
            name: Subcircuit name
            terminals: Port names
            **default_params: Default parameter values

        Yields:
            SubcktBuilder for adding instances
        """
        builder = SubcktBuilder(name, terminals, default_params if default_params else None)
        yield builder
        self._sections.append(builder)

    def model(self, name: str, module: str, **params: str) -> None:
        """Add a model statement.

        Args:
            name: Model name
            module: Module name (e.g., "vsource", "diode")
            **params: Model parameters
        """
        self._sections.append(_ModelDef(name, module, params))

    def inst(self, name: str, model: str, terminals: List[str], **params: str) -> None:
        """Add a top-level instance.

        Args:
            name: Instance name
            model: Model/subcircuit name
            terminals: Terminal connections
            **params: Instance parameters
        """
        self._sections.append(_InstDef(name, model, terminals, params))

    def comment(self, text: str) -> None:
        """Add a comment line."""
        self._sections.append(_Comment(text))

    def blank(self) -> None:
        """Add a blank line."""
        self._sections.append(_Blank())

    def __str__(self) -> str:
        """Render the complete netlist as VACASK format text."""
        lines: List[str] = []

        if self._title:
            lines.append(f"// {self._title}")
            lines.append("")

        if self._globals:
            lines.append("global " + " ".join(self._globals))

        if self._ground:
            lines.append(f"ground {self._ground}")

        if self._globals or self._ground:
            lines.append("")

        for section in self._sections:
            if isinstance(section, SubcktBuilder):
                lines.append(section.render())
                lines.append("")
            elif isinstance(section, _ModelDef):
                lines.append(section.render())
            elif isinstance(section, _InstDef):
                lines.append(section.render())
            elif isinstance(section, _Comment):
                lines.append(f"// {section.text}")
            elif isinstance(section, _Blank):
                lines.append("")

        return "\n".join(lines)


class _ModelDef:
    """Internal: model statement."""

    def __init__(self, name: str, module: str, params: Dict[str, str]):
        self.name = name
        self.module = module
        self.params = params

    def render(self) -> str:
        if self.params:
            param_str = " ".join(f"{k}={v}" for k, v in self.params.items())
            return f"model {self.name} {self.module} {param_str}"
        return f"model {self.name} {self.module}"


class _InstDef:
    """Internal: top-level instance."""

    def __init__(self, name: str, model: str, terminals: List[str], params: Dict[str, str]):
        self.name = name
        self.model = model
        self.terminals = terminals
        self.params = params

    def render(self) -> str:
        terms = " ".join(self.terminals)
        param_str = " ".join(f"{k}={v}" for k, v in self.params.items())
        if param_str:
            return f"{self.name} ({terms}) {self.model} {param_str}"
        return f"{self.name} ({terms}) {self.model}"


class _Comment:
    """Internal: comment line."""

    def __init__(self, text: str):
        self.text = text


class _Blank:
    """Internal: blank line."""

    pass
