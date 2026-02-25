# SPDX-FileCopyrightText: 2025 ChipFlow
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Data structures for representing parsed SPICE netlists.

These classes provide a dialect-independent representation of SPICE
constructs (models, subcircuits, instances) that can be used by the
vcwriter to generate Verilog-A output.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Parameter:
    """A named parameter with optional value.

    Attributes:
        name: Parameter name
        value: Parameter value (may be numeric, string expression, or None)
        unit: Optional unit string (e.g., "V", "A", "F")
    """

    name: str
    value: Any = None
    unit: str | None = None


@dataclass
class Comment:
    """A comment line from the netlist.

    Attributes:
        text: Comment text (without the comment character prefix)
        line_number: Line number in source file
        inline: True if this was an inline comment (// style)
    """

    text: str
    line_number: int | None = None
    inline: bool = False


@dataclass
class ModelDef:
    """A SPICE .model definition.

    Attributes:
        name: Model name
        device_type: Device type (e.g., "nmos", "pnp", "d")
        level: Model level/version (e.g., 54 for BSIM4)
        version: Model version string if specified
        parameters: Dict of parameter name -> value
        source_file: Path to source file (for error reporting)
        line_number: Line number in source file
    """

    name: str
    device_type: str
    level: int | None = None
    version: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    source_file: Path | None = None
    line_number: int | None = None


@dataclass
class Instance:
    """A device instance in a netlist.

    Attributes:
        name: Instance name (e.g., "M1", "R1")
        device_type: Type of device (from prefix or explicit)
        nodes: List of connected node names
        model_name: Model name if specified
        parameters: Instance parameters
        source_file: Path to source file
        line_number: Line number in source file
    """

    name: str
    device_type: str
    nodes: list[str] = field(default_factory=list)
    model_name: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    source_file: Path | None = None
    line_number: int | None = None


@dataclass
class Subcircuit:
    """A .subckt definition containing instances and possibly models.

    Attributes:
        name: Subcircuit name
        ports: List of port names (external nodes)
        parameters: Default parameter values
        instances: Device instances within the subcircuit
        models: Model definitions local to this subcircuit
        subcircuits: Nested subcircuit definitions
        comments: Comments within this subcircuit
        source_file: Path to source file
        line_number: Line number in source file
    """

    name: str
    ports: list[str] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)
    instances: list[Instance] = field(default_factory=list)
    models: list[ModelDef] = field(default_factory=list)
    subcircuits: list["Subcircuit"] = field(default_factory=list)
    comments: list[Comment] = field(default_factory=list)
    source_file: Path | None = None
    line_number: int | None = None


@dataclass
class LibrarySection:
    """A .lib section within a library file.

    Attributes:
        name: Section name
        content: Raw content lines within the section
        models: Parsed models in this section
        subcircuits: Parsed subcircuits in this section
    """

    name: str
    content: list[str] = field(default_factory=list)
    models: list[ModelDef] = field(default_factory=list)
    subcircuits: list[Subcircuit] = field(default_factory=list)


@dataclass
class VariationSpec:
    """A single variation specification in a statistics block.

    Used to represent Spectre-style variation directives like:
        vary vth0 dist=gauss std=0.01

    Attributes:
        parameter: Parameter name being varied
        distribution: Distribution type ("gauss", "uniform", etc.)
        std: Standard deviation (for Gaussian distributions)
        mean: Mean value (optional, defaults to parameter's nominal value)
        variation_type: "process" or "mismatch"
    """

    parameter: str
    distribution: str = "gauss"
    std: str | float | None = None
    mean: str | float | None = None
    variation_type: str = "process"


@dataclass
class StatisticsBlock:
    """A Spectre statistics block for Monte Carlo variation.

    Represents a statistics block containing process and mismatch variations:
        statistics {
            process { vary vth0 dist=gauss std=0.01 }
            mismatch { vary delvto dist=gauss std=0.05 }
        }

    Attributes:
        name: Optional block name
        process_variations: List of process-level variations
        mismatch_variations: List of mismatch (instance-level) variations
        raw_content: Original text for pass-through preservation
    """

    name: str | None = None
    process_variations: list[VariationSpec] = field(default_factory=list)
    mismatch_variations: list[VariationSpec] = field(default_factory=list)
    raw_content: list[str] = field(default_factory=list)


@dataclass
class Netlist:
    """A complete parsed SPICE netlist.

    Attributes:
        title: First line of netlist (title comment)
        comments: Top-level comments (outside subcircuits)
        parameters: Global parameter definitions (.param)
        models: Top-level model definitions
        subcircuits: Top-level subcircuit definitions
        instances: Top-level instances (for testbenches)
        library_sections: Named library sections (.lib name ... .endl)
        includes: List of included files
        statistics_blocks: Spectre statistics blocks for MC variation
        source_file: Path to main source file
    """

    title: str = ""
    comments: list[Comment] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)
    models: list[ModelDef] = field(default_factory=list)
    subcircuits: list[Subcircuit] = field(default_factory=list)
    instances: list[Instance] = field(default_factory=list)
    library_sections: dict[str, LibrarySection] = field(default_factory=dict)
    includes: list[Path] = field(default_factory=list)
    statistics_blocks: list[StatisticsBlock] = field(default_factory=list)
    source_file: Path | None = None

    def get_model(self, name: str) -> ModelDef | None:
        """Find a model by name in the netlist."""
        for model in self.models:
            if model.name.lower() == name.lower():
                return model
        # Search in library sections
        for section in self.library_sections.values():
            for model in section.models:
                if model.name.lower() == name.lower():
                    return model
        return None

    def get_subcircuit(self, name: str) -> Subcircuit | None:
        """Find a subcircuit by name in the netlist."""
        for subckt in self.subcircuits:
            if subckt.name.lower() == name.lower():
                return subckt
        # Search in library sections
        for section in self.library_sections.values():
            for subckt in section.subcircuits:
                if subckt.name.lower() == name.lower():
                    return subckt
        return None
