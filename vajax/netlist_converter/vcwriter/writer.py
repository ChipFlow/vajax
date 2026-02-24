# SPDX-FileCopyrightText: 2025 ChipFlow
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""VACASK output file writer.

Generates VACASK-format files from parsed netlist objects.
"""

from io import StringIO
from pathlib import Path
from typing import Any, TextIO

from ..spiceparser.elements import (
    OsdiModuleInfo,
    get_default_model,
    get_device_type_info,
    get_osdi_module,
)
from ..spiceparser.netlist import Instance, ModelDef, Netlist, Subcircuit
from ..spiceparser.params import format_value, process_terminals


def write_vacask(
    netlist: Netlist,
    output: str | Path | TextIO,
    signature: str = "// Converted by ng2vc converter\n",
    columns: int = 80,
) -> None:
    """Write a netlist in VACASK format.

    Args:
        netlist: Parsed netlist to write
        output: Output file path or file object
        signature: Signature comment to add at top
        columns: Maximum line width for formatting
    """
    writer = VacaskWriter(signature=signature, columns=columns)
    content = writer.write(netlist)

    if isinstance(output, (str, Path)):
        Path(output).write_text(content)
    else:
        output.write(content)


class VacaskWriter:
    """VACASK output file writer.

    Converts parsed SPICE netlists to VACASK format, which uses:
    - Spectre-like syntax with parentheses around node lists
    - `load` directives for OSDI modules
    - `model` statements for model definitions
    - `subckt`/`ends` blocks for subcircuits
    """

    def __init__(
        self,
        signature: str = "// Converted by ng2vc converter\n",
        columns: int = 80,
        default_model_prefix: str = "defmod_",
    ):
        self.signature = signature
        self.columns = columns
        self.default_model_prefix = default_model_prefix

        # Collected OSDI modules needed for this netlist
        self._osdi_modules: dict[str, OsdiModuleInfo] = {}

        # Model name -> OSDI module mapping for instances
        self._model_modules: dict[str, OsdiModuleInfo] = {}

    def write(self, netlist: Netlist) -> str:
        """Write netlist to VACASK format string."""
        buf = StringIO()

        # Write signature
        buf.write(self.signature)
        buf.write("\n")

        # Collect required OSDI modules from models
        self._collect_osdi_modules(netlist)

        # Write load statements
        for osdi_file in sorted(self._osdi_modules.keys()):
            buf.write(f'load "{osdi_file}"\n')

        if self._osdi_modules:
            buf.write("\n")

        # Write models
        for model in netlist.models:
            self._write_model(buf, model)

        # Write models from library sections
        for section in netlist.library_sections.values():
            for model in section.models:
                self._write_model(buf, model)

        # Write subcircuits
        for subckt in netlist.subcircuits:
            self._write_subcircuit(buf, subckt)

        # Write subcircuits from library sections
        for section in netlist.library_sections.values():
            for subckt in section.subcircuits:
                self._write_subcircuit(buf, subckt)

        return buf.getvalue()

    def _collect_osdi_modules(self, netlist: Netlist) -> None:
        """Collect required OSDI modules from netlist models."""
        self._osdi_modules.clear()
        self._model_modules.clear()

        # Process all models
        all_models = list(netlist.models)
        for section in netlist.library_sections.values():
            all_models.extend(section.models)

        for model in all_models:
            osdi_info = self._get_osdi_for_model(model)
            if osdi_info:
                self._osdi_modules[osdi_info.osdi_file] = osdi_info
                self._model_modules[model.name.lower()] = osdi_info

    def _get_osdi_for_model(self, model: ModelDef) -> OsdiModuleInfo | None:
        """Get OSDI module info for a model definition."""
        # Get device type info
        type_info = get_device_type_info(model.device_type)
        if type_info:
            family = type_info.family
        else:
            # Use device type as family if not in registry
            family = model.device_type

        # Get OSDI module for family/level/version
        return get_osdi_module(family, model.level, model.version)

    def _write_model(self, buf: TextIO, model: ModelDef, indent: int = 0) -> None:
        """Write a model definition."""
        prefix = " " * indent

        # Get OSDI module info
        osdi_info = self._get_osdi_for_model(model)
        if osdi_info:
            module_name = osdi_info.module_name
        else:
            # Fallback: use device type as module name
            module_name = f"sp_{model.device_type}"

        # Get device type info for parameter processing
        type_info = get_device_type_info(model.device_type)

        # Model statement: model <module_name> <model_name>
        buf.write(f"{prefix}model {module_name} {model.name}\n")

        # Build parameters, potentially adding extra params and removing level/version
        params = dict(model.parameters)

        # Add extra parameters from device type
        if type_info and type_info.extra_params:
            params.update(type_info.extra_params)

        # Remove level and version if specified by device type
        if type_info:
            if type_info.remove_level:
                params.pop("level", None)
            if type_info.remove_version:
                params.pop("version", None)

        # Add extra parameters from OSDI module
        if osdi_info and osdi_info.extra_params:
            params.update(osdi_info.extra_params)

        # Write parameters
        if params:
            params_str = self._format_params(params)
            buf.write(f"{prefix}    parameters {params_str}\n")

        buf.write("\n")

    def _write_subcircuit(self, buf: TextIO, subckt: Subcircuit, indent: int = 0) -> None:
        """Write a subcircuit definition."""
        prefix = " " * indent

        # Process port names
        ports = process_terminals(subckt.ports)
        ports_str = " ".join(ports)

        # Subcircuit header: subckt name (port1 port2 ...)
        buf.write(f"{prefix}subckt {subckt.name} ({ports_str})\n")

        # Default parameters
        if subckt.parameters:
            for name, value in subckt.parameters.items():
                formatted_value = format_value(str(value))
                buf.write(f"{prefix}    parameters {name}={formatted_value}\n")

        # Nested models
        for model in subckt.models:
            self._write_model(buf, model, indent + 4)

        # Instances
        for inst in subckt.instances:
            self._write_instance(buf, inst, indent + 4)

        buf.write(f"{prefix}ends {subckt.name}\n\n")

    def _write_instance(self, buf: TextIO, inst: Instance, indent: int = 0) -> None:
        """Write a device instance."""
        prefix = " " * indent

        # Process node names
        nodes = process_terminals(inst.nodes)
        nodes_str = " ".join(nodes)

        # Determine the model/type to use
        if inst.model_name:
            type_str = inst.model_name
        else:
            # Check if this is a default model type (R, C, L)
            prefix_char = inst.name[0].lower() if inst.name else ""
            default_osdi = get_default_model(prefix_char)
            if default_osdi:
                # Generate default model name
                type_str = f"{self.default_model_prefix}{prefix_char}"
            else:
                type_str = inst.device_type

        # Build instance line: name (node1 node2 ...) model_or_type
        line = f"{prefix}{inst.name} ({nodes_str}) {type_str}"

        # Add parameters
        if inst.parameters:
            params_str = self._format_params(inst.parameters)
            line += f" {params_str}"

        buf.write(line + "\n")

    def _format_params(self, params: dict[str, Any]) -> str:
        """Format parameters for output."""
        parts = []
        for name, value in params.items():
            if value is not None:
                formatted = format_value(str(value))
                parts.append(f"{name}={formatted}")
        return " ".join(parts)
