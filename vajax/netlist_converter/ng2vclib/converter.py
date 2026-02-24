# SPDX-FileCopyrightText: 2025 ChipFlow
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""SPICE to VACASK netlist converter.

This module provides the Converter class for converting SPICE netlists
to VACASK format. It supports multiple SPICE dialects through the
spiceparser.dialect system.
"""

import os

# Import dialect support from ..spiceparser
from ..spiceparser.dialect import SpiceDialect, get_dialect
from . import dfl
from .m_devices import DevicesMixin
from .m_file import FileLoaderMixin
from .m_inst_d import InstanceDMixin
from .m_inst_n import InstanceNMixin
from .m_inst_passive import InstancePassiveMixin
from .m_inst_q import InstanceQMixin
from .m_inst_v import InstanceVMixin
from .m_inst_x import InstanceXMixin
from .m_masters import MastersMixin
from .m_model import ModelMixin
from .m_output import OutputMixin
from .m_params import ParamsMixin


class Converter(
    FileLoaderMixin,
    MastersMixin,
    OutputMixin,
    ModelMixin,
    InstancePassiveMixin,
    InstanceDMixin,
    InstanceNMixin,
    InstanceQMixin,
    InstanceVMixin,
    InstanceXMixin,
    DevicesMixin,
    ParamsMixin,
):
    """SPICE to VACASK netlist converter.

    Supports multiple SPICE dialects (Ngspice, HSPICE, LTSpice) through the
    dialect parameter.

    Args:
        cfg: Configuration dictionary (see below for keys)
        dialect: SPICE dialect name ("ngspice", "hspice", "ltspice") or
                 SpiceDialect instance. Default is "ngspice".
        indent: Debug indentation level
        debug: Debug verbosity level

    Configuration dictionary keys:
        sourcepath: List of directories for .include and .lib files
        type_map: Model type definitions (SPICE name -> params, family, etc.)
        family_map: Device family to OSDI module mapping
        default_models: Default models for passive elements
        default_model_prefix: Prefix for default model names
        as_toplevel: How to treat input file ("auto", "yes", "no")
        all_models: If True, write all models (not just used ones)
        original_case_subckt: Preserve case of subcircuit names
        original_case_model: Preserve case of model names
        original_case_instance: Preserve case of instance names
        read_depth: Max depth for reading includes (None = unlimited)
        process_depth: Max depth for processing (None = unlimited)
        output_depth: Max depth for output (None = unlimited)
        patch: Dictionary of file patches
        columns: Max columns per line

    Data dictionary (populated during conversion):
        title: First line of toplevel file
        control: Control block lines
        models: Available models by subcircuit
        model_usage: Where models are used
        default_models_needed: Set of device letters needing defaults
        osdi_loads: OSDI files loaded via pre_osdi
        subckts: Subcircuit definitions
        is_toplevel: Whether input is a toplevel netlist
    """

    def __init__(self, cfg=None, dialect="ngspice", indent=4, debug=0):
        # If no config is given, use default config
        if cfg is None:
            cfg = dfl.default_config()
        self.cfg = cfg

        # Set up dialect
        if isinstance(dialect, str):
            self.dialect = get_dialect(dialect)
        elif isinstance(dialect, SpiceDialect):
            self.dialect = dialect
        else:
            raise ValueError(f"Invalid dialect: {dialect}")

        # Store dialect name in config for reference
        self.cfg["dialect"] = self.dialect.name

        self.data = {
            "control": [],
            "models": {},
            "subckts": {},
            "model_usage": {},
            "default_models_needed": set(),
            "osdi_loads": set(),
        }
        self.dbgindent = indent
        self.debug = debug

    def convert(self, fromFile, toFile=None):
        _, deck, canonical_file_path = self.read_file(fromFile)
        self.data["deck"] = deck
        self.data["canonical_input_path"] = canonical_file_path

        self.collect_masters()

        out = self.vacask_file()

        if toFile is None:
            for l in out:
                print(l)
        else:
            if not os.path.isabs(toFile):
                # Relative path, get directory of input file
                dest = os.path.join(os.path.dirname(self.data["canonical_input_path"]), toFile)
            else:
                dest = toFile

            # Create destination directory
            destdir = os.path.dirname(dest)
            os.makedirs(destdir, exist_ok=True)

            with open(dest, "w") as f:
                for l in out:
                    f.write(l)
                    f.write("\n")
