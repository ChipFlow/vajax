# SPDX-FileCopyrightText: 2025 ChipFlow
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""SPICE netlist parser with multi-dialect support.

This package provides a unified interface for parsing SPICE netlists from
different simulators (Ngspice, HSPICE, LTSpice), extracting models and
subcircuits for conversion to Verilog-A.

Supported dialects:
    - ngspice: Ngspice simulator netlists (default)
    - hspice: Synopsys HSPICE netlists with .if/.endif, .ALTER support
    - ltspice: LTSpice netlists with Rser/Lser/Rpar/Cpar parameters

Features:
    - Multi-dialect parsing with automatic dialect detection
    - Device type mapping (NMOS, PMOS, NPN, PNP, etc.)
    - OSDI module resolution for Verilog-A conversion
    - SI prefix conversion (k, meg, m, u, n, p, f)
    - Parameter expression handling
    - Include/library directive processing

Basic usage:
    from spiceparser import parse_netlist

    netlist = parse_netlist("circuit.sp", dialect="ngspice")

Auto-detection:
    from spiceparser import detect_dialect_from_file, parse_netlist

    dialect = detect_dialect_from_file("circuit.sp")
    netlist = parse_netlist("circuit.sp", dialect=dialect)

Device type lookup:
    from spiceparser import get_device_type_info, get_osdi_module

    info = get_device_type_info("nmos")
    # DeviceTypeInfo(family='mos', remove_level=True, remove_version=True, ...)

    osdi = get_osdi_module("mos", level=54)
    # OsdiModuleInfo(osdi_file='bsim4.osdi', module_name='bsim4', ...)
"""

from .dialect import (
    SpiceDialect,
    detect_dialect,
    detect_dialect_from_file,
    get_dialect,
    register_dialect,
)

# Import dialects to register them
from .dialects import ngspice as _ngspice  # noqa: F401
from .elements import (
    DEVICE_TYPES,
    OSDI_MODULES,
    DeviceTypeInfo,
    OsdiModuleInfo,
    get_default_model,
    get_device_type_info,
    get_osdi_module,
)
from .netlist import Instance, ModelDef, Netlist, Subcircuit
from .parser import parse_netlist

__all__ = [
    "SpiceDialect",
    "get_dialect",
    "register_dialect",
    "detect_dialect",
    "detect_dialect_from_file",
    "ModelDef",
    "Subcircuit",
    "Instance",
    "Netlist",
    "parse_netlist",
    "DeviceTypeInfo",
    "OsdiModuleInfo",
    "get_device_type_info",
    "get_osdi_module",
    "get_default_model",
    "DEVICE_TYPES",
    "OSDI_MODULES",
]
