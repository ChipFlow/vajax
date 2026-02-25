# SPDX-FileCopyrightText: 2025 ChipFlow
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""VACASK output writer.

This package generates VACASK-format output files from parsed SPICE netlists.
VACASK uses a Spectre-like syntax with parentheses around node lists and
`load` directives for loading OSDI compiled Verilog-A modules.

Output format example:
    // Converted by netlist_converter converter
    load "spice/resistor.osdi"

    model sp_resistor res1
    parameters r=1k tc1=0 tc2=0

    subckt inverter (in out vdd vss)
    parameters wp=1u wn=500n
        m1 (out in vdd vdd) pmos w=wp l=100n
        m2 (out in vss vss) nmos w=wn l=100n
    ends inverter
"""

from .writer import VacaskWriter, write_vacask

__all__ = [
    "VacaskWriter",
    "write_vacask",
]
