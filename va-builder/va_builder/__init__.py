"""Programmatic VACASK netlist builder.

Build VACASK-format netlists with Python loops and functions
instead of flat enumeration.

Usage:
    from va_builder import Netlist

    nl = Netlist(globals=["vdd", "vss"], ground="0")
    with nl.subckt("inv", ["out", "in"]) as s:
        s.inst("mp", "pmos", ["out", "in", "vdd", "vdd"], w="1u", l="0.2u")
        s.inst("mn", "nmos", ["out", "in", "vss", "vss"], w="0.5u", l="0.2u")
    print(nl)
"""

from va_builder.builder import Netlist, SubcktBuilder

__all__ = ["Netlist", "SubcktBuilder"]
