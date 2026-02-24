"""SPICE netlist converter for VA-JAX.

This module provides tools for converting SPICE netlists from various formats
(ngspice, hspice, ltspice) to VA-JAX/VACASK .sim format.

Originally from VACASK (https://github.com/arpadbuermen/VACASK)
with modifications from https://github.com/robtaylor/VACASK/pull/2

Usage:
    python -m vajax.netlist_converter.ng2vc input.spi output.sim
"""

from .ng2vclib.converter import Converter

__all__ = ["Converter"]
