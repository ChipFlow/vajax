# SPDX-FileCopyrightText: 2025 ChipFlow
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""SPICE dialect implementations.

This package contains dialect-specific parsing rules for different
SPICE simulators:

    - ngspice: Ngspice simulator (reference implementation)
    - hspice: Synopsys HSPICE (priority dialect)
    - ltspice: Analog Devices LTSpice
"""

from .hspice import HspiceDialect
from .ltspice import LtspiceDialect
from .ngspice import NgspiceDialect

__all__ = [
    "NgspiceDialect",
    "HspiceDialect",
    "LtspiceDialect",
]
