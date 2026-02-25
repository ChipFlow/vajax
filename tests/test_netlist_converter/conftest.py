# SPDX-FileCopyrightText: 2025 ChipFlow
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Pytest configuration and fixtures for spiceparser tests."""

from pathlib import Path

import pytest

# Test data directories
TESTS_DIR = Path(__file__).parent
FIXTURES_DIR = TESTS_DIR / "fixtures"

# PDK paths for real-world test data
PDK_BASE = Path.home() / "Code/ChipFlow/PDK"
GF130_HSPICE = PDK_BASE / "pdk-gf130/gf130bcd.pdk/Models/HSPICE"
SKYWATER_MODELS = PDK_BASE / "skywater-pdk/libraries/sky130_fd_pr/latest/models"


@pytest.fixture
def simple_ngspice_netlist() -> str:
    """A simple Ngspice netlist for basic parsing tests."""
    return """\
* Simple RC circuit
.param res_val=1k
.param cap_val=1u

R1 in out {res_val}
C1 out 0 {cap_val}

.model mydiode d is=1e-14 n=1.05

D1 out 0 mydiode

.end
"""


@pytest.fixture
def ngspice_subcircuit() -> str:
    """Ngspice netlist with subcircuit definition."""
    return """\
* Inverter subcircuit
.subckt inverter in out vdd vss wp=1u wn=500n
M1 out in vdd vdd pmos w=wp l=100n
M2 out in vss vss nmos w=wn l=100n
.ends inverter

.model pmos pmos level=1 vto=-0.7
.model nmos nmos level=1 vto=0.7

X1 input output VDD GND inverter wp=2u wn=1u

.end
"""


@pytest.fixture
def hspice_library_section() -> str:
    """HSPICE library with section definitions."""
    return """\
* HSPICE library example
.lib typical
.model nmos_typ nmos level=54 version=4.5
+ tnom=27 toxe=1.8e-9 toxp=1.5e-9
+ toxm=1.8e-9 epsrox=3.9 wint=5e-9
.endl typical

.lib ff
.model nmos_ff nmos level=54 version=4.5
+ tnom=27 toxe=1.7e-9 toxp=1.4e-9
+ toxm=1.7e-9 epsrox=3.9 wint=4e-9
.endl ff

.lib ss
.model nmos_ss nmos level=54 version=4.5
+ tnom=27 toxe=1.9e-9 toxp=1.6e-9
+ toxm=1.9e-9 epsrox=3.9 wint=6e-9
.endl ss
"""


@pytest.fixture
def ltspice_passive() -> str:
    """LTSpice netlist with passive component extensions."""
    return """\
* LTSpice RC circuit with parasitics
V1 in 0 DC 1
R1 in mid 1k Rser=10
C1 mid out 1u Rser=100m Lser=1n Rpar=1meg
L1 out gnd 10u Rser=50m Rpar=100k Cpar=1p
.end
"""


@pytest.fixture
def gf130_hspice_path() -> Path | None:
    """Path to GF130BCD HSPICE models if available."""
    if GF130_HSPICE.exists():
        return GF130_HSPICE
    return None


@pytest.fixture
def skywater_models_path() -> Path | None:
    """Path to SkyWater models if available."""
    if SKYWATER_MODELS.exists():
        return SKYWATER_MODELS
    return None


@pytest.fixture
def fixtures_dir() -> Path:
    """Path to test fixtures directory."""
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    return FIXTURES_DIR


# Spectre fixtures


@pytest.fixture
def simple_spectre_netlist() -> str:
    """A simple Spectre netlist for basic parsing tests."""
    return """\
// Simple Spectre RC circuit
simulator lang=spectre

subckt rc_filter (in out vss) parameters r=1k c=1u
    R1 (in out) resistor r=r
    C1 (out vss) capacitor c=c
ends rc_filter

model resistor resistor
model capacitor capacitor
"""


@pytest.fixture
def spectre_subcircuit() -> str:
    """Spectre netlist with subcircuit and parameters."""
    return """\
// Spectre inverter subcircuit
simulator lang=spectre

subckt inverter (in out vdd vss) parameters wp=1u wn=500n l=100n
    M1 (out in vdd vdd) pch w=wp l=l
    M2 (out in vss vss) nch w=wn l=l
ends inverter

model pch pmos type=p
model nch nmos type=n

// Top-level instantiation
I0 (input output VDD VSS) inverter wp=2u wn=1u
"""


@pytest.fixture
def spectre_library_section() -> str:
    """Spectre library with section definitions."""
    return """\
simulator lang=spectre

library device_lib

section tt
model nch nmos type=n \\
    toxe=1.8e-9 vth0=0.4 \\
    k1=0.5 k2=0.0
model pch pmos type=p \\
    toxe=1.8e-9 vth0=-0.4 \\
    k1=0.5 k2=0.0
endsection tt

section ff
model nch nmos type=n \\
    toxe=1.7e-9 vth0=0.35
model pch pmos type=p \\
    toxe=1.7e-9 vth0=-0.35
endsection ff

section ss
model nch nmos type=n \\
    toxe=1.9e-9 vth0=0.45
model pch pmos type=p \\
    toxe=1.9e-9 vth0=-0.45
endsection ss

endlibrary device_lib
"""


@pytest.fixture
def spectre_mixed_dialect() -> str:
    """Spectre file with simulator lang switching."""
    return """\
// Start in Spectre mode
simulator lang=spectre

subckt inv_spectre (in out vdd vss)
    M1 (out in vdd vdd) pch w=1u l=100n
    M2 (out in vss vss) nch w=500n l=100n
ends inv_spectre

// Switch to SPICE mode for legacy content
simulator lang=spice

.subckt inv_spice in out vdd vss
M1 out in vdd vdd pch w=1u l=100n
M2 out in vss vss nch w=500n l=100n
.ends inv_spice

// Back to Spectre
simulator lang=spectre

model pch pmos type=p
model nch nmos type=n
"""
