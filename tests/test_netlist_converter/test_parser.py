# SPDX-FileCopyrightText: 2025 ChipFlow
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the unified SPICE parser."""

import pytest

from vajax.netlist_converter.spiceparser import parse_netlist


class TestBasicParsing:
    """Basic parsing tests."""

    def test_parse_title(self, simple_ngspice_netlist):
        """First line should be parsed as title."""
        netlist = parse_netlist(simple_ngspice_netlist)
        assert netlist.title == "* Simple RC circuit"

    def test_parse_model_definition(self, simple_ngspice_netlist):
        """Parse a .model definition."""
        netlist = parse_netlist(simple_ngspice_netlist)

        assert len(netlist.models) == 1
        model = netlist.models[0]
        assert model.name == "mydiode"
        assert model.device_type == "d"
        assert "is" in model.parameters
        assert "n" in model.parameters

    def test_parse_instances(self, simple_ngspice_netlist):
        """Parse device instances."""
        netlist = parse_netlist(simple_ngspice_netlist)

        # Should have R1, C1, D1
        assert len(netlist.instances) == 3

        # Check resistor
        r1 = next(i for i in netlist.instances if i.name == "R1")
        assert r1.device_type == "resistor"

        # Check capacitor
        c1 = next(i for i in netlist.instances if i.name == "C1")
        assert c1.device_type == "capacitor"

        # Check diode
        d1 = next(i for i in netlist.instances if i.name == "D1")
        assert d1.device_type == "diode"


class TestSubcircuitParsing:
    """Tests for subcircuit parsing."""

    def test_parse_subcircuit_definition(self, ngspice_subcircuit):
        """Parse a .subckt definition."""
        netlist = parse_netlist(ngspice_subcircuit)

        assert len(netlist.subcircuits) == 1
        subckt = netlist.subcircuits[0]
        assert subckt.name == "inverter"
        assert subckt.ports == ["in", "out", "vdd", "vss"]

    def test_parse_subcircuit_parameters(self, ngspice_subcircuit):
        """Parse subcircuit default parameters."""
        netlist = parse_netlist(ngspice_subcircuit)

        subckt = netlist.subcircuits[0]
        assert "wp" in subckt.parameters
        assert "wn" in subckt.parameters

    def test_parse_subcircuit_instances(self, ngspice_subcircuit):
        """Parse instances within subcircuit."""
        netlist = parse_netlist(ngspice_subcircuit)

        subckt = netlist.subcircuits[0]
        assert len(subckt.instances) == 2

        m1 = next(i for i in subckt.instances if i.name == "M1")
        assert m1.device_type == "mosfet"

    def test_parse_subcircuit_call(self, ngspice_subcircuit):
        """Parse subcircuit instance (X-element)."""
        netlist = parse_netlist(ngspice_subcircuit)

        # Find X1 at top level
        x1 = next((i for i in netlist.instances if i.name == "X1"), None)
        assert x1 is not None
        assert x1.device_type == "subcircuit"


class TestLibrarySections:
    """Tests for library section parsing."""

    def test_parse_lib_sections(self, hspice_library_section):
        """Parse HSPICE-style library sections."""
        netlist = parse_netlist(hspice_library_section)

        # Should have 3 sections: typical, ff, ss
        assert len(netlist.library_sections) == 3
        assert "typical" in netlist.library_sections
        assert "ff" in netlist.library_sections
        assert "ss" in netlist.library_sections

    def test_models_in_lib_section(self, hspice_library_section):
        """Models should be associated with their library section."""
        netlist = parse_netlist(hspice_library_section)

        typical = netlist.library_sections["typical"]
        assert len(typical.models) == 1
        assert typical.models[0].name == "nmos_typ"


class TestContinuationLines:
    """Tests for line continuation handling."""

    def test_continuation_with_plus(self):
        """Lines starting with + should be joined."""
        content = """\
* Test continuation
.model mymos nmos level=54
+ vth0=0.5
+ tox=1.8e-9
.end
"""
        netlist = parse_netlist(content)

        assert len(netlist.models) == 1
        model = netlist.models[0]
        assert model.level == 54
        assert "vth0" in model.parameters
        assert "tox" in model.parameters


class TestRealWorldParsing:
    """Tests with real PDK files (skipped if not available)."""

    def test_parse_gf130_wrapper(self, gf130_hspice_path):
        """Parse GF130BCD HSPICE wrapper file."""
        if gf130_hspice_path is None:
            pytest.skip("GF130 PDK not available")

        wrapper_file = gf130_hspice_path / "130BCD_wrapper.hspice"
        if not wrapper_file.exists():
            pytest.skip("Wrapper file not found")

        netlist = parse_netlist(wrapper_file)

        # Should have library sections for different corners
        assert "typical" in netlist.library_sections or len(netlist.library_sections) > 0
