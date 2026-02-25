# SPDX-FileCopyrightText: 2025 ChipFlow
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for LTSpice dialect implementation."""

from pathlib import Path

import pytest

from vajax.netlist_converter.spiceparser import parse_netlist
from vajax.netlist_converter.spiceparser.dialect import get_dialect
from vajax.netlist_converter.spiceparser.dialects.ltspice import LtspiceDialect


class TestLtspiceDialectRegistry:
    """Tests for LTSpice dialect registration."""

    def test_ltspice_dialect_registered(self):
        """LTSpice dialect should be registered."""
        dialect = get_dialect("ltspice")
        assert dialect.name == "ltspice"
        assert isinstance(dialect, LtspiceDialect)

    def test_case_insensitive(self):
        """Dialect lookup should be case-insensitive."""
        assert get_dialect("LTSPICE").name == "ltspice"
        assert get_dialect("LTspice").name == "ltspice"


class TestLtspiceIncludeDirectives:
    """Tests for LTSpice include directive parsing."""

    @pytest.fixture
    def dialect(self):
        return LtspiceDialect()

    def test_parse_include_unquoted(self, dialect):
        """Parse .include with unquoted path."""
        result = dialect.parse_include(".include models.lib")
        assert result == ("models.lib", None)

    def test_parse_include_quoted(self, dialect):
        """Parse .include with quoted path."""
        result = dialect.parse_include('.include "models/file.lib"')
        assert result == ("models/file.lib", None)

    def test_parse_include_windows_path(self, dialect):
        """Parse .include with Windows-style path."""
        result = dialect.parse_include(r".include C:\users\lib\standard.dio")
        # Backslashes should be converted to forward slashes
        assert result == ("C:/users/lib/standard.dio", None)

    def test_parse_inc_shorthand(self, dialect):
        """Parse .inc shorthand."""
        result = dialect.parse_include(".inc file.mod")
        assert result == ("file.mod", None)


class TestLtspiceLibraryDirectives:
    """Tests for LTSpice library directive parsing."""

    @pytest.fixture
    def dialect(self):
        return LtspiceDialect()

    def test_parse_lib_file_only(self, dialect):
        """Parse .lib with just filename."""
        result = dialect.parse_library(".lib LTC.lib")
        assert result == ("LTC.lib", "")

    def test_parse_lib_with_section(self, dialect):
        """Parse .lib with file and section."""
        result = dialect.parse_library(".lib models.lib typical")
        assert result == ("models.lib", "typical")

    def test_parse_lib_windows_path(self, dialect):
        """Parse .lib with Windows-style path."""
        result = dialect.parse_library(
            r".lib C:\users\brian\My Documents\LTspiceXVII\lib\cmp\standard.dio"
        )
        assert result is not None
        assert result[0] == "C:/users/brian/My"  # Splits on space
        # Note: This is a known limitation - spaces in unquoted paths

    def test_parse_lib_quoted_with_spaces(self, dialect):
        """Parse .lib with quoted path containing spaces."""
        result = dialect.parse_library('.lib "C:/users/brian/My Documents/standard.dio"')
        assert result == ("C:/users/brian/My Documents/standard.dio", "")

    def test_parse_lib_section_only(self, dialect):
        """Parse .lib that starts a section (no file extension)."""
        result = dialect.parse_library(".lib typical")
        assert result is None  # Section start, not file include


class TestLtspiceParameterParsing:
    """Tests for LTSpice parameter value parsing."""

    @pytest.fixture
    def dialect(self):
        return LtspiceDialect()

    def test_parse_si_prefix_standard(self, dialect):
        """Parse standard SI prefixes."""
        assert dialect.parse_parameter_value("1k") == 1000.0
        assert dialect.parse_parameter_value("10meg") == 10e6
        assert dialect.parse_parameter_value("2.8n") == pytest.approx(2.8e-9)

    def test_parse_si_prefix_micro_unicode(self, dialect):
        """Parse µ (Unicode micro) prefix."""
        assert dialect.parse_parameter_value("10µ") == pytest.approx(10e-6)
        assert dialect.parse_parameter_value("100µ") == pytest.approx(100e-6)

    def test_parse_braced_expression(self, dialect):
        """Parse {expression} format."""
        result = dialect.parse_parameter_value("{1+x*3}")
        assert result == "1+x*3"

    def test_parse_plain_number(self, dialect):
        """Parse plain numeric value."""
        assert dialect.parse_parameter_value("121") == 121
        assert dialect.parse_parameter_value("0.1") == 0.1


class TestLtspiceExtraElementParams:
    """Tests for LTSpice extra element parameters."""

    @pytest.fixture
    def dialect(self):
        return LtspiceDialect()

    def test_capacitor_extra_params(self, dialect):
        """Capacitor should have Rser, Lser, Rpar, Cpar."""
        params = dialect.get_extra_element_params("c")
        assert "Rser" in params
        assert "Lser" in params
        assert "Rpar" in params
        assert "Cpar" in params

    def test_inductor_extra_params(self, dialect):
        """Inductor should have Rser, Rpar, Cpar."""
        params = dialect.get_extra_element_params("l")
        assert "Rser" in params
        assert "Rpar" in params
        assert "Cpar" in params

    def test_vsource_extra_params(self, dialect):
        """Voltage source should have Rser."""
        params = dialect.get_extra_element_params("v")
        assert "Rser" in params

    def test_resistor_no_extra_params(self, dialect):
        """Resistor should have no extra params."""
        params = dialect.get_extra_element_params("r")
        assert params == []


class TestLtspiceBackanno:
    """Tests for LTSpice .backanno directive."""

    @pytest.fixture
    def dialect(self):
        return LtspiceDialect()

    def test_is_backanno(self, dialect):
        """Recognize .backanno directive."""
        assert dialect.is_backanno_directive(".backanno") is True
        assert dialect.is_backanno_directive(".BACKANNO") is True
        assert dialect.is_backanno_directive("  .backanno  ") is True

    def test_is_not_backanno(self, dialect):
        """Reject non-.backanno lines."""
        assert dialect.is_backanno_directive(".lib") is False
        assert dialect.is_backanno_directive(".end") is False


class TestLtspiceConditionals:
    """Tests for LTSpice conditional support."""

    @pytest.fixture
    def dialect(self):
        return LtspiceDialect()

    def test_does_not_support_conditionals(self, dialect):
        """LTSpice should not support conditionals."""
        assert dialect.supports_conditional() is False

    def test_parse_conditional_returns_none(self, dialect):
        """parse_conditional should always return None."""
        assert dialect.parse_conditional(".if (x == 1)") is None


class TestLtspiceNetlistParsing:
    """Tests for parsing complete LTSpice netlists."""

    def test_parse_simple_netlist(self):
        """Parse a simple LTSpice netlist."""
        content = """\
* Simple LTSpice circuit
V1 +V 0 15
R1 N002 N001 10K
C1 OUT 0 10µ
.model D D
.end
"""
        netlist = parse_netlist(content, dialect="ltspice")

        assert netlist.title == "* Simple LTSpice circuit"
        assert len(netlist.models) == 1
        assert netlist.models[0].name == "D"

    def test_parse_with_rser_params(self):
        """Parse netlist with Rser parameters."""
        content = """\
* LTSpice with parasitics
V1 IN 0 PWL(0 0 1 10) Rser=0.1
C1 N003 N004 10n Rser=.1
.end
"""
        netlist = parse_netlist(content, dialect="ltspice")

        # Should parse instances with Rser parameters
        assert len(netlist.instances) == 2

    def test_parse_with_lib_reference(self):
        """Parse netlist with .lib reference."""
        content = """\
* LTSpice with library
R1 OUT N001 121
.lib LTC.lib
.backanno
.end
"""
        netlist = parse_netlist(content, dialect="ltspice")

        # Should recognize lib reference (though not actually load it)
        assert len(netlist.instances) == 1


class TestLtspiceRealFiles:
    """Tests with real LTSpice files from spice-datasets."""

    @pytest.fixture
    def ltspice_examples_path(self):
        """Get path to LTSpice examples."""
        path = (
            Path(__file__).parent.parent.parent.parent
            / "VADistiller/tests/fixtures/spice-datasets/ltspice_examples"
        )
        if not path.exists():
            pytest.skip("LTSpice examples not available")
        return path

    def test_parse_1001_net(self, ltspice_examples_path):
        """Parse 1001.net - Precision Absolute Value Circuit."""
        netfile = ltspice_examples_path / "1001.net"
        if not netfile.exists():
            pytest.skip("1001.net not found")

        netlist = parse_netlist(netfile, dialect="ltspice")

        # Should have instances and models
        assert len(netlist.instances) > 0
        assert netlist.title is not None

    def test_parse_1083_net(self, ltspice_examples_path):
        """Parse 1083.net - circuit with Rser on voltage source."""
        netfile = ltspice_examples_path / "1083.net"
        if not netfile.exists():
            pytest.skip("1083.net not found")

        netlist = parse_netlist(netfile, dialect="ltspice")

        # Should parse successfully
        assert len(netlist.instances) > 0
