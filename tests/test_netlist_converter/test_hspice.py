# SPDX-FileCopyrightText: 2025 ChipFlow
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for HSPICE dialect implementation."""

from pathlib import Path

import pytest

from vajax.netlist_converter.spiceparser import parse_netlist
from vajax.netlist_converter.spiceparser.dialect import get_dialect
from vajax.netlist_converter.spiceparser.dialects.hspice import HspiceDialect


class TestHspiceDialectRegistry:
    """Tests for HSPICE dialect registration."""

    def test_hspice_dialect_registered(self):
        """HSPICE dialect should be registered."""
        dialect = get_dialect("hspice")
        assert dialect.name == "hspice"
        assert isinstance(dialect, HspiceDialect)

    def test_case_insensitive(self):
        """Dialect lookup should be case-insensitive."""
        assert get_dialect("HSPICE").name == "hspice"
        assert get_dialect("Hspice").name == "hspice"


class TestHspiceIncludeDirectives:
    """Tests for HSPICE include directive parsing."""

    @pytest.fixture
    def dialect(self):
        return HspiceDialect()

    def test_parse_inc_single_quoted(self, dialect):
        """Parse .inc with single-quoted path."""
        result = dialect.parse_include(".inc 'models/file.hspice'")
        assert result == ("models/file.hspice", None)

    def test_parse_inc_double_quoted(self, dialect):
        """Parse .inc with double-quoted path."""
        result = dialect.parse_include('.inc "models/file.hspice"')
        assert result == ("models/file.hspice", None)

    def test_parse_include_spelled_out(self, dialect):
        """Parse .include directive."""
        result = dialect.parse_include(".include 'file.spi'")
        assert result == ("file.spi", None)


class TestHspiceLibraryDirectives:
    """Tests for HSPICE library directive parsing."""

    @pytest.fixture
    def dialect(self):
        return HspiceDialect()

    def test_parse_lib_with_section(self, dialect):
        """Parse .lib with file and section."""
        result = dialect.parse_library(".lib 'models_lv.hspice' typical")
        assert result == ("models_lv.hspice", "typical")

    def test_parse_lib_double_quoted(self, dialect):
        """Parse .lib with double-quoted path."""
        result = dialect.parse_library('.lib "models.lib" ff')
        assert result == ("models.lib", "ff")

    def test_parse_lib_section_only(self, dialect):
        """Parse .lib that starts a section (no file)."""
        result = dialect.parse_library(".lib typical")
        assert result is None  # Section start, not file include

    def test_parse_lib_unquoted(self, dialect):
        """Parse .lib with unquoted path."""
        result = dialect.parse_library(".lib models.lib corner")
        assert result == ("models.lib", "corner")


class TestHspiceConditionals:
    """Tests for HSPICE conditional directive parsing."""

    @pytest.fixture
    def dialect(self):
        return HspiceDialect()

    def test_supports_conditionals(self, dialect):
        """HSPICE should support conditionals."""
        assert dialect.supports_conditional() is True

    def test_parse_if(self, dialect):
        """Parse .if directive."""
        result = dialect.parse_conditional(".if (corner == 1)")
        assert result == ("if", "corner == 1")

    def test_parse_elseif(self, dialect):
        """Parse .elseif directive."""
        result = dialect.parse_conditional(".elseif (corner == 2)")
        assert result == ("elseif", "corner == 2")

    def test_parse_else(self, dialect):
        """Parse .else directive."""
        result = dialect.parse_conditional(".else")
        assert result == ("else", None)

    def test_parse_endif(self, dialect):
        """Parse .endif directive."""
        result = dialect.parse_conditional(".endif")
        assert result == ("endif", None)

    def test_parse_if_complex_condition(self, dialect):
        """Parse .if with complex condition."""
        result = dialect.parse_conditional(".if (dio_fc_skew >= 0)")
        assert result == ("if", "dio_fc_skew >= 0")


class TestHspiceParameterParsing:
    """Tests for HSPICE parameter value parsing."""

    @pytest.fixture
    def dialect(self):
        return HspiceDialect()

    def test_parse_quoted_expression(self, dialect):
        """Parse single-quoted expression."""
        result = dialect.parse_parameter_value("'1+0.25*x/3'")
        assert result == "1+0.25*x/3"

    def test_parse_si_prefix(self, dialect):
        """Parse value with SI prefix."""
        assert dialect.parse_parameter_value("1k") == 1000.0
        assert dialect.parse_parameter_value("10meg") == 10e6
        assert dialect.parse_parameter_value("2.8e-9") == 2.8e-9

    def test_parse_plain_number(self, dialect):
        """Parse plain numeric value."""
        assert dialect.parse_parameter_value("54") == 54
        assert dialect.parse_parameter_value("4.6") == 4.6


class TestHspiceParamLine:
    """Tests for HSPICE .param line parsing."""

    @pytest.fixture
    def dialect(self):
        return HspiceDialect()

    def test_parse_simple_param(self, dialect):
        """Parse simple .param line."""
        result = dialect.parse_param_line(".param corner=1")
        assert result == {"corner": "1"}

    def test_parse_quoted_param(self, dialect):
        """Parse .param with quoted expression."""
        result = dialect.parse_param_line(".param jsa='1+0.25*x/3'")
        assert result == {"jsa": "1+0.25*x/3"}

    def test_parse_multiple_params(self, dialect):
        """Parse .param with multiple assignments."""
        result = dialect.parse_param_line(".param a=1 b='2+x' c=3")
        assert result == {"a": "1", "b": "2+x", "c": "3"}

    def test_parse_spaces_around_equals(self, dialect):
        """Parse .param with spaces around =."""
        result = dialect.parse_param_line(".param level = 54")
        assert result == {"level": "54"}


class TestHspiceNetlistParsing:
    """Tests for parsing complete HSPICE netlists."""

    def test_parse_hspice_model(self):
        """Parse HSPICE-style model definition."""
        content = """\
* HSPICE model test
.model nmos_1p5 nmos level = 54
+ version = 4.6
+ toxe = '2.8e-9+dtoxe'
+ vth0 = 0.5
.end
"""
        netlist = parse_netlist(content, dialect="hspice")

        assert len(netlist.models) == 1
        model = netlist.models[0]
        assert model.name == "nmos_1p5"
        assert model.device_type == "nmos"
        assert model.level == 54

    def test_parse_hspice_lib_sections(self):
        """Parse HSPICE library sections."""
        content = """\
* HSPICE library
.lib typical
.model nmos_typ nmos level=54
.endl typical

.lib ff
.model nmos_ff nmos level=54
.endl ff
"""
        netlist = parse_netlist(content, dialect="hspice")

        assert "typical" in netlist.library_sections
        assert "ff" in netlist.library_sections


class TestHspiceRealPDK:
    """Tests with real GF130BCD PDK files."""

    @pytest.fixture
    def gf130_path(self):
        """Get path to GF130 HSPICE models."""
        path = Path.home() / "Code/ChipFlow/PDK/pdk-gf130/gf130bcd.pdk/Models/HSPICE"
        if not path.exists():
            pytest.skip("GF130 PDK not available")
        return path

    def test_parse_gf130_wrapper(self, gf130_path):
        """Parse GF130BCD wrapper file."""
        wrapper = gf130_path / "130BCD_wrapper.hspice"
        if not wrapper.exists():
            pytest.skip("Wrapper file not found")

        netlist = parse_netlist(wrapper, dialect="hspice")

        # Should parse library sections
        assert len(netlist.library_sections) > 0 or len(netlist.includes) > 0
