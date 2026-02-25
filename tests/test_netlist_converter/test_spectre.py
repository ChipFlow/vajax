# SPDX-FileCopyrightText: 2025 ChipFlow
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for Spectre dialect implementation."""

import pytest

from vajax.netlist_converter.spiceparser import (
    detect_dialect,
    get_dialect,
    parse_netlist,
)
from vajax.netlist_converter.spiceparser.dialects.spectre import SpectreDialect, SpectreSpiceDialect


class TestSpectreDialectRegistry:
    """Tests for Spectre dialect registration."""

    def test_spectre_dialect_registered(self):
        """Verify SpectreDialect is registered."""
        dialect = get_dialect("spectre")
        assert isinstance(dialect, SpectreDialect)
        assert dialect.name == "spectre"

    def test_spectre_spice_dialect_registered(self):
        """Verify SpectreSpiceDialect is registered."""
        dialect = get_dialect("spectre-spice")
        assert isinstance(dialect, SpectreSpiceDialect)
        assert dialect.name == "spectre-spice"

    def test_case_insensitive(self):
        """Dialect lookup should be case-insensitive."""
        assert get_dialect("SPECTRE").name == "spectre"
        assert get_dialect("Spectre").name == "spectre"
        assert get_dialect("SPECTRE-SPICE").name == "spectre-spice"


class TestSpectreIncludeDirectives:
    """Tests for Spectre include directive parsing."""

    @pytest.fixture
    def dialect(self):
        return SpectreDialect()

    def test_parse_include_spectre_style(self, dialect):
        """Parse Spectre-style include (no dot)."""
        result = dialect.parse_include('include "models.scs"')
        assert result == ("models.scs", None)

    def test_parse_include_with_section(self, dialect):
        """Parse include with section parameter."""
        result = dialect.parse_include('include "lib.scs" section=tt')
        assert result == ("lib.scs", "tt")

    def test_parse_include_spice_style(self, dialect):
        """Parse SPICE-style include (with dot) for compatibility."""
        result = dialect.parse_include('.include "models.scs"')
        assert result == ("models.scs", None)

    def test_parse_include_single_quotes(self, dialect):
        """Parse include with single quotes."""
        result = dialect.parse_include("include 'models.scs'")
        assert result == ("models.scs", None)


class TestSpectreLibrarySections:
    """Tests for Spectre library/section block parsing."""

    @pytest.fixture
    def dialect(self):
        return SpectreDialect()

    def test_parse_library_block_start(self, dialect):
        """Parse library block start."""
        result = dialect.parse_library_block("library mylib")
        assert result == ("library", "mylib")

    def test_parse_library_block_end(self, dialect):
        """Parse library block end."""
        result = dialect.parse_library_block("endlibrary mylib")
        assert result == ("endlibrary", "mylib")

    def test_parse_section_block_start(self, dialect):
        """Parse section block start."""
        result = dialect.parse_section_block("section tt")
        assert result == ("section", "tt")

    def test_parse_section_block_end(self, dialect):
        """Parse section block end."""
        result = dialect.parse_section_block("endsection tt")
        assert result == ("endsection", "tt")


class TestSpectreParameterParsing:
    """Tests for Spectre parameter value parsing."""

    @pytest.fixture
    def dialect(self):
        return SpectreDialect()

    def test_parse_si_prefix_mega(self, dialect):
        """Spectre M = mega (1e6), unlike SPICE where M = milli."""
        result = dialect.parse_parameter_value("1M")
        assert result == 1e6

    def test_parse_si_prefix_milli(self, dialect):
        """Spectre m = milli (1e-3)."""
        result = dialect.parse_parameter_value("1m")
        assert result == 1e-3

    def test_parse_si_prefix_micro(self, dialect):
        """Parse micro prefix."""
        result = dialect.parse_parameter_value("1u")
        assert result == 1e-6

    def test_parse_si_prefix_nano(self, dialect):
        """Parse nano prefix."""
        result = dialect.parse_parameter_value("100n")
        assert result == pytest.approx(1e-7)  # 100 * 1e-9 = 1e-7

    def test_parse_si_prefix_pico(self, dialect):
        """Parse pico prefix."""
        result = dialect.parse_parameter_value("10p")
        assert result == 10e-12

    def test_parse_si_prefix_kilo(self, dialect):
        """Parse kilo prefix (both K and k)."""
        assert dialect.parse_parameter_value("1K") == 1e3
        assert dialect.parse_parameter_value("1k") == 1e3

    def test_parse_plain_number(self, dialect):
        """Parse plain numeric value."""
        assert dialect.parse_parameter_value("1.5e-3") == 1.5e-3
        assert dialect.parse_parameter_value("100") == 100

    def test_parse_quoted_string(self, dialect):
        """Parse quoted string value."""
        assert dialect.parse_parameter_value('"hello"') == "hello"
        assert dialect.parse_parameter_value("'world'") == "world"


class TestSpectreKeywordDetection:
    """Tests for Spectre keyword detection."""

    @pytest.fixture
    def dialect(self):
        return SpectreDialect()

    def test_detect_subckt_keyword(self, dialect):
        """Detect subckt keyword."""
        assert dialect.is_spectre_keyword("subckt inv (in out)") == "subckt"

    def test_detect_ends_keyword(self, dialect):
        """Detect ends keyword."""
        assert dialect.is_spectre_keyword("ends inv") == "ends"

    def test_detect_model_keyword(self, dialect):
        """Detect model keyword."""
        assert dialect.is_spectre_keyword("model nch nmos") == "model"

    def test_detect_include_keyword(self, dialect):
        """Detect include keyword."""
        assert dialect.is_spectre_keyword('include "file.scs"') == "include"

    def test_detect_library_keyword(self, dialect):
        """Detect library keyword."""
        assert dialect.is_spectre_keyword("library mylib") == "library"

    def test_detect_simulator_keyword(self, dialect):
        """Detect simulator keyword."""
        assert dialect.is_spectre_keyword("simulator lang=spectre") == "simulator"

    def test_non_keyword_returns_none(self, dialect):
        """Non-keywords should return None."""
        assert dialect.is_spectre_keyword("M1 (d g s b) nch") is None


class TestSpectreDialectDetection:
    """Tests for Spectre auto-detection."""

    def test_detect_from_simulator_lang(self):
        """Detect Spectre from simulator lang directive."""
        content = """
simulator lang=spectre
subckt inv (in out vdd vss)
ends inv
"""
        assert detect_dialect(content) == "spectre"

    def test_detect_from_library_block(self):
        """Detect Spectre from library/endlibrary blocks."""
        content = """
library mylib
section tt
model nch nmos
endsection tt
endlibrary mylib
"""
        assert detect_dialect(content) == "spectre"

    def test_detect_from_subckt_without_dot(self):
        """Detect Spectre from subckt without dot prefix."""
        content = """
subckt inv (in out)
M1 (out in vdd vdd) pmos
M2 (out in vss vss) nmos
ends inv
"""
        assert detect_dialect(content) == "spectre"

    def test_detect_from_inline_comments(self):
        """Detect Spectre from // inline comments."""
        content = """
// This is a Spectre netlist
* Also has star comments
subckt test (a b)
ends test
"""
        assert detect_dialect(content) == "spectre"

    def test_detect_spice_not_spectre(self):
        """SPICE-style netlists should not be detected as Spectre."""
        content = """
* SPICE netlist
.subckt inv in out vdd vss
M1 out in vdd vdd pmos
.ends inv
"""
        # Should not be detected as spectre
        result = detect_dialect(content)
        assert result != "spectre"


class TestSpectreNetlistParsing:
    """Tests for parsing Spectre netlists."""

    def test_parse_spectre_subcircuit(self):
        """Parse Spectre-style subcircuit."""
        content = """
// Spectre inverter
simulator lang=spectre

subckt inv (in out vdd vss) parameters wp=1u wn=500n
M1 (out in vdd vdd) pmos w=wp l=100n
M2 (out in vss vss) nmos w=wn l=100n
ends inv
"""
        netlist = parse_netlist(content, dialect="spectre")

        assert len(netlist.subcircuits) == 1
        subckt = netlist.subcircuits[0]
        assert subckt.name == "inv"
        assert subckt.ports == ["in", "out", "vdd", "vss"]
        assert "wp" in subckt.parameters
        assert "wn" in subckt.parameters

    def test_parse_spectre_model(self):
        """Parse Spectre-style model definition."""
        content = """
simulator lang=spectre

model nch nmos type=n
model pch pmos type=p
"""
        netlist = parse_netlist(content, dialect="spectre")

        assert len(netlist.models) == 2
        assert netlist.models[0].name == "nch"
        assert netlist.models[0].device_type == "nmos"
        assert netlist.models[1].name == "pch"
        assert netlist.models[1].device_type == "pmos"

    def test_parse_spectre_library_section(self):
        """Parse Spectre library with sections."""
        content = """
simulator lang=spectre

library mylib

section tt
model nch_tt nmos type=n vth0=0.4
endsection tt

section ff
model nch_ff nmos type=n vth0=0.35
endsection ff

endlibrary mylib
"""
        netlist = parse_netlist(content, dialect="spectre")

        assert len(netlist.library_sections) >= 1

    def test_parse_spectre_include(self):
        """Parse Spectre include directive."""
        content = """
simulator lang=spectre

include "models.scs"
include "subcircuits.scs" section=tt

subckt test (a b)
ends test
"""
        netlist = parse_netlist(content, dialect="spectre")
        # Includes are tracked (paths may not resolve in test)
        assert len(netlist.subcircuits) == 1


class TestSpectreSpiceDialect:
    """Tests for SpectreSpiceDialect (mixed-mode)."""

    @pytest.fixture
    def dialect(self):
        return SpectreSpiceDialect()

    def test_spice_style_include(self, dialect):
        """Parse SPICE-style include with dot."""
        result = dialect.parse_include('.include "models.spi"')
        assert result == ("models.spi", None)

    def test_spectre_style_include(self, dialect):
        """Parse Spectre-style include without dot."""
        result = dialect.parse_include('include "models.scs"')
        assert result == ("models.scs", None)

    def test_comment_chars(self, dialect):
        """Both * and // are comment chars."""
        assert "*" in dialect.comment_chars
        assert "//" in dialect.comment_chars


class TestSimulatorLangParsing:
    """Tests for simulator lang= directive parsing."""

    @pytest.fixture
    def dialect(self):
        return SpectreDialect()

    def test_parse_simulator_lang_spectre(self, dialect):
        """Parse simulator lang=spectre."""
        result = dialect.parse_simulator_lang("simulator lang=spectre")
        assert result == "spectre"

    def test_parse_simulator_lang_spice(self, dialect):
        """Parse simulator lang=spice."""
        result = dialect.parse_simulator_lang("simulator lang=spice")
        assert result == "spice"

    def test_parse_simulator_lang_with_spaces(self, dialect):
        """Parse simulator lang = spectre with spaces."""
        result = dialect.parse_simulator_lang("simulator lang = spectre")
        assert result == "spectre"

    def test_non_simulator_lang_returns_none(self, dialect):
        """Non-simulator lines return None."""
        result = dialect.parse_simulator_lang("subckt test (a b)")
        assert result is None
