# SPDX-FileCopyrightText: 2025 ChipFlow
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for dialect base class and registry."""

import pytest

from vajax.netlist_converter.spiceparser.dialect import (
    SpiceDialect,
    detect_dialect,
    get_dialect,
    register_dialect,
)


class TestDialectRegistry:
    """Tests for dialect registration and lookup."""

    def test_get_ngspice_dialect(self):
        """Ngspice dialect should be registered."""
        dialect = get_dialect("ngspice")
        assert dialect.name == "ngspice"

    def test_get_dialect_case_insensitive(self):
        """Dialect lookup should be case-insensitive."""
        dialect1 = get_dialect("ngspice")
        dialect2 = get_dialect("NGSPICE")
        dialect3 = get_dialect("NgSpice")
        assert dialect1.name == dialect2.name == dialect3.name

    def test_get_unknown_dialect_raises(self):
        """Looking up unknown dialect should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown dialect"):
            get_dialect("unknown_dialect")

    def test_register_dialect_decorator(self):
        """Test registering a custom dialect."""

        @register_dialect("test_dialect")
        class TestDialect(SpiceDialect):
            @property
            def name(self) -> str:
                return "test"

            def parse_include(self, line):
                return None

            def parse_library(self, line):
                return None

            def parse_model_type(self, type_str):
                return type_str.lower()

            def get_device_prefix_map(self):
                return {}

            def parse_parameter_value(self, value_str):
                return value_str

        dialect = get_dialect("test_dialect")
        assert dialect.name == "test"


class TestNgspiceDialect:
    """Tests for Ngspice dialect implementation."""

    @pytest.fixture
    def dialect(self):
        return get_dialect("ngspice")

    def test_parse_include_quoted(self, dialect):
        """Parse .include with quoted path."""
        result = dialect.parse_include('.include "path/to/file.spi"')
        assert result == ("path/to/file.spi", None)

    def test_parse_include_single_quoted(self, dialect):
        """Parse .include with single-quoted path."""
        result = dialect.parse_include(".include 'path/to/file.spi'")
        assert result == ("path/to/file.spi", None)

    def test_parse_include_unquoted(self, dialect):
        """Parse .include with unquoted path."""
        result = dialect.parse_include(".include path/to/file.spi")
        assert result == ("path/to/file.spi", None)

    def test_parse_library_with_section(self, dialect):
        """Parse .lib with file and section."""
        result = dialect.parse_library('.lib "models.lib" typical')
        assert result == ("models.lib", "typical")

    def test_parse_library_section_only(self, dialect):
        """Parse .lib that starts a section (no file path)."""
        result = dialect.parse_library(".lib typical")
        assert result is None  # Not a file include

    def test_parse_model_type_nmos(self, dialect):
        """Model type parsing normalizes case."""
        assert dialect.parse_model_type("NMOS") == "nmos"
        assert dialect.parse_model_type("nmos") == "nmos"
        assert dialect.parse_model_type("NMos") == "nmos"

    def test_device_prefix_map(self, dialect):
        """Device prefix map includes standard SPICE prefixes."""
        prefix_map = dialect.get_device_prefix_map()
        assert prefix_map["R"] == "resistor"
        assert prefix_map["C"] == "capacitor"
        assert prefix_map["M"] == "mosfet"
        assert prefix_map["Q"] == "bjt"
        assert prefix_map["D"] == "diode"
        assert prefix_map["X"] == "subcircuit"

    def test_parse_si_prefix_k(self, dialect):
        """Parse value with k (kilo) prefix."""
        assert dialect.parse_parameter_value("1k") == 1000.0

    def test_parse_si_prefix_meg(self, dialect):
        """Parse value with meg (mega) prefix."""
        assert dialect.parse_parameter_value("10meg") == 10e6

    def test_parse_si_prefix_u(self, dialect):
        """Parse value with u (micro) prefix."""
        assert dialect.parse_parameter_value("1u") == 1e-6

    def test_parse_si_prefix_n(self, dialect):
        """Parse value with n (nano) prefix."""
        assert dialect.parse_parameter_value("100n") == pytest.approx(100e-9)

    def test_parse_plain_number(self, dialect):
        """Parse plain numeric value."""
        assert dialect.parse_parameter_value("42") == 42
        assert dialect.parse_parameter_value("3.14") == 3.14
        assert dialect.parse_parameter_value("1e-3") == 0.001

    def test_parse_expression(self, dialect):
        """Parse curly brace expression."""
        assert dialect.parse_parameter_value("{a+b}") == "a+b"

    def test_does_not_support_conditionals(self, dialect):
        """Ngspice doesn't support .if/.else/.endif."""
        assert dialect.supports_conditional() is False


class TestDialectDetection:
    """Tests for dialect auto-detection."""

    def test_detect_ltspice_backanno(self):
        """Detect LTSpice from .backanno directive."""
        content = """.title test
.backanno
M1 out in vdd vdd pmos
"""
        assert detect_dialect(content) == "ltspice"

    def test_detect_ltspice_rser(self):
        """Detect LTSpice from Rser parameter on passives."""
        content = """.title test
C1 out 0 1u Rser=10
"""
        assert detect_dialect(content) == "ltspice"

    def test_detect_ltspice_micro_unicode(self):
        """Detect LTSpice from Unicode micro symbol."""
        content = """.title test
C1 out 0 1µF
"""
        assert detect_dialect(content) == "ltspice"

    def test_detect_ltspice_windows_path(self):
        """Detect LTSpice from Windows-style paths."""
        content = r""".title test
.lib C:\Users\test\models.lib
"""
        assert detect_dialect(content) == "ltspice"

    def test_detect_hspice_conditionals(self):
        """Detect HSPICE from .if/.endif conditionals."""
        content = """.title test
.if (cond == 1)
.param x=1
.endif
"""
        assert detect_dialect(content) == "hspice"

    def test_detect_hspice_alter(self):
        """Detect HSPICE from .ALTER statement."""
        content = """.title test
.op
.alter
.temp 85
.end
"""
        assert detect_dialect(content) == "hspice"

    def test_detect_hspice_prot(self):
        """Detect HSPICE from .prot/.unprot encryption wrappers."""
        content = """.title test
.prot
* encrypted content
.unprot
"""
        assert detect_dialect(content) == "hspice"

    def test_detect_hspice_quoted_expression(self):
        """Detect HSPICE from single-quoted expressions."""
        content = """.title test
.param x='1+y*2'
"""
        assert detect_dialect(content) == "hspice"

    def test_detect_ngspice_default(self):
        """Default to Ngspice when no specific patterns found."""
        content = """.title test
R1 in out 1k
M1 out in vdd vdd pmos
.end
"""
        assert detect_dialect(content) == "ngspice"

    def test_detect_ngspice_control_block(self):
        """Ngspice control blocks don't trigger false positives."""
        content = """.title test
.control
op
print all
.endc
"""
        assert detect_dialect(content) == "ngspice"

    def test_detect_prefers_higher_score(self):
        """When multiple dialects have indicators, prefer higher score."""
        # Content with weak HSPICE indicator (spaces around =)
        # but no strong HSPICE patterns - should default to ngspice
        content = """.title test
.param x = 1
R1 in out 1k
"""
        # Spaces around = gives HSPICE only 1 point, below threshold of 2
        assert detect_dialect(content) == "ngspice"
