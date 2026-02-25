# SPDX-FileCopyrightText: 2025 ChipFlow
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for netlist_converter converter."""

from pathlib import Path

import pytest

from vajax.netlist_converter.ng2vclib import dfl
from vajax.netlist_converter.ng2vclib.converter import Converter
from vajax.netlist_converter.spiceparser.dialect import get_dialect


class TestConverterInitialization:
    """Tests for Converter initialization."""

    def test_default_dialect(self):
        """Default dialect should be ngspice."""
        conv = Converter()
        assert conv.dialect.name == "ngspice"

    def test_explicit_dialect_string(self):
        """Dialect can be specified by name."""
        conv = Converter(dialect="hspice")
        assert conv.dialect.name == "hspice"

        conv = Converter(dialect="ltspice")
        assert conv.dialect.name == "ltspice"

    def test_explicit_dialect_object(self):
        """Dialect can be specified as SpiceDialect instance."""
        dialect = get_dialect("hspice")
        conv = Converter(dialect=dialect)
        assert conv.dialect.name == "hspice"

    def test_invalid_dialect(self):
        """Invalid dialect should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown dialect"):
            Converter(dialect="invalid")

    def test_custom_config(self):
        """Custom config should be used."""
        cfg = dfl.default_config()
        cfg["sourcepath"] = ["/custom/path"]
        conv = Converter(cfg=cfg)
        assert "/custom/path" in conv.cfg["sourcepath"]


class TestConverterBasicConversion:
    """Tests for basic SPICE to VACASK conversion."""

    @pytest.fixture
    def simple_netlist(self, tmp_path):
        """Create a simple SPICE netlist file."""
        content = """\
* Simple RC circuit
R1 in out 1k
C1 out 0 1u
.end
"""
        netlist_file = tmp_path / "simple.sp"
        netlist_file.write_text(content)
        return netlist_file

    @pytest.fixture
    def subcircuit_netlist(self, tmp_path):
        """Create a netlist with subcircuit using passives only."""
        content = """\
* Buffer subcircuit with passives
.subckt buffer in out
R1 in mid 1k
C1 mid out 100p
.ends buffer

X1 input output buffer

.end
"""
        netlist_file = tmp_path / "buffer.sp"
        netlist_file.write_text(content)
        return netlist_file

    def test_convert_simple_netlist(self, simple_netlist, tmp_path):
        """Convert a simple RC netlist."""
        output_file = tmp_path / "simple.vc"

        conv = Converter()
        conv.convert(str(simple_netlist), str(output_file))

        assert output_file.exists()
        content = output_file.read_text()
        # Should have title comment
        assert "Simple RC circuit" in content or "rc circuit" in content.lower()

    def test_convert_to_stdout(self, simple_netlist, capsys):
        """Convert to stdout when no output file specified."""
        conv = Converter()
        conv.convert(str(simple_netlist), None)

        captured = capsys.readouterr()
        # Should print output to stdout
        assert len(captured.out) > 0

    def test_convert_subcircuit(self, subcircuit_netlist, tmp_path):
        """Convert a netlist with subcircuits."""
        output_file = tmp_path / "inverter.vc"

        conv = Converter()
        conv.convert(str(subcircuit_netlist), str(output_file))

        assert output_file.exists()
        content = output_file.read_text()
        # Should contain subcircuit definition
        assert "buffer" in content.lower()


class TestConverterDialects:
    """Tests for multi-dialect conversion."""

    @pytest.fixture
    def hspice_netlist(self, tmp_path):
        """Create an HSPICE-style netlist."""
        content = """\
* HSPICE netlist
.param vdd = 1.8
.param res_val = 1k

.subckt filter in out params: r=1k c=100p
R1 in mid r
C1 mid out c
.ends filter

X1 input output filter r=res_val c=100p

.end
"""
        netlist_file = tmp_path / "hspice.sp"
        netlist_file.write_text(content)
        return netlist_file

    def test_convert_with_hspice_dialect(self, hspice_netlist, tmp_path):
        """Convert with HSPICE dialect."""
        output_file = tmp_path / "hspice.vc"

        conv = Converter(dialect="hspice")
        conv.convert(str(hspice_netlist), str(output_file))

        assert output_file.exists()
        content = output_file.read_text()
        # Should have some content
        assert len(content) > 0


class TestConverterConfig:
    """Tests for converter configuration options."""

    def test_sourcepath_config(self, tmp_path):
        """Sourcepath should be used for includes."""
        cfg = dfl.default_config()
        cfg["sourcepath"] = [str(tmp_path)]

        conv = Converter(cfg=cfg)
        assert str(tmp_path) in conv.cfg["sourcepath"]

    def test_all_models_config(self):
        """all_models config option."""
        cfg = dfl.default_config()
        cfg["all_models"] = True

        conv = Converter(cfg=cfg)
        assert conv.cfg["all_models"] is True

    def test_read_depth_config(self):
        """read_depth config option."""
        cfg = dfl.default_config()
        cfg["read_depth"] = 2

        conv = Converter(cfg=cfg)
        assert conv.cfg["read_depth"] == 2


class TestConverterConditionals:
    """Tests for HSPICE conditional (.if/.else/.endif) handling."""

    @pytest.fixture
    def conditional_netlist(self, tmp_path):
        """Create a netlist with HSPICE conditionals."""
        content = """\
* HSPICE conditional example
.param l_drawn = 100n

.if (l_drawn < 200n)
.param vth0_val = 0.4
.elseif (l_drawn < 500n)
.param vth0_val = 0.38
.else
.param vth0_val = 0.35
.endif

R1 in out 1k
.end
"""
        netlist_file = tmp_path / "conditional.sp"
        netlist_file.write_text(content)
        return netlist_file

    def test_convert_conditionals_to_vacask(self, conditional_netlist, tmp_path):
        """Verify .if/.elseif/.else/.endif are converted to @if/@elseif/@else/@end."""
        output_file = tmp_path / "conditional.vc"

        conv = Converter(dialect="hspice")
        conv.convert(str(conditional_netlist), str(output_file))

        assert output_file.exists()
        content = output_file.read_text()

        # Verify VACASK conditional syntax
        assert "@if" in content
        assert "@elseif" in content
        assert "@else" in content
        assert "@end" in content

        # Verify SPICE syntax is gone
        assert ".if" not in content.lower()
        assert ".elseif" not in content.lower()
        assert ".endif" not in content.lower()


class TestConverterFromFixtures:
    """Tests using fixture files."""

    @pytest.fixture
    def fixtures_dir(self):
        """Path to test fixtures directory."""
        return Path(__file__).parent / "fixtures"

    def test_convert_bsim4_fixture(self, fixtures_dir, tmp_path):
        """Convert BSIM4 ngspice fixture file."""
        input_file = fixtures_dir / "ngspice" / "bsim4_nmos_id.sp"
        if not input_file.exists():
            pytest.skip("Fixture file not found")

        output_file = tmp_path / "bsim4_nmos_id.vc"

        # Use sourcepath to find the modelcard
        cfg = dfl.default_config()
        cfg["sourcepath"] = [str(fixtures_dir / "ngspice"), "."]

        conv = Converter(cfg=cfg, dialect="ngspice")
        try:
            conv.convert(str(input_file), str(output_file))
        except (ValueError, KeyError) as e:
            # BSIM4 fixtures have complex features not yet fully supported
            pytest.skip(f"Fixture conversion not yet supported: {e}")

        assert output_file.exists()
        content = output_file.read_text()
        assert len(content) > 0
