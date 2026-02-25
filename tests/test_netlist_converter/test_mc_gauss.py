# SPDX-FileCopyrightText: 2025 ChipFlow
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for Monte Carlo and gauss function support.

This module tests:
- gauss/agauss function detection in expressions
- Preservation of gauss expressions across dialects
- Spectre statistics block parsing
- Pass-through to VACASK output
"""

from io import StringIO
from pathlib import Path

import pytest

from vajax.netlist_converter.spiceparser import parse_netlist
from vajax.netlist_converter.spiceparser.params import (
    contains_gauss_function,
    extract_gauss_calls,
    format_value,
)
from vajax.netlist_converter.vcwriter import write_vacask

# ============================================================================
# Gauss Function Detection Tests
# ============================================================================


class TestGaussFunctionDetection:
    """Tests for gauss/agauss function detection in expressions."""

    def test_detect_gauss_simple(self):
        """Detect simple gauss() function call."""
        assert contains_gauss_function("gauss(1k, 0.1, 3)")

    def test_detect_gauss_quoted(self):
        """Detect gauss() in quoted expression."""
        assert contains_gauss_function("'gauss(1k, 0.1, 3)'")

    def test_detect_agauss(self):
        """Detect agauss() function call."""
        assert contains_gauss_function("agauss(0, 0.05, 3)")

    def test_detect_agauss_uppercase(self):
        """Detect AGAUSS() (case-insensitive)."""
        assert contains_gauss_function("AGAUSS(0u, 0.1u, 3)")

    def test_detect_gauss_in_expression(self):
        """Detect gauss in complex expression."""
        assert contains_gauss_function("1k + gauss(0, 0.1, 3)")
        assert contains_gauss_function("vth0 + agauss(0, sigma, 3)")

    def test_no_gauss_plain_value(self):
        """Non-gauss plain values return False."""
        assert not contains_gauss_function("1k")
        assert not contains_gauss_function("0.5")

    def test_no_gauss_expression(self):
        """Non-gauss expressions return False."""
        assert not contains_gauss_function("1+x*2")
        assert not contains_gauss_function("vth0 * 1.1")

    def test_no_gauss_partial_word(self):
        """Words containing 'gauss' but not function calls return False."""
        assert not contains_gauss_function("gaussian")
        assert not contains_gauss_function("degauss")


# ============================================================================
# Gauss Function Extraction Tests
# ============================================================================


class TestGaussFunctionExtraction:
    """Tests for extracting gauss function parameters."""

    def test_extract_simple_gauss(self):
        """Extract parameters from simple gauss call."""
        calls = extract_gauss_calls("gauss(1k, 0.1, 3)")
        assert len(calls) == 1
        assert calls[0]["function"] == "gauss"
        assert calls[0]["mean"] == "1k"
        assert calls[0]["std_dev"] == "0.1"
        assert calls[0]["num_sigma"] == 3.0

    def test_extract_agauss(self):
        """Extract parameters from agauss call."""
        calls = extract_gauss_calls("agauss(0, 0.05, 3)")
        assert len(calls) == 1
        assert calls[0]["function"] == "agauss"
        assert calls[0]["mean"] == "0"
        assert calls[0]["std_dev"] == "0.05"

    def test_extract_without_num_sigma(self):
        """Extract gauss call without optional num_sigma."""
        calls = extract_gauss_calls("gauss(mean_val, sigma)")
        assert len(calls) == 1
        assert calls[0]["num_sigma"] is None

    def test_extract_multiple_calls(self):
        """Extract multiple gauss calls from one expression."""
        expr = "gauss(1k, 0.1, 3) + agauss(0, 0.05, 2)"
        calls = extract_gauss_calls(expr)
        assert len(calls) == 2
        assert calls[0]["function"] == "gauss"
        assert calls[1]["function"] == "agauss"

    def test_raw_expression_preserved(self):
        """Raw expression is captured for pass-through."""
        calls = extract_gauss_calls("agauss(0, 0.1, 3)")
        assert calls[0]["raw"] == "agauss(0, 0.1, 3)"


# ============================================================================
# Format Value Tests (Gauss Preservation)
# ============================================================================


class TestFormatValueGaussPreservation:
    """Tests for preserving gauss expressions in format_value."""

    def test_preserve_gauss_expression(self):
        """Gauss expressions should be preserved without SI conversion."""
        result = format_value("gauss(1k, 0.1, 3)")
        assert "gauss" in result.lower()
        # Should NOT convert 1k to 1000
        assert "1k" in result

    def test_preserve_agauss_expression(self):
        """Agauss expressions should be preserved."""
        result = format_value("agauss(0, 0.05, 3)")
        assert "agauss" in result.lower()

    def test_preserve_gauss_in_braces(self):
        """Gauss in curly braces should be preserved."""
        result = format_value("{gauss(1k, 0.1, 3)}")
        assert "gauss" in result.lower()
        # Braces should be removed but gauss preserved
        assert not result.startswith("{")

    def test_non_gauss_si_converted(self):
        """Non-gauss values should still have SI conversion."""
        result = format_value("10meg")
        assert "10M" in result


# ============================================================================
# Ngspice Gauss Preservation Tests
# ============================================================================


class TestNgspiceGaussPreservation:
    """Tests for preserving gauss in Ngspice netlists."""

    def test_parse_param_with_gauss(self):
        """Parse .param with gauss expression."""
        content = """\
* Test
.param r_var='gauss(1k, 0.1, 3)'
.end
"""
        netlist = parse_netlist(content, dialect="ngspice")
        # Should parse without error
        assert netlist is not None

    def test_parse_instance_with_agauss(self):
        """Parse instance with agauss parameter."""
        content = """\
* Test
.model nmos nmos level=54
.subckt test_ckt in out
M1 out in 0 0 nmos w=1u l=100n delvto='agauss(0, 0.1, 3)'
.ends test_ckt
.end
"""
        netlist = parse_netlist(content, dialect="ngspice")
        assert len(netlist.subcircuits) >= 1


# ============================================================================
# HSPICE Gauss Preservation Tests
# ============================================================================


class TestHspiceGaussPreservation:
    """Tests for preserving GAUSS in HSPICE netlists."""

    def test_parse_hspice_gauss(self):
        """Parse HSPICE .PARAM with GAUSS."""
        content = """\
* Test
.PARAM val=GAUSS(1k, 0.05, 3)
.end
"""
        netlist = parse_netlist(content, dialect="hspice")
        assert netlist is not None

    def test_parse_hspice_agauss(self):
        """Parse HSPICE .PARAM with AGAUSS."""
        content = """\
* Test
.PARAM offset=AGAUSS(0u, 0.1u, 3)
.end
"""
        netlist = parse_netlist(content, dialect="hspice")
        assert netlist is not None


# ============================================================================
# Spectre Statistics Block Tests
# ============================================================================


class TestSpectreStatisticsBlock:
    """Tests for Spectre statistics block parsing."""

    def test_parse_statistics_block(self):
        """Parse basic Spectre statistics block."""
        content = """\
* Spectre test
simulator lang=spectre

statistics {
    process {
        vary vth0 dist=gauss std=0.01
    }
    mismatch {
        vary delvto dist=gauss std=0.05
    }
}
"""
        netlist = parse_netlist(content, dialect="spectre")

        assert len(netlist.statistics_blocks) == 1
        block = netlist.statistics_blocks[0]
        assert len(block.process_variations) >= 1
        assert len(block.mismatch_variations) >= 1

    def test_statistics_process_variation(self):
        """Verify process variation details."""
        content = """\
* Test
simulator lang=spectre
statistics {
    process {
        vary toxe dist=gauss std=0.001
    }
}
"""
        netlist = parse_netlist(content, dialect="spectre")

        block = netlist.statistics_blocks[0]
        var = block.process_variations[0]
        assert var.parameter == "toxe"
        assert var.distribution == "gauss"
        assert var.std == "0.001"

    def test_statistics_mismatch_variation(self):
        """Verify mismatch variation details."""
        content = """\
* Test
simulator lang=spectre
statistics {
    mismatch {
        vary dvth0 dist=gauss std=0.03
    }
}
"""
        netlist = parse_netlist(content, dialect="spectre")

        block = netlist.statistics_blocks[0]
        var = block.mismatch_variations[0]
        assert var.parameter == "dvth0"
        assert var.distribution == "gauss"
        assert var.std == "0.03"
        assert var.variation_type == "mismatch"


# ============================================================================
# VACASK Output Tests (Pass-through)
# ============================================================================


class TestGaussPassThrough:
    """Tests for gauss expression pass-through to VACASK output."""

    def test_statistics_block_output(self):
        """Statistics blocks should appear in VACASK output."""
        content = """\
* Test
simulator lang=spectre
statistics {
    process { vary vth0 dist=gauss std=0.01 }
}
"""
        netlist = parse_netlist(content, dialect="spectre")

        output = StringIO()
        write_vacask(netlist, output)
        result = output.getvalue()

        assert "statistics" in result
        assert "process" in result
        assert "vary" in result
        assert "vth0" in result

    def test_statistics_block_format(self):
        """Statistics blocks should have correct format."""
        content = """\
* Test
simulator lang=spectre
statistics {
    process {
        vary vth0 dist=gauss std=0.01
    }
    mismatch {
        vary delvto dist=gauss std=0.05
    }
}
"""
        netlist = parse_netlist(content, dialect="spectre")

        output = StringIO()
        write_vacask(netlist, output)
        result = output.getvalue()

        # Should have both process and mismatch sections
        assert "process {" in result
        assert "mismatch {" in result
        # Should have both vary directives
        assert "vary vth0" in result
        assert "vary delvto" in result


# ============================================================================
# Fixture File Tests
# ============================================================================


class TestMCFixtureFiles:
    """Tests using actual fixture files."""

    @pytest.fixture
    def fixtures_dir(self):
        """Path to test fixtures directory."""
        return Path(__file__).parent / "fixtures"

    def test_ngspice_mc_fixture(self, fixtures_dir):
        """Parse Ngspice MC fixture file."""
        fixture_path = fixtures_dir / "ngspice" / "mc_gauss_params.sp"
        if fixture_path.exists():
            netlist = parse_netlist(fixture_path, dialect="ngspice")
            assert len(netlist.subcircuits) >= 1

    def test_hspice_mc_fixture(self, fixtures_dir):
        """Parse HSPICE MC fixture file."""
        fixture_path = fixtures_dir / "hspice" / "mc_gauss_params.sp"
        if fixture_path.exists():
            netlist = parse_netlist(fixture_path, dialect="hspice")
            assert len(netlist.subcircuits) >= 1

    def test_spectre_statistics_fixture(self, fixtures_dir):
        """Parse Spectre statistics block fixture file."""
        fixture_path = fixtures_dir / "spectre" / "statistics_block.scs"
        if fixture_path.exists():
            netlist = parse_netlist(fixture_path, dialect="spectre")
            assert len(netlist.statistics_blocks) >= 1
