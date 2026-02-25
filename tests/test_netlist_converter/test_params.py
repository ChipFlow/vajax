# SPDX-FileCopyrightText: 2025 ChipFlow
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for parameter handling utilities."""

from vajax.netlist_converter.spiceparser.params import (
    convert_si_prefixes,
    format_params_line,
    format_value,
    process_expressions,
    process_instance_params,
    process_terminals,
    remove_params,
    split_params,
)


class TestSiPrefixConversion:
    """Tests for SI prefix conversion."""

    def test_convert_meg(self):
        """meg prefix should convert to M."""
        assert convert_si_prefixes("10meg") == "10M"
        assert convert_si_prefixes("1.5meg") == "1.5M"

    def test_convert_g(self):
        """g prefix should convert to G."""
        assert convert_si_prefixes("1g") == "1G"

    def test_convert_t(self):
        """t prefix should convert to T."""
        assert convert_si_prefixes("2t") == "2T"

    def test_convert_mil(self):
        """mil prefix should convert to multiplication."""
        assert convert_si_prefixes("100mil") == "(100*25.4e-6)"

    def test_preserve_other_prefixes(self):
        """Standard SI prefixes should pass through."""
        # Note: k, m, u, n, p, f are handled by simulator, not converted here
        assert convert_si_prefixes("1k") == "1k"

    def test_case_insensitive(self):
        """SI prefix conversion should be case-insensitive."""
        assert convert_si_prefixes("10MEG") == "10M"
        assert convert_si_prefixes("10Meg") == "10M"


class TestFormatValue:
    """Tests for parameter value formatting."""

    def test_remove_braces(self):
        """Curly braces should be removed from expressions."""
        assert format_value("{a+b}") == "a+b"
        assert format_value("{1+2}") == "1+2"

    def test_convert_si_in_value(self):
        """SI prefixes in values should be converted."""
        assert format_value("10meg") == "10M"
        assert format_value("{10meg+5}") == "10M+5"

    def test_plain_value(self):
        """Plain values should pass through."""
        assert format_value("1.5") == "1.5"
        assert format_value("42") == "42"


class TestSplitParams:
    """Tests for parameter splitting."""

    def test_split_key_value(self):
        """Split key=value parameters."""
        result = split_params(["r=1k", "tc1=0"])
        assert result == [("r", "1k"), ("tc1", "0")]

    def test_boolean_params(self):
        """Boolean parameters should get value 1."""
        result = split_params(["off", "on"])
        assert result == [("off", "1"), ("on", "1")]

    def test_handle_m_factor(self):
        """m parameter should be renamed when requested."""
        result = split_params(["m=2"], handle_m=True)
        assert result == [("$mfactor", "2")]

        result = split_params(["_mfactor=3"], handle_m=True)
        assert result == [("$mfactor", "3")]

    def test_no_handle_m(self):
        """m parameter should not be renamed when not requested."""
        result = split_params(["m=2"], handle_m=False)
        assert result == [("m", "2")]


class TestRemoveParams:
    """Tests for parameter removal."""

    def test_remove_specified(self):
        """Specified parameters should be removed."""
        params = [("r", "1k"), ("ic", "0"), ("tc1", "0")]
        result = remove_params(params, {"ic"})
        assert result == [("r", "1k"), ("tc1", "0")]

    def test_remove_multiple(self):
        """Multiple parameters can be removed."""
        params = [("r", "1k"), ("ic", "0"), ("off", "1")]
        result = remove_params(params, {"ic", "off"})
        assert result == [("r", "1k")]

    def test_case_insensitive(self):
        """Removal should be case-insensitive."""
        params = [("R", "1k"), ("IC", "0")]
        result = remove_params(params, {"ic"})
        # Removal is case-insensitive - IC is removed even with lowercase 'ic'
        assert result == [("R", "1k")]


class TestProcessExpressions:
    """Tests for expression processing."""

    def test_replace_temper(self):
        """temper should be replaced with $temp."""
        params = [("r", "1k*(1+tc1*temper)")]
        result = process_expressions(params)
        assert result == [("r", "1k*(1+tc1*$temp)")]

    def test_preserve_other(self):
        """Other expressions should pass through."""
        params = [("r", "1k+2k")]
        result = process_expressions(params)
        assert result == [("r", "1k+2k")]


class TestProcessTerminals:
    """Tests for terminal name processing."""

    def test_replace_exclamation(self):
        """Exclamation marks should be replaced with underscores."""
        result = process_terminals(["VDD!", "VSS!", "in", "out"])
        assert result == ["VDD_", "VSS_", "in", "out"]

    def test_preserve_normal(self):
        """Normal names should pass through."""
        result = process_terminals(["in", "out", "vdd", "gnd"])
        assert result == ["in", "out", "vdd", "gnd"]


class TestProcessInstanceParams:
    """Tests for full instance parameter processing."""

    def test_diode_removes_ic_off(self):
        """Diode should have ic and off removed."""
        params = ["is=1e-14", "ic=0", "off"]
        result = process_instance_params(params, "d")
        names = [p[0] for p in result]
        assert "ic" not in names
        assert "off" not in names
        assert "is" in names

    def test_capacitor_removes_ic(self):
        """Capacitor should have ic removed."""
        params = ["c=1u", "ic=0"]
        result = process_instance_params(params, "c")
        names = [p[0] for p in result]
        assert "ic" not in names
        assert "c" in names


class TestFormatParamsLine:
    """Tests for parameter line formatting."""

    def test_single_line(self):
        """Short params should fit on one line."""
        params = [("r", "1k"), ("tc1", "0")]
        result, was_split = format_params_line(params)
        assert result == "r=1k tc1=0"
        assert was_split is False

    def test_empty_params(self):
        """Empty params should return empty string."""
        result, was_split = format_params_line([])
        assert result == ""
        assert was_split is False
