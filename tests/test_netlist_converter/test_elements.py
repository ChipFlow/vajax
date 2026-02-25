# SPDX-FileCopyrightText: 2025 ChipFlow
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for device element definitions and OSDI module mappings."""

from vajax.netlist_converter.spiceparser.elements import (
    get_default_model,
    get_device_type_info,
    get_osdi_module,
)


class TestDeviceTypes:
    """Tests for device type registry."""

    def test_nmos_type_info(self):
        """NMOS should be in MOS family with type=1."""
        info = get_device_type_info("nmos")
        assert info is not None
        assert info.family == "mos"
        assert info.extra_params == {"type": "1"}
        assert info.remove_level is True
        assert info.remove_version is True

    def test_pmos_type_info(self):
        """PMOS should be in MOS family with type=-1."""
        info = get_device_type_info("pmos")
        assert info is not None
        assert info.family == "mos"
        assert info.extra_params == {"type": "-1"}

    def test_npn_type_info(self):
        """NPN should be in BJT family."""
        info = get_device_type_info("npn")
        assert info is not None
        assert info.family == "bjt"
        assert info.extra_params == {"type": "1"}
        assert info.remove_level is True
        assert info.remove_version is False

    def test_diode_type_info(self):
        """Diode should be in D family."""
        info = get_device_type_info("d")
        assert info is not None
        assert info.family == "d"
        assert info.extra_params == {}
        assert info.remove_level is False

    def test_resistor_type_info(self):
        """Resistor should be in R family."""
        info = get_device_type_info("r")
        assert info is not None
        assert info.family == "r"

    def test_case_insensitive(self):
        """Device type lookup should be case-insensitive."""
        assert get_device_type_info("NMOS") == get_device_type_info("nmos")
        assert get_device_type_info("NPN") == get_device_type_info("npn")

    def test_unknown_type(self):
        """Unknown device type should return None."""
        assert get_device_type_info("unknown_device") is None


class TestOsdiModules:
    """Tests for OSDI module registry."""

    def test_resistor_osdi(self):
        """Resistor should map to sp_resistor module."""
        osdi = get_osdi_module("r")
        assert osdi is not None
        assert osdi.osdi_file == "spice/resistor.osdi"
        assert osdi.module_name == "sp_resistor"

    def test_diode_osdi(self):
        """Diode should map to sp_diode module."""
        osdi = get_osdi_module("d")
        assert osdi is not None
        assert osdi.osdi_file == "spice/diode.osdi"
        assert osdi.module_name == "sp_diode"

    def test_mos_level_1(self):
        """MOS level 1 should map to sp_mos1."""
        osdi = get_osdi_module("mos", level=1)
        assert osdi is not None
        assert osdi.module_name == "sp_mos1"

    def test_mos_level_54(self):
        """MOS level 54 (BSIM4) should map to sp_bsim4."""
        osdi = get_osdi_module("mos", level=54)
        assert osdi is not None
        assert osdi.osdi_file == "spice/bsim4.osdi"
        assert osdi.module_name == "sp_bsim4"

    def test_mos_level_8_version_32(self):
        """MOS level 8 version 3.2 should have version param."""
        osdi = get_osdi_module("mos", level=8, version="3.2")
        assert osdi is not None
        assert osdi.module_name == "sp_bsim3v2"
        assert osdi.extra_params.get("version") == '"3.2"'

    def test_bjt_level_1(self):
        """BJT level 1 (Gummel-Poon) should map to sp_bjt."""
        osdi = get_osdi_module("bjt", level=1)
        assert osdi is not None
        assert osdi.module_name == "sp_bjt"

    def test_bjt_level_4(self):
        """BJT level 4 (VBIC) should map to sp_vbic."""
        osdi = get_osdi_module("bjt", level=4)
        assert osdi is not None
        assert osdi.module_name == "sp_vbic"

    def test_fallback_to_level_none(self):
        """Unknown level should fall back to level=None mapping."""
        osdi = get_osdi_module("d", level=99)
        assert osdi is not None
        # Should fall back to ("d", None, None)
        assert osdi.module_name == "sp_diode"

    def test_unknown_family(self):
        """Unknown family should return None."""
        assert get_osdi_module("unknown_family") is None


class TestDefaultModels:
    """Tests for default model mappings (passives)."""

    def test_resistor_default(self):
        """Resistor should have default model."""
        osdi = get_default_model("r")
        assert osdi is not None
        assert osdi.module_name == "sp_resistor"

    def test_capacitor_default(self):
        """Capacitor should have default model."""
        osdi = get_default_model("c")
        assert osdi is not None
        assert osdi.module_name == "sp_capacitor"

    def test_inductor_default(self):
        """Inductor should have default model."""
        osdi = get_default_model("l")
        assert osdi is not None
        assert osdi.module_name == "sp_inductor"

    def test_diode_no_default(self):
        """Diode should not have default model (requires .model)."""
        assert get_default_model("d") is None

    def test_mosfet_no_default(self):
        """MOSFET should not have default model (requires .model)."""
        assert get_default_model("m") is None
