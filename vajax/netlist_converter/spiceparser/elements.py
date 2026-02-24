# SPDX-FileCopyrightText: 2025 ChipFlow
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Device element definitions and OSDI module mappings.

This module defines the mapping between SPICE device types and their
corresponding OSDI (Open Source Device Interface) modules for VACASK output.

Extracted from ng2vclib/dfl.py to provide a shared, dialect-independent
device registry.
"""

from dataclasses import dataclass, field


@dataclass
class DeviceTypeInfo:
    """Information about a SPICE device type.

    Attributes:
        model_type: The SPICE model type string (e.g., "nmos", "npn")
        family: Device family (e.g., "mos", "bjt", "d")
        extra_params: Parameters to add when converting (e.g., {"type": "1"})
        remove_level: Whether to remove level parameter in output
        remove_version: Whether to remove version parameter in output
    """

    model_type: str
    family: str
    extra_params: dict[str, str] = field(default_factory=dict)
    remove_level: bool = False
    remove_version: bool = False


@dataclass
class OsdiModuleInfo:
    """OSDI module mapping for a device family/level/version.

    Attributes:
        osdi_file: Path to OSDI file (e.g., "spice/resistor.osdi")
        module_name: Module name in OSDI file (e.g., "sp_resistor")
        extra_params: Additional parameters for this module
    """

    osdi_file: str
    module_name: str
    extra_params: dict[str, str] = field(default_factory=dict)


# Device type registry: model_type -> DeviceTypeInfo
# This maps SPICE model types to their device families and properties
DEVICE_TYPES: dict[str, DeviceTypeInfo] = {
    # Passives
    "r": DeviceTypeInfo("r", "r"),
    "res": DeviceTypeInfo("res", "r"),
    "c": DeviceTypeInfo("c", "c"),
    "l": DeviceTypeInfo("l", "l"),
    # Diodes
    "d": DeviceTypeInfo("d", "d"),
    # BJTs
    "npn": DeviceTypeInfo("npn", "bjt", {"type": "1"}, remove_level=True),
    "pnp": DeviceTypeInfo("pnp", "bjt", {"type": "-1"}, remove_level=True),
    # JFETs
    "njf": DeviceTypeInfo("njf", "jfet", {"type": "1"}, remove_level=True),
    "pjf": DeviceTypeInfo("pjf", "jfet", {"type": "-1"}, remove_level=True),
    # MESFETs
    "nmf": DeviceTypeInfo("nmf", "mes", {"type": "1"}, remove_level=True),
    "pmf": DeviceTypeInfo("pmf", "mes", {"type": "-1"}, remove_level=True),
    # HEMTs
    "nhfet": DeviceTypeInfo("nhfet", "hemt", {"type": "1"}, remove_level=True),
    "phfet": DeviceTypeInfo("phfet", "hemt", {"type": "-1"}, remove_level=True),
    # MOSFETs
    "nmos": DeviceTypeInfo("nmos", "mos", {"type": "1"}, remove_level=True, remove_version=True),
    "pmos": DeviceTypeInfo("pmos", "mos", {"type": "-1"}, remove_level=True, remove_version=True),
    # SOI MOSFETs
    "nsoi": DeviceTypeInfo("nsoi", "soi", {"type": "1"}, remove_level=True, remove_version=True),
    "psoi": DeviceTypeInfo("psoi", "soi", {"type": "-1"}, remove_level=True, remove_version=True),
}


# OSDI module registry: (family, level, version) -> OsdiModuleInfo
# This maps device family/level/version to OSDI modules
OSDI_MODULES: dict[tuple[str, int | None, str | None], OsdiModuleInfo] = {
    # Passives
    ("r", None, None): OsdiModuleInfo("spice/resistor.osdi", "sp_resistor"),
    ("c", None, None): OsdiModuleInfo("spice/capacitor.osdi", "sp_capacitor"),
    ("l", None, None): OsdiModuleInfo("spice/inductor.osdi", "sp_inductor"),
    # Diodes
    ("d", None, None): OsdiModuleInfo("spice/diode.osdi", "sp_diode"),
    ("d", 1, None): OsdiModuleInfo("spice/diode.osdi", "sp_diode"),
    ("d", 3, None): OsdiModuleInfo("spice/diode.osdi", "sp_diode"),
    # BJTs - Gummel-Poon
    ("bjt", None, None): OsdiModuleInfo("spice/bjt.osdi", "sp_bjt"),
    ("bjt", 1, None): OsdiModuleInfo("spice/bjt.osdi", "sp_bjt"),
    # BJTs - VBIC
    ("bjt", 4, None): OsdiModuleInfo("spice/vbic.osdi", "sp_vbic"),
    ("bjt", 9, None): OsdiModuleInfo("spice/vbic.osdi", "sp_vbic"),
    # JFETs
    ("jfet", None, None): OsdiModuleInfo("spice/jfet1.osdi", "sp_jfet1"),
    ("jfet", 1, None): OsdiModuleInfo("spice/jfet1.osdi", "sp_jfet1"),
    ("jfet", 2, None): OsdiModuleInfo("spice/jfet2.osdi", "sp_jfet2"),
    # MESFETs
    ("mes", None, None): OsdiModuleInfo("spice/mes1.osdi", "sp_mes1"),
    ("mes", 1, None): OsdiModuleInfo("spice/mes1.osdi", "sp_mes1"),
    # MOSFETs - Basic levels
    ("mos", None, None): OsdiModuleInfo("spice/mos1.osdi", "sp_mos1"),
    ("mos", 1, None): OsdiModuleInfo("spice/mos1.osdi", "sp_mos1"),
    ("mos", 2, None): OsdiModuleInfo("spice/mos2.osdi", "sp_mos2"),
    ("mos", 3, None): OsdiModuleInfo("spice/mos3.osdi", "sp_mos3"),
    ("mos", 6, None): OsdiModuleInfo("spice/mos6.osdi", "sp_mos6"),
    ("mos", 9, None): OsdiModuleInfo("spice/mos9.osdi", "sp_mos9"),
    # MOSFETs - BSIM3 (level 8 and 49)
    ("mos", 8, None): OsdiModuleInfo("spice/bsim3v3.osdi", "sp_bsim3v3"),
    ("mos", 8, "3.3"): OsdiModuleInfo("spice/bsim3v3.osdi", "sp_bsim3v3"),
    ("mos", 8, "3.3.0"): OsdiModuleInfo("spice/bsim3v3.osdi", "sp_bsim3v3"),
    ("mos", 8, "3.2"): OsdiModuleInfo("spice/bsim3v2.osdi", "sp_bsim3v2", {"version": '"3.2"'}),
    ("mos", 8, "3.20"): OsdiModuleInfo("spice/bsim3v2.osdi", "sp_bsim3v2", {"version": '"3.20"'}),
    ("mos", 8, "3.2.2"): OsdiModuleInfo("spice/bsim3v2.osdi", "sp_bsim3v2", {"version": '"3.2.2"'}),
    ("mos", 8, "3.22"): OsdiModuleInfo("spice/bsim3v2.osdi", "sp_bsim3v2", {"version": '"3.22"'}),
    ("mos", 8, "3.2.3"): OsdiModuleInfo("spice/bsim3v2.osdi", "sp_bsim3v2", {"version": '"3.2.3"'}),
    ("mos", 8, "3.23"): OsdiModuleInfo("spice/bsim3v2.osdi", "sp_bsim3v2", {"version": '"3.23"'}),
    ("mos", 8, "3.2.4"): OsdiModuleInfo("spice/bsim3v2.osdi", "sp_bsim3v2", {"version": '"3.2.4"'}),
    ("mos", 8, "3.24"): OsdiModuleInfo("spice/bsim3v2.osdi", "sp_bsim3v2", {"version": '"3.24"'}),
    ("mos", 8, "3.1"): OsdiModuleInfo("spice/bsim3v1.osdi", "sp_bsim3v1"),
    ("mos", 8, "3.0"): OsdiModuleInfo("spice/bsim3v0.osdi", "sp_bsim3v0"),
    # Level 49 (same as level 8 for BSIM3)
    ("mos", 49, None): OsdiModuleInfo("spice/bsim3v3.osdi", "sp_bsim3v3"),
    ("mos", 49, "3.3"): OsdiModuleInfo("spice/bsim3v3.osdi", "sp_bsim3v3"),
    ("mos", 49, "3.3.0"): OsdiModuleInfo("spice/bsim3v3.osdi", "sp_bsim3v3"),
    ("mos", 49, "3.2"): OsdiModuleInfo("spice/bsim3v2.osdi", "sp_bsim3v2", {"version": '"3.2"'}),
    ("mos", 49, "3.20"): OsdiModuleInfo("spice/bsim3v2.osdi", "sp_bsim3v2", {"version": '"3.20"'}),
    ("mos", 49, "3.2.2"): OsdiModuleInfo(
        "spice/bsim3v2.osdi", "sp_bsim3v2", {"version": '"3.2.2"'}
    ),
    ("mos", 49, "3.22"): OsdiModuleInfo("spice/bsim3v2.osdi", "sp_bsim3v2", {"version": '"3.22"'}),
    ("mos", 49, "3.2.3"): OsdiModuleInfo(
        "spice/bsim3v2.osdi", "sp_bsim3v2", {"version": '"3.2.3"'}
    ),
    ("mos", 49, "3.23"): OsdiModuleInfo("spice/bsim3v2.osdi", "sp_bsim3v2", {"version": '"3.23"'}),
    ("mos", 49, "3.2.4"): OsdiModuleInfo(
        "spice/bsim3v2.osdi", "sp_bsim3v2", {"version": '"3.2.4"'}
    ),
    ("mos", 49, "3.24"): OsdiModuleInfo("spice/bsim3v2.osdi", "sp_bsim3v2", {"version": '"3.24"'}),
    ("mos", 49, "3.1"): OsdiModuleInfo("spice/bsim3v1.osdi", "sp_bsim3v1"),
    ("mos", 49, "3.0"): OsdiModuleInfo("spice/bsim3v0.osdi", "sp_bsim3v0"),
    # BSIM4 (level 14 and 54) - commonly used in modern PDKs
    ("mos", 14, None): OsdiModuleInfo("spice/bsim4.osdi", "sp_bsim4"),
    ("mos", 54, None): OsdiModuleInfo("spice/bsim4.osdi", "sp_bsim4"),
    ("mos", 54, "4.5"): OsdiModuleInfo("spice/bsim4.osdi", "sp_bsim4"),
    ("mos", 54, "4.6"): OsdiModuleInfo("spice/bsim4.osdi", "sp_bsim4"),
    ("mos", 54, "4.7"): OsdiModuleInfo("spice/bsim4.osdi", "sp_bsim4"),
    ("mos", 54, "4.8"): OsdiModuleInfo("spice/bsim4.osdi", "sp_bsim4"),
}


# Default models for passive elements without explicit .model
DEFAULT_MODELS: dict[str, OsdiModuleInfo] = {
    "r": OsdiModuleInfo("spice/resistor.osdi", "sp_resistor"),
    "c": OsdiModuleInfo("spice/capacitor.osdi", "sp_capacitor"),
    "l": OsdiModuleInfo("spice/inductor.osdi", "sp_inductor"),
}


# Parameters to remove from instances by device type
REMOVE_INSTANCE_PARAMS: dict[str, set[str]] = {
    "c": {"ic"},
    "l": {"ic"},
    "d": {"ic", "off"},
    "q": {"ic", "off"},
}


# Parameters that should be merged as vectors
MERGE_VECTOR_INSTANCE_PARAMS: dict[str, set[str]] = {
    "q": {"ic"},
}


def get_device_type_info(model_type: str) -> DeviceTypeInfo | None:
    """Get device type information by model type string.

    Args:
        model_type: SPICE model type (e.g., "nmos", "npn", "d")

    Returns:
        DeviceTypeInfo or None if not found
    """
    return DEVICE_TYPES.get(model_type.lower())


def get_osdi_module(
    family: str, level: int | None = None, version: str | None = None
) -> OsdiModuleInfo | None:
    """Get OSDI module info for a device family/level/version.

    Tries exact match first, then falls back to less specific matches.

    Args:
        family: Device family (e.g., "mos", "bjt", "d")
        level: Model level (e.g., 54 for BSIM4)
        version: Model version string

    Returns:
        OsdiModuleInfo or None if not found
    """
    # Try exact match
    key = (family, level, version)
    if key in OSDI_MODULES:
        return OSDI_MODULES[key]

    # Try without version
    key = (family, level, None)
    if key in OSDI_MODULES:
        return OSDI_MODULES[key]

    # Try without level
    key = (family, None, None)
    if key in OSDI_MODULES:
        return OSDI_MODULES[key]

    return None


def get_default_model(prefix: str) -> OsdiModuleInfo | None:
    """Get default OSDI module for a device prefix.

    Used for passive elements (R, C, L) without explicit .model.

    Args:
        prefix: Device prefix letter (e.g., "r", "c", "l")

    Returns:
        OsdiModuleInfo or None if not a default model type
    """
    return DEFAULT_MODELS.get(prefix.lower())
