# SPDX-FileCopyrightText: 2025 ChipFlow
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Default configuration for ng2vc converter.

This module provides the default configuration and uses shared definitions
from ..spiceparser.elements for device type and OSDI module mappings.
"""

from ..spiceparser.elements import (
    DEFAULT_MODELS,
    DEVICE_TYPES,
    MERGE_VECTOR_INSTANCE_PARAMS,
    OSDI_MODULES,
    REMOVE_INSTANCE_PARAMS,
)


def _build_type_map():
    """Build type_map from ..spiceparser.elements.DEVICE_TYPES.

    Converts DeviceTypeInfo dataclasses to the tuple format expected by ng2vclib.
    """
    type_map = {}
    for name, info in DEVICE_TYPES.items():
        type_map[name] = (
            dict(info.extra_params),  # params dict
            info.family,
            info.remove_level,
            info.remove_version,
        )
    return type_map


def _build_family_map():
    """Build family_map from ..spiceparser.elements.OSDI_MODULES.

    Converts OsdiModuleInfo dataclasses to the tuple format expected by ng2vclib.
    """
    family_map = {}
    for key, info in OSDI_MODULES.items():
        family_map[key] = (info.osdi_file, info.module_name, dict(info.extra_params))
    return family_map


def _build_default_models():
    """Build default_models from ..spiceparser.elements.DEFAULT_MODELS.

    Converts OsdiModuleInfo dataclasses to the tuple format expected by ng2vclib.
    """
    return {prefix: (info.osdi_file, info.module_name) for prefix, info in DEFAULT_MODELS.items()}


def default_config():
    """Returns a default configuration.

    Uses shared definitions from ..spiceparser.elements for device mappings,
    ensuring consistency between ng2vc and the multi-dialect parser.
    """
    return {
        "signature": "// Converted by ng2vc converter\n",
        "sourcepath": ["."],
        "merge_vector_instance_params": dict(MERGE_VECTOR_INSTANCE_PARAMS),
        "remove_instance_params": {k: set(v) for k, v in REMOVE_INSTANCE_PARAMS.items()},
        "merge_vector_model_params": {},
        "remove_model_params": {},
        "type_map": _build_type_map(),
        "family_map": _build_family_map(),
        "all_models": False,
        "default_models": _build_default_models(),
        "default_model_prefix": "defmod_",
        "as_toplevel": "auto",
        "read_depth": None,  # Fully recursive
        "process_depth": None,  # Fully recursive
        "output_depth": None,  # Fully recursive
        "patch": {},
        "columns": 80,
    }
