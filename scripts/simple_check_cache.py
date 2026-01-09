#!/usr/bin/env python3
"""Simple check of cache structure."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import openvaf_py
import json

# Compile capacitor
modules = openvaf_py.compile_va(str(Path(__file__).parent.parent / 'vendor/VACASK/devices/capacitor.va'))
cap = modules[0]

# Get init MIR
init_mir_dict = cap.get_init_mir_instructions()

print("Keys in init_mir_dict:", list(init_mir_dict.keys()))

if 'cache_mapping' in init_mir_dict:
    print("\ncache_mapping:")
    print(json.dumps(init_mir_dict['cache_mapping'], indent=2))
else:
    print("\nNo cache_mapping key")

# Get metadata
metadata = cap.get_codegen_metadata()
print("\nCache info from metadata:")
print(json.dumps(metadata['cache_info'], indent=2))
