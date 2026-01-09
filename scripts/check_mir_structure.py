#!/usr/bin/env python3
"""Check MIR structure."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import openvaf_py
import json

# Compile capacitor
modules = openvaf_py.compile_va(str(Path(__file__).parent.parent / 'vendor/VACASK/devices/capacitor.va'))
cap = modules[0]

# Get MIR
eval_mir_dict = cap.get_mir_instructions()

print("Type of eval_mir_dict:", type(eval_mir_dict))
print("\nKeys in eval_mir_dict:", eval_mir_dict.keys() if hasattr(eval_mir_dict, 'keys') else 'N/A')

if 'instructions' in eval_mir_dict:
    print("\nType of instructions:", type(eval_mir_dict['instructions']))
    print("Length of instructions:", len(eval_mir_dict['instructions']))
    if eval_mir_dict['instructions']:
        print("\nFirst instruction:")
        print(json.dumps(eval_mir_dict['instructions'][0], indent=2))
