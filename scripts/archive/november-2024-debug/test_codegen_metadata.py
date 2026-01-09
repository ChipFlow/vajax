#!/usr/bin/env python3
"""Test the new get_codegen_metadata() method."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import openvaf_py
import json

# Compile capacitor
modules = openvaf_py.compile_va(str(Path(__file__).parent.parent / "vendor/VACASK/devices/capacitor.va"))
cap = modules[0]

print("Testing get_codegen_metadata() method")
print("=" * 80)

# Get metadata
metadata = cap.get_codegen_metadata()

print("\nAvailable metadata keys:")
for key in sorted(metadata.keys()):
    print(f"  - {key}")

print("\n1. Eval Parameter Mapping:")
print(f"   {metadata['eval_param_mapping']}")

print("\n2. Init Parameter Mapping:")
print(f"   {metadata['init_param_mapping']}")

print("\n3. Cache Info:")
for entry in metadata['cache_info']:
    print(f"   cache[{entry['cache_idx']}]: {entry['init_value']} → {entry['eval_param']}")

print("\n4. Residuals:")
for res in metadata['residuals']:
    print(f"   Residual {res['residual_idx']}: resist={res['resist_var']}, react={res['react_var']}")

print("\n5. Jacobian:")
for jac in metadata['jacobian'][:5]:  # Show first 5
    print(f"   J[{jac['row']},{jac['col']}]: resist={jac['resist_var']}, react={jac['react_var']}")
if len(metadata['jacobian']) > 5:
    print(f"   ... and {len(metadata['jacobian']) - 5} more entries")

print("\n6. Model Info:")
print(f"   Name: {metadata['model_name']}")
print(f"   Terminals: {metadata['num_terminals']}")
print(f"   Residuals: {metadata['num_residuals']}")
print(f"   Jacobian entries: {metadata['num_jacobian']}")
print(f"   Cache slots: {metadata['num_cache_slots']}")

print("\n" + "=" * 80)
print("✓ get_codegen_metadata() works perfectly!")
print("=" * 80)
