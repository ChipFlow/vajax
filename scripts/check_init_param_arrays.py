#!/usr/bin/env python3
"""Check what init_param_names actually contains."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import openvaf_py

# Compile capacitor
modules = openvaf_py.compile_va(str(Path(__file__).parent.parent / 'vendor/VACASK/devices/capacitor.va'))
cap = modules[0]

print("="*80)
print("INIT PARAMETER ARRAYS (from openvaf-py)")
print("="*80)

print("\nInit param arrays:")
print(f"  init_param_names:         {cap.init_param_names}")
print(f"  init_param_value_indices: {cap.init_param_value_indices}")
print(f"  init_param_kinds:         {cap.init_param_kinds}")

print("\nZipped:")
for name, value_idx, kind in zip(cap.init_param_names, cap.init_param_value_indices, cap.init_param_kinds):
    print(f"  {name:20} v{value_idx:3}  kind={kind}")

print("\n" + "="*80)
print("METADATA INIT_PARAM_MAPPING")
print("="*80)

metadata = cap.get_codegen_metadata()
print("\nMetadata init_param_mapping:")
for name, var in metadata['init_param_mapping'].items():
    print(f"  {name:20} → {var}")

print("\n" + "="*80)
print("ANALYSIS")
print("="*80)

print("""
If init_param_names = ['c', 'mfactor', 'c']
and value_indices = [18, 20, 32]
and kinds = ['param', 'sysfun', 'param_given']

Then the mapping should be:
  c (param) → v18
  mfactor → v20
  c (param_given) → v32

But metadata only shows 2 entries!
This is because metadata generation filters by UNIQUE NAMES.
Both v18 and v32 map to 'c', so the second one overwrites the first!

The fix: metadata needs to distinguish between:
  - c (value)
  - c_given (flag)

OpenVAF should use different names, or we need to handle param_given specially.
""")

print("\nChecking for duplicate names:")
name_count = {}
for name in cap.init_param_names:
    name_count[name] = name_count.get(name, 0) + 1

for name, count in name_count.items():
    if count > 1:
        print(f"  ⚠️  '{name}' appears {count} times!")
