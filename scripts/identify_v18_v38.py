#!/usr/bin/env python3
"""Identify what v18 and v38 are in the init MIR."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import openvaf_py

# Compile capacitor
modules = openvaf_py.compile_va(str(Path(__file__).parent.parent / 'vendor/VACASK/devices/capacitor.va'))
cap = modules[0]

# Get init MIR
init_mir_dict = cap.get_init_mir_instructions()

print("="*80)
print("IDENTIFYING v18 and v38")
print("="*80)

# v33 = phi with operands v18 and v38
print("\nv33 = phi from:")
print("  v18 (block8)")
print("  v38 (block14)")

# Check params
print("\nInit MIR params:")
for i, param in enumerate(init_mir_dict.get('params', [])):
    print(f"  {i}: {param}")

    # Check if v18 or v38 is a param
    if param == 'v18':
        print("    → v18 IS A PARAMETER!")
        # Find semantic name
        for name, value_idx in zip(cap.param_names, cap.param_value_indices):
            if value_idx == 18:
                print(f"       Semantic name: {name}")
    if param == 'v38':
        print("    → v38 IS A PARAMETER!")
        for name, value_idx in zip(cap.param_names, cap.param_value_indices):
            if value_idx == 38:
                print(f"       Semantic name: {name}")

# Build instruction lookup
inst_by_result = {}
for inst in init_mir_dict.get('instructions', []):
    if 'result' in inst:
        inst_by_result[inst['result']] = inst

# Check if v18 is computed
print("\nChecking if v18 is computed:")
if 'v18' in inst_by_result:
    inst = inst_by_result['v18']
    print(f"  v18 = {inst.get('opcode')} {inst.get('operands', [])}")
else:
    print("  v18 is NOT computed (must be param or constant)")

# Check if v38 is computed
print("\nChecking if v38 is computed:")
if 'v38' in inst_by_result:
    inst = inst_by_result['v38']
    print(f"  v38 = {inst.get('opcode')} {inst.get('operands', [])}")
else:
    print("  v38 is NOT computed (must be param or constant)")

# Check constants
print("\nChecking constants:")
for const_name, const_val in init_mir_dict.get('constants', {}).items():
    if const_name in ['v18', 'v38']:
        print(f"  {const_name} = {const_val}")

print("\n" + "="*80)
print("FULL PARAMETER MAPPING")
print("="*80)

print("\nAll semantic parameters:")
for name, value_idx, kind in zip(cap.param_names, cap.param_value_indices, cap.param_kinds):
    print(f"  {name:20} v{value_idx:3}  kind={kind}")

print("\n" + "="*80)
print("HYPOTHESIS")
print("="*80)

print("""
If v18 is a parameter and corresponds to 'c' (capacitance),
then the phi node selects between:
- v18 = user-provided capacitance (if c_given=True)
- v38 = default capacitance (if c_given=False)

This is a common pattern: phi selects between user value and default.
The cache should store mfactor * selected_capacitance.
""")

# Check if either is 'c' or relates to it
print("\nCapacitor parameters in metadata:")
metadata = cap.get_codegen_metadata()
for name, var in metadata['init_param_mapping'].items():
    print(f"  {name} → {var}")
