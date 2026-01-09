#!/usr/bin/env python3
"""Investigate MIR to find derivative calls and understand Jacobian variable mapping."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import openvaf_py

# Compile capacitor
print("Compiling capacitor model...")
modules = openvaf_py.compile_va(str(Path(__file__).parent.parent / 'vendor/VACASK/devices/capacitor.va'))
cap = modules[0]

# Get metadata
print("\n" + "="*80)
print("METADATA from get_codegen_metadata()")
print("="*80)
metadata = cap.get_codegen_metadata()
print("\nJacobian metadata:")
for jac_info in metadata['jacobian']:
    print(f"  J[{jac_info['row']},{jac_info['col']}]: resist={jac_info['resist_var']}, react={jac_info['react_var']}")

# Get MIR
print("\n" + "="*80)
print("EVAL MIR ANALYSIS")
print("="*80)
eval_mir_dict = cap.get_mir_instructions()

# Look for derivative function declarations
print("\nSearching for derivative function declarations (ddx_node_*)...")
for func_decl in eval_mir_dict.get('function_decls', []):
    if 'ddx_node' in str(func_decl):
        print(f"  {func_decl}")

print("\nSearching for derivative calls (calls to ddx_node functions)...")
derivative_calls = []
for inst in eval_mir_dict.get('instructions', []):
    if inst.get('opcode') == 'call':
        # Check if the function being called is a derivative function
        operands = inst.get('operands', [])
        func_arg = operands[0] if operands else None
        if func_arg and 'ddx' in str(func_arg):
            derivative_calls.append(inst)
            print(f"  {inst['result']} = call {func_arg}({operands[1:]})")

print(f"\nFound {len(derivative_calls)} derivative calls")

# Check what the metadata is pointing to
print("\n" + "="*80)
print("CHECKING METADATA VARIABLES")
print("="*80)

# Build a lookup dict for instructions by result variable
inst_by_result = {}
for inst in eval_mir_dict.get('instructions', []):
    if 'result' in inst:
        inst_by_result[inst['result']] = inst

for jac_info in metadata['jacobian'][:2]:  # Check first 2
    react_var = jac_info['react_var']
    print(f"\nJacobian[{jac_info['row']},{jac_info['col']}] react_var = {react_var}")

    # Find this variable in MIR
    if react_var in inst_by_result:
        inst_info = inst_by_result[react_var]
        print(f"  MIR: {react_var} = {inst_info.get('opcode')} {inst_info.get('operands', [])}")

        # Check if it's a derivative call
        if inst_info.get('opcode') == 'call':
            operands = inst_info.get('operands', [])
            func_arg = operands[0] if operands else None
            if func_arg and 'ddx' in str(func_arg):
                print(f"  ✓ This IS a derivative call!")
            else:
                print(f"  ✗ This is NOT a derivative call (calling {func_arg})")
        else:
            print(f"  ✗ This is NOT a derivative (opcode={inst_info.get('opcode')})")
    else:
        print(f"  ✗ Variable {react_var} not found in MIR instructions")

# Get DAE system info
print("\n" + "="*80)
print("DAE SYSTEM INFO")
print("="*80)
dae_system = cap.get_dae_system()
print(f"\nNumber of Jacobian entries: {len(dae_system['jacobian_rows'])}")
print("\nFirst 2 Jacobian entries:")
for i in range(min(2, len(dae_system['jacobian_rows']))):
    print(f"  Entry {i}:")
    print(f"    Row: {dae_system['jacobian_rows'][i]}")
    print(f"    Col: {dae_system['jacobian_cols'][i]}")
    print(f"    Resist index: {dae_system['jacobian_resist_indices'][i]}")
    print(f"    React index: {dae_system['jacobian_react_indices'][i]}")

    # Check if these indices correspond to derivative calls
    resist_var = f"v{dae_system['jacobian_resist_indices'][i]}"
    react_var = f"v{dae_system['jacobian_react_indices'][i]}"

    print(f"    Resist var: {resist_var}")
    if resist_var in inst_by_result:
        inst_info = inst_by_result[resist_var]
        operands = inst_info.get('operands', [])[:2]
        print(f"      {resist_var} = {inst_info.get('opcode')} {operands}")

    print(f"    React var: {react_var}")
    if react_var in inst_by_result:
        inst_info = inst_by_result[react_var]
        operands = inst_info.get('operands', [])[:2]
        print(f"      {react_var} = {inst_info.get('opcode')} {operands}")

# Compare metadata vs DAE system
print("\n" + "="*80)
print("METADATA vs DAE SYSTEM COMPARISON")
print("="*80)
print("\nChecking if metadata matches DAE system...")
for i, jac_info in enumerate(metadata['jacobian']):
    meta_resist = jac_info['resist_var']
    meta_react = jac_info['react_var']
    dae_resist = f"v{dae_system['jacobian_resist_indices'][i]}"
    dae_react = f"v{dae_system['jacobian_react_indices'][i]}"

    match = (meta_resist == dae_resist and meta_react == dae_react)
    symbol = "✓" if match else "✗"
    print(f"  Entry {i}: {symbol}")
    if not match:
        print(f"    Metadata: resist={meta_resist}, react={meta_react}")
        print(f"    DAE:      resist={dae_resist}, react={dae_react}")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print("\nIf the variables in metadata ARE derivative calls:")
print("  → OpenVAF is correctly identifying derivatives")
print("  → Problem is in our generated code execution or extraction")
print("\nIf the variables in metadata ARE NOT derivative calls:")
print("  → OpenVAF metadata is pointing to wrong MIR values")
print("  → Need to find the actual derivative values in MIR")
