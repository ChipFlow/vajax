#!/usr/bin/env python3
"""Check what init computes for cache values."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import openvaf_py

# Compile capacitor
modules = openvaf_py.compile_va(str(Path(__file__).parent.parent / 'vendor/VACASK/devices/capacitor.va'))
cap = modules[0]

print("="*80)
print("INIT MIR - CACHE VALUE COMPUTATION")
print("="*80)

# Get init MIR
init_mir_dict = cap.get_init_mir_instructions()

# Build instruction lookup
inst_by_result = {}
for inst in init_mir_dict.get('instructions', []):
    if 'result' in inst:
        inst_by_result[inst['result']] = inst

# Trace cache values
cache_mapping = init_mir_dict.get('cache_mapping', [])
print("\nCache values computed by init:")
for i, entry in enumerate(cache_mapping):
    init_value = entry['init_value']
    eval_param_idx = entry['eval_param']

    print(f"\nCache[{i}]: {init_value} → eval param {eval_param_idx}")

    # Trace computation
    current_var = init_value
    depth = 0
    max_depth = 10

    while current_var in inst_by_result and depth < max_depth:
        inst = inst_by_result[current_var]
        opcode = inst.get('opcode')
        operands = inst.get('operands', [])

        indent = "  " * (depth + 1)
        print(f"{indent}{current_var} = {opcode} {operands}")

        # Trace operands if they're computed
        if opcode == 'optbarrier' and len(operands) == 1:
            current_var = operands[0]
            depth += 1
        elif len(operands) > 0:
            # Show what operands are
            for op in operands:
                if op in inst_by_result:
                    op_inst = inst_by_result[op]
                    print(f"{indent}  {op} = {op_inst.get('opcode')} {op_inst.get('operands', [])}")
            break
        else:
            break

print("\n" + "="*80)
print("INIT MIR PARAMETERS")
print("="*80)
print("\nInit function parameters:")
for i, param in enumerate(init_mir_dict.get('params', [])):
    # Find semantic name
    for name, value_idx in zip(cap.param_names, cap.param_value_indices):
        if f"v{value_idx}" == param:
            print(f"  Param {i}: {param} = {name}")
            break
    else:
        print(f"  Param {i}: {param} (unknown semantic name)")

print("\n" + "="*80)
print("TEST: Run init with c=1e-9")
print("="*80)

# Call native init
print("\nCalling cap.run_init_eval with c=1e-9, V=0.0...")
ref_residuals, ref_jacobian = cap.run_init_eval({'c': 1e-9, 'c_given': True, 'V_A_B': 0.0, 'mfactor': 1.0})

print(f"\nReference Jacobian (should show dQ/dV = c = 1e-9):")
for row, col, resist, react in ref_jacobian:
    print(f"  J[{row},{col}]: resist={resist:12.6e}, react={react:12.6e}")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print("""
For a capacitor with c=1e-9:
- dQ/dV = c = 1e-9 (constant, independent of voltage)
- This should be computed by init and stored in cache
- Then eval uses cache to fill Jacobian

The cache values v27 and v28 should both equal ±1e-9.
Let me verify by generating and running the init function.
""")
