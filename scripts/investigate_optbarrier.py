#!/usr/bin/env python3
"""Investigate what optbarrier wraps in the capacitor MIR."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import openvaf_py

# Compile capacitor
print("Compiling capacitor model...")
modules = openvaf_py.compile_va(str(Path(__file__).parent.parent / 'vendor/VACASK/devices/capacitor.va'))
cap = modules[0]

# Get metadata and MIR
metadata = cap.get_codegen_metadata()
eval_mir_dict = cap.get_mir_instructions()

# Build instruction lookup
inst_by_result = {}
for inst in eval_mir_dict.get('instructions', []):
    if 'result' in inst:
        inst_by_result[inst['result']] = inst

print("="*80)
print("TRACING JACOBIAN REACTIVE VARIABLES")
print("="*80)

# Trace back from Jacobian variables
for jac_info in metadata['jacobian']:
    react_var = jac_info['react_var']
    print(f"\nJacobian[{jac_info['row']},{jac_info['col']}] react = {react_var}")

    # Trace through the computation
    current_var = react_var
    depth = 0
    max_depth = 10

    while current_var in inst_by_result and depth < max_depth:
        inst = inst_by_result[current_var]
        opcode = inst.get('opcode')
        operands = inst.get('operands', [])

        print(f"  {'  ' * depth}{current_var} = {opcode} {operands}")

        # If it's a simple pass-through operation, trace the operand
        if opcode in ['optbarrier', 'copy', 'cast'] and len(operands) == 1:
            current_var = operands[0]
            depth += 1
        else:
            # Hit a real computation
            print(f"  {'  ' * depth}→ This is the actual computation!")
            break

print("\n" + "="*80)
print("CHECKING DAE SYSTEM")
print("="*80)
dae_system = cap.get_dae_system()
print("\nDAE system keys:", list(dae_system.keys()))

if 'jacobian' in dae_system:
    print(f"\nJacobian entries: {len(dae_system['jacobian'])}")
    print("\nFirst 2 entries:")
    for i, entry in enumerate(dae_system['jacobian'][:2]):
        print(f"  Entry {i}: {entry}")

print("\n" + "="*80)
print("ALL EVAL MIR INSTRUCTIONS")
print("="*80)
print("\nAll instructions (showing opcode and result):")
for i, inst in enumerate(eval_mir_dict.get('instructions', [])):
    result = inst.get('result', 'no_result')
    opcode = inst.get('opcode')
    operands = inst.get('operands', [])
    print(f"  {i}: {result} = {opcode} {operands}")
