#!/usr/bin/env python3
"""Debug why init computes 0 for cache."""

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
print("INIT MIR PARAMETERS")
print("="*80)

print("\nInit MIR params:")
for i, param in enumerate(init_mir_dict.get('params', [])):
    print(f"  {i}: {param}")

# Find semantic names
print("\nMapping to semantic names:")
for i, param in enumerate(init_mir_dict.get('params', [])):
    for name, value_idx in zip(cap.param_names, cap.param_value_indices):
        if f"v{value_idx}" == param:
            print(f"  {param} = {name}")
            break
    else:
        print(f"  {param} = ???")

print("\n" + "="*80)
print("INIT MIR INSTRUCTIONS")
print("="*80)

# Show all init instructions
for i, inst in enumerate(init_mir_dict.get('instructions', [])):
    result = inst.get('result', 'no_result')
    opcode = inst.get('opcode')
    operands = inst.get('operands', [])
    print(f"  {result:8} = {opcode:15} {operands}")

print("\n" + "="*80)
print("CACHE COMPUTATION TRACE")
print("="*80)

# Build instruction lookup
inst_by_result = {}
for inst in init_mir_dict.get('instructions', []):
    if 'result' in inst:
        inst_by_result[inst['result']] = inst

# Trace cache value computation
cache_mapping = init_mir_dict.get('cache_mapping', [])
for i, entry in enumerate(cache_mapping):
    init_value = entry['init_value']
    print(f"\nCache[{i}] = {init_value}")

    # Full recursive trace
    def trace(var, depth=0):
        indent = "  " * (depth + 1)
        if var in inst_by_result:
            inst = inst_by_result[var]
            opcode = inst.get('opcode')
            operands = inst.get('operands', [])
            print(f"{indent}{var} = {opcode} {operands}")
            for op in operands:
                trace(op, depth + 1)
        elif var in [p for p in init_mir_dict.get('params', [])]:
            print(f"{indent}{var} = [PARAM]")
        else:
            # Might be a constant or phi argument
            print(f"{indent}{var} = [CONST/PHI]")

    trace(init_value)

print("\n" + "="*80)
print("EXPECTED VALUES")
print("="*80)
print("""
For cache[0] (v27 = optbarrier(v19)):
  v19 = fmul(v20, v33)
  v20 = mfactor = 1.0
  v33 = phi (from some block)

For cache[1] (v28 = optbarrier(v22)):
  v22 = fmul(v20, v17)
  v20 = mfactor = 1.0
  v17 = fneg(v33)

So both depend on v33. What is v33?
""")

print("\nChecking v33:")
if 'v33' in inst_by_result:
    inst = inst_by_result['v33']
    print(f"  v33 = {inst.get('opcode')} {inst.get('operands', [])}")
else:
    print("  v33 not found in instructions")

    # Check if v33 is a param
    if 'v33' in init_mir_dict.get('params', []):
        print("  v33 is a parameter!")
        for name, value_idx in zip(cap.param_names, cap.param_value_indices):
            if value_idx == 33:
                print(f"    v33 = {name}")

print("\n" + "="*80)
print("BLOCKS AND PHI NODES")
print("="*80)
print("\nInit MIR blocks:")
for block_name, block_info in init_mir_dict.get('blocks', {}).items():
    print(f"\n  {block_name}:")
    print(f"    preds: {block_info.get('preds', [])}")
    print(f"    succs: {block_info.get('succs', [])}")

print("\nLooking for phi nodes:")
for inst in init_mir_dict.get('instructions', []):
    if inst.get('opcode') == 'phi':
        result = inst.get('result')
        operands = inst.get('operands', [])
        print(f"  {result} = phi {operands}")
