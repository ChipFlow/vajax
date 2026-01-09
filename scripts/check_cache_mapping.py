#!/usr/bin/env python3
"""Check cache mapping to understand v37 and v40."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import openvaf_py

# Compile capacitor
print("Compiling capacitor model...")
modules = openvaf_py.compile_va(str(Path(__file__).parent.parent / 'vendor/VACASK/devices/capacitor.va'))
cap = modules[0]

# Get metadata
metadata = cap.get_codegen_metadata()
print("\n" + "="*80)
print("CACHE MAPPING")
print("="*80)
print("\nCache info from metadata:")
for cache_info in metadata['cache_info']:
    print(f"  Cache[{cache_info['cache_idx']}]: init computes {cache_info['init_value']}, eval reads as {cache_info['eval_param']}")

# Get init MIR
print("\n" + "="*80)
print("INIT MIR - WHAT GETS STORED IN CACHE")
print("="*80)
init_mir_dict = cap.get_init_mir_instructions()

# Build instruction lookup
inst_by_result = {}
for inst in init_mir_dict.get('instructions', []):
    if 'result' in inst:
        inst_by_result[inst['result']] = inst

print("\nTracing cache values:")
for cache_info in metadata['cache_info']:
    init_value = cache_info['init_value']
    eval_param = cache_info['eval_param']

    print(f"\nCache[{cache_info['cache_idx']}]:")
    print(f"  Init computes: {init_value}")
    print(f"  Eval reads as: {eval_param}")

    # Trace what init_value computes
    if init_value in inst_by_result:
        inst = inst_by_result[init_value]
        opcode = inst.get('opcode')
        operands = inst.get('operands', [])
        print(f"  Computation: {init_value} = {opcode} {operands}")

        # Trace operands
        for op in operands:
            if op in inst_by_result:
                op_inst = inst_by_result[op]
                print(f"    {op} = {op_inst.get('opcode')} {op_inst.get('operands', [])}")

print("\n" + "="*80)
print("CHECKING JACOBIAN VARIABLES")
print("="*80)
print("\nJacobian reactive variables and their cache sources:")
for jac_info in metadata['jacobian']:
    react_var = jac_info['react_var']
    print(f"\nJ[{jac_info['row']},{jac_info['col']}] react = {react_var}")

    # Check if this is a cache variable
    for cache_info in metadata['cache_info']:
        if cache_info['eval_param'] == react_var:
            print(f"  → Comes from cache[{cache_info['cache_idx']}], computed by init as {cache_info['init_value']}")
            break
    else:
        print(f"  → Not a cache variable (should be computed in eval)")

print("\n" + "="*80)
print("KEY INSIGHT")
print("="*80)
print("""
If v37 and v40 come from cache (eval params), then:
1. They are CONSTANTS computed during init
2. For a capacitor, dQ/dV = C (constant)
3. The cache should contain the derivatives, not the charge values

Let's check if v37/v40 are in the cache mapping.
""")

eval_mir_dict = cap.get_mir_instructions()
print("Eval MIR params (should include cache values):")
for param in eval_mir_dict.get('params', []):
    print(f"  {param}")
