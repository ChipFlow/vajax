#!/usr/bin/env python3
"""Trace where v37 and v40 come from - are they cache values?"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import openvaf_py

# Compile capacitor
print("Compiling capacitor model...")
modules = openvaf_py.compile_va(str(Path(__file__).parent.parent / 'vendor/VACASK/devices/capacitor.va'))
cap = modules[0]

print("\n" + "="*80)
print("HYPOTHESIS: v37 and v40 are cache-derived params")
print("="*80)

# Get cache mapping
init_mir_dict = cap.get_init_mir_instructions()
cache_mapping = init_mir_dict.get('cache_mapping', [])

print("\nCache mapping:")
for entry in cache_mapping:
    print(f"  Cache[{entry['cache_idx']}]: init={entry['init_value']} → eval={entry['eval_param']}")

print("\n" + "="*80)
print("CHECKING PARAM VALUE INDICES")
print("="*80)

# Check if there's a pattern between cache and v37/v40
print("\nSemantic params with their value indices:")
for name, value_idx in zip(cap.param_names, cap.param_value_indices):
    print(f"  {name:20} → v{value_idx}")

print("\nEval MIR params:")
eval_mir_dict = cap.get_mir_instructions()
for i, param in enumerate(eval_mir_dict.get('params', [])):
    print(f"  MIR param {i}: {param}")

    # Check if this might be a cache param
    for cache_entry in cache_mapping:
        if cache_entry['eval_param'] == param:
            print(f"    → Comes from cache[{cache_entry['cache_idx']}]")
            break

print("\n" + "="*80)
print("INIT MIR - Looking for v37/v40 computations")
print("="*80)

init_instructions = init_mir_dict.get('instructions', [])
inst_by_result = {}
for inst in init_instructions:
    if 'result' in inst:
        inst_by_result[inst['result']] = inst

# Look for any instruction that produces v37 or v40
print("\nSearching for v37 computation in init MIR:")
if 'v37' in inst_by_result:
    inst = inst_by_result['v37']
    print(f"  FOUND: v37 = {inst.get('opcode')} {inst.get('operands', [])}")
else:
    print(f"  NOT FOUND in init MIR")

print("\nSearching for v40 computation in init MIR:")
if 'v40' in inst_by_result:
    inst = inst_by_result['v40']
    print(f"  FOUND: v40 = {inst.get('opcode')} {inst.get('operands', [])}")
else:
    print(f"  NOT FOUND in init MIR")

print("\n" + "="*80)
print("CHECKING EVAL MIR CONSTANTS")
print("="*80)

print("\nEval MIR constants:")
for const_name, const_val in eval_mir_dict.get('constants', {}).items():
    print(f"  {const_name} = {const_val}")

print("\n" + "="*80)
print("ALTERNATIVE HYPOTHESIS: Derivatives stored as constants")
print("="*80)

# Check OpenVAF's internal derivative info
print("\nLet's check the full DAE system for derivative info:")
dae_system = cap.get_dae_system()

print("\nResidual info:")
for i, res in enumerate(dae_system['residuals']):
    print(f"  Residual {i}: {res}")

print("\nJacobian info (showing react_var):")
for i, jac in enumerate(dae_system['jacobian']):
    print(f"  J[{jac['row_node_name']},{jac['col_node_name']}]: react_var={jac['react_var']}")

print("\n" + "="*80)
print("KEY QUESTION")
print("="*80)
print("""
v37 and v40 are eval MIR params but:
1. NOT semantic parameters (no entry in param_names)
2. NOT cache values (v5/v6 are the cache params)
3. NOT constants

Where do they come from? Options:
A. They're derivative parameters computed by autodiff
B. They're implicit parameters created by OpenVAF
C. There's a mapping we're missing

Let me check if v5 or v6 map to v37/v40...
""")

# Check if cache params are related
print("Cache params in eval: v5, v6")
print("Mystery params in eval: v37, v40")
print(f"Difference: 37-5={37-5}, 40-6={40-6}")
print("\nLet me check init MIR to see what produces outputs at those indices...")

print("\nAll init MIR params:")
for i, param in enumerate(init_mir_dict.get('params', [])):
    print(f"  Init param {i}: {param}")
