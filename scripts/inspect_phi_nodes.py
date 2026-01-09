#!/usr/bin/env python3
"""Inspect phi node structure in detail."""

import sys
from pathlib import Path
import json
sys.path.insert(0, str(Path(__file__).parent.parent))

import openvaf_py

# Compile capacitor
modules = openvaf_py.compile_va(str(Path(__file__).parent.parent / 'vendor/VACASK/devices/capacitor.va'))
cap = modules[0]

# Get init MIR
init_mir_dict = cap.get_init_mir_instructions()

print("="*80)
print("PHI NODES IN DETAIL")
print("="*80)

# Find all phi nodes
phi_nodes = []
for inst in init_mir_dict.get('instructions', []):
    if inst.get('opcode') == 'phi':
        phi_nodes.append(inst)

print(f"\nFound {len(phi_nodes)} phi nodes")

for i, phi_inst in enumerate(phi_nodes):
    print(f"\nPhi node {i}:")
    print(json.dumps(phi_inst, indent=2))

print("\n" + "="*80)
print("CHECKING v33 SPECIFICALLY")
print("="*80)

for inst in init_mir_dict.get('instructions', []):
    if inst.get('result') == 'v33':
        print("\nFound v33:")
        print(json.dumps(inst, indent=2))

        # Check if it has phi_args
        if 'phi_args' in inst:
            print(f"\nphi_args: {inst['phi_args']}")
        if 'phi_preds' in inst:
            print(f"phi_preds: {inst['phi_preds']}")

print("\n" + "="*80)
print("BLOCK STRUCTURE")
print("="*80)

print("\nBlocks:")
for block_name, block_info in init_mir_dict.get('blocks', {}).items():
    print(f"\n  {block_name}:")
    for key, value in block_info.items():
        print(f"    {key}: {value}")

print("\n" + "="*80)
print("FULL INSTRUCTION LIST")
print("="*80)

print("\nShowing block assignments:")
for inst in init_mir_dict.get('instructions', []):
    result = inst.get('result', 'no_result')
    opcode = inst.get('opcode')
    block = inst.get('block', '???')
    operands = inst.get('operands', [])
    print(f"  {block:10} {result:8} = {opcode:15} {operands}")
