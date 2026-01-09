#!/usr/bin/env python3
"""Check if generated code has phi handling."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import openvaf_py
from jax_spice.codegen.mir_parser import parse_mir_dict
from jax_spice.codegen.setup_instance_mir_codegen import generate_setup_instance_from_mir

# Compile capacitor
modules = openvaf_py.compile_va(str(Path(__file__).parent.parent / 'vendor/VACASK/devices/capacitor.va'))
cap = modules[0]

# Get init MIR
init_mir_dict = cap.get_init_mir_instructions()

# Parse MIR
print("Parsing MIR...")
init_mir = parse_mir_dict(init_mir_dict)

# Check if phi nodes have phi_args
print("\nChecking parsed PHI nodes:")
for block in init_mir.blocks:
    for inst in block.instructions:
        if inst.opcode == 'phi':
            print(f"\nPHI in {block.name}:")
            print(f"  result: {inst.result}")
            print(f"  phi_args: {inst.phi_args}")
            if not inst.phi_args:
                print("  ⚠️  PHI_ARGS IS EMPTY!")

# Generate code
print("\n" + "="*80)
print("Generating setup_instance code...")
print("="*80)

init_param_map = {p.name: p.name for p in init_mir.params}
for const_name in init_mir.constants.keys():
    init_param_map[const_name] = const_name

cache_tuples = [(entry['init_value'], entry['eval_param']) for entry in init_mir_dict['cache_mapping']]

setup_instance_code = generate_setup_instance_from_mir(
    init_mir,
    init_param_map,
    cache_tuples,
    'capacitor'
)

# Search for phi handling in generated code
print("\nSearching for PHI in generated code...")
lines = setup_instance_code.split('\n')
for i, line in enumerate(lines):
    if 'phi' in line.lower() or 'prev_block' in line:
        print(f"  Line {i+1}: {line}")

# Show the full generated function
print("\n" + "="*80)
print("FULL GENERATED FUNCTION")
print("="*80)
print(setup_instance_code)
