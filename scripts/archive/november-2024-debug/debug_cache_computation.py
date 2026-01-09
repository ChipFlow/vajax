#!/usr/bin/env python3
"""Debug why cache depends on voltage."""

import sys
import math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import openvaf_py
from jax_spice.codegen.mir_parser import parse_mir_dict
from jax_spice.codegen.setup_instance_mir_codegen import generate_setup_instance_from_mir

# Compile capacitor
modules = openvaf_py.compile_va(str(Path(__file__).parent.parent / 'vendor/VACASK/devices/capacitor.va'))
cap = modules[0]

# Get MIR
init_mir_dict = cap.get_init_mir_instructions()
init_mir = parse_mir_dict(init_mir_dict)

# Get metadata
metadata = cap.get_codegen_metadata()

# Create parameter mapping
init_param_map = {p.name: p.name for p in init_mir.params}
for const_name in init_mir.constants.keys():
    init_param_map[const_name] = const_name

cache_tuples = [(entry['init_value'], entry['eval_param']) for entry in init_mir_dict['cache_mapping']]

# Generate init function
print("Generating setup_instance_capacitor...")
setup_instance_code = generate_setup_instance_from_mir(
    init_mir,
    init_param_map,
    cache_tuples,
    'capacitor'
)

# Execute
namespace = {'math': math}
exec(setup_instance_code, namespace)
setup_instance_fn = namespace['setup_instance_capacitor']

print("\n" + "="*80)
print("TEST 1: V=0.0")
print("="*80)

# Call with V=0
init_kwargs = {}
for semantic_name in metadata['init_param_mapping'].keys():
    if semantic_name == 'c':
        init_kwargs[semantic_name] = 1e-9
    elif semantic_name == 'mfactor':
        init_kwargs[semantic_name] = 1.0
    else:
        init_kwargs[semantic_name] = 0.0

print(f"\nCalling setup_instance_capacitor with: {init_kwargs}")
cache_v0 = setup_instance_fn(**init_kwargs)
print(f"Cache result: {cache_v0}")

print("\n" + "="*80)
print("TEST 2: V=1.0")
print("="*80)

# Call with V=1
init_kwargs_v1 = init_kwargs.copy()
# Note: init shouldn't use V at all!

print(f"\nCalling setup_instance_capacitor with: {init_kwargs_v1}")
cache_v1 = setup_instance_fn(**init_kwargs_v1)
print(f"Cache result: {cache_v1}")

print("\n" + "="*80)
print("ANALYSIS")
print("="*80)

print(f"\nCache at V=0: {cache_v0}")
print(f"Cache at V=1: {cache_v1}")
print(f"Are they equal? {cache_v0 == cache_v1}")

print("\n" + "="*80)
print("INIT MIR PARAMS")
print("="*80)
print("\nWhat parameters does init actually take?")
for name, value_idx in zip(cap.param_names, cap.param_value_indices):
    for param_name in init_param_map.keys():
        if f"v{value_idx}" == param_name:
            print(f"  {param_name} = {name}")

print("\nInit param mapping from metadata:")
for name, var in metadata['init_param_mapping'].items():
    print(f"  {name} → {var}")

print("\n" + "="*80)
print("KEY QUESTION")
print("="*80)
print("""
If cache is constant (dQ/dV = C), init should not depend on voltage.
But our generated init function might be receiving voltage as a param.

Let me check the init function signature...
""")

import inspect
print(f"\nsetup_instance_capacitor signature: {inspect.signature(setup_instance_fn)}")

# Check what params init MIR expects
print("\nInit MIR params:")
for i, p in enumerate(init_mir.params):
    print(f"  {i}: {p.name}")

# Look for V in the init params
has_voltage = any('V' in p.name or 'voltage' in p.name.lower() for p in init_mir.params)
print(f"\nDoes init have voltage param? {has_voltage}")
