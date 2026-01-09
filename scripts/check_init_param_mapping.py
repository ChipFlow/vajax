#!/usr/bin/env python3
"""Check correct init parameter mapping."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import openvaf_py

# Compile capacitor
modules = openvaf_py.compile_va(str(Path(__file__).parent.parent / 'vendor/VACASK/devices/capacitor.va'))
cap = modules[0]

print("="*80)
print("UNDERSTANDING INIT PARAMETERS")
print("="*80)

# Get metadata
metadata = cap.get_codegen_metadata()

print("\nMetadata init_param_mapping:")
for name, var in metadata['init_param_mapping'].items():
    print(f"  {name} → {var}")

# Get init MIR
init_mir_dict = cap.get_init_mir_instructions()

print("\nInit MIR params:")
for i, param in enumerate(init_mir_dict.get('params', [])):
    print(f"  Param {i}: {param}")

print("\n" + "="*80)
print("SEMANTIC TO MIR MAPPING")
print("="*80)

print("\nAll model parameters:")
for name, value_idx, kind in zip(cap.param_names, cap.param_value_indices, cap.param_kinds):
    print(f"  {name:20} v{value_idx:3}  kind={kind}")

print("\n" + "="*80)
print("ANALYSIS")
print("="*80)

print("""
Init MIR expects: v18, v20, v32
Metadata says: c → v32, mfactor → v20

So what is v18?

Looking at the generated code:
- if v32: use v18 (c_given=True path)
- else: use v38=1e-12 (c_given=False path)

So v32 is the c_given FLAG, not the value!
And v18 is the actual capacitance VALUE!

The correct mapping should be:
- v18 = c (the value)
- v32 = c_given (the flag)
- v20 = mfactor

But metadata says c → v32, which is wrong!
Let me check what v32 actually is in the semantic parameters...
""")

for name, value_idx in zip(cap.param_names, cap.param_value_indices):
    if value_idx == 32:
        print(f"\nv32 corresponds to: {name}")
    if value_idx == 18:
        print(f"v18 corresponds to: {name}")
    if value_idx == 20:
        print(f"v20 corresponds to: {name}")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)

print("""
If v32 = c_given (a boolean flag), then metadata is mapping the WRONG thing.
Metadata maps semantic 'c' to v32, but v32 is the given flag, not the value!

The init function needs BOTH:
1. The capacitance value (v18)
2. The given flag (v32)

We need to pass BOTH to setup_instance.
""")
