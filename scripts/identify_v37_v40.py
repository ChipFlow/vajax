#!/usr/bin/env python3
"""Identify what v37 and v40 are in the parameter list."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import openvaf_py

# Compile capacitor
print("Compiling capacitor model...")
modules = openvaf_py.compile_va(str(Path(__file__).parent.parent / 'vendor/VACASK/devices/capacitor.va'))
cap = modules[0]

print("\n" + "="*80)
print("FULL PARAMETER MAPPING")
print("="*80)

# Get all parameter info
print("\nAll parameters (semantic name, value index, kind):")
for name, value_idx, kind in zip(cap.param_names, cap.param_value_indices, cap.param_kinds):
    print(f"  {name:20} → v{value_idx:3}    kind={kind}")

print("\n" + "="*80)
print("IDENTIFYING v37 and v40")
print("="*80)

# Find what v37 and v40 are
v37_name = None
v40_name = None
for name, value_idx, kind in zip(cap.param_names, cap.param_value_indices, cap.param_kinds):
    if value_idx == 37:
        v37_name = name
        v37_kind = kind
    if value_idx == 40:
        v40_name = name
        v40_kind = kind

print(f"\nv37 = {v37_name} (kind={v37_kind if v37_name else 'N/A'})")
print(f"v40 = {v40_name} (kind={v40_kind if v40_name else 'N/A'})")

# Get eval MIR params
print("\n" + "="*80)
print("EVAL MIR PARAMS")
print("="*80)
eval_mir_dict = cap.get_mir_instructions()
print("\nParams in eval MIR:")
for i, param in enumerate(eval_mir_dict.get('params', [])):
    print(f"  Param {i}: {param}")

# Check metadata
print("\n" + "="*80)
print("METADATA PARAM MAPPING")
print("="*80)
metadata = cap.get_codegen_metadata()
print("\nEval param mapping (excludes hidden_state):")
for name, var in metadata['eval_param_mapping'].items():
    print(f"  {name:20} → {var}")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print("""
The Jacobian react variables are:
  v30 = optbarrier(v37)
  v38 = optbarrier(v40)
  v32 = optbarrier(v40)
  v41 = optbarrier(v37)

If v37 and v40 are hidden_state parameters, then:
1. They were filtered out of eval_param_mapping
2. They should be derivatives (dQ/dV)
3. Setting them to 0.0 makes Jacobian = 0 (wrong!)

If v37 and v40 are something else, we need to understand what.
""")
