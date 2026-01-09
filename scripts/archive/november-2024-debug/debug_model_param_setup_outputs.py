#!/usr/bin/env python3
"""Debug: inspect the model_param_setup intern structure."""

import openvaf_py

# Compile resistor
modules = openvaf_py.compile_va("vendor/VACASK/devices/resistor.va")
resistor = modules[0]

print("="*80)
print("Debug: model_param_setup parameter structure")
print("="*80)

print(f"\nmodel_setup_num_params: {resistor.model_setup_num_params}")
print(f"\nmodel_setup_param_names:")
for i, name in enumerate(resistor.model_setup_param_names):
    kind = resistor.model_setup_param_kinds[i] if i < len(resistor.model_setup_param_kinds) else "?"
    print(f"  [{i}] {kind:15} -> {name}")

print(f"\nmodel_param_setup outputs (from debug method):")
outputs = resistor.debug_model_setup_outputs()
for kind_str, value_idx in outputs:
    print(f"  {kind_str} -> v{value_idx}")

print(f"\nAll PHI nodes in model_param_setup:")
phi_nodes = resistor.debug_model_setup_phi_nodes()
for value_idx in phi_nodes:
    print(f"  v{value_idx}")

print("\n" + "="*80)
print("Test simple case:")
print("="*80)

# Simple test: r=100, given
result = resistor.run_model_param_setup({
    'r': 100.0,
    'r_given': 1.0,
    'has_noise': 5.0,
    'has_noise_given': 1.0
})

print(f"Input: r=100 (given), has_noise=5 (given)")
print(f"Result: {result}")
print(f"Expected: r=100, has_noise=5")

# The issue is probably in how we're matching Parameter to param names
# The interner params has both param and param_given for the same param name
# We need to match correctly
