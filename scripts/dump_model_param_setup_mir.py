#!/usr/bin/env python3
"""Dump model_param_setup MIR to understand its structure."""

import sys
import openvaf_py

# Compile resistor
modules = openvaf_py.compile_va("vendor/VACASK/devices/resistor.va")
resistor = modules[0]

print("="*80)
print("Resistor model_param_setup Function Info")
print("="*80)

print(f"\nNumber of params: {resistor.model_setup_num_params}")
print(f"Param names: {resistor.model_setup_param_names}")
print(f"Param kinds: {resistor.model_setup_param_kinds}")

print("\n" + "="*80)
print("model_param_intern.params mapping:")
print("="*80)
# We can't directly access model_param_intern from Python, but we extracted the names

print("\nFrom compiled extraction:")
for i, (name, kind) in enumerate(zip(resistor.model_setup_param_names, resistor.model_setup_param_kinds)):
    print(f"  Param{i}: {kind:15} -> {name}")

# Now let's understand the structure by looking at defaults
print("\n" + "="*80)
print("Parameter defaults:")
print("="*80)
defaults = resistor.get_param_defaults()
for name, val in defaults.items():
    print(f"  {name}: {val}")

print("\n" + "="*80)
print("Test: What does run_model_param_setup return for different inputs?")
print("="*80)

# Test 1: All given
print("\nTest 1: All parameters given")
result = resistor.run_model_param_setup({
    'r': 1000.0,
    'r_given': 1.0,
    'has_noise': 1.0,
    'has_noise_given': 1.0
})
print(f"  Input: r=1000, r_given=1, has_noise=1, has_noise_given=1")
print(f"  Result: {result}")

# Test 2: r not given
print("\nTest 2: r not given (should default to 1.0)")
result = resistor.run_model_param_setup({
    'r': 999.0,  # Should be ignored
    'r_given': 0.0,
    'has_noise': 1.0,
    'has_noise_given': 1.0
})
print(f"  Input: r=999 (ignored), r_given=0, has_noise=1, has_noise_given=1")
print(f"  Result: {result}")
print(f"  Expected r=1.0, got r={result.get('r', 'MISSING')}")

# The issue: we're reading Param values (inputs) instead of output values
# We need to identify which Values are the "output" parameters
