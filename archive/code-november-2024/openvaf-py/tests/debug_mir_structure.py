#!/usr/bin/env python3
"""Debug script to examine MIR structure for hidden_state mapping."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openvaf_jax import OpenVAFToJAX

# Load PSP103 model
va_path = os.path.join(os.path.dirname(__file__), '..', 'vendor', 'OpenVAF', 'integration_tests', 'PSP103', 'psp103.va')

print("Loading PSP103 model...")
translator = OpenVAFToJAX.from_file(va_path)
module = translator.module

# Check what keys are in init_mir_data
print(f"\n=== Init MIR Data Keys ===")
init_mir = translator.init_mir_data
print(f"Keys: {list(init_mir.keys())}")

# Check cache_mapping in detail
print(f"\n=== Cache Mapping ===")
cache_mapping = init_mir.get('cache_mapping', [])
print(f"Count: {len(cache_mapping)}")
if cache_mapping:
    # What's the range of eval_param indices?
    eval_params = [m['eval_param'] for m in cache_mapping]
    print(f"Min eval_param: {min(eval_params)}")
    print(f"Max eval_param: {max(eval_params)}")
    print(f"\nFirst 5 entries:")
    for m in cache_mapping[:5]:
        print(f"  init_value={m['init_value']} -> eval_param={m['eval_param']}")

# Check if there's hidden_state mapping in MIR
print(f"\n=== Looking for hidden_state mapping in MIR ===")
for key in init_mir.keys():
    if 'hidden' in key.lower() or 'state' in key.lower():
        print(f"  Found: {key}")

# Check init params structure
print(f"\n=== Init Params Info ===")
init_params = init_mir.get('func_params', [])
print(f"func_params count: {len(init_params)}")

# Look at param_names in both init and eval MIR
print(f"\n=== Comparing Init and Eval Params ===")
eval_mir = translator.mir_data
eval_param_names = list(module.param_names)
init_param_names = list(module.init_param_names)

print(f"Eval param_names count: {len(eval_param_names)}")
print(f"Init param_names count: {len(init_param_names)}")

# Find hidden_state params in eval that have matching names in init
param_kinds = list(module.param_kinds)
hidden_state_indices = [i for i, k in enumerate(param_kinds) if k == 'hidden_state']

print(f"\n=== Searching for Name-Based Mapping ===")
# For hidden_state params like TOXO_i, the corresponding init value might be TOXO
# Let's see if we can find the mapping

matches_found = []
for idx in hidden_state_indices[:30]:
    eval_name = eval_param_names[idx] if idx < len(eval_param_names) else f"idx_{idx}"
    # Try to find matching init param by name pattern
    base_name = eval_name.replace('_i', '').replace('_p', '').lower()

    for i, init_name in enumerate(init_param_names):
        if init_name.lower() == base_name:
            matches_found.append((idx, eval_name, i, init_name))
            break

print(f"Name-based matches found: {len(matches_found)}")
for idx, eval_name, init_idx, init_name in matches_found[:10]:
    print(f"  eval[{idx}] {eval_name} <- init[{init_idx}] {init_name}")

# Check the init MIR for output structure
print(f"\n=== Init Function Outputs ===")
if 'outputs' in init_mir:
    print(f"outputs: {init_mir['outputs'][:10]}...")
if 'return_values' in init_mir:
    print(f"return_values: {init_mir['return_values'][:10]}...")

# Check for a "values" or similar mapping
print(f"\n=== All Init MIR Keys with Values ===")
for key, val in init_mir.items():
    if isinstance(val, (list, dict)):
        if isinstance(val, list):
            print(f"  {key}: list[{len(val)}]")
        else:
            print(f"  {key}: dict with keys {list(val.keys())[:5]}...")
    else:
        print(f"  {key}: {type(val).__name__}")

print("\n=== Done ===")
