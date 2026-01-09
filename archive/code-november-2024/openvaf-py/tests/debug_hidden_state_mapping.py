#!/usr/bin/env python3
"""Debug script to understand hidden_state variable mapping between init and eval."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openvaf_jax import OpenVAFToJAX

# Load PSP103 model
va_path = os.path.join(os.path.dirname(__file__), '..', 'vendor', 'OpenVAF', 'integration_tests', 'PSP103', 'psp103.va')

print("Loading PSP103 model...")
translator = OpenVAFToJAX.from_file(va_path)
module = translator.module

# Get eval param metadata
param_names = list(module.param_names)
param_kinds = list(module.param_kinds)

# Check the params array (variable IDs in eval)
print(f"\n=== Eval params array (translator.params) ===")
print(f"Length: {len(translator.params)}")
print(f"First 50 entries (corresponding to param indices 0-49):")
for i in range(min(50, len(translator.params))):
    var_id = translator.params[i]
    kind = param_kinds[i] if i < len(param_kinds) else 'unknown'
    name = param_names[i] if i < len(param_names) else f"idx_{i}"
    print(f"  params[{i}] = {var_id} -> {name} ({kind})")

# Check init's params array
init_param_names = list(module.init_param_names)
init_params = list(translator.init_params) if hasattr(translator, 'init_params') else []
print(f"\n=== Init params array (translator.init_params) ===")
print(f"Length: {len(init_params) if init_params else 'N/A'}")
if init_params:
    print(f"First 30 entries:")
    for i in range(min(30, len(init_params))):
        var_id = init_params[i]
        name = init_param_names[i] if i < len(init_param_names) else f"idx_{i}"
        print(f"  init_params[{i}] = {var_id} -> {name}")

# Check cache_mapping - this maps init outputs to eval params
print(f"\n=== Cache Mapping Analysis ===")
print(f"cache_mapping links init values to eval param indices")
# Generate init to populate cache_mapping
init_fn, init_metadata = translator.translate_init_array()
print(f"cache_mapping count: {len(translator.cache_mapping)}")

# Look for hidden_state params in cache_mapping
print(f"\nCache mapping entries for hidden_state params (by eval_param index):")
hidden_state_indices = [i for i, k in enumerate(param_kinds) if k == 'hidden_state']
mapped_hidden_states = []
for mapping in translator.cache_mapping:
    eval_idx = mapping['eval_param']
    if eval_idx in hidden_state_indices:
        name = param_names[eval_idx] if eval_idx < len(param_names) else f"idx_{eval_idx}"
        mapped_hidden_states.append((eval_idx, mapping['init_value'], name))
        if len(mapped_hidden_states) <= 20:
            print(f"  eval[{eval_idx}] {name} <- init {mapping['init_value']}")

print(f"\nTotal hidden_state params mapped via cache_mapping: {len(mapped_hidden_states)}")
print(f"Total hidden_state params (param_kinds): {len(hidden_state_indices)}")

# Check what _build_hidden_state_assignments does
print(f"\n=== Value-Number Matching Analysis ===")
print("This is how _generate_core_code links init outputs to eval hidden_state")

# Get init defined vars
init_code_lines = translator._generate_init_code_array()
init_defined = set()
for line in init_code_lines:
    if ' = ' in line and line.strip().startswith('v'):
        # Extract variable name before ' = '
        var = line.strip().split(' = ')[0].strip()
        init_defined.add(var)

print(f"Init defines {len(init_defined)} variables")

# Now check: for each hidden_state param, does eval use the same var ID as init computes?
print(f"\nChecking if eval hidden_state var IDs match init computed var IDs:")
matches = []
mismatches = []
for idx in hidden_state_indices[:50]:  # First 50
    if idx < len(translator.params):
        eval_var = translator.params[idx]
        name = param_names[idx] if idx < len(param_names) else f"idx_{idx}"
        in_init = eval_var in init_defined
        if in_init:
            matches.append((idx, eval_var, name))
        else:
            mismatches.append((idx, eval_var, name))

print(f"\nMatches (init computes same var ID as eval expects):")
for idx, var, name in matches[:10]:
    print(f"  [{idx}] {name}: eval uses {var}, init computes {var} ✓")

print(f"\nMismatches (init does NOT compute the var ID eval expects):")
for idx, var, name in mismatches[:10]:
    print(f"  [{idx}] {name}: eval uses {var}, init does NOT compute {var} ✗")

print(f"\nSummary: {len(matches)} matches, {len(mismatches)} mismatches (of first 50 hidden_state params)")

# The key insight: even if var IDs match, they may compute different things!
# Let's check what v177 computes in init vs what it should be for TOXO_i
print(f"\n=== Checking v177 (TOXO_i) specifically ===")
for line in init_code_lines:
    if 'v177' in line and '=' in line:
        print(f"Init code: {line.strip()}")
        break

# Look for TOXO input in init
print(f"\nTOXO input in init:")
for i, name in enumerate(init_param_names):
    if name.upper() == 'TOXO':
        print(f"  init_param[{i}] = TOXO")
        # Check what var this maps to in the inputs
        for line in init_code_lines:
            if f'inputs[{i}]' in line:
                print(f"  Init code: {line.strip()}")
                break

print("\n=== Done ===")
