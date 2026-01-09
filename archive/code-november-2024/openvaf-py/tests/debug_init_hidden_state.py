#!/usr/bin/env python3
"""Debug script to trace init function hidden_state computation for PSP103."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax.numpy as jnp
from openvaf_jax import OpenVAFToJAX

# Load PSP103 model
va_path = os.path.join(os.path.dirname(__file__), '..', 'vendor', 'OpenVAF', 'integration_tests', 'PSP103', 'psp103.va')

print("Loading PSP103 model...")
translator = OpenVAFToJAX.from_file(va_path)
module = translator.module

# Get eval param metadata
print(f"\n=== Eval Param Metadata ===")
param_names = list(module.param_names)
param_kinds = list(module.param_kinds)
print(f"param_names count: {len(param_names)}")

# Count param kinds
kind_counts = {}
for kind in param_kinds:
    kind_counts[kind] = kind_counts.get(kind, 0) + 1
print(f"\nParam kind counts: {kind_counts}")

# Find hidden_state params and their indices
hidden_state_indices = [i for i, k in enumerate(param_kinds) if k == 'hidden_state']
print(f"\nHidden state indices count: {len(hidden_state_indices)}")

# Check first few hidden_state params
print(f"\nFirst 20 hidden_state params:")
for idx in hidden_state_indices[:20]:
    name = param_names[idx] if idx < len(param_names) else f"idx_{idx}"
    print(f"  [{idx}] {name}")

# Generate init function (this populates cache_mapping and hidden_state_cache_mapping)
print(f"\n=== Init Function Analysis ===")
init_fn, init_metadata = translator.translate_init_array()

# Get code for inspection
init_code_lines = translator._generate_init_code_array()
init_code = '\n'.join(init_code_lines)

# Check cache_mapping
print(f"\ncache_mapping count: {len(translator.cache_mapping)}")
if translator.cache_mapping:
    print(f"First 5 cache_mapping entries:")
    for i, m in enumerate(translator.cache_mapping[:5]):
        print(f"  [{i}] init_value={m['init_value']} -> eval_param={m['eval_param']}")

# Check hidden_state_cache_mapping
hidden_state_mapping = getattr(translator, '_hidden_state_cache_mapping', [])
print(f"\nhidden_state_cache_mapping count: {len(hidden_state_mapping)}")
if hidden_state_mapping:
    print(f"First 20 hidden_state_cache_mapping entries:")
    for i, (eval_var, param_idx) in enumerate(hidden_state_mapping[:20]):
        pname = param_names[param_idx] if param_idx < len(param_names) else f"idx_{param_idx}"
        print(f"  cache[{462 + i}] = {eval_var} (param[{param_idx}] = {pname})")

# Find TOXO_i specifically
print(f"\n=== Looking for TOXO_i ===")
for i, (eval_var, param_idx) in enumerate(hidden_state_mapping):
    pname = param_names[param_idx] if param_idx < len(param_names) else f"idx_{param_idx}"
    if 'TOX' in pname.upper():
        print(f"  cache[{462 + i}] = {eval_var} -> param[{param_idx}] = {pname}")

# Check init code for specific variables
print(f"\n=== Searching init code for v177, v178, v181 ===")
for i, line in enumerate(init_code.split('\n')):
    if ('v177' in line or 'v178' in line or 'v181' in line) and '=' in line:
        print(f"  Line {i+1}: {line.strip()}")

# Check what init function input params look like
print(f"\n=== Init function input params ===")
init_param_names = list(module.init_param_names)
init_param_kinds = list(module.init_param_kinds)
print(f"Init param count: {len(init_param_names)}")
print(f"\nLooking for TOX* params in init inputs:")
for i, name in enumerate(init_param_names):
    if 'TOX' in name.upper():
        kind = init_param_kinds[i] if i < len(init_param_kinds) else 'unknown'
        print(f"  init_param[{i}] {name} (kind={kind})")

# Use the compiled init function directly
print(f"\n=== Init Function Ready ===")

# Get param defaults
param_defaults = dict(module.get_param_defaults())
print(f"Got {len(param_defaults)} param defaults")

# Show TOX defaults
print(f"\nTOX-related defaults:")
for name, val in param_defaults.items():
    if 'TOX' in name.upper():
        print(f"  {name} = {val:.6e}")

# Build init params array
print(f"\n=== Build Init Params Array ===")
init_params = []
for i, name in enumerate(init_param_names):
    kind = init_param_kinds[i] if i < len(init_param_kinds) else 'unknown'
    if kind == 'param':
        # Try to find this param in defaults
        val = param_defaults.get(name.lower(), param_defaults.get(name, 0.0))
        init_params.append(val)
    elif kind == 'temperature':
        init_params.append(300.15)  # Room temp
    elif kind == 'param_given':
        # Check if param was given
        base_name = name.replace('_given', '')
        init_params.append(1.0 if base_name.lower() in param_defaults else 0.0)
    else:
        init_params.append(0.0)

init_params = jnp.array(init_params)
print(f"Built init_params array with shape {init_params.shape}")

# Show TOX values in array
print(f"\nTOX values in init_params:")
for i, name in enumerate(init_param_names):
    if 'TOX' in name.upper():
        val = float(init_params[i])
        print(f"  init_params[{i}] {name} = {val:.6e}")

# Call init
print(f"\n=== Calling Init Function ===")
try:
    cache_result, collapse_result = init_fn(init_params)
    print(f"cache result shape: {cache_result.shape}")
    print(f"collapse result shape: {collapse_result.shape}")

    # Check the hidden_state values
    print(f"\nHidden state values in cache (starting at index 462):")
    for i, (eval_var, param_idx) in enumerate(hidden_state_mapping[:20]):
        pname = param_names[param_idx] if param_idx < len(param_names) else f"idx_{param_idx}"
        cache_idx = 462 + i
        if cache_idx < len(cache_result):
            val = float(cache_result[cache_idx])
            print(f"  cache[{cache_idx}] = {eval_var} ({pname}): {val:.6e}")
        else:
            print(f"  cache[{cache_idx}] = {eval_var} ({pname}): OUT OF BOUNDS")

    # Check for -inf or nan
    has_inf = jnp.any(jnp.isinf(cache_result))
    has_nan = jnp.any(jnp.isnan(cache_result))
    print(f"\nCache has inf: {has_inf}, has nan: {has_nan}")

    if has_inf:
        inf_indices = jnp.where(jnp.isinf(cache_result))[0]
        print(f"Inf at indices: {list(inf_indices[:10].tolist())}...")

    # Check specific cache values
    print(f"\nFirst 10 cache values:")
    for i in range(min(10, len(cache_result))):
        print(f"  cache[{i}] = {float(cache_result[i]):.6e}")

except Exception as e:
    print(f"Init function error: {e}")
    import traceback
    traceback.print_exc()

print("\n=== Done ===")
