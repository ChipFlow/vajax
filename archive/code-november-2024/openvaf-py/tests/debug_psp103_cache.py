#!/usr/bin/env python3
"""Debug PSP103 init cached values to find NaN source."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import openvaf_py
import json
import numpy as np

# Compile PSP103
psp103_va = Path(__file__).parent.parent / "vendor" / "OpenVAF" / "integration_tests" / "PSP103" / "psp103.va"
modules = openvaf_py.compile_va(str(psp103_va))
psp103 = modules[0]

print("=" * 80)
print("PSP103 Init Cache Value Inspection")
print("=" * 80)
print()

# Load model parameters
params_file = Path(__file__).parent.parent.parent / "scripts" / "psp103_model_params.json"
with open(params_file) as f:
    all_params = json.load(f)
pmos_model_params = all_params['pmos']

# Build init params
init_params = {}
for name in psp103.init_param_names:
    if name == '$temperature':
        init_params[name] = 300.0
    elif name.upper() == 'TYPE':
        init_params[name] = -1.0  # PMOS
    elif name.upper() == 'W':
        init_params[name] = 20e-6
    elif name.upper() == 'L':
        init_params[name] = 1e-6
    elif name.lower() == 'mfactor':
        init_params[name] = 1.0
    elif name in pmos_model_params:
        init_params[name] = pmos_model_params[name]
    else:
        init_params[name] = 0.0

print(f"Init params: {len(init_params)} entries")
print()

# Get cached values from init
print("Running init and extracting cached values...")
cache_values = psp103.debug_init_cache(init_params)
print(f"Found {len(cache_values)} cached values")
print()

# Check for NaN and inf
nan_values = [(idx, val) for idx, val in cache_values if np.isnan(val)]
inf_values = [(idx, val) for idx, val in cache_values if np.isinf(val)]
zero_values = [(idx, val) for idx, val in cache_values if val == 0.0]

print(f"NaN values: {len(nan_values)}")
if nan_values:
    print("ALL NaN cached values:")
    for idx, val in nan_values:
        print(f"  cache[{idx}] = {val}")
    print()

print(f"Inf values: {len(inf_values)}")
if inf_values:
    print("First 10 Inf cached values:")
    for idx, val in inf_values[:10]:
        print(f"  cache[{idx}] = {val}")
    print()

print(f"Zero values: {len(zero_values)}")
print()

# Show statistics
vals = [val for _, val in cache_values]
print("Cache value statistics:")
print(f"  Min: {np.min(vals):.6e}")
print(f"  Max: {np.max(vals):.6e}")
print(f"  Mean: {np.mean(vals):.6e}")
print(f"  Std: {np.std(vals):.6e}")
print()

# Show first 20 cached values
print("First 20 cached values:")
for idx, val in cache_values[:20]:
    print(f"  cache[{idx}] = {val:.6e}")
