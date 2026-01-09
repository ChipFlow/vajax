#!/usr/bin/env python3
"""Test PSP103 with proper default parameter values from VACASK models.inc."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import openvaf_py
import json
import numpy as np
import re

# Compile PSP103
psp103_va = Path(__file__).parent.parent / "vendor" / "OpenVAF" / "integration_tests" / "PSP103" / "psp103.va"
modules = openvaf_py.compile_va(str(psp103_va))
psp103 = modules[0]

print("=" * 80)
print("PSP103 Test with VACASK Default Parameters")
print("=" * 80)
print()

# Parse VACASK models.inc to extract default parameters
models_inc = Path("/Users/roberttaylor/Code/ChipFlow/reference/VACASK/benchmark/ring/vacask/models.inc")
with open(models_inc) as f:
    content = f.read()

# Extract psp103p model parameters
psp103p_match = re.search(r'model psp103p psp103va \((.*?)\)', content, re.DOTALL)
if not psp103p_match:
    print("ERROR: Could not find psp103p model in models.inc")
    sys.exit(1)

param_block = psp103p_match.group(1)
default_params = {}

# Parse parameter=value lines
for line in param_block.strip().split('\n'):
    line = line.strip()
    if '=' in line:
        # Remove trailing comment if present
        if '//' in line:
            line = line.split('//')[0].strip()
        parts = line.split('=')
        if len(parts) == 2:
            param_name = parts[0].strip()
            param_value = parts[1].strip()
            try:
                default_params[param_name] = float(param_value)
            except ValueError:
                print(f"Warning: Could not parse value for {param_name}: {param_value}")

print(f"Extracted {len(default_params)} default parameters from VACASK models.inc")
print()

# Load our model parameters (if any)
params_file = Path(__file__).parent.parent.parent / "scripts" / "psp103_model_params.json"
if params_file.exists():
    with open(params_file) as f:
        all_params = json.load(f)
    pmos_model_params = all_params.get('pmos', {})
    print(f"Loaded {len(pmos_model_params)} parameters from psp103_model_params.json")
else:
    pmos_model_params = {}
    print("No psp103_model_params.json found, using only VACASK defaults")
print()

# Build init params with proper defaults
init_params = {}
using_default = 0
using_model_file = 0
unmapped_params = []

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
        using_model_file += 1
    elif name.lower() in default_params:
        init_params[name] = default_params[name.lower()]
        using_default += 1
    else:
        # Still set to 0.0 if not found anywhere
        init_params[name] = 0.0
        unmapped_params.append(name)

print(f"Parameter sources:")
print(f"  From model file:     {using_model_file}")
print(f"  From VACASK default: {using_default}")
print(f"  Still using 0.0:     {len(unmapped_params)}")
print()

if unmapped_params:
    print("First 30 unmapped parameters (set to 0.0):")
    for name in unmapped_params[:30]:
        print(f"  {name}")
    print()

# Test init
print("Running init with proper defaults...")
cache_values = psp103.debug_init_cache(init_params)
print(f"Got {len(cache_values)} cached values")
print()

# Check for NaN
nan_values = [(idx, val) for idx, val in cache_values if np.isnan(val)]
inf_values = [(idx, val) for idx, val in cache_values if np.isinf(val)]

print(f"NaN values: {len(nan_values)}")
if nan_values:
    print("  First 10 NaN:")
    for idx, val in nan_values[:10]:
        print(f"    cache[{idx}] = {val}")
else:
    print("  ✅ No NaN values!")
print()

print(f"Inf values: {len(inf_values)}")
if inf_values:
    print("  First 10 Inf:")
    for idx, val in inf_values[:10]:
        print(f"    cache[{idx}] = {val}")
else:
    print("  ✅ No Inf values!")
print()

# If no NaN/Inf, test eval
if not nan_values and not inf_values:
    print("Testing eval at Vgs=-1.2V (PMOS should conduct)...")

    # Build eval params
    eval_params = init_params.copy()
    # Voltages: Vd=0, Vg=0, Vs=1.2, Vb=1.2 → Vgs=-1.2V
    for name in psp103.param_names:
        if name == 'V(d)':
            eval_params[name] = 0.0
        elif name == 'V(g)':
            eval_params[name] = 0.0
        elif name == 'V(s)':
            eval_params[name] = 1.2
        elif name == 'V(b)':
            eval_params[name] = 1.2
        elif name not in eval_params:
            eval_params[name] = 0.0

    try:
        residuals, jacobian = psp103.run_init_eval(eval_params)

        print(f"  Got {len(residuals)} residuals")
        print("  First 4 terminal currents:")
        for i in range(min(4, len(residuals))):
            resist, react = residuals[i]
            print(f"    Terminal {i}: I={resist:.6e} A, Q={react:.6e} C")

        if all(abs(r) < 1e-30 and abs(q) < 1e-30 for r, q in residuals):
            print("\n  ❌ ERROR: All currents are zero (should have drain current!)")
        else:
            print("\n  ✅ SUCCESS: Non-zero currents detected!")
    except Exception as e:
        print(f"\n  ❌ ERROR in eval: {e}")
        import traceback
        traceback.print_exc()
