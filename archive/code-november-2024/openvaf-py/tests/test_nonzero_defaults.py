#!/usr/bin/env python3
"""Test PSP103 with non-zero defaults for unmapped parameters."""

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
print("PSP103 Test with Non-Zero Defaults for Unmapped Parameters")
print("=" * 80)
print()

# Parse VACASK models.inc
models_inc = Path("/Users/roberttaylor/Code/ChipFlow/reference/VACASK/benchmark/ring/vacask/models.inc")
with open(models_inc) as f:
    content = f.read()

psp103p_match = re.search(r'model psp103p psp103va \((.*?)\)', content, re.DOTALL)
param_block = psp103p_match.group(1)
default_params = {}

for line in param_block.strip().split('\n'):
    line = line.strip()
    if '=' in line and not line.startswith('//'):
        if '//' in line:
            line = line.split('//')[0].strip()
        parts = line.split('=')
        if len(parts) == 2:
            param_name = parts[0].strip()
            param_value = parts[1].strip()
            try:
                default_params[param_name] = float(param_value)
            except:
                pass

print(f"Extracted {len(default_params)} parameters from VACASK models.inc\n")

# Build init params with NON-ZERO defaults for unmapped params
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
    elif name.lower() in default_params:
        init_params[name] = default_params[name.lower()]
    else:
        # Instead of 0.0, use 1e-99 (tiny but non-zero)
        init_params[name] = 1e-99

# Test init
print("Running init...")
cache_values = psp103.debug_init_cache(init_params)
print(f"Got {len(cache_values)} cached values\n")

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

# If successful, test eval
if not nan_values:
    print("Testing eval at Vgs=-1.2V...")

    eval_params = init_params.copy()
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
            eval_params[name] = 1e-99

    try:
        residuals, jacobian = psp103.run_init_eval(eval_params)
        print(f"  Got {len(residuals)} residuals")
        print("  First 4 terminal currents:")
        for i in range(min(4, len(residuals))):
            resist, react = residuals[i]
            print(f"    Terminal {i}: I={resist:.6e} A")

        if all(abs(r) < 1e-30 for r, q in residuals):
            print("\n  ❌ All currents still zero")
        else:
            print("\n  ✅ Non-zero currents!")
    except Exception as e:
        print(f"\n  ❌ Error: {e}")
