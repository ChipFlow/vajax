#!/usr/bin/env python3
"""Debug PSP103 init function to see what cached values it produces."""

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
print("PSP103 Init Function Value Inspection")
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

# Try to get access to the actual init interpreter state
# We need to modify lib.rs to expose this...
print("To debug this properly, we need to:")
print("1. Add a method to VaModule that returns init cached values")
print("2. Check for NaN/inf in those values")
print("3. Find which cache value is NaN and trace back to its computation")
print()
print("The issue is likely:")
print("  - Division by zero in init (e.g., 1/L where L=0)")
print("  - log/sqrt of negative number")
print("  - Missing parameter causing NaN propagation")
