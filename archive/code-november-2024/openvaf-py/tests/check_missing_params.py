#!/usr/bin/env python3
"""Check if we're missing any required PSP103 init parameters."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import openvaf_py
import json

# Compile PSP103
psp103_va = Path(__file__).parent.parent / "vendor" / "OpenVAF" / "integration_tests" / "PSP103" / "psp103.va"
modules = openvaf_py.compile_va(str(psp103_va))
psp103 = modules[0]

# Load model parameters
params_file = Path(__file__).parent.parent.parent / "scripts" / "psp103_model_params.json"
with open(params_file) as f:
    all_params = json.load(f)
pmos_model_params = all_params['pmos']

print(f"Total init params expected: {len(psp103.init_param_names)}")
print()

# Build init params
init_params = {}
missing_params = []

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
        missing_params.append(name)

print(f"Parameters set to 0.0 (not in model file): {len(missing_params)}")
if missing_params:
    print("\nFirst 20 missing parameters:")
    for name in missing_params[:20]:
        print(f"  {name}")
    print()

# Check specific early params that might affect cache indices 12340-12350
print("\nFirst 30 init parameters:")
for i, name in enumerate(psp103.init_param_names[:30]):
    value = init_params.get(name, "NOT SET")
    print(f"  [{i:3d}] {name:20s} = {value}")
