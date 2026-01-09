#!/usr/bin/env python3
"""Debug PSP103 MIR interpreter to see where it fails."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import openvaf_py
import json

# Compile PSP103
psp103_va = Path(__file__).parent.parent / "vendor" / "OpenVAF" / "integration_tests" / "PSP103" / "psp103.va"
modules = openvaf_py.compile_va(str(psp103_va))
psp103 = modules[0]

print("=" * 80)
print("PSP103 MIR Interpreter Debug")
print("=" * 80)
print()

# Load model parameters
params_file = Path(__file__).parent.parent.parent / "scripts" / "psp103_model_params.json"
with open(params_file) as f:
    all_params = json.load(f)
pmos_model_params = all_params['pmos']

# Build params dict - try NMOS first (type=1) at V=0
params = {}
for name in psp103.param_names:
    if name == '$temperature':
        params[name] = 300.0
    elif name.upper() == 'TYPE':
        params[name] = 1.0  # NMOS
    elif name.upper() == 'W':
        params[name] = 20e-6
    elif name.upper() == 'L':
        params[name] = 1e-6
    elif name.lower() == 'mfactor':
        params[name] = 1.0
    elif 'V(' in name:
        params[name] = 0.0  # All voltages at zero
    elif name in pmos_model_params:
        # Use model params but this is NMOS so values might differ
        params[name] = pmos_model_params[name]
    else:
        params[name] = 0.0

print(f"Testing NMOS (type=1) at V=0 (all terminals grounded)")
print(f"  Params dict has {len(params)} entries")
print(f"  Expected behavior: Vgs=0, NMOS OFF, Ids≈0")
print()

# Call interpreter
try:
    residuals, jacobian = psp103.run_init_eval(params)

    print("Results:")
    print(f"  Residuals: {len(residuals)} entries")
    print(f"  First 4 residuals (terminal currents):")
    for i in range(min(4, len(residuals))):
        resist, react = residuals[i]
        print(f"    [{i}]: I={resist:.6e} A, Q={react:.6e} C")

    print()
    print(f"  Jacobian: {len(jacobian)} entries")
    print(f"  First 4 Jacobian entries:")
    for i in range(min(4, len(jacobian))):
        row, col, resist, react = jacobian[i]
        print(f"    J[{row},{col}]: dI/dV={resist:.6e} S, dQ/dV={react:.6e} F")

    # Check if all zeros
    all_zero = all(abs(r) < 1e-30 and abs(q) < 1e-30 for r, q in residuals)
    if all_zero:
        print()
        print("⚠️  ALL RESIDUALS ARE ZERO!")
        print("This is the bug - even at V=0, there should be some non-zero entries")
        print("(leakage currents, capacitances, etc.)")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
