#!/usr/bin/env python3
"""Check what parameters native openvaf-py init expects."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import openvaf_py

# Compile capacitor
modules = openvaf_py.compile_va(str(Path(__file__).parent.parent / 'vendor/VACASK/devices/capacitor.va'))
cap = modules[0]

print("="*80)
print("NATIVE OPENVAF-PY TEST")
print("="*80)

# Try calling with different parameter combinations
test_cases = [
    {"c": 1e-9, "c_given": True, "mfactor": 1.0},
    {"c": 1e-9, "c_given": False, "mfactor": 1.0},
    {"c": 2e-9, "c_given": True, "mfactor": 2.0},
]

for i, params in enumerate(test_cases):
    print(f"\nTest {i+1}: {params}")
    try:
        residuals, jacobian = cap.run_init_eval(params)
        print(f"  ✓ Success")
        print(f"    Jacobian[0,0] react: {jacobian[0][3]}")

        # The reactive jacobian should be ±c*mfactor
        expected = params['c'] * params['mfactor'] if params['c_given'] else 1e-12 * params['mfactor']
        actual = abs(jacobian[0][3])
        print(f"    Expected: {expected}, Actual: {actual}, Match: {abs(expected - actual) < 1e-15}")
    except Exception as e:
        print(f"  ✗ Failed: {e}")

print("\n" + "="*80)
print("UNDERSTANDING INIT PARAMETER TRANSFORMATION")
print("="*80)

print("""
OpenVAF internally transforms parameters between eval and init.
For capacitor:

Semantic parameters (from VA file):
  - c (value)
  - c_given (flag)
  - mfactor

Eval MIR (runtime parameters):
  - v16 = c
  - v17 = V(A,B)
  - v25 = mfactor

Init MIR (compile-time parameters):
  - v18 = ? (some parameter)
  - v20 = ? (some parameter)
  - v32 = ? (some parameter)

The metadata init_param_mapping says:
  - c → v32
  - mfactor → v20

So v32 is mapped to 'c'. But looking at the generated code,
v32 is used as a boolean condition (if v32:), suggesting it's c_given!

This means OpenVAF's metadata might be incorrect, or we're misinterpreting it.

Let me check if v32 in init corresponds to c_given by checking the control flow...
""")

# Check if there's a param kind that tells us about given flags
print("\nChecking parameter kinds:")
for name, value_idx, kind in zip(cap.param_names, cap.param_value_indices, cap.param_kinds):
    print(f"  {name:20} v{value_idx:3}  kind={kind}")
    if 'given' in name.lower():
        print(f"    → This is a given flag!")

print("\n" + "="*80)
print("HYPOTHESIS")
print("="*80)

print("""
The init MIR signature is (v18, v20, v32).
Based on control flow analysis:
  - v32 is used as a boolean (if v32:) → likely c_given
  - v18 is selected when v32=True → likely c value
  - v20 is used in multiplication → likely mfactor

So the CORRECT mapping should be:
  - v18 → c (value)
  - v20 → mfactor
  - v32 → c_given (flag)

But metadata says c → v32, which conflicts with this analysis.

The bug is either:
1. Metadata is wrong (OpenVAF bug)
2. We're misinterpreting the metadata
3. Init parameters are NOT semantic parameters but transformed versions
""")
