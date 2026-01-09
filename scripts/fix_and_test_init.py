#!/usr/bin/env python3
"""Fix init parameter mapping and test."""

import sys
import math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import openvaf_py
from jax_spice.codegen.mir_parser import parse_mir_dict
from jax_spice.codegen.setup_instance_mir_codegen import generate_setup_instance_from_mir

# Compile capacitor
modules = openvaf_py.compile_va(str(Path(__file__).parent.parent / 'vendor/VACASK/devices/capacitor.va'))
cap = modules[0]

# Get init MIR
init_mir_dict = cap.get_init_mir_instructions()
init_mir = parse_mir_dict(init_mir_dict)

# Try MANUAL parameter mapping based on control flow analysis
# Based on the generated code:
# - v18 appears to be the capacitance value
# - v20 is mfactor
# - v32 is c_given flag (used as boolean in if statement)

manual_param_map = {
    'v18': 'c',           # Capacitance value
    'v20': 'mfactor',     # Multiplication factor
    'v32': 'c_given',     # Given flag
}

# Also add constants
for const_name in init_mir.constants.keys():
    manual_param_map[const_name] = const_name

cache_tuples = [(entry['init_value'], entry['eval_param']) for entry in init_mir_dict['cache_mapping']]

print("="*80)
print("GENERATING WITH MANUAL PARAMETER MAPPING")
print("="*80)
print("\nManual mapping:")
for mir_name, semantic_name in manual_param_map.items():
    if mir_name.startswith('v') and len(mir_name) <= 4:  # Only show vNN params, not constants
        print(f"  {mir_name} → {semantic_name}")

setup_instance_code = generate_setup_instance_from_mir(
    init_mir,
    manual_param_map,
    cache_tuples,
    'capacitor'
)

# Save the generated code
output_file = Path(__file__).parent / 'generated_setup_instance_capacitor_fixed.py'
output_file.write_text(setup_instance_code)
print(f"\nSaved to: {output_file}")

# Execute the generated code
namespace = {'math': math}
exec(setup_instance_code, namespace)
setup_instance_fn = namespace['setup_instance_capacitor']

print("\n" + "="*80)
print("TESTING GENERATED FUNCTION")
print("="*80)

test_cases = [
    {"name": "c=1e-9, c_given=True",  "params": {'c': 1e-9, 'c_given': True, 'mfactor': 1.0}},
    {"name": "c=1e-9, c_given=False", "params": {'c': 1e-9, 'c_given': False, 'mfactor': 1.0}},
    {"name": "c=2e-9, c_given=True",  "params": {'c': 2e-9, 'c_given': True, 'mfactor': 2.0}},
]

for test_case in test_cases:
    name = test_case['name']
    params = test_case['params']

    print(f"\nTest: {name}")
    print(f"  Params: {params}")

    # Call generated function
    try:
        cache = setup_instance_fn(**params)
        print(f"  Generated cache: {cache}")

        # Expected: ±(c * mfactor) if c_given, else ±(1e-12 * mfactor)
        if params.get('c_given', False):
            expected_mag = params['c'] * params['mfactor']
        else:
            expected_mag = 1e-12 * params['mfactor']

        actual_mag = abs(cache[0])
        match = abs(expected_mag - actual_mag) < 1e-15

        print(f"  Expected magnitude: {expected_mag}")
        print(f"  Actual magnitude:   {actual_mag}")
        print(f"  Match: {match} {'✓' if match else '✗'}")

    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "="*80)
print("COMPARING WITH NATIVE")
print("="*80)

for test_case in test_cases:
    name = test_case['name']
    params = test_case['params']

    print(f"\nTest: {name}")

    # Native
    residuals, jacobian = cap.run_init_eval(params)
    native_jac_react = jacobian[0][3]

    # Generated
    try:
        cache = setup_instance_fn(**params)
        gen_cache0 = cache[0]

        print(f"  Native Jacobian[0,0] react:  {native_jac_react}")
        print(f"  Generated cache[0]:          {gen_cache0}")
        print(f"  Match: {abs(native_jac_react - gen_cache0) < 1e-15} {'✓' if abs(native_jac_react - gen_cache0) < 1e-15 else '✗'}")
    except Exception as e:
        print(f"  Generated error: {e}")
