#!/usr/bin/env python3
"""Debug why Jacobian values are zero in generated code."""

import sys
import math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import openvaf_py
from jax_spice.codegen.mir_parser import parse_mir_dict
from jax_spice.codegen.eval_mir_codegen import generate_eval_from_mir

# Compile capacitor
modules = openvaf_py.compile_va(str(Path(__file__).parent.parent / 'vendor/VACASK/devices/capacitor.va'))
cap = modules[0]

# Get metadata
metadata = cap.get_codegen_metadata()
print("Jacobian metadata:")
for jac_info in metadata['jacobian']:
    print(f"  J[{jac_info['row']},{jac_info['col']}]: resist={jac_info['resist_var']}, react={jac_info['react_var']}")

# Get MIR
init_mir_dict = cap.get_init_mir_instructions()
eval_mir_dict = cap.get_mir_instructions()
eval_mir = parse_mir_dict(eval_mir_dict)

# Create trivial param map
eval_param_map = {p.name: p.name for p in eval_mir.params}
for const_name in eval_mir.constants.keys():
    eval_param_map[const_name] = const_name

cache_param_indices = [entry['eval_param'] for entry in init_mir_dict['cache_mapping']]

# Generate code
eval_code = generate_eval_from_mir(eval_mir, eval_param_map, cache_param_indices, 'capacitor')

# Execute
namespace = {'math': math}
exec(eval_code, namespace)
eval_fn = namespace['eval_capacitor']

# Call with test values: c=1e-9, V_A_B=1.0, mfactor=1.0, cache=[1e-9, -1e-9]
print("\nCalling eval_capacitor(c=1e-9, V=1.0, Q=0, q=0, mfactor=1.0, cache=[1e-9, -1e-9])")
result = eval_fn(1e-9, 1.0, 0.0, 0.0, 1.0, cache=[1e-9, -1e-9])

print(f"\nGenerated code returned {len(result)} values")
print("\nChecking Jacobian reactive values from metadata:")
for jac_info in metadata['jacobian']:
    react_var = jac_info['react_var']
    value = result.get(react_var, 'NOT FOUND')
    print(f"  {react_var} = {value}")

print("\nChecking expected values:")
print(f"  c (v16) = {result.get('v16', 'NOT FOUND')}")
print(f"  V_A_B (v17) = {result.get('v17', 'NOT FOUND')}")
print(f"  mfactor (v25) = {result.get('v25', 'NOT FOUND')}")
print(f"  cache[0] (v37) = {result.get('v37', 'NOT FOUND')}")
print(f"  cache[1] (v40) = {result.get('v40', 'NOT FOUND')}")

# Check reference computation
print("\nReference computation:")
ref_residuals, ref_jacobian = cap.run_init_eval({'c': 1e-9, 'c_given': True, 'V_A_B': 1.0, 'mfactor': 1.0})
print(f"  Reference Jacobian: {ref_jacobian}")
