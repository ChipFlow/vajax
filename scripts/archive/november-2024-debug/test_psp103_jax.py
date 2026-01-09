#!/usr/bin/env python3
"""Test PSP103 using JAX translator (the production code path)."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "openvaf-py"))

import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import numpy as np
import json
import openvaf_py
import openvaf_jax

def main():
    print("=" * 80)
    print("PSP103 JAX Evaluation Test")
    print("=" * 80)
    print()

    # Compile PSP103
    psp103_va = Path(__file__).parent.parent / "openvaf-py" / "vendor" / "OpenVAF" / "integration_tests" / "PSP103" / "psp103.va"
    modules = openvaf_py.compile_va(str(psp103_va))
    psp103 = modules[0]

    print(f"✓ Compiled: {psp103.name}")
    print()

    # Load model parameters
    params_file = Path(__file__).parent / 'psp103_model_params.json'
    with open(params_file) as f:
        all_params = json.load(f)
    pmos_model_params = all_params['pmos']

    print(f"Loaded {len(pmos_model_params)} PMOS model parameters")
    print()

    # Create JAX translator
    translator = openvaf_jax.OpenVAFToJAX(psp103)
    jax_fn = translator.translate()

    print("✓ Created JAX function")
    print()

    # Build input array following the same pattern as conftest.py
    inputs = []
    for name, kind in zip(psp103.param_names, psp103.param_kinds):
        if kind == 'voltage':
            # Set PSP103 voltages for PMOS at ring oscillator operating point
            # Vs=1.2V (reference), Vd=0V, Vg=0V, Vb=1.2V → Vds=-1.2V, Vgs=-1.2V, Vbs=0V
            if 'GP,SI' in name:
                inputs.append(-1.2)  # V(GP,SI) = Vgs
            elif 'DI,SI' in name:
                inputs.append(-1.2)  # V(DI,SI) = Vds
            elif 'SI,BP' in name or 'SI,BS' in name:
                inputs.append(0.0)   # V(SI,BP) = Vbs
            elif 'GP' in name and 'SI' not in name:
                inputs.append(-1.2)  # V(GP) = -1.2V
            elif 'DI' in name and 'SI' not in name and 'BD' not in name:
                inputs.append(-1.2)  # V(DI) = -1.2V
            else:
                inputs.append(0.0)
        elif kind == 'param':
            if '$temperature' in name.lower():
                inputs.append(300.0)
            elif name.upper() == 'TYPE':
                inputs.append(-1.0)  # PMOS
            elif name.upper() == 'W':
                inputs.append(20e-6)
            elif name.upper() == 'L':
                inputs.append(1e-6)
            elif name.upper() in ('AS', 'AD'):
                inputs.append(20e-6 * 0.5e-6)
            elif name.upper() in ('PS', 'PD'):
                inputs.append(2*(20e-6 + 0.5e-6))
            elif name.lower() == 'mfactor':
                inputs.append(1.0)
            elif name in pmos_model_params:
                inputs.append(pmos_model_params[name])
            else:
                inputs.append(0.0)
        elif kind == 'hidden_state':
            # Hidden states will be computed by the JAX function
            inputs.append(0.0)
        else:
            inputs.append(0.0)

    print(f"Input array: {len(inputs)} parameters")
    print()

    # Evaluate
    print("Evaluating PMOS at Vgs=-1.2V, Vds=-1.2V...")
    residuals, jacobian = jax_fn(inputs)

    print()
    print("Results:")
    print(f"  Residuals: {len(residuals)} nodes")
    print(f"  Jacobian: {len(jacobian)} entries")
    print()

    # Show terminal currents
    print("Terminal currents:")
    for i, (node, res) in enumerate(list(residuals.items())[:4]):
        resist = float(res['resist'])
        react = float(res['react'])
        print(f"  Node {node}: I={resist:.6e} A, Q={react:.6e} C")

    # Check for PMOS conduction
    print()
    first_node_current = float(list(residuals.values())[0]['resist'])
    if abs(first_node_current) > 1e-12:
        print(f"✓ PMOS is conducting! I = {first_node_current:.6e} A")
    elif abs(first_node_current) == 0:
        print(f"✗ PMOS shows ZERO current (same bug as MIR interpreter)")
    else:
        print(f"⚠️  Small current: I = {first_node_current:.6e} A")

    return 0

if __name__ == "__main__":
    sys.exit(main())
