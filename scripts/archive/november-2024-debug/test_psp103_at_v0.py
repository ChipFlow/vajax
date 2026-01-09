#!/usr/bin/env python3
"""Test PSP103 model evaluation at V=0 using openvaf_py.

This will help us understand why NR fails from V=0 initial condition.
VACASK converges in 3-5 iterations, but JAX-SPICE completely fails.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "openvaf-py"))

import numpy as np
import json

# Import openvaf_py
try:
    import openvaf_py
    import openvaf_jax
except ImportError as e:
    print(f"Error importing openvaf modules: {e}")
    print("Make sure openvaf-py is built and in Python path")
    sys.exit(1)

def main():
    print("="*80)
    print("PSP103 Model Evaluation at V=0")
    print("="*80)
    print()

    # Compile PSP103 model
    psp103_va = Path(__file__).parent.parent / "openvaf-py" / "vendor" / "OpenVAF" / "integration_tests" / "PSP103" / "psp103.va"

    if not psp103_va.exists():
        print(f"PSP103 model not found at {psp103_va}")
        return 1

    print(f"Compiling PSP103 from: {psp103_va}")
    modules = openvaf_py.compile_va(str(psp103_va))

    if not modules:
        print("Failed to compile PSP103!")
        return 1

    psp103 = modules[0]
    print(f"✓ Compiled PSP103 module: {psp103.name}")
    print()

    # Test case: NMOS at V=0 (should be off)
    print("="*80)
    print("Test 1: NMOS at V=0 (all terminals grounded)")
    print("="*80)
    print()
    print("Expected behavior:")
    print("  Vgs=0, Vds=0, Vbs=0")
    print("  NMOS should be OFF")
    print("  Ids ≈ 0 (subthreshold leakage only)")
    print()

    # Get default NMOS parameters
    # For a minimal test, we need at least W, L, and type
    # Let's use default parameters from the model

    # Create MIR interpreter input
    params = {
        'V(D,S)': 0.0,  # Drain-source voltage
        'V(G,S)': 0.0,  # Gate-source voltage
        'V(B,S)': 0.0,  # Bulk-source voltage
        '$temperature': 300.0,
        'mfactor': 1.0,
    }

    # Add model parameters - we need these from the model card
    # For now, use some typical values
    # TODO: Extract actual parameters from ring oscillator netlist

    print("Running PSP103 eval at V=0...")
    print(f"  Inputs: Vds=0V, Vgs=0V, Vbs=0V, T=300K")

    try:
        # Run the model
        residuals, jacobian = psp103.run_init_eval(params)

        print()
        print("Results:")
        print(f"  Number of residuals: {len(residuals)}")
        print(f"  Number of Jacobian entries: {len(jacobian)}")
        print()

        # Analyze results
        # residuals is a list of (resist, react) tuples
        if residuals:
            print("Residual breakdown (resist, react):")
            for i, (resist, react) in enumerate(residuals):
                print(f"    Terminal [{i}]: resist={resist:.6e} A, react={react:.6e} C")

        print()
        print("This tells us what OpenVAF thinks PSP103 should do at V=0.")
        print("Next: Compare with VACASK's OSDI PSP103 at same operating point.")

    except Exception as e:
        print(f"Error running PSP103: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Test case 2: PMOS at V=0 with actual ring oscillator parameters
    print()
    print("="*80)
    print("Test 2: PMOS at V=0 (Ring Oscillator Configuration)")
    print("="*80)
    print()
    print("Configuration from ring oscillator:")
    print("  Instance: W=20u, L=1u (pfact=2, so W=10u*2)")
    print("  Model: PSP103 PMOS (type=-1)")
    print("  Operating point: Vd=0V (out), Vg=0V (in), Vs=1.2V (VDD), Vb=1.2V (VDD)")
    print()
    print("Expected behavior:")
    print("  Vgs = Vg - Vs = 0 - 1.2 = -1.2V")
    print("  For PMOS with Vth ≈ -0.3V to -0.5V:")
    print("  Vgs < Vth → PMOS is ON")
    print("  Vds = Vd - Vs = 0 - 1.2 = -1.2V")
    print("  PMOS should conduct: Ids < 0 (current flows from S to D)")
    print("  This charges the output node toward VDD")
    print()

    # Load ALL 269 parameters from model card
    params_file = Path(__file__).parent / 'psp103_model_params.json'
    if not params_file.exists():
        print(f"Model parameters file not found: {params_file}")
        print("Run: python scripts/parse_psp103_model_card.py")
        return 1

    with open(params_file) as f:
        all_params = json.load(f)

    pmos_model_params = all_params['pmos']
    print(f"Loaded {len(pmos_model_params)} PMOS model parameters from card")
    print()

    # Build complete parameter dict for ALL param_names
    # Ring oscillator operating point: Vd=0V, Vg=0V, Vs=1.2V, Vb=1.2V
    # Using Vs as reference: Vgs=-1.2V, Vds=-1.2V, Vbs=0V
    pmos_params = {}

    for name in psp103.param_names:
        if name == '$temperature':
            pmos_params[name] = 300.0
        elif name.upper() == 'TYPE':
            pmos_params[name] = -1.0  # PMOS
        elif name.upper() == 'W':
            pmos_params[name] = 20e-6
        elif name.upper() == 'L':
            pmos_params[name] = 1e-6
        elif name.upper() in ('AS', 'AD'):
            pmos_params[name] = 20e-6 * 0.5e-6
        elif name.upper() in ('PS', 'PD'):
            pmos_params[name] = 2*(20e-6 + 0.5e-6)
        elif name.lower() == 'mfactor':
            pmos_params[name] = 1.0
        elif 'V(' in name:
            # Voltage parameters - assume all external=internal (no parasitics)
            # Set terminal voltages relative to source (S=0 reference)
            if 'GP' in name or 'G' in name:
                # Gate voltages: Vg = 0V, Vs = 1.2V → V(G,S) = -1.2V
                # V(G) = -1.2 (relative to source at 0)
                if name == 'V(G,GP)':
                    pmos_params[name] = 0.0  # No parasitic drop
                elif name == 'V(GP)':
                    pmos_params[name] = -1.2  # Gate at -1.2V relative to Si
                elif 'SI' in name:
                    pmos_params[name] = -1.2  # V(GP,SI) = -1.2V
                else:
                    pmos_params[name] = 0.0
            elif 'DI' in name or 'D' in name:
                # Drain voltages: Vd = 0V, Vs = 1.2V → V(D,S) = -1.2V
                if name == 'V(D,DI)':
                    pmos_params[name] = 0.0
                elif name == 'V(DI)':
                    pmos_params[name] = -1.2
                elif 'SI' in name:
                    pmos_params[name] = -1.2  # V(DI,SI) = -1.2V
                elif 'BD' in name:
                    pmos_params[name] = 0.0
                else:
                    pmos_params[name] = 0.0
            elif 'SI' in name or 'S' in name:
                # Source = reference = 0V
                if name == 'V(S,SI)':
                    pmos_params[name] = 0.0
                elif name == 'V(SI)':
                    pmos_params[name] = 0.0
                elif 'BP' in name or 'BS' in name:
                    pmos_params[name] = 0.0  # V(SI,BP) = V(SI,BS) = 0
                else:
                    pmos_params[name] = 0.0
            elif 'BP' in name or 'BS' in name or 'BD' in name or 'BI' in name or 'B' in name:
                # Bulk voltages: Vb = 1.2V, Vs = 1.2V → V(B,S) = 0V
                pmos_params[name] = 0.0
            elif 'NOI' in name:
                # Noise node - should be zero
                pmos_params[name] = 0.0
            else:
                pmos_params[name] = 0.0
        elif name in pmos_model_params:
            # Use model card value
            pmos_params[name] = pmos_model_params[name]
        else:
            # Default to zero
            pmos_params[name] = 0.0

    # Debug: Check what voltage parameters we set
    print()
    print("Debug: Voltage parameters set:")
    for name, value in sorted(pmos_params.items()):
        if 'V(' in name and value != 0.0:
            print(f"  {name} = {value}V")
    print()

    print("Running PSP103 PMOS eval...")
    print(f"  Vds (internal) = V(DI,SI) = {pmos_params.get('V(DI,SI)', 0)}V")
    print(f"  Vgs (internal) = V(GP,SI) = {pmos_params.get('V(GP,SI)', 0)}V")
    print(f"  Vbs (internal) = V(BP,SI) = {pmos_params.get('V(SI,BP)', 0)}V")
    print(f"  W/L = {pmos_params.get('W', pmos_params.get('w', 0))*1e6:.1f}u/{pmos_params.get('L', pmos_params.get('l', 0))*1e6:.1f}u")
    print(f"  TYPE = {pmos_params.get('TYPE', pmos_params.get('type', 0))}")
    print()

    try:
        residuals_pmos, jacobian_pmos = psp103.run_init_eval(pmos_params)

        print()
        print("Results:")
        print(f"  Number of residuals: {len(residuals_pmos)}")
        print(f"  Number of Jacobian entries: {len(jacobian_pmos)}")
        print()

        # PSP103 has 4 external terminals: D, G, S, B
        # Residuals are currents INTO each terminal
        if len(residuals_pmos) >= 4:
            print("Terminal currents (resistive component):")
            terminal_names = ['D (drain)', 'G (gate)', 'S (source)', 'B (bulk)']
            for i in range(min(4, len(residuals_pmos))):
                resist, react = residuals_pmos[i]
                name = terminal_names[i] if i < len(terminal_names) else f'Node {i}'
                print(f"  I({name}): {resist:.6e} A (react: {react:.6e} C)")

            # Check if PMOS is conducting
            I_drain = residuals_pmos[0][0]  # Current into drain
            I_source = residuals_pmos[2][0]  # Current into source

            print()
            if abs(I_drain) > 1e-12:
                print(f"✓ PMOS is conducting!")
                print(f"  I_drain = {I_drain:.6e} A")
                print(f"  I_source = {I_source:.6e} A")
                print(f"  Current flow: {'Source→Drain' if I_source < 0 else 'Drain→Source'}")
                print()
                print("This is CORRECT - PMOS at Vgs=-1.2V should conduct.")
                print("In DC solver from V=0, this current charges output nodes toward VDD.")
            else:
                print(f"⚠️  WARNING: PMOS shows no current!")
                print(f"  I_drain = {I_drain:.6e} A")
                print(f"  This is WRONG - PMOS at Vgs=-1.2V should conduct strongly.")
                print()
                print("If OpenVAF returns Ids=0 but VACASK returns Ids>0,")
                print("this would explain why DC solver fails from V=0!")

        # Check Jacobian
        print()
        print("Jacobian (first few entries):")
        for i, (row, col, resist, react) in enumerate(list(jacobian_pmos)[:5]):
            print(f"  J[{row},{col}]: resist={resist:.6e} S, react={react:.6e} F")

    except Exception as e:
        print(f"Error running PMOS PSP103: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
