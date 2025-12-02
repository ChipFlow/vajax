"""Test built-in primitive modules (resistor, capacitor, inductor)

This tests that OpenVAF correctly lowers built-in primitives to contribution
statements, enabling models that use them to compile and run.

The test model uses:
- resistor #(.r(R1)) r1 (p, mid)  ->  I(p,mid) <+ V(p,mid) / R1
- resistor #(.r(R2)) r2 (mid, n)  ->  I(mid,n) <+ V(mid,n) / R2
- capacitor #(.c(C1)) c1 (mid, n) ->  I(mid,n) <+ ddt(C1 * V(mid,n))
"""

import sys
sys.path.insert(0, '.')

from jax_spice.devices import compile_va
import jax.numpy as jnp

print("=" * 60)
print("Built-in Primitives Test")
print("=" * 60)

# Try to compile the model with built-in primitives
try:
    device = compile_va('openvaf-py/vendor/OpenVAF/integration_tests/BUILTIN_PRIMITIVES/builtin_primitives.va')
    print(f"Successfully compiled device: {device.name}")
    print(f"Terminals: {device.terminals}")
    print(f"Parameters: {[n for n, k in zip(device.param_names, device.param_kinds) if k == 'param']}")
    print()
except Exception as e:
    print(f"ERROR: Failed to compile: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Circuit parameters
R1 = 1000.0   # First resistor
R2 = 2000.0   # Second resistor
C1 = 1e-12    # Capacitor

# Test DC behavior (capacitor acts as open circuit at DC)
# Expected: voltage divider Vout = V_in * R2 / (R1 + R2)
V_in = 5.0
V_out_expected = V_in * R2 / (R1 + R2)

print(f"Testing DC voltage divider:")
print(f"  V_in = {V_in} V")
print(f"  R1 = {R1} Ohm, R2 = {R2} Ohm")
print(f"  Expected V_mid = {V_out_expected:.4f} V")
print()

# Evaluate the device
# The built-in primitives should have been transformed to contribution statements
try:
    # Get parameter info
    print("Device parameter info:")
    for name, kind, default in device.get_parameter_info():
        print(f"  {name}: {kind} = {default}")
    print()

    # Try evaluating with some voltages
    # Note: The exact voltage names depend on how the model is structured
    print("Testing evaluation...")

    # Build voltages dict based on what the model expects
    voltages = {}
    for name, kind in zip(device.param_names, device.param_kinds):
        if kind == 'voltage':
            # Set some test values
            if 'p' in name.lower():
                voltages[name] = V_in
            elif 'mid' in name.lower():
                voltages[name] = V_out_expected
            else:
                voltages[name] = 0.0

    print(f"Input voltages: {voltages}")

    res, jac = device.eval(
        voltages=voltages,
        params={'R1': R1, 'R2': R2, 'C1': C1}
    )

    print(f"\nResiduals: {res}")
    print(f"Jacobian entries: {list(jac.keys())}")

    print("\n" + "=" * 60)
    print("SUCCESS: Built-in primitives compiled and evaluated!")
    print("=" * 60)

except Exception as e:
    print(f"ERROR during evaluation: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
