"""Test resistor divider circuit using VerilogADevice

This demonstrates using OpenVAF-compiled resistor models in a simple circuit.
The circuit is a voltage divider: VDD -> R1 -> Vout -> R2 -> GND

Expected: Vout = VDD * R2 / (R1 + R2)
"""

import sys
sys.path.insert(0, '.')

from jax_spice.devices import compile_va
import jax.numpy as jnp

print("=" * 60)
print("Resistor Divider Test")
print("=" * 60)

# Compile resistor model
resistor = compile_va('OpenVAF/integration_tests/RESISTOR/resistor.va')
print(f"Compiled device: {resistor.name}")
print(f"Terminals: {resistor.terminals}")
print()

# Circuit parameters
VDD = 5.0       # Supply voltage
R1 = 1000.0     # Upper resistor
R2 = 2000.0     # Lower resistor

# Expected output
Vout_expected = VDD * R2 / (R1 + R2)
print(f"Circuit: VDD={VDD}V, R1={R1}Ω, R2={R2}Ω")
print(f"Expected Vout = {Vout_expected}V")
print()

# Simple Newton-Raphson to solve for Vout
# KCL at Vout: I_R1 + I_R2 = 0
# I_R1 = (VDD - Vout) / R1 (current into node)
# I_R2 = (Vout - 0) / R2 (current out of node, so negative)
# (VDD - Vout) / R1 - Vout / R2 = 0

def evaluate_kcl(Vout):
    """Evaluate KCL residual at Vout node"""
    # R1: between VDD and Vout
    V_R1 = VDD - Vout
    res1, jac1 = resistor.eval(
        voltages={'V(A,B)': V_R1, 'vres': V_R1},
        params={'R': R1}
    )
    I_R1 = float(res1['sim_node0']['resist'])
    G_R1 = float(jac1[('sim_node0', 'sim_node0')]['resist'])

    # R2: between Vout and GND
    V_R2 = Vout
    res2, jac2 = resistor.eval(
        voltages={'V(A,B)': V_R2, 'vres': V_R2},
        params={'R': R2}
    )
    I_R2 = float(res2['sim_node0']['resist'])
    G_R2 = float(jac2[('sim_node0', 'sim_node0')]['resist'])

    # KCL: current into node = current out of node
    # I_R1 flows into Vout, I_R2 flows out
    residual = I_R1 - I_R2

    # Jacobian: d(residual)/d(Vout)
    # d(I_R1)/d(Vout) = -G_R1 (V_R1 = VDD - Vout, so dV_R1/dVout = -1)
    # d(I_R2)/d(Vout) = G_R2
    jacobian = -G_R1 - G_R2

    return residual, jacobian

# Newton-Raphson iteration
Vout = VDD / 2  # Initial guess
print("Newton-Raphson iteration:")
for i in range(10):
    res, jac = evaluate_kcl(Vout)
    if abs(res) < 1e-12:
        print(f"  Converged at iteration {i}")
        break
    delta = -res / jac
    Vout += delta
    print(f"  Iter {i}: Vout={Vout:.6f}V, residual={res:.6e}")

print()
print("=" * 60)
print("Results:")
print(f"  Computed Vout: {Vout:.6f} V")
print(f"  Expected Vout: {Vout_expected:.6f} V")
print(f"  Error: {abs(Vout - Vout_expected):.6e} V")
print("=" * 60)

# Verify with direct calculation
I_expected = VDD / (R1 + R2)
print(f"\nDirect calculation:")
print(f"  I = VDD/(R1+R2) = {I_expected*1000:.6f} mA")
print(f"  Vout = I*R2 = {I_expected * R2:.6f} V")
