"""Test diode half-wave rectifier circuit using VerilogADevice

This demonstrates using OpenVAF-compiled diode model in a circuit.
The circuit is a half-wave rectifier: Vin -> D -> Vout -> R -> GND

For Vin > 0: Vout ≈ Vin - Vd (where Vd is diode forward voltage)
For Vin < 0: Vout ≈ 0 (diode blocks reverse current)
"""

import sys
sys.path.insert(0, '.')

from jax_spice.devices import compile_va
import jax.numpy as jnp
import math

print("=" * 60)
print("Diode Half-Wave Rectifier Test")
print("=" * 60)

# Compile models
diode = compile_va('OpenVAF/integration_tests/DIODE/diode.va')
resistor = compile_va('OpenVAF/integration_tests/RESISTOR/resistor.va')

print(f"Diode: {diode.name}")
print(f"Resistor: {resistor.name}")
print()

# Set diode parameters
diode.set_parameters(**{
    'is': 1e-14,      # Saturation current
    'n': 1.0,         # Ideality factor
    'rs': 0.0,        # Series resistance
    'rth': 0.0,       # Thermal resistance
    'cj0': 0.0,       # Junction capacitance
})

# Circuit parameters
R_load = 1000.0  # Load resistor

def evaluate_circuit(Vin, Vout):
    """Evaluate KCL residual at Vout node for given Vin"""
    # Diode: anode at Vin, cathode at Vout
    # V(A,CI) is the voltage across the junction (Vin - Vout)
    V_diode = Vin - Vout

    res_d, jac_d = diode.eval(
        voltages={
            'V(A,CI)': V_diode,
            'V(CI,C)': 0.0,  # No series resistance drop
            'V(dT)': 0.0,
            'V(A)': Vin,
        },
        params={'is': 1e-14, 'n': 1.0, 'rs': 0.0, 'rth': 0.0}
    )
    I_diode = float(res_d['sim_node0']['resist'])

    # Get diode conductance from jacobian
    # The diode's d(I)/d(V) appears in the jacobian
    G_diode = 0.0
    for (row, col), entry in jac_d.items():
        if 'sim_node0' in row and 'sim_node0' in col:
            G_diode = abs(float(entry.get('resist', 0.0)))
            break

    # Load resistor: between Vout and GND
    res_r, jac_r = resistor.eval(
        voltages={'V(A,B)': Vout, 'vres': Vout},
        params={'R': R_load}
    )
    I_load = float(res_r['sim_node0']['resist'])

    # KCL at Vout: I_diode = I_load
    # The diode current flows into Vout, load current flows out
    residual = I_diode - I_load

    # Jacobian: d(residual)/d(Vout)
    # d(I_diode)/d(Vout) = -G_diode (since V_diode = Vin - Vout)
    # d(I_load)/d(Vout) = G_load
    G_load = 1.0 / R_load
    jacobian = -G_diode - G_load

    return residual, jacobian, I_diode, I_load

def solve_for_vout(Vin, initial_guess=0.0):
    """Solve for Vout given Vin using Newton-Raphson"""
    Vout = initial_guess

    for i in range(100):
        res, jac, I_d, I_r = evaluate_circuit(Vin, Vout)

        if abs(res) < 1e-10:
            break

        # Newton step with damping
        if abs(jac) > 1e-15:
            delta = -res / jac
            # Limit step size to prevent overshooting
            max_step = 0.1
            if abs(delta) > max_step:
                delta = max_step if delta > 0 else -max_step
        else:
            delta = 0.01 if res < 0 else -0.01

        Vout += delta

        # Keep Vout in reasonable bounds
        Vout = max(min(Vout, Vin + 0.1), -0.1)

    return Vout, I_d

# Test at various input voltages
print("Vin (V)    Vout (V)   I_diode (mA)   I_load (mA)")
print("-" * 55)

Vin_values = [-2.0, -1.0, -0.5, 0.0, 0.3, 0.5, 0.7, 1.0, 2.0, 5.0]
previous_Vout = 0.0

for Vin in Vin_values:
    Vout, I_diode = solve_for_vout(Vin, initial_guess=previous_Vout)
    I_load = Vout / R_load

    print(f"{Vin:7.2f}    {Vout:7.4f}    {I_diode*1000:10.6f}    {I_load*1000:10.6f}")
    previous_Vout = Vout

print()
print("=" * 60)
print("Analysis:")
print("- For Vin < 0: Diode is reverse biased, Vout ≈ 0")
print("- For Vin > 0: Diode conducts, Vout ≈ Vin - Vd")
print("- Vd (forward voltage) ≈ 0.6-0.7V for silicon diode")
print("=" * 60)
