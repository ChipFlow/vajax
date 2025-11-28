"""Debug CMOS inverter convergence"""

import sys
sys.path.insert(0, '/Users/roberttaylor/Code/ChipFlow/reference/jax-spice')

import jax.numpy as jnp
from jax_spice import Circuit, MOSFETSimple

# Create inverter
ckt = Circuit()

pmos = MOSFETSimple(W=20e-6, L=0.25e-6, Vth0=-0.4, pmos=True)
nmos = MOSFETSimple(W=10e-6, L=0.25e-6, Vth0=0.4, pmos=False)

ckt.add_device("M1", pmos, connections={'d': 'out', 'g': 'in', 's': 'vdd', 'b': 'vdd'})
ckt.add_device("M2", nmos, connections={'d': 'out', 'g': 'in', 's': 'gnd', 'b': 'gnd'})
ckt.add_vsource("VDD", 'vdd', 'gnd', 2.5)
ckt.add_vsource("VIN", 'in', 'gnd', 1.25)
ckt.set_ground('gnd')

print("Circuit:", ckt)
print("Nodes:", ckt.nodes)
print("Node indices:", {name: idx for name, idx in ckt.nodes.items()})
print("Num unknowns:", ckt.num_unknowns())
print()

# Reasonable initial guess: out=1.25V (middle), in=1.25V, vdd=2.5V
V_guess = jnp.array([1.25, 1.25, 2.5])  # out, in, vdd
print("Initial guess:", V_guess)
print("  out =", V_guess[0], "V")
print("  in  =", V_guess[1], "V")
print("  vdd =", V_guess[2], "V")
print()

# Build system
residual, jacobian = ckt.build_system(V_guess)
print("Residual:", residual)
print("  (should be currents at nodes)")
print()
print("Jacobian:")
print(jacobian)
print()

# Check if Jacobian is singular
try:
    det = jnp.linalg.det(jacobian)
    print(f"Jacobian determinant: {det}")
    if abs(det) < 1e-10:
        print("  WARNING: Jacobian is singular or nearly singular!")
except:
    print("  Could not compute determinant")

# Try one Newton step manually
try:
    delta_V = jnp.linalg.solve(jacobian, -residual)
    print(f"\nNewton step: δV = {delta_V}")
    V_new = V_guess + delta_V
    print(f"New voltage: {V_new}")
except Exception as e:
    print(f"\nFailed to solve: {e}")
