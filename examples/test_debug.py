"""Debug voltage source handling"""

import sys
sys.path.insert(0, '/Users/roberttaylor/Code/ChipFlow/reference/jax-spice')

import jax.numpy as jnp
from jax_spice import Circuit, MOSFETSimple
from jax_spice.analysis import dc_operating_point

# Simple test: single NMOS with fixed gate and drain voltages
ckt = Circuit()

nmos = MOSFETSimple(W=10e-6, L=0.25e-6, pmos=False)
ckt.add_device("M1", nmos, connections={'d': 'vd', 'g': 'vg', 's': 'gnd', 'b': 'gnd'})
ckt.add_vsource("VD", 'vd', 'gnd', 1.2)
ckt.add_vsource("VG", 'vg', 'gnd', 1.5)
ckt.set_ground('gnd')

print("Circuit:", ckt)
print("Nodes:", ckt.nodes)
print("Ground node:", ckt.ground_node)
print("Num unknowns:", ckt.num_unknowns())

# Try initial guess
V_guess = jnp.array([1.2, 1.5])  # vd=1.2, vg=1.5
print("\nInitial guess:", V_guess)

# Build system
residual, jacobian = ckt.build_system(V_guess)
print("\nResidual:", residual)
print("Jacobian:")
print(jacobian)

# Try to solve
print("\nAttempting DC operating point...")
V_sol, conv, iters = dc_operating_point(ckt, V_initial=V_guess, verbose=True, max_iter=10)
print(f"Solution: {V_sol}, Converged: {conv}")
