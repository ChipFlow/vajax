"""Simple test of JAX-SPICE core functionality"""

import sys
sys.path.insert(0, '/Users/roberttaylor/Code/ChipFlow/reference/jax-spice')

# Test imports
print("Testing imports...")
try:
    import jax
    import jax.numpy as jnp
    print(f"  ✓ JAX {jax.__version__}")
except ImportError as e:
    print(f"  ✗ JAX not available: {e}")
    sys.exit(1)

try:
    from jax_spice import Circuit, MOSFETSimple
    from jax_spice.analysis import dc_operating_point
    print("  ✓ JAX-SPICE modules")
except ImportError as e:
    print(f"  ✗ JAX-SPICE import failed: {e}")
    sys.exit(1)

# Test MOSFET model
print("\nTesting MOSFET model...")
nmos = MOSFETSimple(W=10e-6, L=0.25e-6, pmos=False)
print(f"  Created: {nmos}")

# Test evaluation
voltages = {'d': 1.2, 'g': 1.5, 's': 0.0, 'b': 0.0}
stamps = nmos.evaluate(voltages)
print(f"  Ids = {stamps.currents['d']:.6e} A")
print(f"  gm = {stamps.conductances[('d', 'g')]:.6e} S")
print(f"  gds = {stamps.conductances[('d', 'd')]:.6e} S")

# Test circuit
print("\nTesting circuit creation...")
ckt = Circuit()
ckt.add_device("M1", nmos, connections={'d': 'vd', 'g': 'vg', 's': 'gnd', 'b': 'gnd'})
ckt.add_vsource("VDD", 'vd', 'gnd', 1.2)
ckt.add_vsource("VG", 'vg', 'gnd', 1.5)
ckt.set_ground('gnd')
print(f"  {ckt}")

# Test DC operating point
print("\nTesting DC operating point...")
V_solution, converged, num_iter = dc_operating_point(ckt, verbose=True)
print(f"  Solution: {V_solution}")
print(f"  Converged: {converged} in {num_iter} iterations")

print("\n✓ All tests passed!")
