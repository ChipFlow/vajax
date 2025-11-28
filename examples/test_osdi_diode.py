"""Test OSDI wrapper with GF130 diode model"""

import sys
sys.path.insert(0, '/Users/roberttaylor/Code/ChipFlow/reference/jax-spice')

from pathlib import Path

# Test OSDI library loading
print("=" * 60)
print("Testing OSDI Wrapper with GF130 Diode")
print("=" * 60)

osdi_path = Path("/Users/roberttaylor/Code/ChipFlow/reference/jax-spice/pdks/gf130/diode_rr.osdi")

if not osdi_path.exists():
    print(f"ERROR: OSDI file not found: {osdi_path}")
    sys.exit(1)

print(f"\n1. Loading OSDI library: {osdi_path}")
try:
    from jax_spice.devices.osdi import OSDILibrary, OSDIDevice
    lib = OSDILibrary(str(osdi_path))
    print(f"   ✓ OSDI version: {lib.version_major}.{lib.version_minor}")
    print(f"   ✓ Number of models: {lib.num_descriptors}")

    desc = lib.get_descriptor()
    print(f"   ✓ Model name: {desc.name.decode()}")
    print(f"   ✓ Terminals: {desc.num_terminals}")
    print(f"   ✓ Total nodes: {desc.num_nodes}")
    print(f"   ✓ Jacobian entries: {desc.num_resistive_jacobian_entries}")

    # Print terminal names
    terminals = []
    for i in range(desc.num_terminals):
        node = desc.nodes[i]
        terminals.append(node.name.decode())
    print(f"   ✓ Terminal names: {terminals}")

except Exception as e:
    print(f"   ✗ Failed to load: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test device creation
print("\n2. Creating OSDIDevice with GF130 parameters...")
try:
    # GF130 np_30p0_iso diode parameters (from PDK)
    gf130_params = {
        'tref': 25,              # Reference temperature (°C)
        'js': 5.5195e-007,       # Saturation current (A)
        'jsw': 4.2728e-012,      # Sidewall saturation current (A/m)
        'n': 1,                  # Ideality factor
        'tau': 0,                # Disable reverse recovery for DC
        'tt': 0,                 # Transit time
        'tm': 2.0e-08,           # Reverse recovery time constant
        'area': 1e-10,           # Junction area (m²)
        'pj': 4e-5,              # Junction perimeter (m)
    }

    diode = OSDIDevice(
        osdi_path=str(osdi_path),
        params=gf130_params,
        temperature=300.0,  # 27°C in Kelvin
    )
    print(f"   ✓ Created: {diode}")
    print(f"   ✓ Terminals: {diode.terminals}")

except Exception as e:
    print(f"   ✗ Failed to create device: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test evaluation
print("\n3. Testing device evaluation...")
try:
    from jax_spice.analysis.context import AnalysisContext

    # Test at forward bias
    voltages = {'A': 0.7, 'C': 0.0}  # Assume A=anode, C=cathode

    # Try to figure out terminal names
    print(f"   Device terminals: {diode.terminals}")

    # Use actual terminal names from OSDI
    if len(diode.terminals) >= 2:
        anode = diode.terminals[0]
        cathode = diode.terminals[1]
        voltages = {anode: 0.7, cathode: 0.0}
        print(f"   Using: {anode}={voltages[anode]}V, {cathode}={voltages[cathode]}V")

    context = AnalysisContext.dc()
    stamps = diode.evaluate(voltages, context=context)

    print(f"   ✓ Currents: {stamps.currents}")
    print(f"   ✓ Conductances: {stamps.conductances}")

except Exception as e:
    print(f"   ✗ Evaluation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test DC sweep
print("\n4. DC I-V sweep...")
try:
    import numpy as np

    V_sweep = np.linspace(0, 1.0, 21)
    I_diode = []

    anode = diode.terminals[0]
    cathode = diode.terminals[1]

    for V in V_sweep:
        voltages = {anode: V, cathode: 0.0}
        stamps = diode.evaluate(voltages, context=AnalysisContext.dc())
        I = stamps.currents[anode]
        I_diode.append(I)

    print(f"   {'V(V)':>8s}  {'I(A)':>12s}")
    print(f"   {'-'*8}  {'-'*12}")
    for V, I in zip(V_sweep[::4], I_diode[::4]):  # Print every 4th point
        print(f"   {V:8.3f}  {I:12.4e}")

    print("   ✓ DC sweep completed")

except Exception as e:
    print(f"   ✗ DC sweep failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✓ All OSDI wrapper tests passed!")
print("=" * 60)
