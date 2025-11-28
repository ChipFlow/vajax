"""Test the MIR-to-JAX translator with the diode model"""

import sys
sys.path.insert(0, '/Users/roberttaylor/Code/ChipFlow/reference/jax-spice')

from pathlib import Path
from jax_spice.mir.translator import MIRToJAX
import jax.numpy as jnp
import numpy as np

# Load diode snapshots
openvaf_path = Path("/Users/roberttaylor/Code/ChipFlow/reference/OpenVAF")
mir_path = openvaf_path / "openvaf/test_data/dae/diode_va_mir.snap"
system_path = openvaf_path / "openvaf/test_data/dae/diode_va_system.snap"

print("Loading diode MIR snapshots...")
with open(mir_path) as f:
    mir_text = f.read()
with open(system_path) as f:
    system_text = f.read()

translator = MIRToJAX.from_snapshots(mir_text, system_text)
eval_fn = translator.translate()
print("✓ JAX function generated!")

# From detailed tracing of MIR:
# v64 = v59 / v42 where v42 = vt * n
# So v59 = vd (junction voltage for limexp)!
# v59 is the ACTUAL junction voltage used in exp(vd/(n*vt))

# v16 might be V(A) (anode voltage)
# v17 = 0 (threshold)
# v19 = temperature (K)
# v20 = self-heating temp contribution
# v28, v29 = Is related
# v30 = tnom
# v33 = zetais
# v35 = n
# v59 = vd (junction voltage!) - THIS IS KEY!

params = translator.mir.params
print(f"\nParameter indices:")
for i, p in enumerate(params):
    print(f"  [{i:2d}] {p}")

# Core parameters
param_values = {
    'v17': 0.0,      # threshold (rth comparison)
    'v19': 300.15,   # temperature K
    'v20': 0.0,      # self-heating contribution
    'v28': 1e-14,    # Is related
    'v29': 1e-14,    # Is (saturation current)
    'v30': 300.0,    # tnom
    'v33': 3.0,      # zetais
    'v35': 1.0,      # n (emission coefficient)
    'v40': 0.0,      # junction param
    'v47': 0.0,      # threshold
    'v48': 0.0,      # rs (no series resistance)
    'v50': 0.5,      # exponent
    'v53': 0.0,      # junction
    'v55': 0.5,      # m
    'v58': 0.0,      # cj0 related
    # v59 is junction voltage - set dynamically
    'v60': 0.0,
    'v61': 0.0,
    'v62': 0.0,
    'v76': 0.0,
    'v77': 1.0,      # vj
    'v81': 0.5,      # m
    'v86': 1.11,     # ea
    'v95': 0.0,
    'v100': 0.0,
    'v107': 0.0,     # rth
    'v108': 0.0,     # rth
    'v122': 0.0,
    'v201': 0.0,
    'v274': 0.0,
    'v276': 0.0,
    'v283': 0.0,
    'v361': 1.0,     # scale
    'v362': 1.0,     # scale
    'v403': 1.0,     # scale
}

def build_inputs(vd):
    """Build input array for given junction voltage"""
    inputs = []
    for p in params:
        if p == 'v16':
            inputs.append(vd)     # V(A) - might be same as junction voltage
        elif p == 'v59':
            inputs.append(vd)     # Junction voltage for limexp!
        elif p in param_values:
            inputs.append(param_values[p])
        else:
            inputs.append(1.0)
    return inputs

print("\nTesting diode I-V characteristic:")
print("-" * 50)

voltages = [0.0, 0.3, 0.5, 0.6, 0.7]
for vd in voltages:
    inputs = build_inputs(vd)
    try:
        residuals, jacobian = eval_fn(inputs)
        i_anode = float(residuals['sim_node0']['resist'])
        g_diode = float(jacobian[('sim_node0', 'sim_node0')]['resist'])

        if np.isfinite(i_anode) and np.isfinite(g_diode):
            print(f"V = {vd:.2f}V: I = {i_anode:12.6e} A, G = {g_diode:12.6e} S")
        else:
            print(f"V = {vd:.2f}V: I = {i_anode}, G = {g_diode} (overflow)")
    except Exception as e:
        print(f"V = {vd:.2f}V: Error - {e}")

# Expected values
print("\nExpected (ideal diode Is=1e-14, n=1):")
print("-" * 50)
Is = 1e-14
n = 1.0
vt = 0.02586

for vd in voltages:
    i_expected = Is * (np.exp(vd / (n * vt)) - 1)
    g_expected = Is / (n * vt) * np.exp(vd / (n * vt))
    print(f"V = {vd:.2f}V: I = {i_expected:12.6e} A, G = {g_expected:12.6e} S")

print("\n" + "="*60)
