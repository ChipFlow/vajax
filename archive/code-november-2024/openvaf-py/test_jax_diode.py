"""Test JAX translator with diode model"""
import openvaf_py
import openvaf_jax
import jax.numpy as jnp
import math

print("="*70)
print("Testing JAX translator with diode")
print("="*70)

# Compile the diode
modules = openvaf_py.compile_va(
    "/Users/roberttaylor/Code/ChipFlow/reference/jax-spice/OpenVAF/integration_tests/DIODE/diode.va"
)
diode = modules[0]

print(f"Module: {diode.name}")
print(f"Params: {len(diode.param_names)}, Init params: {diode.init_num_params}")
print(f"Cached values: {diode.num_cached_values}")

# Create JAX translator
translator = openvaf_jax.OpenVAFToJAX(diode)
eval_fn = translator.translate()
print("JAX function compiled!")

def test_diode(V_diode, temperature=300.15):
    """Test diode at given forward voltage"""
    # Physical constants
    k = 1.380649e-23  # Boltzmann
    q = 1.602176634e-19  # electron charge
    vt = k * temperature / q  # thermal voltage ~26mV at 300K

    # Diode parameters (defaults from typical model)
    Is = 1e-14  # Saturation current
    N = 1.0     # Ideality factor
    Rs = 0.0    # Series resistance
    Rth = 0.0   # Thermal resistance

    # Build parameter dict for interpreter
    params = {}
    for i, (name, kind) in enumerate(zip(diode.param_names, diode.param_kinds)):
        if kind == 'voltage':
            if name == 'V(A,CI)':
                params[name] = V_diode  # Main diode voltage
            elif name == 'V(CI,C)':
                params[name] = 0.0  # Rs voltage drop
            elif name == 'V(dT)':
                params[name] = 0.0  # Temperature rise
            elif name == 'V(A)':
                params[name] = V_diode
            else:
                params[name] = 0.0
        elif kind == 'temperature':
            params[name] = temperature
        elif kind == 'param':
            # Set model parameters
            if name == 'is':
                params[name] = Is
            elif name == 'n':
                params[name] = N
            elif name == 'rs':
                params[name] = Rs
            elif name == 'rth':
                params[name] = Rth
            elif name == 'tnom':
                params[name] = 300.0
            elif name == 'minr':
                params[name] = 1e-6
            elif name in ('zetais', 'zetars', 'zetarth'):
                params[name] = 0.0
            elif name == 'ea':
                params[name] = 1.11  # Silicon bandgap
            elif name == 'vj':
                params[name] = 0.9  # Junction voltage
            elif name == 'm':
                params[name] = 0.5  # Grading coefficient
            elif name == 'cj0':
                params[name] = 0.0  # Zero junction capacitance
            else:
                params[name] = 0.0
        elif kind == 'hidden_state':
            # Compute hidden states
            if name == 'tdev':
                params[name] = temperature
            elif name == 'vt':
                params[name] = vt
            elif name == 'is_t':
                params[name] = Is  # Temperature-adjusted Is
            elif name == 'rs_t':
                params[name] = Rs
            elif name == 'rth_t':
                params[name] = Rth
            elif name == 'vd':
                params[name] = V_diode
            elif name == 'vr':
                params[name] = 0.0
            elif name == 'id':
                # Ideal diode current
                params[name] = Is * (math.exp(V_diode / (N * vt)) - 1)
            elif name in ('vf', 'x', 'y', 'vd_smooth', 'qd', 'pterm', 'cd', 'gd'):
                params[name] = 0.0
            else:
                params[name] = 0.0
        elif kind == 'current':
            params[name] = 0.0
        elif kind == 'sysfun':
            if name == 'mfactor':
                params[name] = 1.0
            else:
                params[name] = 0.0
        else:
            params[name] = 0.0

    # Run interpreter
    try:
        interp_res, interp_jac = diode.run_init_eval(params)
        interp_I = interp_res[0][0]  # First residual, resistive part
    except Exception as e:
        print(f"Interpreter error: {e}")
        interp_I = float('nan')

    # Build JAX inputs array
    jax_inputs = []
    for name in diode.param_names:
        jax_inputs.append(params.get(name, 0.0))

    # Run JAX
    try:
        jax_res, jax_jac = eval_fn(jax_inputs)
        jax_I = float(jax_res['sim_node0']['resist'])
    except Exception as e:
        print(f"JAX error: {e}")
        jax_I = float('nan')

    # Expected (ideal diode equation)
    expected_I = Is * (math.exp(V_diode / (N * vt)) - 1)

    return interp_I, jax_I, expected_I

print("\n=== Diode I-V Curve Test ===")
print(f"{'V (V)':<10} {'Interp I':<15} {'JAX I':<15} {'Expected I':<15} {'Match':<8}")
print("-" * 70)

for V in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
    interp_I, jax_I, expected_I = test_diode(V)

    # Check if JAX matches interpreter
    if math.isnan(interp_I) or math.isnan(jax_I):
        match = "NaN"
    elif abs(interp_I) < 1e-20 and abs(jax_I) < 1e-20:
        match = "✓"
    elif abs(interp_I) > 1e-20:
        rel_err = abs(jax_I - interp_I) / abs(interp_I)
        match = "✓" if rel_err < 1e-4 else f"{rel_err:.2e}"
    else:
        match = "~"

    print(f"{V:<10.2f} {interp_I:<15.6e} {jax_I:<15.6e} {expected_I:<15.6e} {match:<8}")
