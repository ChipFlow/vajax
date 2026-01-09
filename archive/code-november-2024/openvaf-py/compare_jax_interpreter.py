"""Compare JAX translator results with MIR interpreter"""
import openvaf_py
import openvaf_jax
import numpy as np

print("="*70)
print("Comparing JAX translator vs MIR interpreter for resistor")
print("="*70)

# Load the module
modules = openvaf_py.compile_va("/Users/roberttaylor/Code/ChipFlow/reference/OpenVAF/integration_tests/RESISTOR/resistor.va")
res = modules[0]

# Create JAX translator
translator = openvaf_jax.OpenVAFToJAX(res)
jax_eval = translator.translate()

def compare_test(V, R, temperature, tnom, zeta, mfactor):
    """Compare JAX vs interpreter results"""
    print(f"\nTest: V={V}, R={R}, T={temperature}, tnom={tnom}, zeta={zeta}, mfactor={mfactor}")

    # Run MIR interpreter
    params = {
        'V(A,B)': V,
        'vres': V,
        'R': R,
        '$temperature': temperature,
        'tnom': tnom,
        'zeta': zeta,
        'res': R,
        'mfactor': mfactor,
    }
    interp_residuals, interp_jacobian = res.run_init_eval(params)

    # Run JAX translator
    jax_inputs = [V, V, R, temperature, tnom, zeta, R, mfactor]
    jax_residuals, jax_jacobian = jax_eval(jax_inputs)

    # Compare residuals
    interp_I = interp_residuals[0][0]  # node0 resist
    jax_I = float(jax_residuals['sim_node0']['resist'])

    # Compute expected
    resistance = R * (temperature/tnom)**zeta
    expected_I = mfactor * V / resistance

    print(f"  Expected I     = {expected_I:.9f}")
    print(f"  Interpreter I  = {interp_I:.9f}")
    print(f"  JAX I          = {jax_I:.9f}")
    print(f"  Match (interp) = {abs(expected_I - interp_I) < 1e-12}")
    print(f"  Match (JAX)    = {abs(expected_I - jax_I) < 1e-6}")
    print(f"  JAX≈Interp     = {abs(jax_I - interp_I) < 1e-6}")

    # Compare Jacobian
    interp_jac_00 = interp_jacobian[0][2]  # row=0, col=0, resist
    jax_jac_00 = float(jax_jacobian[('sim_node0', 'sim_node0')]['resist'])
    expected_jac = mfactor / resistance

    print(f"  Jacobian (0,0): interp={interp_jac_00:.9f}, jax={jax_jac_00:.9f}, expected={expected_jac:.9f}")

    return abs(jax_I - interp_I) < 1e-6

# Run comparison tests
results = []

results.append(compare_test(V=1.0, R=1000.0, temperature=300.0, tnom=300.0, zeta=0.0, mfactor=1.0))
results.append(compare_test(V=5.0, R=1000.0, temperature=300.0, tnom=300.0, zeta=0.0, mfactor=1.0))
results.append(compare_test(V=1.0, R=470.0, temperature=300.0, tnom=300.0, zeta=0.0, mfactor=1.0))
results.append(compare_test(V=1.0, R=1000.0, temperature=350.0, tnom=300.0, zeta=1.0, mfactor=1.0))
results.append(compare_test(V=1.0, R=1000.0, temperature=350.0, tnom=300.0, zeta=2.0, mfactor=1.0))
results.append(compare_test(V=1.0, R=1000.0, temperature=300.0, tnom=300.0, zeta=0.0, mfactor=2.0))
results.append(compare_test(V=1.0, R=1000.0, temperature=250.0, tnom=300.0, zeta=1.0, mfactor=1.0))
results.append(compare_test(V=0.1, R=100.0, temperature=273.15, tnom=300.0, zeta=1.5, mfactor=0.5))

print("\n" + "="*70)
print(f"Results: {sum(results)}/{len(results)} tests passed")
print("="*70)

if all(results):
    print("\nSUCCESS: JAX translator matches MIR interpreter!")
else:
    print("\nFAILURE: Some tests did not match!")
