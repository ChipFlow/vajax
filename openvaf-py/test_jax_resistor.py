"""Test JAX resistor with various parameters"""
import openvaf_jax
import jax.numpy as jnp

print("="*60)
print("Testing OpenVAF to JAX translator with resistor")
print("="*60)

translator = openvaf_jax.OpenVAFToJAX.from_file(
    "/Users/roberttaylor/Code/ChipFlow/reference/OpenVAF/integration_tests/RESISTOR/resistor.va"
)

eval_fn = translator.translate()

def test_resistor(V, R, temperature, tnom, zeta, mfactor):
    """Test resistor with given parameters"""
    # Build inputs array: V(A,B), vres, R, $temp, tnom, zeta, res, mfactor
    inputs = [V, V, R, temperature, tnom, zeta, R, mfactor]
    residuals, jacobian = eval_fn(inputs)

    # Compute expected (mfactor multiplies the contribution)
    res = R * (temperature/tnom)**zeta
    expected_I = mfactor * V / res
    actual_I = float(residuals['sim_node0']['resist'])

    print(f"\nTest: V={V}, R={R}, T={temperature}, tnom={tnom}, zeta={zeta}, mfactor={mfactor}")
    print(f"  res = R * (T/tnom)^zeta = {R} * ({temperature}/{tnom})^{zeta} = {res:.6f}")
    print(f"  Expected I = mfactor*V/res = {mfactor}*{V}/{res:.3f} = {expected_I:.6f}")
    print(f"  Actual I   = {actual_I:.6f}")
    print(f"  Match: {abs(expected_I - actual_I) < 1e-6}")

# Test 1: Basic resistor at nominal temperature
test_resistor(V=1.0, R=1000.0, temperature=300.0, tnom=300.0, zeta=0.0, mfactor=1.0)

# Test 2: Higher voltage
test_resistor(V=5.0, R=1000.0, temperature=300.0, tnom=300.0, zeta=0.0, mfactor=1.0)

# Test 3: Different resistance
test_resistor(V=1.0, R=470.0, temperature=300.0, tnom=300.0, zeta=0.0, mfactor=1.0)

# Test 4: Temperature coefficient (zeta=1)
test_resistor(V=1.0, R=1000.0, temperature=350.0, tnom=300.0, zeta=1.0, mfactor=1.0)

# Test 5: Temperature coefficient (zeta=2)
test_resistor(V=1.0, R=1000.0, temperature=350.0, tnom=300.0, zeta=2.0, mfactor=1.0)

# Test 6: With multiplier
test_resistor(V=1.0, R=1000.0, temperature=300.0, tnom=300.0, zeta=0.0, mfactor=2.0)

# Test 7: Cold temperature
test_resistor(V=1.0, R=1000.0, temperature=250.0, tnom=300.0, zeta=1.0, mfactor=1.0)

print("\n" + "="*60)
print("JAX resistor tests complete!")
print("="*60)
