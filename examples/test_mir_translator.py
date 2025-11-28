"""Test the MIR-to-JAX translator with a resistor model"""

import sys
sys.path.insert(0, '/Users/roberttaylor/Code/ChipFlow/reference/jax-spice')

from jax_spice.mir.device import create_resistor_device

def test_resistor():
    """Test that the JAX resistor produces correct results"""
    print("Creating JAX resistor device from MIR...")
    resistor = create_resistor_device(R=1000.0, tnom=300.15, zeta=0.0)
    print(f"  Device: {resistor.name}")
    print(f"  Terminals: {resistor.terminals}")
    print(f"  Parameters: {resistor.parameters}")

    print("\nTesting with different voltages:")
    for v in [0.0, 1.0, 5.0, 10.0]:
        voltages = {'node0': v, 'node1': 0.0}
        residuals, jacobian = resistor.eval(voltages)

        # Extract values
        i_node0 = residuals['sim_node0']['resist']
        i_node1 = residuals['sim_node1']['resist']
        g_00 = jacobian[('sim_node0', 'sim_node0')]['resist']
        g_01 = jacobian[('sim_node0', 'sim_node1')]['resist']

        # Expected: I = V/R = v/1000, G = 1/R = 0.001
        expected_i = v / 1000.0
        expected_g = 1.0 / 1000.0

        print(f"  V = {v}V:")
        print(f"    I(node0) = {i_node0:.6f} A (expected {expected_i:.6f})")
        print(f"    I(node1) = {i_node1:.6f} A (expected {-expected_i:.6f})")
        print(f"    G(0,0) = {g_00:.6f} S (expected {expected_g:.6f})")
        print(f"    G(0,1) = {g_01:.6f} S (expected {-expected_g:.6f})")

        # Verify
        assert abs(i_node0 - expected_i) < 1e-10, f"I(node0) mismatch: {i_node0} != {expected_i}"
        assert abs(i_node1 + expected_i) < 1e-10, f"I(node1) mismatch: {i_node1} != {-expected_i}"
        assert abs(g_00 - expected_g) < 1e-10, f"G(0,0) mismatch: {g_00} != {expected_g}"
        assert abs(g_01 + expected_g) < 1e-10, f"G(0,1) mismatch: {g_01} != {-expected_g}"

    print("\n✓ All basic tests passed!")

    # Test temperature coefficient
    print("\nTesting temperature coefficient (zeta):")
    resistor_tc = create_resistor_device(R=1000.0, tnom=300.15, zeta=1.0)
    resistor_tc.set_parameters(tnom=300.15)

    # At T=300.15K (tnom), R should be 1000
    voltages = {'node0': 1.0, 'node1': 0.0}
    residuals, _ = resistor_tc.eval(voltages, temperature=300.15)
    i_at_tnom = residuals['sim_node0']['resist']
    print(f"  At T=tnom (300.15K): I = {i_at_tnom:.6f} A (expected 0.001)")

    # At T=330K, R = 1000 * (330/300.15)^1 = 1099.5
    residuals, _ = resistor_tc.eval(voltages, temperature=330.0)
    i_at_330 = residuals['sim_node0']['resist']
    expected_r_330 = 1000.0 * (330.0 / 300.15) ** 1.0
    expected_i_330 = 1.0 / expected_r_330
    print(f"  At T=330K: I = {i_at_330:.6f} A (expected {expected_i_330:.6f})")
    print(f"    R = {1.0/i_at_330:.1f} ohms (expected {expected_r_330:.1f})")

    assert abs(i_at_330 - expected_i_330) < 1e-10, f"TC mismatch"

    print("\n✓ Temperature coefficient test passed!")

    # Test get_stamps interface
    print("\nTesting get_stamps interface:")
    node_indices = {'node0': 0, 'node1': 1}
    voltages = {'node0': 5.0, 'node1': 0.0}
    G_stamps, I_stamps = resistor.get_stamps(node_indices, voltages)

    print(f"  G_stamps: {G_stamps}")
    print(f"  I_stamps: {I_stamps}")

    # Check stamp structure
    assert (0, 0) in G_stamps, "Missing G(0,0)"
    assert (0, 1) in G_stamps, "Missing G(0,1)"
    assert (1, 0) in G_stamps, "Missing G(1,0)"
    assert (1, 1) in G_stamps, "Missing G(1,1)"
    assert 0 in I_stamps, "Missing I(0)"
    assert 1 in I_stamps, "Missing I(1)"

    print("\n✓ get_stamps test passed!")

    print("\n" + "="*50)
    print("All MIR-to-JAX translator tests passed!")
    print("="*50)


if __name__ == '__main__':
    test_resistor()
