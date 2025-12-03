"""Test DC operating point analysis - adapted from VACASK test_op.sim

This is a direct port of VACASK's test_op.sim which tests a simple
resistor divider circuit.

Circuit:
    V1 (10V) -- node1 -- R1 (1kOhm) -- node2 -- R2 (9kOhm) -- GND

Expected:
    V(1) = 10V (fixed by voltage source)
    V(2) = 10V * 9k / (1k + 9k) = 9V (voltage divider)

This test does NOT use any tweaked parameters - it uses the default
solver settings to validate basic functionality.
"""

import pytest
import jax.numpy as jnp

from jax_spice.devices.base import DeviceStamps
from jax_spice.analysis.mna import MNASystem, DeviceInfo
from jax_spice.analysis.dc import dc_operating_point


def resistor_eval(voltages, params, context):
    """Resistor evaluation function matching VACASK resistor.va"""
    Vp = voltages.get('p', 0.0)
    Vn = voltages.get('n', 0.0)
    R = float(params.get('r', 1000.0))

    # Ensure minimum resistance
    R = max(R, 1e-12)
    G = 1.0 / R
    I = G * (Vp - Vn)

    return DeviceStamps(
        currents={'p': jnp.array(I), 'n': jnp.array(-I)},
        conductances={
            ('p', 'p'): jnp.array(G), ('p', 'n'): jnp.array(-G),
            ('n', 'p'): jnp.array(-G), ('n', 'n'): jnp.array(G)
        }
    )


def vsource_eval(voltages, params, context):
    """Voltage source evaluation function

    Uses large conductance method (G_big) to force voltage.
    """
    Vp = voltages.get('p', 0.0)
    Vn = voltages.get('n', 0.0)
    V_target = float(params.get('dc', 0.0))
    V_actual = Vp - Vn

    # Large conductance to force voltage
    # Note: Using 1e12 matches SPICE convention but may cause conditioning issues
    # in large circuits. For small circuits like this, it's fine.
    G_big = 1e12
    I = G_big * (V_actual - V_target)

    return DeviceStamps(
        currents={'p': jnp.array(I), 'n': jnp.array(-I)},
        conductances={
            ('p', 'p'): jnp.array(G_big), ('p', 'n'): jnp.array(-G_big),
            ('n', 'p'): jnp.array(-G_big), ('n', 'n'): jnp.array(G_big)
        }
    )


class TestVACASKOperatingPoint:
    """Tests from VACASK test_op.sim"""

    def test_resistor_divider(self):
        """Test voltage divider from VACASK test_op.sim

        Circuit: V1(10V) -- R1(1k) -- node2 -- R2(9k) -- GND
        Expected: V(1) = 10V, V(2) = 9V
        """
        # Create MNA system
        # Nodes: 0=GND, 1=V1_positive, 2=mid
        system = MNASystem(
            num_nodes=3,
            node_names={'0': 0, '1': 1, '2': 2}
        )

        # V1: 10V from node 1 to ground
        system.devices.append(DeviceInfo(
            name='v1',
            model_name='vsource',
            terminals=['p', 'n'],
            node_indices=[1, 0],
            params={'dc': 10.0},
            eval_fn=vsource_eval
        ))

        # R1: 1kOhm from node 1 to node 2
        system.devices.append(DeviceInfo(
            name='r1',
            model_name='resistor',
            terminals=['p', 'n'],
            node_indices=[1, 2],
            params={'r': 1000.0},
            eval_fn=resistor_eval
        ))

        # R2: 9kOhm from node 2 to ground
        system.devices.append(DeviceInfo(
            name='r2',
            model_name='resistor',
            terminals=['p', 'n'],
            node_indices=[2, 0],
            params={'r': 9000.0},
            eval_fn=resistor_eval
        ))

        # Solve without tweaking any parameters
        solution, info = dc_operating_point(system)

        # Check convergence
        assert info['converged'], f"Did not converge: {info}"

        # Extract voltages
        V1 = float(solution[1])  # Node 1
        V2 = float(solution[2])  # Node 2

        # Expected values from VACASK
        V1_expected = 10.0
        V2_expected = 10.0 * 9000.0 / (1000.0 + 9000.0)  # 9.0V

        # Check V(1) = 10V (relative tolerance 0.1%)
        rel_err_V1 = abs(V1 - V1_expected) / V1_expected
        assert rel_err_V1 < 1e-3, f"V(1)={V1}, expected={V1_expected}, rel_err={rel_err_V1}"

        # Check V(2) = 9V (relative tolerance 0.1%)
        rel_err_V2 = abs(V2 - V2_expected) / V2_expected
        assert rel_err_V2 < 1e-3, f"V(2)={V2}, expected={V2_expected}, rel_err={rel_err_V2}"

        print(f"V(1) = {V1:.6f}V (expected {V1_expected}V)")
        print(f"V(2) = {V2:.6f}V (expected {V2_expected}V)")
        print(f"Converged in {info['iterations']} iterations")


class TestVACASKResistor:
    """Tests from VACASK test_resistor.sim"""

    def test_single_resistor_current(self):
        """Test single resistor with voltage source

        Circuit: V1(1V) -- R1(2k) -- GND
        Expected: I(R1) = V/R = 1V / 2kOhm = 0.5mA
        """
        system = MNASystem(
            num_nodes=2,
            node_names={'0': 0, '1': 1}
        )

        # V1: 1V from node 1 to ground
        system.devices.append(DeviceInfo(
            name='v1',
            model_name='vsource',
            terminals=['p', 'n'],
            node_indices=[1, 0],
            params={'dc': 1.0},
            eval_fn=vsource_eval
        ))

        # R1: 2kOhm from node 1 to ground
        system.devices.append(DeviceInfo(
            name='r1',
            model_name='resistor',
            terminals=['p', 'n'],
            node_indices=[1, 0],
            params={'r': 2000.0},
            eval_fn=resistor_eval
        ))

        # Solve
        solution, info = dc_operating_point(system)

        assert info['converged'], f"Did not converge: {info}"

        # Check voltage
        V1 = float(solution[1])
        assert abs(V1 - 1.0) < 1e-3, f"V(1)={V1}, expected=1.0"

        # The current through R1 should be I = V/R = 1V / 2kOhm = 0.5mA
        expected_current = 1.0 / 2000.0  # 0.5mA

        # In VACASK, they check i(v1) = -1/2k * mfactor
        # The voltage source current equals the resistor current (KCL)
        # Since we don't track branch currents explicitly, we verify via voltage
        print(f"V(1) = {V1:.6f}V")
        print(f"Expected current through R1: {expected_current*1000:.3f} mA")
        print(f"Converged in {info['iterations']} iterations")


class TestMultipleOperatingPoints:
    """Test multiple operating points without parameter tweaking"""

    def test_various_voltage_dividers(self):
        """Test voltage dividers with various ratios

        Tests that the solver works for different R1/R2 ratios
        without needing any parameter tweaks.
        """
        test_cases = [
            (1000.0, 1000.0, 5.0),    # 50% divider -> 5V
            (1000.0, 9000.0, 9.0),    # 90% divider -> 9V
            (9000.0, 1000.0, 1.0),    # 10% divider -> 1V
            (100.0, 100.0, 5.0),      # Low resistance
            (100000.0, 100000.0, 5.0), # High resistance
        ]

        for R1, R2, V_expected in test_cases:
            system = MNASystem(
                num_nodes=3,
                node_names={'0': 0, '1': 1, '2': 2}
            )

            system.devices.append(DeviceInfo(
                name='v1',
                model_name='vsource',
                terminals=['p', 'n'],
                node_indices=[1, 0],
                params={'dc': 10.0},
                eval_fn=vsource_eval
            ))

            system.devices.append(DeviceInfo(
                name='r1',
                model_name='resistor',
                terminals=['p', 'n'],
                node_indices=[1, 2],
                params={'r': R1},
                eval_fn=resistor_eval
            ))

            system.devices.append(DeviceInfo(
                name='r2',
                model_name='resistor',
                terminals=['p', 'n'],
                node_indices=[2, 0],
                params={'r': R2},
                eval_fn=resistor_eval
            ))

            solution, info = dc_operating_point(system)

            assert info['converged'], \
                f"Did not converge for R1={R1}, R2={R2}: {info}"

            V2 = float(solution[2])
            rel_err = abs(V2 - V_expected) / V_expected

            assert rel_err < 1e-3, \
                f"R1={R1}, R2={R2}: V(2)={V2}, expected={V_expected}"

            print(f"R1={R1}, R2={R2}: V(2)={V2:.6f}V (expected {V_expected}V) - OK")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
