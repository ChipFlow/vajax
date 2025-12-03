"""Test diode DC operating point analysis - adapted from VACASK test_diode.sim

This tests a nonlinear circuit with a diode, which requires Newton-Raphson
iteration to converge to the DC operating point.

Circuit:
    V1 -- R1 (1 Ohm) -- D1 (diode) -- GND

The VACASK test sweeps:
- Is from 1e-12 to 1e-6
- V1 from -50V to +10V

For this test, we use a single operating point to validate convergence
without any parameter tweaking.
"""

import pytest
import jax.numpy as jnp
import numpy as np

from jax_spice.devices.base import DeviceStamps
from jax_spice.analysis.mna import MNASystem, DeviceInfo
from jax_spice.analysis.dc import dc_operating_point


def resistor_eval(voltages, params, context):
    """Resistor evaluation function"""
    Vp = voltages.get('p', 0.0)
    Vn = voltages.get('n', 0.0)
    R = float(params.get('r', 1.0))
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
    """Voltage source evaluation function"""
    Vp = voltages.get('p', 0.0)
    Vn = voltages.get('n', 0.0)
    V_target = float(params.get('dc', 0.0))
    V_actual = Vp - Vn
    G_big = 1e12
    I = G_big * (V_actual - V_target)

    return DeviceStamps(
        currents={'p': jnp.array(I), 'n': jnp.array(-I)},
        conductances={
            ('p', 'p'): jnp.array(G_big), ('p', 'n'): jnp.array(-G_big),
            ('n', 'p'): jnp.array(-G_big), ('n', 'n'): jnp.array(G_big)
        }
    )


def diode_eval(voltages, params, context):
    """Diode evaluation function implementing Shockley equation

    I = Is * (exp(V/(n*Vt)) - 1)

    Uses limited exponential to prevent overflow and adds gmin for
    numerical stability.
    """
    Vp = voltages.get('p', 0.0)  # Anode
    Vn = voltages.get('n', 0.0)  # Cathode
    Vd = Vp - Vn

    # Parameters
    Is = float(params.get('is', 1e-12))
    n = float(params.get('n', 2.0))
    Rs = float(params.get('rs', 0.0))  # Series resistance (ignored for now)

    # Constants
    Vt = 0.02585  # Thermal voltage at 300K (kT/q)
    gmin = 1e-12  # Minimum conductance for stability

    # Limited exponential to prevent overflow
    Vd_max = 40 * n * Vt  # Limit for exp argument
    if Vd > Vd_max:
        # Linear extrapolation beyond Vd_max
        exp_max = np.exp(Vd_max / (n * Vt))
        Id = Is * (exp_max - 1) + Is * exp_max / (n * Vt) * (Vd - Vd_max)
        gd = Is * exp_max / (n * Vt)
    elif Vd < -40 * Vt:
        # Reverse bias: saturates to -Is
        Id = -Is
        gd = gmin
    else:
        # Normal operation
        exp_val = np.exp(Vd / (n * Vt))
        Id = Is * (exp_val - 1)
        gd = Is * exp_val / (n * Vt)

    # Add gmin for numerical stability
    Id = Id + gmin * Vd
    gd = gd + gmin

    return DeviceStamps(
        currents={'p': jnp.array(Id), 'n': jnp.array(-Id)},
        conductances={
            ('p', 'p'): jnp.array(gd), ('p', 'n'): jnp.array(-gd),
            ('n', 'p'): jnp.array(-gd), ('n', 'n'): jnp.array(gd)
        }
    )


class TestVACASKDiode:
    """Tests from VACASK test_diode.sim"""

    def test_diode_forward_bias(self):
        """Test diode in forward bias

        Circuit: V1(1V) -- R1(1 Ohm) -- D1 -- GND
        Expected: Diode should be forward biased with Vd ~ 0.6-0.7V
        """
        system = MNASystem(
            num_nodes=3,
            node_names={'0': 0, '1': 1, '2': 2}
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

        # R1: 1 Ohm from node 1 to node 2
        system.devices.append(DeviceInfo(
            name='r1',
            model_name='resistor',
            terminals=['p', 'n'],
            node_indices=[1, 2],
            params={'r': 1.0},
            eval_fn=resistor_eval
        ))

        # D1: diode from node 2 to ground (Is=1e-12, n=2)
        system.devices.append(DeviceInfo(
            name='d1',
            model_name='diode',
            terminals=['p', 'n'],
            node_indices=[2, 0],
            params={'is': 1e-12, 'n': 2.0},
            eval_fn=diode_eval
        ))

        # Solve
        solution, info = dc_operating_point(system)

        assert info['converged'], f"Did not converge: {info}"

        V1 = float(solution[1])
        V2 = float(solution[2])  # Diode voltage

        print(f"V(1) = {V1:.6f}V")
        print(f"V(2) = Vd = {V2:.6f}V")
        print(f"I = (V1 - Vd) / R = {(V1 - V2) / 1.0 * 1000:.3f} mA")
        print(f"Converged in {info['iterations']} iterations")

        # V1 should be close to 1.0V
        assert abs(V1 - 1.0) < 0.01, f"V(1) = {V1}, expected ~1.0V"

        # Diode voltage should be positive (forward bias)
        assert V2 > 0, f"Vd = {V2}, should be positive"

        # Diode voltage should be reasonable
        # For Is=1e-12, n=2, Vt=25.85mV, and I~0.25mA:
        # I = Is * exp(Vd/(n*Vt)) => Vd = n*Vt * ln(I/Is) = 2*0.02585*ln(0.25e-3/1e-12) = ~1.0V
        # So Vd close to 1V is expected for this low Is and n=2
        assert 0.3 < V2 < 1.1, f"Vd = {V2}, expected 0.3-1.1V"

    def test_diode_reverse_bias(self):
        """Test diode in reverse bias

        Circuit: V1(-1V) -- R1(1k) -- D1 -- GND
        Expected: Diode should be reverse biased with very small current
        """
        system = MNASystem(
            num_nodes=3,
            node_names={'0': 0, '1': 1, '2': 2}
        )

        system.devices.append(DeviceInfo(
            name='v1',
            model_name='vsource',
            terminals=['p', 'n'],
            node_indices=[1, 0],
            params={'dc': -1.0},  # Reverse bias
            eval_fn=vsource_eval
        ))

        system.devices.append(DeviceInfo(
            name='r1',
            model_name='resistor',
            terminals=['p', 'n'],
            node_indices=[1, 2],
            params={'r': 1000.0},  # 1k to limit current
            eval_fn=resistor_eval
        ))

        system.devices.append(DeviceInfo(
            name='d1',
            model_name='diode',
            terminals=['p', 'n'],
            node_indices=[2, 0],
            params={'is': 1e-12, 'n': 2.0},
            eval_fn=diode_eval
        ))

        solution, info = dc_operating_point(system)

        assert info['converged'], f"Did not converge: {info}"

        V1 = float(solution[1])
        V2 = float(solution[2])

        print(f"V(1) = {V1:.6f}V")
        print(f"V(2) = Vd = {V2:.6f}V")
        print(f"Converged in {info['iterations']} iterations")

        # V1 should be close to -1.0V
        assert abs(V1 - (-1.0)) < 0.01, f"V(1) = {V1}, expected ~-1.0V"

        # Diode should be reverse biased (negative voltage)
        assert V2 < 0, f"Vd = {V2}, should be negative (reverse bias)"

    def test_diode_saturation_current_sweep(self):
        """Test with different saturation currents (from VACASK Is sweep)

        VACASK sweeps Is = [1e-12, 1e-10, 1e-8, 1e-6]
        Higher Is should give higher diode voltage for same current.
        """
        Is_values = [1e-12, 1e-10, 1e-8, 1e-6]
        Vd_results = []

        for Is in Is_values:
            system = MNASystem(
                num_nodes=3,
                node_names={'0': 0, '1': 1, '2': 2}
            )

            system.devices.append(DeviceInfo(
                name='v1',
                model_name='vsource',
                terminals=['p', 'n'],
                node_indices=[1, 0],
                params={'dc': 1.0},
                eval_fn=vsource_eval
            ))

            system.devices.append(DeviceInfo(
                name='r1',
                model_name='resistor',
                terminals=['p', 'n'],
                node_indices=[1, 2],
                params={'r': 1.0},
                eval_fn=resistor_eval
            ))

            system.devices.append(DeviceInfo(
                name='d1',
                model_name='diode',
                terminals=['p', 'n'],
                node_indices=[2, 0],
                params={'is': Is, 'n': 2.0},
                eval_fn=diode_eval
            ))

            solution, info = dc_operating_point(system)

            assert info['converged'], f"Did not converge for Is={Is}: {info}"

            Vd = float(solution[2])
            Vd_results.append(Vd)

            print(f"Is = {Is:.0e}: Vd = {Vd:.4f}V")

        # Higher Is means lower Vd for same current
        # (same I = Is * exp(Vd/nVt), higher Is -> lower exp needed -> lower Vd)
        for i in range(len(Vd_results) - 1):
            assert Vd_results[i+1] < Vd_results[i], \
                f"Vd should decrease with Is: Vd[Is={Is_values[i]:.0e}]={Vd_results[i]:.4f} " \
                f"> Vd[Is={Is_values[i+1]:.0e}]={Vd_results[i+1]:.4f}"


class TestDiodeVoltageSweep:
    """Test diode with voltage sweep (like VACASK v1 sweep)"""

    def test_voltage_sweep_convergence(self):
        """Test that solver converges for various input voltages

        VACASK sweeps V1 from -50V to +10V
        We test a subset to validate convergence.
        """
        # Subset of VACASK sweep
        V_values = [-5.0, -1.0, 0.0, 0.5, 1.0, 2.0, 5.0, 10.0]

        for V in V_values:
            system = MNASystem(
                num_nodes=3,
                node_names={'0': 0, '1': 1, '2': 2}
            )

            system.devices.append(DeviceInfo(
                name='v1',
                model_name='vsource',
                terminals=['p', 'n'],
                node_indices=[1, 0],
                params={'dc': V},
                eval_fn=vsource_eval
            ))

            system.devices.append(DeviceInfo(
                name='r1',
                model_name='resistor',
                terminals=['p', 'n'],
                node_indices=[1, 2],
                params={'r': 1.0},
                eval_fn=resistor_eval
            ))

            system.devices.append(DeviceInfo(
                name='d1',
                model_name='diode',
                terminals=['p', 'n'],
                node_indices=[2, 0],
                params={'is': 1e-12, 'n': 2.0},
                eval_fn=diode_eval
            ))

            solution, info = dc_operating_point(system)

            # Must converge for all voltages without tweaking
            assert info['converged'], \
                f"Did not converge for V1={V}V: {info}"

            V1 = float(solution[1])
            Vd = float(solution[2])
            I = (V1 - Vd) / 1.0  # Current through circuit

            print(f"V1={V:+6.1f}V: Vd={Vd:+.4f}V, I={I*1000:+.4f}mA, " \
                  f"iter={info['iterations']}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
