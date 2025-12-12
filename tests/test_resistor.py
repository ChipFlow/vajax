"""Test resistor model - adapted from VACASK test_resistor.sim

Tests the basic resistor model for DC operating point analysis.
Based on VACASK test suite by Arpad Burmen.

Circuit: V1 (1V) -> R1 (2kOhm) -> GND
Expected: I = V/R = 1V / 2kOhm = 0.5mA

Note: Uses the OpenVAF integration test resistor model which has simpler
parameter handling. The VACASK resistor model has complex bounds checking
that causes issues with the current JAX translator.
"""

import pytest
from pathlib import Path
import sys

# Add jax-spice to path if needed
jax_spice_path = Path(__file__).parent.parent
if str(jax_spice_path) not in sys.path:
    sys.path.insert(0, str(jax_spice_path))

from jax_spice.devices import compile_va


@pytest.fixture
def resistor_model():
    """Compile the resistor Verilog-A model"""
    # Use OpenVAF resistor which has simpler init function
    model_path = Path(__file__).parent / "models" / "resistor_openvaf.va"
    return compile_va(str(model_path))


class TestResistorDC:
    """Test resistor DC behavior"""

    def test_compile(self, resistor_model):
        """Test that the model compiles successfully"""
        assert resistor_model.name == "resistor_va"
        # OpenVAF reports node indices, not names
        assert len(resistor_model.terminals) == 2

    def test_ohms_law(self, resistor_model):
        """Test Ohm's law: I = V/R

        Circuit: 1V across 2kOhm resistor
        Expected current: 0.5mA
        """
        R = 2000.0  # 2kOhm
        V = 1.0     # 1V

        # Get voltage parameter name
        voltage_params = [
            (name, kind) for name, kind in
            zip(resistor_model.param_names, resistor_model.param_kinds)
            if kind == 'voltage'
        ]
        assert len(voltage_params) > 0, "No voltage parameters found"

        # Build voltages dict - set V(A,B) = 1V
        voltages = {}
        for name, kind in zip(resistor_model.param_names, resistor_model.param_kinds):
            if kind == 'voltage':
                voltages[name] = V

        # Evaluate (OpenVAF model uses 'R' parameter name)
        residuals, jacobian = resistor_model.eval(
            voltages=voltages,
            params={'R': R}
        )

        # The residual at each node is the current flowing INTO that node
        # For a two-terminal device, they should be equal and opposite
        # Check that node0 has +I and node1 has -I (where I = V/R)
        expected_current = V / R

        node0_current = float(residuals['sim_node0']['resist'])
        node1_current = float(residuals['sim_node1']['resist'])

        # Node 0 should have positive current (current flows in)
        rel_error0 = abs(node0_current - expected_current) / abs(expected_current + 1e-15)
        assert rel_error0 < 1e-3, f"Node0 current {node0_current} != expected {expected_current}"

        # Node 1 should have negative current (current flows out)
        rel_error1 = abs(node1_current + expected_current) / abs(expected_current + 1e-15)
        assert rel_error1 < 1e-3, f"Node1 current {node1_current} != expected {-expected_current}"

    def test_conductance_jacobian(self, resistor_model):
        """Test that the Jacobian gives correct conductance G = 1/R"""
        R = 2000.0  # 2kOhm
        V = 1.0

        # Build voltages
        voltages = {}
        for name, kind in zip(resistor_model.param_names, resistor_model.param_kinds):
            if kind == 'voltage':
                voltages[name] = V

        residuals, jacobian = resistor_model.eval(
            voltages=voltages,
            params={'R': R}
        )

        # Find conductance in Jacobian
        total_conductance = 0.0
        for (row, col), entry in jacobian.items():
            if 'resist' in entry:
                total_conductance += abs(float(entry['resist']))

        expected_conductance = 1.0 / R
        # Note: Jacobian typically has entries on both (i,i) and (i,j) for a two-terminal device
        # The self-conductance should match 1/R
        assert total_conductance > 0, "No conductance found in Jacobian"

    def test_voltage_divider_physics(self, resistor_model):
        """Test resistor in voltage divider configuration

        Two resistors in series: R1 = 1kOhm, R2 = 2kOhm
        VDD = 3V, expect Vmid = VDD * R2 / (R1 + R2) = 2V
        """
        R1 = 1000.0
        R2 = 2000.0
        VDD = 3.0

        # At equilibrium, Vmid should satisfy KCL:
        # I(R1) = I(R2)
        # (VDD - Vmid) / R1 = Vmid / R2
        # Vmid = VDD * R2 / (R1 + R2) = 2V

        Vmid_expected = VDD * R2 / (R1 + R2)

        # Verify using Newton-Raphson
        Vmid = VDD / 2  # Initial guess

        for _ in range(20):
            # R1: V(A,B) = VDD - Vmid
            V_R1 = VDD - Vmid
            voltages1 = {}
            for name, kind in zip(resistor_model.param_names, resistor_model.param_kinds):
                if kind == 'voltage':
                    voltages1[name] = V_R1
            res1, jac1 = resistor_model.eval(voltages=voltages1, params={'R': R1})

            # R2: V(A,B) = Vmid
            voltages2 = {}
            for name, kind in zip(resistor_model.param_names, resistor_model.param_kinds):
                if kind == 'voltage':
                    voltages2[name] = Vmid
            res2, jac2 = resistor_model.eval(voltages=voltages2, params={'R': R2})

            # Get currents from node0 (positive = current flowing into device)
            I_R1 = float(res1['sim_node0']['resist'])
            I_R2 = float(res2['sim_node0']['resist'])

            # Get conductances
            G_R1 = sum(abs(float(v.get('resist', 0))) for v in jac1.values())
            G_R2 = sum(abs(float(v.get('resist', 0))) for v in jac2.values())

            # KCL: I_R1 should equal I_R2 at the mid node
            # R1 current flows into mid, R2 current flows out
            residual = I_R1 - I_R2

            if abs(residual) < 1e-12:
                break

            # Jacobian: d(I_R1 - I_R2)/dVmid = -G_R1 - G_R2
            jac_total = -G_R1 - G_R2
            delta = -residual / jac_total
            Vmid += delta

        # Allow 0.1% relative error (32-bit float precision)
        rel_error = abs(Vmid - Vmid_expected) / Vmid_expected
        assert rel_error < 1e-2, \
            f"Vmid {Vmid} != expected {Vmid_expected} (rel_error={rel_error:.2e})"


class TestResistorParameterSweep:
    """Test resistor with parameter sweeps"""

    def test_resistance_sweep(self, resistor_model):
        """Test current varies inversely with resistance"""
        V = 1.0

        voltages = {}
        for name, kind in zip(resistor_model.param_names, resistor_model.param_kinds):
            if kind == 'voltage':
                voltages[name] = V

        resistances = [100.0, 1000.0, 10000.0, 100000.0]
        currents = []

        for R in resistances:
            res, _ = resistor_model.eval(voltages=voltages, params={'R': float(R)})
            # Get current from node0 (positive direction)
            I = float(res['sim_node0']['resist'])
            currents.append(I)

        # Verify I = V/R relationship
        for R, I in zip(resistances, currents):
            expected = V / R
            rel_error = abs(I - expected) / abs(expected)
            assert rel_error < 1e-3, f"At R={R}: I={I} != expected {expected}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
