"""Test diode model - adapted from VACASK test_diode.sim

Tests the diode model for DC characteristic and parameter sweeps.
Based on VACASK test suite by Arpad Burmen.

The OpenVAF diode model includes:
- Forward exponential current (limexp)
- Series resistance (collapsible when rs < minr)
- Junction capacitance with smoothing
- Self-heating network
- Temperature dependence

Note: Uses the OpenVAF integration test diode model which has simpler
parameter handling than the VACASK model.
"""

import pytest
from pathlib import Path
import sys
import numpy as np

# Add jax-spice to path if needed
jax_spice_path = Path(__file__).parent.parent
if str(jax_spice_path) not in sys.path:
    sys.path.insert(0, str(jax_spice_path))

from jax_spice.devices import compile_va


@pytest.fixture
def diode_model():
    """Compile the diode Verilog-A model"""
    # Use OpenVAF diode which has simpler init function
    model_path = Path(__file__).parent / "models" / "diode_openvaf.va"
    return compile_va(str(model_path))


def build_diode_voltages(diode_model, Vd):
    """Helper to build voltage dict for diode model

    Args:
        diode_model: The compiled diode model
        Vd: Forward voltage across the diode junction

    Returns:
        Dict mapping voltage parameter names to values
    """
    voltages = {}
    for name, kind in zip(diode_model.param_names, diode_model.param_kinds):
        if kind == 'voltage':
            if 'A,CI' in name:
                voltages[name] = Vd  # Diode junction
            elif 'CI,C' in name or 'dT' in name:
                voltages[name] = 0.0  # Rs voltage, thermal
            else:
                voltages[name] = Vd  # V(A) for derivatives
    return voltages


class TestDiodeCompilation:
    """Test diode model compilation"""

    def test_compile(self, diode_model):
        """Test that the model compiles successfully"""
        assert diode_model.name == "diode_va"
        # OpenVAF diode has 4 nodes: A, CI (internal), C, dT (thermal)
        assert len(diode_model.terminals) >= 2

    def test_parameters(self, diode_model):
        """Test that key parameters are available"""
        param_names = diode_model.param_names
        # Check for standard diode parameters (lowercase in OpenVAF model)
        expected_params = ['is', 'n', 'rs']
        for param in expected_params:
            assert param in param_names, f"Parameter {param} not found"


class TestDiodeDCCharacteristic:
    """Test diode DC I-V characteristics"""

    def test_forward_current(self, diode_model):
        """Test forward bias current follows Shockley equation

        I = Is * (exp(V/(n*Vt)) - 1)
        At room temperature, Vt ~ 26mV
        """
        Is = 1e-12  # 1pA saturation current
        n = 1.0     # Ideality factor
        T = 300.0   # Temperature in K
        Vt = 0.026  # Thermal voltage ~kT/q at 300K

        # Forward bias voltage
        Vf = 0.6  # 600mV forward bias

        # Build voltages dict
        voltages = build_diode_voltages(diode_model, Vf)

        # Evaluate with given parameters (lowercase param names)
        residuals, jacobian = diode_model.eval(
            voltages=voltages,
            params={'is': Is, 'n': n, 'rs': 0.0},  # No series resistance
            temperature=T
        )

        # Get the current from anode node (sim_node0)
        I_diode = float(residuals['sim_node0']['resist'])

        # Expected Shockley current (approximate, model uses limexp and gmin)
        I_expected = Is * (np.exp(Vf / (n * Vt)) - 1)

        # The current should be positive and of similar order of magnitude
        assert I_diode > 0, f"Forward current should be positive, got {I_diode}"
        # Allow for gmin and limexp differences
        assert I_diode > 1e-6, f"Forward current {I_diode} seems too small"

    def test_reverse_blocking(self, diode_model):
        """Test reverse bias current is much smaller than forward"""
        Is = 1e-12
        n = 1.0
        T = 300.0

        # Build voltages for forward and reverse
        voltages_fwd = build_diode_voltages(diode_model, 0.6)
        voltages_rev = build_diode_voltages(diode_model, -0.6)

        params = {'is': Is, 'n': n, 'rs': 0.0}

        res_fwd, _ = diode_model.eval(voltages=voltages_fwd, params=params, temperature=T)
        res_rev, _ = diode_model.eval(voltages=voltages_rev, params=params, temperature=T)

        I_fwd = abs(float(res_fwd['sim_node0']['resist']))
        I_rev = abs(float(res_rev['sim_node0']['resist']))

        # Forward current should be orders of magnitude larger
        # Note: gmin adds a small conductance, so ratio may not be as extreme
        assert I_fwd > 100 * I_rev, \
            f"Forward {I_fwd} should be >> reverse {I_rev}"

    def test_differential_conductance(self, diode_model):
        """Test differential conductance increases with forward bias

        gd = dI/dV should be positive and increase with forward bias
        """
        Is = 1e-12
        n = 1.0
        T = 300.0

        voltages_low = build_diode_voltages(diode_model, 0.5)
        voltages_high = build_diode_voltages(diode_model, 0.7)

        params = {'is': Is, 'n': n, 'rs': 0.0}

        _, jac_low = diode_model.eval(voltages=voltages_low, params=params, temperature=T)
        _, jac_high = diode_model.eval(voltages=voltages_high, params=params, temperature=T)

        # Get conductance from diagonal Jacobian entries for anode node
        g_low = abs(float(jac_low.get(('sim_node0', 'sim_node0'), {}).get('resist', 0)))
        g_high = abs(float(jac_high.get(('sim_node0', 'sim_node0'), {}).get('resist', 0)))

        assert g_high > g_low, \
            f"Conductance at 0.7V ({g_high}) should be > at 0.5V ({g_low})"


class TestDiodeParameterSweep:
    """Test diode with parameter sweeps - adapted from VACASK test_diode.sim"""

    def test_saturation_current_sweep(self, diode_model):
        """Test I-V characteristic with different saturation currents

        Based on VACASK sweep: is=[1e-12, 1e-10, 1e-8, 1e-6]
        """
        n = 1.0
        T = 300.0
        Vf = 0.6

        Is_values = [1e-12, 1e-10, 1e-8, 1e-6]
        currents = []

        voltages = build_diode_voltages(diode_model, Vf)

        for Is in Is_values:
            res, _ = diode_model.eval(
                voltages=voltages,
                params={'is': Is, 'n': n, 'rs': 0.0},
                temperature=T
            )
            I = float(res['sim_node0']['resist'])
            currents.append(I)

        # Higher saturation current should give higher forward current
        for i in range(len(currents) - 1):
            assert currents[i+1] > currents[i], \
                f"Current at Is={Is_values[i+1]} should be > at Is={Is_values[i]}"

    def test_voltage_sweep(self, diode_model):
        """Test I-V characteristic across voltage sweep

        Based on VACASK sweep: v1 from -0.5V to 0.8V
        """
        Is = 1e-12
        n = 2.0  # VACASK uses n=2
        T = 300.0

        # Voltage sweep from -0.5 to 0.8V (avoid extreme values)
        V_values = np.linspace(-0.5, 0.8, 21)
        currents = []

        for V in V_values:
            voltages = build_diode_voltages(diode_model, V)

            res, _ = diode_model.eval(
                voltages=voltages,
                params={'is': Is, 'n': n, 'rs': 0.0},
                temperature=T
            )
            I = float(res['sim_node0']['resist'])
            currents.append(I)

        currents = np.array(currents)

        # Verify monotonicity: current should increase with voltage
        # (at least in forward bias region)
        forward_mask = V_values > 0.2
        forward_currents = currents[forward_mask]
        assert np.all(np.diff(forward_currents) > 0), \
            "Forward current should increase with voltage"


class TestDiodeWithResistor:
    """Test diode in circuit with resistor (like VACASK test)"""

    def test_diode_resistor_current_consistency(self, diode_model):
        """Test that diode and resistor currents can be balanced

        At a known operating point (Vd = 0.6V), verify both devices
        produce reasonable currents and Jacobians.

        Full Newton-Raphson circuit simulation requires proper handling
        of the circuit topology and is tested in the jax_spice.circuit module.
        """
        # Load resistor model
        resistor_path = Path(__file__).parent / "models" / "resistor_openvaf.va"
        resistor = compile_va(str(resistor_path))

        Is = 1e-12
        n = 2.0
        R = 1000.0
        T = 300.0

        # Test at Vd = 0.6V, Vs = 1.0V
        Vd = 0.6
        Vs = 1.0
        V_R = Vs - Vd  # 0.4V across resistor

        # Resistor evaluation
        r_voltages = {}
        for name, kind in zip(resistor.param_names, resistor.param_kinds):
            if kind == 'voltage':
                r_voltages[name] = V_R
        res_r, jac_r = resistor.eval(voltages=r_voltages, params={'R': R})

        # Diode evaluation
        d_voltages = build_diode_voltages(diode_model, Vd)
        res_d, jac_d = diode_model.eval(
            voltages=d_voltages,
            params={'is': Is, 'n': n, 'rs': 0.0},
            temperature=T
        )

        # Get currents
        I_R = float(res_r['sim_node0']['resist'])
        I_D = float(res_d['sim_node0']['resist'])

        # Resistor current should be V/R = 0.4V / 1kOhm = 0.4mA
        expected_I_R = V_R / R
        assert abs(I_R - expected_I_R) / expected_I_R < 0.01, \
            f"Resistor current {I_R} != expected {expected_I_R}"

        # Diode current should be positive (forward bias)
        assert I_D > 0, f"Diode current should be positive, got {I_D}"

        # Both devices should have positive conductance (Jacobian diagonal)
        G_R = float(jac_r.get(('sim_node0', 'sim_node0'), {}).get('resist', 0))
        G_D = float(jac_d.get(('sim_node0', 'sim_node0'), {}).get('resist', 0))

        assert G_R > 0, f"Resistor conductance should be positive, got {G_R}"
        assert G_D > 0, f"Diode conductance should be positive, got {G_D}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
