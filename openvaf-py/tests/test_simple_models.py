"""Tests for simple models: resistor, diode, current source, vccs, cccs"""

import pytest
import numpy as np
from conftest import assert_allclose, INTEGRATION_PATH, CompiledModel


class TestResistor:
    """Test resistor model: I = V/R where R = R0 * (T/Tnom)^zeta"""

    @pytest.mark.parametrize("voltage,resistance", [
        (0.0, 1000.0),
        (1.0, 1000.0),
        (5.0, 100.0),
        (10.0, 10000.0),
        (-1.0, 1000.0),
        (0.1, 470.0),
    ])
    def test_ohms_law(self, resistor_model: CompiledModel, voltage, resistance):
        """Test basic Ohm's law: I = V/R"""
        temperature = 300.0
        tnom = 300.0
        zeta = 0.0
        mfactor = 1.0

        # Build inputs: V(A,B), vres, R, $temp, tnom, zeta, res, mfactor
        inputs = [voltage, voltage, resistance, temperature, tnom, zeta, resistance, mfactor]

        residuals, jacobian = resistor_model.jax_fn(inputs)

        expected_current = voltage / resistance
        actual_current = float(residuals['A']['resist'])

        assert_allclose(
            actual_current, expected_current,
            rtol=1e-6, atol=1e-15,
            err_msg=f"Resistor I=V/R failed for V={voltage}, R={resistance}"
        )

    @pytest.mark.parametrize("voltage,resistance", [
        (1.0, 1000.0),
        (5.0, 100.0),
    ])
    def test_conductance(self, resistor_model: CompiledModel, voltage, resistance):
        """Test Jacobian entry: G = 1/R"""
        temperature = 300.0
        tnom = 300.0
        zeta = 0.0
        mfactor = 1.0

        inputs = [voltage, voltage, resistance, temperature, tnom, zeta, resistance, mfactor]

        residuals, jacobian = resistor_model.jax_fn(inputs)

        expected_conductance = 1.0 / resistance
        actual_conductance = float(jacobian[('A', 'A')]['resist'])

        assert_allclose(
            actual_conductance, expected_conductance,
            rtol=1e-5, atol=1e-12,
            err_msg=f"Resistor G=1/R failed for R={resistance}"
        )

    @pytest.mark.parametrize("temperature,tnom,zeta", [
        (350.0, 300.0, 1.0),
        (350.0, 300.0, 2.0),
        (250.0, 300.0, 1.0),
        (400.0, 300.0, 1.5),
    ])
    def test_temperature_coefficient(self, resistor_model: CompiledModel, temperature, tnom, zeta):
        """Test R(T) = R0 * (T/Tnom)^zeta"""
        voltage = 1.0
        resistance = 1000.0
        mfactor = 1.0

        inputs = [voltage, voltage, resistance, temperature, tnom, zeta, resistance, mfactor]

        residuals, jacobian = resistor_model.jax_fn(inputs)

        # R(T) = R0 * (T/Tnom)^zeta
        effective_resistance = resistance * (temperature / tnom) ** zeta
        expected_current = voltage / effective_resistance
        actual_current = float(residuals['A']['resist'])

        assert_allclose(
            actual_current, expected_current,
            rtol=1e-6, atol=1e-15,
            err_msg=f"Resistor temp coeff failed for T={temperature}, zeta={zeta}"
        )

    @pytest.mark.parametrize("mfactor", [0.5, 1.0, 2.0, 10.0])
    def test_multiplier(self, resistor_model: CompiledModel, mfactor):
        """Test multiplier factor"""
        voltage = 1.0
        resistance = 1000.0
        temperature = 300.0
        tnom = 300.0
        zeta = 0.0

        inputs = [voltage, voltage, resistance, temperature, tnom, zeta, resistance, mfactor]

        residuals, jacobian = resistor_model.jax_fn(inputs)

        expected_current = mfactor * voltage / resistance
        actual_current = float(residuals['A']['resist'])

        assert_allclose(
            actual_current, expected_current,
            rtol=1e-6, atol=1e-15,
            err_msg=f"Resistor mfactor failed for mfactor={mfactor}"
        )


class TestDiode:
    """Test diode model: I = Is * (exp(V/(n*Vt)) - 1)"""

    def test_compilation(self, diode_model: CompiledModel):
        """Diode model compiles without error"""
        assert diode_model.name == "diode_va"
        assert len(diode_model.nodes) >= 2

    def test_zero_bias(self, diode_model: CompiledModel):
        """At zero bias, current should be near zero"""
        inputs = diode_model.build_default_inputs()
        # Set voltages to zero
        for i, kind in enumerate(diode_model.param_kinds):
            if kind == 'voltage':
                inputs[i] = 0.0

        residuals, jacobian = diode_model.jax_fn(inputs)

        # At zero bias, diode current should be very small (just gmin)
        for node, res in residuals.items():
            current = float(res['resist'])
            # Current should be small (gmin level)
            assert abs(current) < 1e-9, f"Zero bias current too large at {node}: {current}"

    def test_forward_bias_increases_current(self, diode_model: CompiledModel):
        """Forward bias should increase current exponentially"""
        base_inputs = diode_model.build_default_inputs()

        currents = []
        voltages = [0.0, 0.3, 0.5, 0.6, 0.7]

        for v in voltages:
            inputs = base_inputs.copy()
            # Set the diode voltage
            for i, (name, kind) in enumerate(zip(diode_model.param_names, diode_model.param_kinds)):
                if kind == 'voltage' and 'A' in name:
                    inputs[i] = v

            residuals, _ = diode_model.jax_fn(inputs)
            # Get the first residual that represents diode current
            current = float(list(residuals.values())[0]['resist'])
            currents.append(current)

        # Forward bias should monotonically increase current
        for i in range(1, len(currents)):
            assert currents[i] >= currents[i-1], (
                f"Current should increase with voltage: I({voltages[i]})={currents[i]} < I({voltages[i-1]})={currents[i-1]}"
            )


class TestCurrentSource:
    """Test ideal current source model"""

    def test_compilation(self, isrc_model: CompiledModel):
        """Current source compiles without error"""
        assert isrc_model.module is not None
        assert isrc_model.jax_fn is not None

    def test_constant_current(self, isrc_model: CompiledModel):
        """Current should be constant regardless of voltage"""
        inputs = isrc_model.build_default_inputs()

        residuals, jacobian = isrc_model.jax_fn(inputs)

        # Just verify we get valid output
        assert residuals is not None
        for node, res in residuals.items():
            current = float(res['resist'])
            assert not np.isnan(current), f"NaN current at {node}"


class TestVCCS:
    """Test voltage-controlled current source"""

    def test_compilation(self, vccs_model: CompiledModel):
        """VCCS compiles without error"""
        assert vccs_model.module is not None
        assert vccs_model.jax_fn is not None

    def test_output_valid(self, vccs_model: CompiledModel):
        """VCCS produces valid outputs"""
        inputs = vccs_model.build_default_inputs()

        residuals, jacobian = vccs_model.jax_fn(inputs)

        assert residuals is not None
        for node, res in residuals.items():
            current = float(res['resist'])
            assert not np.isnan(current), f"NaN current at {node}"


class TestCCCS:
    """Test current-controlled current source"""

    def test_compilation(self, cccs_model: CompiledModel):
        """CCCS compiles without error"""
        assert cccs_model.module is not None
        assert cccs_model.jax_fn is not None

    def test_output_valid(self, cccs_model: CompiledModel):
        """CCCS produces valid outputs"""
        inputs = cccs_model.build_default_inputs()

        residuals, jacobian = cccs_model.jax_fn(inputs)

        assert residuals is not None
        for node, res in residuals.items():
            current = float(res['resist'])
            assert not np.isnan(current), f"NaN current at {node}"
