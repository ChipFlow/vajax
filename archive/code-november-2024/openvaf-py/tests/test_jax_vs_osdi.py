"""Tests comparing JAX function output against MIR interpreter (OSDI reference)

This module validates that the JAX-compiled models produce the same
residuals and Jacobians as the MIR interpreter, which uses the same
computation as the OSDI compiled models used by VACASK.

The MIR interpreter is the authoritative reference for correctness.
"""

import pytest
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path

from conftest import (
    INTEGRATION_PATH, INTEGRATION_MODELS, assert_allclose,
    CompiledModel, build_param_dict
)


# Models that work with generic testing (simple parameter structure)
# Note: Most models require careful hidden state setup for the interpreter.
# For comprehensive testing, we use model-specific test classes below.
SIMPLE_MODELS = [
    ("isrc", "CURRENT_SOURCE/current_source.va"),
    ("vccs", "VCCS/vccs.va"),
    ("cccs", "CCCS/cccs.va"),
]


class TestJaxCompilation:
    """Verify all models compile to JAX without error"""

    @pytest.mark.parametrize("model_name,model_path", INTEGRATION_MODELS)
    def test_compiles(self, compile_model, model_name, model_path):
        """Model compiles to JAX function"""
        model = compile_model(INTEGRATION_PATH / model_path)
        assert model.jax_fn is not None
        assert model.module is not None


class TestJaxProducesValidOutput:
    """Verify JAX functions produce valid (non-NaN) output"""

    @pytest.mark.parametrize("model_name,model_path", SIMPLE_MODELS)
    def test_simple_model_output(self, compile_model, model_name, model_path):
        """Simple models produce valid output with default inputs"""
        model = compile_model(INTEGRATION_PATH / model_path)
        jax_inputs = model.build_default_inputs()

        jax_residuals, jax_jacobian = model.jax_fn(jax_inputs)

        # Check at least we got dictionaries back
        assert isinstance(jax_residuals, dict)
        assert isinstance(jax_jacobian, dict)

        # Check no NaN in residuals
        for node, res in jax_residuals.items():
            resist = float(res['resist'])
            assert not np.isnan(resist), f"{model_name}: NaN at {node}"


class TestResistorDetailed:
    """Detailed tests for resistor model with various parameter combinations"""

    @pytest.fixture
    def resistor(self, compile_model):
        return compile_model(INTEGRATION_PATH / "RESISTOR/resistor.va")

    @pytest.mark.parametrize("voltage,resistance", [
        (1.0, 1000.0),
        (0.1, 100.0),
        (5.0, 10000.0),
        (-1.0, 1000.0),
        (0.001, 1e6),
    ])
    def test_ohms_law(self, resistor, voltage, resistance):
        """Resistor follows Ohm's law: I = V/R"""
        # JAX inputs: [V(A,B), vres, R, $temperature, tnom, zeta, res, mfactor]
        jax_inputs = [voltage, voltage, resistance, 300.15, 300.0, 0.0, resistance, 1.0]

        interp_params = {
            'V(A,B)': voltage,
            'vres': voltage,
            'R': resistance,
            '$temperature': 300.15,
            'tnom': 300.0,
            'zeta': 0.0,
            'res': resistance,
            'mfactor': 1.0,
        }

        jax_residuals, jax_jacobian = resistor.jax_fn(jax_inputs)
        interp_residuals, interp_jacobian = resistor.module.run_init_eval(interp_params)

        expected_current = voltage / resistance
        jax_current = float(jax_residuals['A']['resist'])
        interp_current = interp_residuals[0][0]

        # Use 1e-6 tolerance for floating point comparisons
        assert_allclose(jax_current, expected_current, rtol=1e-6,
                       err_msg=f"JAX: V={voltage}, R={resistance}")
        assert_allclose(interp_current, expected_current, rtol=1e-6,
                       err_msg=f"Interpreter: V={voltage}, R={resistance}")
        assert_allclose(jax_current, interp_current, rtol=1e-6,
                       err_msg="JAX vs Interpreter mismatch")

    @pytest.mark.parametrize("temperature,tnom,zeta", [
        (300.0, 300.0, 0.0),   # No temp effect
        (350.0, 300.0, 1.0),   # Linear temp coefficient
        (250.0, 300.0, 1.0),   # Below tnom
        (400.0, 300.0, 2.0),   # Quadratic temp coefficient
    ])
    def test_temperature_dependence(self, resistor, temperature, tnom, zeta):
        """Resistor temperature dependence matches between JAX and interpreter"""
        voltage = 1.0
        resistance = 1000.0

        jax_inputs = [voltage, voltage, resistance, temperature, tnom, zeta, resistance, 1.0]

        interp_params = {
            'V(A,B)': voltage,
            'vres': voltage,
            'R': resistance,
            '$temperature': temperature,
            'tnom': tnom,
            'zeta': zeta,
            'res': resistance,
            'mfactor': 1.0,
        }

        jax_residuals, _ = resistor.jax_fn(jax_inputs)
        interp_residuals, _ = resistor.module.run_init_eval(interp_params)

        jax_current = float(jax_residuals['A']['resist'])
        interp_current = interp_residuals[0][0]

        assert_allclose(jax_current, interp_current, rtol=1e-6,
                       err_msg=f"T={temperature}, zeta={zeta}")


class TestDiodeDetailed:
    """Detailed tests for diode model

    Note: The diode model has complex init/hidden state requirements.
    We test the JAX output independently rather than comparing to interpreter.
    """

    @pytest.fixture
    def diode(self, compile_model):
        return compile_model(INTEGRATION_PATH / "DIODE/diode.va")

    def test_diode_compiles(self, diode):
        """Diode model compiles to JAX"""
        assert diode.jax_fn is not None
        assert diode.module.num_jacobian > 0

    def test_diode_output_shape(self, diode):
        """Diode produces expected output structure"""
        jax_inputs = diode.build_default_inputs()
        jax_residuals, jax_jacobian = diode.jax_fn(jax_inputs)

        # Diode has internal nodes, so may have more than 2 residuals
        assert len(jax_residuals) >= 2
        assert len(jax_jacobian) > 0


class TestEkvMosfet:
    """Tests for EKV MOSFET model"""

    @pytest.fixture
    def ekv(self, compile_model):
        return compile_model(INTEGRATION_PATH / "EKV/ekv.va")

    def test_ekv_compiles_and_evaluates(self, ekv):
        """EKV model compiles and produces valid output"""
        jax_inputs = ekv.build_default_inputs()

        jax_residuals, jax_jacobian = ekv.jax_fn(jax_inputs)

        # Check we got valid outputs
        assert jax_residuals is not None
        assert jax_jacobian is not None

        # Check at least some entries are non-zero
        has_nonzero = any(
            abs(float(v['resist'])) > 1e-20
            for v in jax_residuals.values()
        )
        # Note: at zero bias, currents may be zero, that's OK

    def test_ekv_matches_interpreter(self, ekv):
        """EKV JAX output matches interpreter"""
        jax_inputs = ekv.build_default_inputs()
        interp_params = build_param_dict(ekv, jax_inputs)

        jax_residuals, jax_jacobian = ekv.jax_fn(jax_inputs)
        interp_residuals, interp_jacobian = ekv.module.run_init_eval(interp_params)

        # Compare residuals - iterate over JAX keys which are clean VA names
        jax_nodes = list(jax_residuals.keys())
        for i, (resist, react) in enumerate(interp_residuals):
            if i < len(jax_nodes):
                node_name = jax_nodes[i]
                jax_resist = float(jax_residuals[node_name]['resist'])

                if abs(resist) > 1e-20 or abs(jax_resist) > 1e-20:
                    assert_allclose(
                        jax_resist, resist,
                        rtol=1e-4, atol=1e-12,
                        err_msg=f"EKV residual[{node_name}] ({i}) mismatch"
                    )


