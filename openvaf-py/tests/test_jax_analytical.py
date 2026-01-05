"""Tests comparing JAX output against analytical formulas

This mirrors OpenVAF's integration test approach: use specific physical
constants and verify the computed residuals/Jacobians match analytical formulas.

These tests use the same methodology as:
  openvaf/openvaf/tests/integration.rs
  openvaf/openvaf/tests/mock_sim/mod.rs
"""

import pytest
import numpy as np
from math import exp, sqrt, log, pow

from conftest import (
    INTEGRATION_PATH, assert_allclose,
    CompiledModel
)


# Physical constants (same as OpenVAF tests)
KB = 1.3806488e-23      # Boltzmann constant [J/K]
Q = 1.602176565e-19     # Elementary charge [C]
T_ROOM = 300.0          # Room temperature [K]
VT = KB * T_ROOM / Q    # Thermal voltage at 300K (~25.85mV)


class TestResistorAnalytical:
    """Test resistor model against Ohm's law"""

    @pytest.fixture
    def resistor(self, compile_model):
        return compile_model(INTEGRATION_PATH / "RESISTOR/resistor.va")

    @pytest.mark.parametrize("voltage,resistance", [
        (1.0, 1000.0),
        (0.5, 500.0),
        (-1.0, 1000.0),
        (0.01, 100.0),
    ])
    def test_ohms_law_current(self, resistor, voltage, resistance):
        """I = V/R"""
        # JAX inputs: [V(A,B), vres, R, $temperature, tnom, zeta, res, mfactor]
        jax_inputs = [voltage, voltage, resistance, T_ROOM, T_ROOM, 0.0, resistance, 1.0]

        jax_residuals, _ = resistor.jax_fn(jax_inputs)

        expected_current = voltage / resistance
        actual_current = float(jax_residuals['A']['resist'])

        assert_allclose(actual_current, expected_current, rtol=1e-6,
                       err_msg=f"V={voltage}, R={resistance}")

    @pytest.mark.parametrize("voltage,resistance", [
        (1.0, 1000.0),
        (5.0, 100.0),
    ])
    def test_ohms_law_conductance(self, resistor, voltage, resistance):
        """dI/dV = 1/R (Jacobian)"""
        jax_inputs = [voltage, voltage, resistance, T_ROOM, T_ROOM, 0.0, resistance, 1.0]

        _, jax_jacobian = resistor.jax_fn(jax_inputs)

        expected_conductance = 1.0 / resistance
        actual_conductance = float(jax_jacobian[('A', 'A')]['resist'])

        assert_allclose(actual_conductance, expected_conductance, rtol=1e-6,
                       err_msg=f"R={resistance}")

    @pytest.mark.parametrize("temperature,zeta", [
        (T_ROOM, 0.0),      # No temp dependence
        (350.0, 1.0),       # Linear: R * (T/Tnom)^1
        (400.0, 2.0),       # Quadratic: R * (T/Tnom)^2
    ])
    def test_temperature_scaling(self, resistor, temperature, zeta):
        """R_eff = R * (T/Tnom)^zeta"""
        voltage = 1.0
        R_nominal = 1000.0
        tnom = T_ROOM

        # Effective resistance at temperature
        R_eff = R_nominal * pow(temperature / tnom, zeta)
        expected_current = voltage / R_eff

        jax_inputs = [voltage, voltage, R_nominal, temperature, tnom, zeta, R_eff, 1.0]

        jax_residuals, _ = resistor.jax_fn(jax_inputs)
        actual_current = float(jax_residuals['A']['resist'])

        assert_allclose(actual_current, expected_current, rtol=1e-6,
                       err_msg=f"T={temperature}, zeta={zeta}")


class TestCurrentSourceAnalytical:
    """Test current source model"""

    @pytest.fixture
    def isrc(self, compile_model):
        return compile_model(INTEGRATION_PATH / "CURRENT_SOURCE/current_source.va")

    def test_current_source_compiles(self, isrc):
        """Current source compiles and produces output"""
        jax_inputs = isrc.build_default_inputs()
        jax_residuals, jax_jacobian = isrc.jax_fn(jax_inputs)

        assert isinstance(jax_residuals, dict)
        assert len(jax_residuals) >= 2
